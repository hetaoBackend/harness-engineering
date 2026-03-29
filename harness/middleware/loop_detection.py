"""LoopDetectionMiddleware: tracks per-file edit counts to detect doom loops.

When an agent repeatedly edits the same file without making progress it is
stuck in a doom loop.  This middleware:
1. Counts edits per file via awrap_tool_call.
2. Injects a doom-loop warning into the system prompt when the threshold is
   exceeded, nudging the model to reconsider its approach.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

EDIT_TOOLS = {"edit_file", "write_file"}
LOOP_THRESHOLD = 3  # Warn after this many edits to a single file

DOOM_LOOP_WARNING = """
<doom_loop_warning>
⚠️  You have edited the following files multiple times without apparent progress:
{files}

This suggests you may be stuck in a doom loop.  Before making another edit:
1. Re-read the task requirements carefully.
2. Consider a fundamentally different approach.
3. Check whether the tests actually validate what you think they do.
</doom_loop_warning>
"""


class LoopState(AgentState):
    file_edit_counts: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]


class LoopDetectionMiddleware(AgentMiddleware):
    """Detects repeated edits to the same file and warns the model."""

    state_schema = LoopState

    def __init__(self, threshold: int = LOOP_THRESHOLD) -> None:
        self.threshold = threshold

    # --- tool-call hook: count edits ---

    def _update_counts(self, request: ToolCallRequest) -> dict[str, int] | None:
        tool_name = request.tool_call.get("name", "")
        if tool_name not in EDIT_TOOLS:
            return None
        file_path: str = request.tool_call.get("args", {}).get("path", "") or \
                         request.tool_call.get("args", {}).get("file_path", "unknown")
        counts: dict[str, int] = dict(request.state.get("file_edit_counts") or {})
        counts[file_path] = counts.get(file_path, 0) + 1
        return counts

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        counts = self._update_counts(request)
        result = handler(request)
        if counts is not None and isinstance(result, ToolMessage):
            # Attach updated counts to state via Command if possible; otherwise
            # store on the tool message metadata for the next before_model hook.
            result.additional_kwargs["_file_edit_counts"] = counts
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        counts = self._update_counts(request)
        result = await handler(request)
        if counts is not None and isinstance(result, ToolMessage):
            result.additional_kwargs["_file_edit_counts"] = counts
        return result

    # --- model-call hook: inject warning if looping ---

    def _extract_counts(self, request: ModelRequest) -> dict[str, int]:
        """Pull file edit counts from the most recent ToolMessages in history."""
        counts: dict[str, int] = dict(request.state.get("file_edit_counts") or {})
        for msg in reversed(request.messages):
            if hasattr(msg, "additional_kwargs"):
                stored = msg.additional_kwargs.get("_file_edit_counts")
                if stored:
                    # Merge – take the max per file
                    for k, v in stored.items():
                        counts[k] = max(counts.get(k, 0), v)
            break  # only look at the most recent tool message
        return counts

    def _maybe_inject_warning(self, request: ModelRequest) -> ModelRequest:
        counts = self._extract_counts(request)
        looping_files = [f for f, n in counts.items() if n >= self.threshold]
        if not looping_files:
            return request
        file_list = "\n".join(f"  - {f} ({counts[f]} edits)" for f in looping_files)
        warning = DOOM_LOOP_WARNING.format(files=file_list)
        existing = request.system_message.content if request.system_message else ""
        merged = f"{existing}\n\n{warning}" if existing else warning
        return request.override(system_message=SystemMessage(content=merged))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._maybe_inject_warning(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._maybe_inject_warning(request))
