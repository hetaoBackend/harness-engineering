"""LocalContextMiddleware: proactively injects environment context into every model call.

Inspired by the blog post's description of proactively delivering:
- Directory structures and available tooling inventories
- Coding best practices
- Environment state (Python version, git status, etc.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from typing import Annotated, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


class LocalContextState(AgentState):
    env_context: NotRequired[Annotated[str, PrivateStateAttr]]


LOCAL_CONTEXT_TEMPLATE = """
<environment_context>
Working directory: {cwd}

Directory contents:
{dir_listing}

Available tools:
{tool_inventory}
</environment_context>
"""


def _detect_tools() -> str:
    """Detect available shell tools and runtimes."""
    lines: list[str] = []
    checks = {
        "python": ["python3", "--version"],
        "git": ["git", "--version"],
        "node": ["node", "--version"],
        "make": ["make", "--version"],
        "curl": ["curl", "--version"],
    }
    for name, cmd in checks.items():
        if shutil.which(cmd[0]):
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).splitlines()[0]
                lines.append(f"  - {name}: {out.strip()}")
            except Exception:
                lines.append(f"  - {name}: available")
        else:
            lines.append(f"  - {name}: NOT FOUND")
    return "\n".join(lines)


def _dir_listing(cwd: str, max_entries: int = 30) -> str:
    """List top-level directory contents."""
    try:
        entries = sorted(os.listdir(cwd))[:max_entries]
        return "\n".join(f"  {'[d]' if os.path.isdir(os.path.join(cwd, e)) else '[f]'} {e}" for e in entries)
    except Exception as exc:
        return f"  (could not list directory: {exc})"


class LocalContextMiddleware(AgentMiddleware):
    """Injects cwd + tool inventory into every model call system prompt.

    The context is gathered once before the agent starts and cached in state.
    """

    state_schema = LocalContextState

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd or os.getcwd()

    def before_agent(self, state: LocalContextState, runtime: Runtime, config: RunnableConfig) -> dict | None:
        if "env_context" in state:
            return None
        context = LOCAL_CONTEXT_TEMPLATE.format(
            cwd=self.cwd,
            dir_listing=_dir_listing(self.cwd),
            tool_inventory=_detect_tools(),
        )
        return {"env_context": context}

    async def abefore_agent(self, state: LocalContextState, runtime: Runtime, config: RunnableConfig) -> dict | None:
        return self.before_agent(state, runtime, config)

    def _inject(self, request: ModelRequest) -> ModelRequest:
        context: str = request.state.get("env_context", "")
        if not context:
            return request
        existing = request.system_message.content if request.system_message else ""
        merged = f"{context}\n\n{existing}" if existing else context
        return request.override(system_message=SystemMessage(content=merged))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._inject(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._inject(request))
