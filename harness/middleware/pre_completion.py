"""PreCompletionChecklistMiddleware: forces a verification pass before the agent exits.

When the model produces a response with no tool calls (about to finish), this
middleware intercepts and appends a verification checklist, prompting the agent
to re-check its work before declaring success.

This implements the blog post's "build-verify loop":
  planning → build (with tests) → verification → fix
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

# Sentinel so we don't run the checklist twice
_CHECKLIST_SENTINEL = "[[HARNESS:VERIFICATION_CHECKLIST]]"

VERIFICATION_CHECKLIST = f"""{_CHECKLIST_SENTINEL}

Before finishing, please complete this verification checklist:

1. **Run the tests**: Execute the test suite and confirm all tests pass.
2. **Check edge cases**: Have you handled error conditions and edge cases?
3. **Review requirements**: Re-read the original task. Did you fulfill every requirement?
4. **Code quality**: Is the code readable, well-structured, and free of obvious bugs?

If any check fails, fix the issue before finishing.
If all checks pass, respond with a brief summary of what you built and verified.
"""


class PreCompletionChecklistMiddleware(AgentMiddleware):
    """Intercepts no-tool-call responses and forces a verification pass.

    The checklist is injected exactly once per agent run (tracked by presence
    of _CHECKLIST_SENTINEL in message history).
    """

    def _checklist_already_run(self, request: ModelRequest) -> bool:
        return any(
            isinstance(m, HumanMessage) and _CHECKLIST_SENTINEL in str(m.content)
            for m in request.messages
        )

    def _agent_wants_to_stop(self, response: ModelResponse) -> bool:
        """Return True if the AI message has no tool calls."""
        for msg in response.result:
            if isinstance(msg, AIMessage):
                return not bool(getattr(msg, "tool_calls", None))
        return False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        if self._agent_wants_to_stop(response) and not self._checklist_already_run(request):
            checklist_request = request.override(
                messages=list(request.messages)
                + list(response.result)
                + [HumanMessage(content=VERIFICATION_CHECKLIST)],
            )
            return handler(checklist_request)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        response = await handler(request)
        if self._agent_wants_to_stop(response) and not self._checklist_already_run(request):
            checklist_request = request.override(
                messages=list(request.messages)
                + list(response.result)
                + [HumanMessage(content=VERIFICATION_CHECKLIST)],
            )
            return await handler(checklist_request)
        return response
