"""ReasoningSandwichMiddleware: adaptive thinking budget allocation per phase.

Blog post finding: maximizing reasoning at every step scores *worse* (53.9%)
due to timeouts.  Instead, use a "reasoning sandwich":

  - Turn 1 (planning):      extended thinking
  - Turns 2..N-1 (impl):   standard thinking  (fast, cheap)
  - Final turn (verify):    extended thinking  (triggered by PreCompletionChecklist)

Detection heuristics:
- Turn 1 is always planning.
- If the latest HumanMessage contains the PreCompletionChecklist sentinel it is
  the verification turn.
- Everything else is implementation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage

from harness.middleware.pre_completion import _CHECKLIST_SENTINEL

# --------------------------------------------------------------------------- #
# Thinking budget presets (values are illustrative – tune to your model tier)  #
# --------------------------------------------------------------------------- #
THINKING_EXTENDED = {"type": "enabled", "budget_tokens": 10_000}
THINKING_STANDARD = {"type": "disabled"}


def _is_planning_turn(request: ModelRequest) -> bool:
    """True if this is the very first model call (no prior AI messages)."""
    from langchain_core.messages import AIMessage
    return not any(isinstance(m, AIMessage) for m in request.messages)


def _is_verification_turn(request: ModelRequest) -> bool:
    """True if the checklist sentinel is in the most recent human message."""
    for msg in reversed(request.messages):
        if isinstance(msg, HumanMessage):
            return _CHECKLIST_SENTINEL in str(msg.content)
    return False


class ReasoningSandwichMiddleware(AgentMiddleware):
    """Sets per-turn thinking budgets to balance quality vs. cost/latency."""

    def __init__(
        self,
        thinking_extended: dict | None = None,
        thinking_standard: dict | None = None,
    ) -> None:
        self.thinking_extended = thinking_extended or THINKING_EXTENDED
        self.thinking_standard = thinking_standard or THINKING_STANDARD

    def _choose_thinking(self, request: ModelRequest) -> dict:
        if _is_planning_turn(request) or _is_verification_turn(request):
            return self.thinking_extended
        return self.thinking_standard

    def _apply(self, request: ModelRequest) -> ModelRequest:
        thinking = self._choose_thinking(request)
        phase = (
            "planning" if _is_planning_turn(request)
            else "verification" if _is_verification_turn(request)
            else "implementation"
        )
        # Attach thinking via model_settings; langchain-anthropic passes these
        # as extra kwargs to the underlying API call.
        new_settings = {**request.model_settings, "thinking": thinking}
        # Optionally surface phase info for debugging
        new_settings["_harness_phase"] = phase
        return request.override(model_settings=new_settings)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._apply(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._apply(request))
