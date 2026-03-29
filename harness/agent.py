"""Factory for the harness-engineered deep agent.

Wires up all three middleware layers in the correct order:

    LocalContext → LoopDetection → PreCompletionChecklist → ReasoningSandwich

Ordering matters because deepagents composes middleware as a stack (first = outermost):
- LocalContext runs first so its env context is in place for all later layers.
- LoopDetection sits next to observe every tool call.
- PreCompletionChecklist intercepts stopping responses before reasoning decisions.
- ReasoningSandwich is innermost, closest to the model call.
"""

from __future__ import annotations

import os

from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph

from harness.middleware import (
    LocalContextMiddleware,
    LoopDetectionMiddleware,
    PreCompletionChecklistMiddleware,
    ReasoningSandwichMiddleware,
)

HARNESS_SYSTEM_PROMPT = """You are a capable autonomous coding agent.

Follow this four-phase workflow for every task:
1. **Plan** – understand the codebase and write a clear plan before touching any files.
2. **Build** – implement the plan; write tests alongside the code.
3. **Verify** – run the tests, fix failures, check edge cases.
4. **Report** – summarise what you built and how you verified it.

If you find yourself editing the same file repeatedly without progress, stop and
reconsider your approach from scratch.
"""


def create_harness_agent(
    model_name: str = "claude-sonnet-4-5-20250929",
    cwd: str | None = None,
    loop_threshold: int = 3,
    enable_reasoning_sandwich: bool = True,
) -> CompiledStateGraph:
    """Create a deep agent with harness engineering middleware.

    Args:
        model_name: Anthropic model identifier.
        cwd: Working directory for the agent (defaults to current directory).
        loop_threshold: Number of edits to the same file before doom-loop warning.
        enable_reasoning_sandwich: Toggle adaptive thinking budget allocation.

    Returns:
        Compiled LangGraph agent ready for ainvoke / astream.
    """
    model = ChatAnthropic(
        model_name=model_name,
        max_tokens=16_000,
        temperature=0,
    )

    middleware = [
        LocalContextMiddleware(cwd=cwd or os.getcwd()),
        LoopDetectionMiddleware(threshold=loop_threshold),
        PreCompletionChecklistMiddleware(),
    ]

    if enable_reasoning_sandwich:
        middleware.append(ReasoningSandwichMiddleware())

    return create_deep_agent(
        model=model,
        system_prompt=HARNESS_SYSTEM_PROMPT,
        middleware=middleware,
    )
