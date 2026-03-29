"""Factory for the harness-engineered deep agent.

Wires up all three middleware layers in the correct order:

    LocalContext → LoopDetection → PreCompletionChecklist → ReasoningSandwich

Ordering matters because deepagents composes middleware as a stack (first = outermost):
- LocalContext runs first so its env context is in place for all later layers.
- LoopDetection sits next to observe every tool call.
- PreCompletionChecklist intercepts stopping responses before reasoning decisions.
- ReasoningSandwich is innermost, closest to the model call (Anthropic only).

Provider selection
------------------
Set ``provider="anthropic"`` (default) to use ChatAnthropic, or
``provider="openai"`` to use ChatOpenAI with an OpenAI-compatible endpoint.

For OpenAI-compatible endpoints (e.g. DeepSeek, Together AI, local Ollama):

    create_harness_agent(
        provider="openai",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-...",
    )

Or set the ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` environment variables and
omit those arguments.

ReasoningSandwichMiddleware is automatically disabled for the ``"openai"``
provider because it relies on Anthropic-specific extended thinking parameters.
"""

from __future__ import annotations

import os
from typing import Literal

from deepagents import create_deep_agent
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
    provider: Literal["anthropic", "openai"] = "anthropic",
    base_url: str | None = None,
    api_key: str | None = None,
    cwd: str | None = None,
    loop_threshold: int = 3,
    enable_reasoning_sandwich: bool | None = None,
) -> CompiledStateGraph:
    """Create a deep agent with harness engineering middleware.

    Args:
        model_name: Model identifier (e.g. "claude-sonnet-4-5-20250929" or
            "gpt-4o" or any OpenAI-compatible model name).
        provider: ``"anthropic"`` (default) or ``"openai"`` for any
            OpenAI-compatible endpoint.
        base_url: Override the API base URL (OpenAI provider only).
            Falls back to the ``OPENAI_BASE_URL`` env var, then the default
            OpenAI endpoint.
        api_key: API key override.  Falls back to ``OPENAI_API_KEY`` or
            ``ANTHROPIC_API_KEY`` depending on provider.
        cwd: Working directory for the agent (defaults to current directory).
        loop_threshold: Number of edits to the same file before doom-loop warning.
        enable_reasoning_sandwich: Toggle adaptive thinking budget allocation.
            Defaults to ``True`` for Anthropic, ``False`` for OpenAI (because
            extended thinking is Anthropic-specific).

    Returns:
        Compiled LangGraph agent ready for ainvoke / astream.
    """
    model = _build_model(provider, model_name, base_url, api_key)

    # Default: reasoning sandwich only makes sense for Anthropic
    if enable_reasoning_sandwich is None:
        enable_reasoning_sandwich = provider == "anthropic"

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


def _build_model(
    provider: str,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
):
    """Instantiate the correct LangChain chat model for the given provider."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict = dict(
            model_name=model_name,
            max_tokens=16_000,
            temperature=0,
        )
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        kwargs = dict(
            model=model_name,
            max_tokens=16_000,
            temperature=0,
        )
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if resolved_api_key:
            kwargs["api_key"] = resolved_api_key
        return ChatOpenAI(**kwargs)

    raise ValueError(
        f"Unknown provider {provider!r}. Choose 'anthropic' or 'openai'."
    )
