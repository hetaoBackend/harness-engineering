"""Factory for the harness-engineered deep agent.

Wires up all four middleware layers in the correct order:

    LocalContext → LoopDetection → PreCompletionChecklist → ReasoningSandwich

Ordering matters because deepagents composes middleware as a stack (first = outermost).

Provider selection
------------------
Pass a ``ProviderConfig`` loaded from ``providers.toml``:

    from harness.config import get_provider
    from harness.agent import create_harness_agent

    cfg = get_provider("deepseek")
    agent = create_harness_agent(provider_config=cfg)

Or use the legacy keyword arguments directly:

    agent = create_harness_agent(provider="openai", model_name="gpt-4o",
                                 base_url="https://api.deepseek.com/v1")

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
    provider_config=None,
    *,
    # Legacy / override kwargs
    model_name: str | None = None,
    provider: Literal["anthropic", "openai"] = "anthropic",
    base_url: str | None = None,
    api_key: str | None = None,
    # Common options
    cwd: str | None = None,
    loop_threshold: int = 3,
    enable_reasoning_sandwich: bool | None = None,
) -> CompiledStateGraph:
    """Create a deep agent with harness engineering middleware.

    Args:
        provider_config: A ``ProviderConfig`` loaded from providers.toml.
            When provided, ``provider``, ``base_url``, and ``api_key`` are
            ignored (use ``model_name`` to override just the model).
        model_name: Override the model in ``provider_config``, or set the
            model when using legacy kwargs.
        provider: Legacy — ``"anthropic"`` or ``"openai"``.
        base_url: Legacy — API endpoint override (openai only).
        api_key: Legacy — explicit API key.
        cwd: Working directory shown to the agent (defaults to cwd).
        loop_threshold: Edits before doom-loop warning fires.
        enable_reasoning_sandwich: Defaults to ``True`` for Anthropic,
            ``False`` for OpenAI.
    """
    if provider_config is not None:
        _provider = provider_config.type
        _model = model_name or provider_config.model
        _base_url = provider_config.base_url
        _api_key = provider_config.resolved_api_key
    else:
        _provider = provider
        _model = model_name or (
            "claude-sonnet-4-5-20250929" if provider == "anthropic" else "gpt-4o"
        )
        _base_url = base_url
        _api_key = api_key

    model = _build_model(_provider, _model, _base_url, _api_key)

    if enable_reasoning_sandwich is None:
        enable_reasoning_sandwich = _provider == "anthropic"

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


def _build_model(provider: str, model_name: str, base_url: str | None, api_key: str | None):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs: dict = dict(model_name=model_name, max_tokens=16_000, temperature=0)
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        kwargs = dict(model=model_name, max_tokens=16_000, temperature=0)
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if resolved_api_key:
            kwargs["api_key"] = resolved_api_key
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unknown provider {provider!r}. Use 'anthropic' or 'openai'.")
