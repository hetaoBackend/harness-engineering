"""Factory for the harness-engineered deep agent.

Middleware stack (first = outermost):

    LocalContext → LoopDetection → PreCompletionChecklist → ReasoningSandwich

Each layer can be individually toggled for ablation experiments:

    create_harness_agent(
        provider_config=cfg,
        enable_local_context=True,
        enable_loop_detection=False,   # ablation: remove loop detection
        enable_pre_completion=True,
        enable_reasoning_sandwich=None,  # auto: True for Anthropic, False otherwise
    )
"""

from __future__ import annotations

import os
from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
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
    # Working directory
    cwd: str | None = None,
    # Per-middleware toggles (all True by default)
    enable_local_context: bool = True,
    enable_loop_detection: bool = True,
    enable_pre_completion: bool = True,
    enable_reasoning_sandwich: bool | None = None,
    loop_threshold: int = 3,
) -> CompiledStateGraph:
    """Create a deep agent with harness engineering middleware.

    Args:
        provider_config: A ``ProviderConfig`` from providers.toml.  When set,
            ``provider``, ``base_url``, and ``api_key`` are ignored.
        model_name: Override the model (works with both provider_config and
            legacy kwargs).
        provider: Legacy — ``"anthropic"`` or ``"openai"``.
        base_url: Legacy — API endpoint override (openai only).
        api_key: Legacy — explicit API key.
        cwd: Working directory shown to the agent.
        enable_local_context: Inject cwd + tool inventory into system prompt.
        enable_loop_detection: Warn when the same file is edited repeatedly.
        enable_pre_completion: Force a verification pass before the agent exits.
        enable_reasoning_sandwich: Extended thinking on planning + verification
            turns.  Defaults to ``True`` for Anthropic, ``False`` otherwise.
        loop_threshold: Edits before doom-loop warning fires.
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

    middleware = []
    if enable_local_context:
        middleware.append(LocalContextMiddleware(cwd=cwd or os.getcwd()))
    if enable_loop_detection:
        middleware.append(LoopDetectionMiddleware(threshold=loop_threshold))
    if enable_pre_completion:
        middleware.append(PreCompletionChecklistMiddleware())
    if enable_reasoning_sandwich:
        middleware.append(ReasoningSandwichMiddleware())

    backend = FilesystemBackend(root_dir=cwd or os.getcwd(), virtual_mode=False)

    return create_deep_agent(
        model=model,
        system_prompt=HARNESS_SYSTEM_PROMPT,
        middleware=middleware,
        backend=backend,
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
