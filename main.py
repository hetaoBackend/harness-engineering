#!/usr/bin/env python3
"""Entry point for the harness-engineered deep agent.

Providers are configured in providers.toml (project root) or
~/.config/harness-engineering/providers.toml.

Usage:
    # Use the default provider (first entry in providers.toml)
    uv run main.py "Write a prime-checker with tests"

    # Pick a specific provider by name
    uv run main.py "Fix the bug" --provider deepseek
    uv run main.py "Refactor auth.py" --provider anthropic

    # Override just the model for a run
    uv run main.py "..." --provider openai --model gpt-4o-mini

    # Use a custom config file
    uv run main.py "..." --config ~/my-providers.toml --provider local

    # Point at a specific project directory
    uv run main.py "Add docstrings" --cwd /path/to/myproject --provider deepseek
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig


async def run(
    task: str,
    cwd: str,
    provider_name: str,
    config_path: str | None,
    model_override: str | None,
    thread_id: str,
) -> None:
    from harness.agent import create_harness_agent
    from harness.config import get_provider

    cfg = get_provider(provider_name, config_path=config_path, model_override=model_override)
    cfg.check_api_key()

    agent = create_harness_agent(provider_config=cfg, cwd=cwd)

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"Task    : {task}")
    print(f"CWD     : {cwd}")
    print(f"Provider: {cfg.name}  ({cfg.type})")
    print(f"Model   : {cfg.model}")
    if cfg.base_url:
        print(f"Base URL: {cfg.base_url}")
    print(f"{'='*60}\n")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config=config,
    )

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            print("\n--- Agent Response ---")
            print(msg.content if isinstance(msg.content, str) else str(msg.content))
            break


def _default_provider(config_path: str | None) -> str:
    """Return the first provider name in the config file."""
    try:
        from harness.config import load_providers
        providers = load_providers(config_path)
        return next(iter(providers))
    except Exception:
        return "anthropic"


def main() -> None:
    # Do a quick pre-parse for --config so we can use it to populate the
    # --provider choices dynamically.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    try:
        from harness.config import load_providers
        providers = load_providers(pre_args.config)
        provider_names = sorted(providers)
        provider_default = next(iter(providers))
    except Exception:
        provider_names = None   # no restriction — validated at runtime
        provider_default = "anthropic"

    parser = argparse.ArgumentParser(
        description="Harness-engineered deep agent (TerminalBench-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task", help="Task for the agent to complete")
    parser.add_argument(
        "--provider",
        default=provider_default,
        choices=provider_names,
        metavar="NAME",
        help=(
            f"Provider name from providers.toml (default: {provider_default!r}). "
            + (f"Available: {', '.join(provider_names)}" if provider_names else "")
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model defined in providers.toml for this run",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to providers TOML config (default: ./providers.toml)",
    )
    parser.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory shown to the agent (default: current dir)",
    )
    parser.add_argument("--thread-id", default="harness-run", help="LangGraph thread ID")

    args = parser.parse_args()

    asyncio.run(
        run(
            task=args.task,
            cwd=args.cwd,
            provider_name=args.provider,
            config_path=args.config,
            model_override=args.model,
            thread_id=args.thread_id,
        )
    )


if __name__ == "__main__":
    main()
