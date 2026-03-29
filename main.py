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

from langchain_core.runnables import RunnableConfig

_DIM = "\033[90m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


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

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": task}]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if isinstance(content, str):
                    print(content, end="", flush=True)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block["text"], end="", flush=True)

        elif kind == "on_chat_model_end":
            # Newline after each LLM turn
            print()

        elif kind == "on_tool_start":
            tool_name = event.get("name", "tool")
            inp = event["data"].get("input", {})
            # Show a brief one-liner so the user knows what's happening
            summary = _tool_summary(tool_name, inp)
            print(f"\n{_CYAN}▶ {tool_name}{_RESET}  {_DIM}{summary}{_RESET}", flush=True)

        elif kind == "on_tool_end":
            tool_name = event.get("name", "tool")
            output = event["data"].get("output", "")
            preview = str(output)[:120].replace("\n", " ")
            print(f"{_DIM}  ↳ {preview}{_RESET}", flush=True)


def _tool_summary(tool_name: str, inp: dict) -> str:
    """Return a short human-readable description of a tool call."""
    if not isinstance(inp, dict):
        return str(inp)[:80]
    path = inp.get("path") or inp.get("file_path") or inp.get("filename") or ""
    cmd  = inp.get("command") or inp.get("cmd") or ""
    if path:
        return path
    if cmd:
        return cmd[:80]
    # fallback: first value
    first = next(iter(inp.values()), "")
    return str(first)[:80]


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
