#!/usr/bin/env python3
"""Entry point for the harness-engineered deep agent.

Usage:
    # Anthropic (default)
    uv run main.py "Write a Python function that checks if a number is prime, with tests"
    uv run main.py "Refactor auth.py to use JWT" --cwd /path/to/project

    # OpenAI
    uv run main.py "Fix the bug in main.py" --provider openai --model gpt-4o

    # OpenAI-compatible (DeepSeek, Together, Ollama, etc.)
    uv run main.py "..." --provider openai --model deepseek-chat \\
        --base-url https://api.deepseek.com/v1

    Environment variables:
        ANTHROPIC_API_KEY   Required when --provider anthropic (default)
        OPENAI_API_KEY      Required when --provider openai
        OPENAI_BASE_URL     Optional base URL override for openai provider
"""

import argparse
import asyncio
import os
import sys

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig


async def run(
    task: str,
    cwd: str,
    model: str,
    provider: str,
    base_url: str | None,
    thread_id: str = "default",
) -> None:
    # Import here so the module can be imported without side-effects
    from harness.agent import create_harness_agent

    agent = create_harness_agent(
        model_name=model,
        provider=provider,
        base_url=base_url,
        cwd=cwd,
    )

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
    }

    print(f"\n{'='*60}")
    print(f"Task    : {task}")
    print(f"CWD     : {cwd}")
    print(f"Provider: {provider}")
    print(f"Model   : {model}")
    if base_url:
        print(f"Base URL: {base_url}")
    print(f"{'='*60}\n")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config=config,
    )

    # Print final AI response
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            print("\n--- Agent Response ---")
            print(msg.content if isinstance(msg.content, str) else str(msg.content))
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harness-engineered deep agent (TerminalBench-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task", help="Task for the agent to complete")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory (default: cwd)")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="Model provider: 'anthropic' (default) or 'openai' for OpenAI-compatible APIs",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Defaults to 'claude-sonnet-4-5-20250929' for anthropic "
            "or 'gpt-4o' for openai."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        dest="base_url",
        help=(
            "API base URL override (openai provider only). "
            "E.g. https://api.deepseek.com/v1. "
            "Falls back to OPENAI_BASE_URL env var."
        ),
    )
    parser.add_argument("--thread-id", default="harness-run", help="LangGraph thread ID")
    args = parser.parse_args()

    # Resolve model default per provider
    model = args.model
    if model is None:
        model = (
            "claude-sonnet-4-5-20250929" if args.provider == "anthropic" else "gpt-4o"
        )

    # Validate API key presence
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if args.provider == "openai":
        # Allow key via env or OPENAI_BASE_URL override (some local servers don't need a key)
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "WARNING: OPENAI_API_KEY is not set. "
                "Set it if your endpoint requires authentication.",
                file=sys.stderr,
            )

    asyncio.run(
        run(
            task=args.task,
            cwd=args.cwd,
            model=model,
            provider=args.provider,
            base_url=args.base_url,
            thread_id=args.thread_id,
        )
    )


if __name__ == "__main__":
    main()
