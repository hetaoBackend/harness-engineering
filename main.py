#!/usr/bin/env python3
"""Entry point for the harness-engineered deep agent.

Usage:
    uv run main.py "Write a Python function that checks if a number is prime, with tests"
    uv run main.py "Refactor auth.py to use JWT" --cwd /path/to/project
"""

import argparse
import asyncio
import os
import sys

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig


async def run(task: str, cwd: str, model: str, thread_id: str = "default") -> None:
    # Import here so the module can be imported without side-effects
    from harness.agent import create_harness_agent

    agent = create_harness_agent(model_name=model, cwd=cwd)

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
    }

    print(f"\n{'='*60}")
    print(f"Task : {task}")
    print(f"CWD  : {cwd}")
    print(f"Model: {model}")
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
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model name (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument("--thread-id", default="harness-run", help="LangGraph thread ID")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args.task, args.cwd, args.model, args.thread_id))


if __name__ == "__main__":
    main()
