#!/usr/bin/env python3
"""Lightweight harness benchmark runner.

For each task:
  1. Creates an isolated temp sandbox directory.
  2. Calls task.setup() to plant any initial files.
  3. Runs the harness agent with the task description.
  4. Calls task.verify() and records pass / fail.
  5. Prints a results table.

Usage:
    # Run all tasks with default provider (first in providers.toml)
    uv run benchmark/run.py

    # Pick a provider
    uv run benchmark/run.py --provider deepseek

    # Run a subset of tasks
    uv run benchmark/run.py --tasks prime_with_tests fix_buggy_bsearch

    # Compare harness vs no-harness on the same provider
    uv run benchmark/run.py --provider anthropic
    uv run benchmark/run.py --provider anthropic --no-harness

    # Show available task IDs
    uv run benchmark/run.py --list

    # Save results to JSON
    uv run benchmark/run.py --out results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.tasks import ALL_TASKS, TASK_MAP, Task


# ── result dataclass ─────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    reason: str
    elapsed_s: float
    error: str = ""   # set if the agent itself crashed


# ── agent runner ─────────────────────────────────────────────────────────────

async def run_task(
    task: Task,
    sandbox: Path,
    provider_name: str,
    config_path: str | None,
    model_override: str | None,
    no_harness: bool,
    timeout_s: int,
) -> TaskResult:
    from harness.config import get_provider
    from harness.agent import create_harness_agent

    cfg = get_provider(provider_name, config_path=config_path, model_override=model_override)

    agent = create_harness_agent(
        provider_config=cfg,
        cwd=str(sandbox),
        # Disable all middleware for baseline comparison
        loop_threshold=999 if no_harness else 3,
        enable_reasoning_sandwich=False if no_harness else None,
    )
    # For a true no-harness baseline we also skip pre-completion & local-context
    # by rebuilding without them.
    if no_harness:
        from deepagents import create_deep_agent
        from harness.agent import _build_model, HARNESS_SYSTEM_PROMPT
        model = _build_model(cfg.type, cfg.model, cfg.base_url, cfg.resolved_api_key)
        agent = create_deep_agent(model=model, system_prompt=HARNESS_SYSTEM_PROMPT, middleware=[])

    from langchain_core.runnables import RunnableConfig
    config: RunnableConfig = {"configurable": {"thread_id": f"bench-{task.id}"}}

    t0 = time.monotonic()
    error = ""
    try:
        await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": task.description}]},
                config=config,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        error = f"agent timed out after {timeout_s}s"
    except Exception as exc:
        error = f"agent error: {exc}"
    elapsed = time.monotonic() - t0

    if error:
        return TaskResult(task.id, passed=False, reason="", error=error, elapsed_s=elapsed)

    passed, reason = task.verify(sandbox)
    return TaskResult(task.id, passed=passed, reason=reason, elapsed_s=elapsed)


# ── table printer ─────────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_RED   = "\033[31m"
_GREY  = "\033[90m"
_RESET = "\033[0m"


def _status(r: TaskResult) -> str:
    if r.error:
        return f"{_RED}ERROR{_RESET}"
    return f"{_GREEN}PASS{_RESET}" if r.passed else f"{_RED}FAIL{_RESET}"


def print_table(results: list[TaskResult], provider: str, model: str, no_harness: bool) -> None:
    harness_label = "no-harness (baseline)" if no_harness else "full harness"
    print(f"\n{'─'*65}")
    print(f"  Provider : {provider}   Model: {model}   Mode: {harness_label}")
    print(f"{'─'*65}")
    print(f"  {'Task':<28}  {'Result':<8}  {'Time':>6}  Note")
    print(f"{'─'*65}")
    for r in results:
        note = r.error or r.reason
        note_short = note[:25] + "…" if len(note) > 26 else note
        print(f"  {r.task_id:<28}  {_status(r):<17}  {r.elapsed_s:>5.1f}s  {_GREY}{note_short}{_RESET}")
    print(f"{'─'*65}")
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    pct    = 100 * passed // total if total else 0
    print(f"  Score: {passed}/{total}  ({pct}%)")
    print(f"{'─'*65}\n")


# ── main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if args.list:
        print("\nAvailable tasks:")
        for t in ALL_TASKS:
            tags = ", ".join(t.tags) if t.tags else ""
            print(f"  {t.id:<30}  [{tags}]")
        print()
        return

    # Resolve tasks to run
    if args.tasks:
        unknown = [tid for tid in args.tasks if tid not in TASK_MAP]
        if unknown:
            print(f"ERROR: unknown task IDs: {', '.join(unknown)}", file=sys.stderr)
            sys.exit(1)
        tasks = [TASK_MAP[tid] for tid in args.tasks]
    else:
        tasks = ALL_TASKS

    # Resolve provider
    from harness.config import get_provider, load_providers
    cfg = get_provider(args.provider, config_path=args.config, model_override=args.model)
    cfg.check_api_key()

    print(f"\nBenchmark: {len(tasks)} tasks  |  provider={cfg.name}  model={cfg.model}"
          f"  harness={'OFF (baseline)' if args.no_harness else 'ON'}")

    results: list[TaskResult] = []

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task.id} ...", end="", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"bench_{task.id}_") as tmp:
            sandbox = Path(tmp)
            task.setup(sandbox)
            result = await run_task(
                task=task,
                sandbox=sandbox,
                provider_name=cfg.name,
                config_path=args.config,
                model_override=args.model,
                no_harness=args.no_harness,
                timeout_s=args.timeout,
            )
        results.append(result)
        status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
        print(f" {status}  ({result.elapsed_s:.1f}s)")

    print_table(results, provider=cfg.name, model=cfg.model, no_harness=args.no_harness)

    if args.out:
        out_path = Path(args.out)
        payload = {
            "provider": cfg.name,
            "model": cfg.model,
            "no_harness": args.no_harness,
            "results": [asdict(r) for r in results],
            "passed": sum(1 for r in results if r.passed),
            "total": len(results),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    # Quick pre-parse to get --config for dynamic provider choices
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    try:
        from harness.config import load_providers
        providers = load_providers(pre_args.config)
        provider_names = sorted(providers)
        provider_default = next(iter(providers))
    except Exception:
        provider_names = None
        provider_default = "anthropic"

    p = argparse.ArgumentParser(
        description="Lightweight harness benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--provider", default=provider_default, choices=provider_names, metavar="NAME",
        help=f"Provider from providers.toml (default: {provider_default!r})"
             + (f". Available: {', '.join(provider_names)}" if provider_names else ""),
    )
    p.add_argument("--model", default=None, help="Override model for this run")
    p.add_argument("--config", default=None, metavar="PATH", help="Path to providers.toml")
    p.add_argument(
        "--tasks", nargs="+", metavar="ID",
        help="Run only these task IDs (use --list to see all)",
    )
    p.add_argument(
        "--no-harness", action="store_true",
        help="Disable all middleware (baseline comparison)",
    )
    p.add_argument(
        "--timeout", type=int, default=180,
        help="Per-task agent timeout in seconds (default: 180)",
    )
    p.add_argument("--out", metavar="FILE", help="Save results to JSON file")
    p.add_argument("--list", action="store_true", help="List available task IDs and exit")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
