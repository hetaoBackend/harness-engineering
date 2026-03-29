#!/usr/bin/env python3
"""Lightweight harness benchmark runner.

Normal mode — run tasks with a single configuration:
    uv run benchmark/run.py --provider deepseek
    uv run benchmark/run.py --provider anthropic --no-harness
    uv run benchmark/run.py --tasks prime_with_tests fix_buggy_bsearch

Ablation mode — auto-run all middleware combinations, print comparison matrix:
    uv run benchmark/run.py --provider deepseek --ablation
    uv run benchmark/run.py --provider anthropic --ablation --tasks prime_with_tests lru_cache

Local tracing (no LangSmith needed):
    uv run benchmark/run.py --provider deepseek --trace
    uv run benchmark/run.py --provider deepseek --trace --trace-dir my_traces

Other options:
    uv run benchmark/run.py --list               # show task IDs
    uv run benchmark/run.py --out results.json   # save to JSON
    uv run benchmark/run.py --timeout 240        # per-task seconds (default 180)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.tasks import ALL_TASKS, TASK_MAP, Task


# ── result types ─────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    reason: str
    elapsed_s: float
    error: str = ""


@dataclass
class RunResult:
    """Results for one complete configuration run."""
    label: str            # e.g. "full_harness" or "no_local_context"
    provider: str
    model: str
    flags: dict           # middleware toggles used
    results: list[TaskResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def score_pct(self) -> int:
        return 100 * self.passed // self.total if self.total else 0


# ── ablation configurations ───────────────────────────────────────────────────
# Each entry: (label, middleware kwargs dict)
# Ordered from weakest to strongest so the table reads naturally.

ABLATION_CONFIGS: list[tuple[str, dict]] = [
    ("baseline",              dict(enable_local_context=False, enable_loop_detection=False,
                                   enable_pre_completion=False, enable_reasoning_sandwich=False)),
    ("+local_context",        dict(enable_local_context=True,  enable_loop_detection=False,
                                   enable_pre_completion=False, enable_reasoning_sandwich=False)),
    ("+loop_detection",       dict(enable_local_context=True,  enable_loop_detection=True,
                                   enable_pre_completion=False, enable_reasoning_sandwich=False)),
    ("+pre_completion",       dict(enable_local_context=True,  enable_loop_detection=True,
                                   enable_pre_completion=True,  enable_reasoning_sandwich=False)),
    ("full_harness",          dict(enable_local_context=True,  enable_loop_detection=True,
                                   enable_pre_completion=True,  enable_reasoning_sandwich=None)),
]


# ── agent runner ─────────────────────────────────────────────────────────────

async def run_task(
    task: Task,
    sandbox: Path,
    cfg,                      # ProviderConfig
    mw_flags: dict,
    timeout_s: int,
    run_id: str = "",
    traces_dir: Path | None = None,
) -> TaskResult:
    from harness.agent import create_harness_agent

    agent = create_harness_agent(provider_config=cfg, cwd=str(sandbox), **mw_flags)

    from langchain_core.runnables import RunnableConfig
    lc_config: RunnableConfig = {"configurable": {"thread_id": f"bench-{task.id}-{time.monotonic_ns()}"}}

    tracer = None
    if traces_dir is not None and run_id:
        from harness.tracer import LocalTracer
        tracer = LocalTracer(task_id=task.id, run_id=run_id, traces_dir=traces_dir)
        lc_config["callbacks"] = [tracer]

    t0 = time.monotonic()
    error = ""
    try:
        await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": task.description}]},
                config=lc_config,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        error = f"timed out after {timeout_s}s"
    except Exception as exc:
        error = f"agent error: {type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - t0

    if error:
        if tracer:
            tracer.set_verdict(passed=False, reason="", error=error)
            tracer.save()
        return TaskResult(task.id, passed=False, reason="", error=error, elapsed_s=elapsed)

    passed, reason = task.verify(sandbox)
    if tracer:
        tracer.set_verdict(passed=passed, reason=reason)
        trace_path = tracer.save()
        # Print trace path quietly so user knows where to look
        print(f" [trace → {trace_path}]", end="")
    return TaskResult(task.id, passed=passed, reason=reason, elapsed_s=elapsed)


async def run_config(
    tasks: list[Task],
    label: str,
    cfg,
    mw_flags: dict,
    timeout_s: int,
    run_id: str = "",
    traces_dir: Path | None = None,
    verbose: bool = True,
) -> RunResult:
    run = RunResult(label=label, provider=cfg.name, model=cfg.model, flags=mw_flags)
    if verbose:
        print(f"\n  [{label}]")
    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"    [{i}/{len(tasks)}] {task.id} ...", end="", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"bench_{task.id}_") as tmp:
            sandbox = Path(tmp)
            task.setup(sandbox)
            result = await run_task(
                task, sandbox, cfg, mw_flags, timeout_s,
                run_id=f"{run_id}_{label}" if run_id else "",
                traces_dir=traces_dir,
            )
        run.results.append(result)
        if verbose:
            status = "PASS" if result.passed else ("ERR" if result.error else "FAIL")
            note = result.error or result.reason
            print(f" {status}  ({result.elapsed_s:.1f}s)  {note[:40]}")
    return run


# ── printers ──────────────────────────────────────────────────────────────────

_G = "\033[32m"; _R = "\033[31m"; _Y = "\033[33m"; _DIM = "\033[90m"; _RST = "\033[0m"

def _cell(passed: bool, error: bool) -> str:
    if error:   return f"{_Y}ERR{_RST}"
    if passed:  return f"{_G}PASS{_RST}"
    return f"{_R}FAIL{_RST}"


def print_single(run: RunResult, no_harness: bool) -> None:
    mode = "no-harness (baseline)" if no_harness else "full harness"
    print(f"\n{'─'*65}")
    print(f"  {run.provider}  |  {run.model}  |  {mode}")
    print(f"{'─'*65}")
    print(f"  {'Task':<30} {'Result':<8} {'Time':>6}  Note")
    print(f"{'─'*65}")
    for r in run.results:
        note = (r.error or r.reason)[:28]
        print(f"  {r.task_id:<30} {_cell(r.passed, bool(r.error)):<17} {r.elapsed_s:>5.1f}s  {_DIM}{note}{_RST}")
    print(f"{'─'*65}")
    print(f"  Score: {run.passed}/{run.total}  ({run.score_pct}%)")
    print(f"{'─'*65}\n")


def print_ablation_matrix(runs: list[RunResult], tasks: list[Task]) -> None:
    task_ids = [t.id for t in tasks]
    col_w = 8

    # Header
    label_w = max(len(r.label) for r in runs) + 2
    header = f"  {'Task':<30}" + "".join(f" {r.label:>{col_w}}" for r in runs)
    print(f"\n{'─' * len(header)}")
    print(f"  Ablation matrix  |  {runs[0].provider}  {runs[0].model}")
    print(f"{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for tid in task_ids:
        row = f"  {tid:<30}"
        for run in runs:
            r = next((x for x in run.results if x.task_id == tid), None)
            if r is None:
                cell = f"{'—':>{col_w}}"
            elif r.error:
                cell = f"{_Y}{'ERR':>{col_w}}{_RST}"
            elif r.passed:
                cell = f"{_G}{'✓':>{col_w}}{_RST}"
            else:
                cell = f"{_R}{'✗':>{col_w}}{_RST}"
            row += " " + cell
        print(row)

    print(f"{'─' * len(header)}")
    score_row = f"  {'SCORE':<30}"
    for run in runs:
        pct = f"{run.passed}/{run.total}"
        score_row += f" {pct:>{col_w}}"
    print(score_row)
    pct_row = f"  {'':30}"
    for run in runs:
        pct_row += f" {run.score_pct:>{col_w-1}}%"
    print(pct_row)
    print(f"{'─' * len(header)}\n")


# ── main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if args.list:
        print("\nAvailable tasks:")
        for t in ALL_TASKS:
            print(f"  {t.id:<32} [{', '.join(t.tags)}]")
        print()
        return

    # Resolve tasks
    if args.tasks:
        unknown = [tid for tid in args.tasks if tid not in TASK_MAP]
        if unknown:
            print(f"ERROR: unknown task IDs: {', '.join(unknown)}", file=sys.stderr)
            sys.exit(1)
        tasks = [TASK_MAP[tid] for tid in args.tasks]
    else:
        tasks = ALL_TASKS

    from harness.config import get_provider
    cfg = get_provider(args.provider, config_path=args.config, model_override=args.model)
    cfg.check_api_key()

    all_runs: list[RunResult] = []
    traces_dir = Path(args.trace_dir) if args.trace else None
    run_id = f"{cfg.name}_{int(time.time())}" if traces_dir else ""
    if traces_dir:
        print(f"  Tracing → {traces_dir}/{run_id}/")

    if args.ablation:
        print(f"\nAblation run: {len(ABLATION_CONFIGS)} configs × {len(tasks)} tasks"
              f"  |  provider={cfg.name}  model={cfg.model}")
        for label, mw_flags in ABLATION_CONFIGS:
            run = await run_config(tasks, label, cfg, mw_flags, args.timeout,
                                   run_id=run_id, traces_dir=traces_dir)
            all_runs.append(run)
        print_ablation_matrix(all_runs, tasks)
    else:
        mw_flags = dict(
            enable_local_context=not args.no_harness,
            enable_loop_detection=not args.no_harness,
            enable_pre_completion=not args.no_harness,
            enable_reasoning_sandwich=False if args.no_harness else None,
        )
        label = "no_harness" if args.no_harness else "full_harness"
        print(f"\nBenchmark: {len(tasks)} tasks  |  provider={cfg.name}  model={cfg.model}"
              f"  harness={'OFF' if args.no_harness else 'ON'}")
        run = await run_config(tasks, label, cfg, mw_flags, args.timeout,
                               run_id=run_id, traces_dir=traces_dir)
        all_runs.append(run)
        print_single(run, args.no_harness)

    if args.out:
        out_path = Path(args.out)
        payload = {
            "provider": cfg.name,
            "model": cfg.model,
            "runs": [
                {**{"label": r.label, "passed": r.passed, "total": r.total,
                    "score_pct": r.score_pct, "flags": r.flags},
                 "results": [asdict(t) for t in r.results]}
                for r in all_runs
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved → {out_path}")


def _build_parser() -> argparse.ArgumentParser:
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
    p.add_argument("--config", default=None, metavar="PATH")
    p.add_argument("--tasks", nargs="+", metavar="ID",
                   help="Run only these task IDs (--list to see all)")
    p.add_argument("--ablation", action="store_true",
                   help="Run all middleware combinations and print comparison matrix")
    p.add_argument("--no-harness", action="store_true",
                   help="Disable all middleware (baseline)")
    p.add_argument("--timeout", type=int, default=180,
                   help="Per-task agent timeout in seconds (default: 180)")
    p.add_argument("--out", metavar="FILE", help="Save results to JSON")
    p.add_argument("--trace", action="store_true",
                   help="Save local traces to traces/ (no LangSmith needed)")
    p.add_argument("--trace-dir", default="traces", metavar="DIR",
                   help="Directory for trace files (default: traces/)")
    p.add_argument("--list", action="store_true", help="List task IDs and exit")
    return p


if __name__ == "__main__":
    asyncio.run(main(_build_parser().parse_args()))
