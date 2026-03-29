#!/usr/bin/env python3
"""Analyze local trace files to find failure patterns — no LangSmith needed.

Reads JSON trace files produced by LocalTracer and surfaces:
  - Which tasks failed and why
  - Which middleware fired (doom-loop warning, pre-completion checklist)
  - Common failure patterns across runs
  - Per-task turn counts and tool call breakdown

Usage:
    # Analyze all traces in the default directory
    uv run benchmark/analyze.py

    # Analyze a specific run
    uv run benchmark/analyze.py --run-id deepseek_1234567890

    # Analyze only failed tasks
    uv run benchmark/analyze.py --failed-only

    # Detailed view: print input/output for each LLM turn
    uv run benchmark/analyze.py --run-id deepseek_1234567890 --task prime_with_tests --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_G = "\033[32m"; _R = "\033[31m"; _Y = "\033[33m"; _B = "\033[1m"; _DIM = "\033[90m"; _RST = "\033[0m"


# ── loading ───────────────────────────────────────────────────────────────────

def load_traces(traces_dir: Path, run_id: str | None = None) -> list[dict]:
    """Return all trace dicts under traces_dir, optionally filtered by run_id."""
    traces = []
    search_root = traces_dir / run_id if run_id else traces_dir
    for path in sorted(search_root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            data["_run_id"] = path.parent.name
            traces.append(data)
        except Exception:
            pass
    return traces


# ── pattern detection ─────────────────────────────────────────────────────────

def _detect_patterns(trace: dict) -> list[str]:
    """Return a list of human-readable failure pattern labels."""
    patterns = []
    turns = trace.get("turns", [])
    verdict = trace.get("verdict", {})
    summary = trace.get("summary", {})

    if verdict.get("error"):
        patterns.append(f"TIMEOUT/CRASH: {verdict['error'][:60]}")
        return patterns

    if verdict.get("passed"):
        return []

    # Did the agent ever run tests?
    tool_names = [t.get("name", "") for t in turns if t.get("type") == "tool"]
    ran_tests = any(
        n in ("bash", "run_command", "execute_command", "shell")
        and ("pytest" in str(t.get("result_preview", "")) or "python -m" in str(t.get("result_preview", "")))
        for t, n in zip(turns, tool_names)
    ) or any("pytest" in n or "test" in n for n in tool_names)
    if not ran_tests:
        patterns.append("never ran tests")

    # Did doom-loop warning fire?
    mw_flags_seen = set()
    for t in turns:
        for f in t.get("middleware_flags", []):
            mw_flags_seen.add(f)
    if "loop_detection_triggered" in mw_flags_seen:
        patterns.append("doom-loop detected")

    # Did pre-completion checklist fire?
    if "pre_completion" not in mw_flags_seen:
        patterns.append("pre-completion checklist not triggered")

    # Did agent write any files?
    wrote_files = any(t.get("name") in ("write_file", "edit_file") for t in turns if t.get("type") == "tool")
    if not wrote_files:
        patterns.append("no files written")

    # Tool errors
    tool_errors = [t for t in turns if t.get("type") == "tool_error"]
    if tool_errors:
        patterns.append(f"{len(tool_errors)} tool error(s)")

    # Very few turns — agent gave up early
    llm_turns = summary.get("llm_turns", len([t for t in turns if t.get("type") == "llm"]))
    if llm_turns <= 2:
        patterns.append(f"gave up early ({llm_turns} LLM turns)")

    if not patterns:
        patterns.append("unknown — check verbose output")

    return patterns


# ── report ────────────────────────────────────────────────────────────────────

def print_summary(traces: list[dict], failed_only: bool) -> None:
    if not traces:
        print("No traces found.")
        return

    # Group by run_id
    by_run: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_run[t["_run_id"]].append(t)

    all_patterns: list[str] = []

    for run_id, run_traces in sorted(by_run.items()):
        passed_count = sum(1 for t in run_traces if t.get("verdict", {}).get("passed"))
        total = len(run_traces)
        print(f"\n{'═'*68}")
        print(f"  Run: {_B}{run_id}{_RST}   Score: {passed_count}/{total}")
        print(f"{'═'*68}")
        print(f"  {'Task':<30} {'Result':<8} {'Turns':>5} {'Tools':>5}  Patterns")
        print(f"{'─'*68}")

        for trace in sorted(run_traces, key=lambda t: t["task_id"]):
            verdict = trace.get("verdict", {})
            passed = verdict.get("passed", False)
            error = verdict.get("error", "")
            summary = trace.get("summary", {})
            llm_turns = summary.get("llm_turns", "?")
            tool_calls = summary.get("tool_calls", "?")
            patterns = _detect_patterns(trace)
            all_patterns.extend(patterns)

            if failed_only and passed:
                continue

            if error:
                status = f"{_Y}ERR{_RST}"
            elif passed:
                status = f"{_G}PASS{_RST}"
            else:
                status = f"{_R}FAIL{_RST}"

            pattern_str = "; ".join(patterns) if patterns else "—"
            print(f"  {trace['task_id']:<30} {status:<17} {llm_turns!s:>5} {tool_calls!s:>5}  "
                  f"{_DIM}{pattern_str[:40]}{_RST}")

    # Pattern frequency summary
    if all_patterns:
        counts = Counter(p for p in all_patterns if "PASS" not in p)
        if counts:
            print(f"\n{'─'*68}")
            print(f"  {_B}Most common failure patterns across all runs:{_RST}")
            for pattern, count in counts.most_common(8):
                bar = "█" * count
                print(f"  {count:>3}x  {pattern:<40} {_DIM}{bar}{_RST}")
    print()


def print_verbose(trace: dict) -> None:
    """Print full LLM turn content for a single trace."""
    print(f"\n{'═'*68}")
    print(f"  Task: {_B}{trace['task_id']}{_RST}  Run: {trace['_run_id']}")
    verdict = trace.get("verdict", {})
    status = f"{_G}PASS{_RST}" if verdict.get("passed") else f"{_R}FAIL{_RST}"
    print(f"  Result: {status}  ({trace.get('elapsed_s', '?')}s)")
    if verdict.get("reason"):
        print(f"  Reason: {verdict['reason']}")
    if verdict.get("error"):
        print(f"  Error:  {_Y}{verdict['error']}{_RST}")
    print(f"{'─'*68}")

    for i, turn in enumerate(trace.get("turns", []), 1):
        ttype = turn.get("type", "?")
        elapsed = turn.get("elapsed_ms", 0)
        if ttype == "llm":
            mw = turn.get("middleware_flags", [])
            tc = turn.get("tool_calls_requested", [])
            print(f"\n  {_B}[LLM turn {i}]{_RST}  {elapsed}ms"
                  + (f"  middleware={mw}" if mw else ""))
            print(f"  {_DIM}output:{_RST} {turn.get('output_text', '')[:300]}")
            if tc:
                print(f"  {_DIM}tool calls:{_RST} {[t['name'] for t in tc]}")
        elif ttype == "tool":
            print(f"\n  {_B}[tool]{_RST} {turn.get('name', '?')}  {elapsed}ms")
            print(f"  {_DIM}result:{_RST} {turn.get('result_preview', '')[:200]}")
        elif ttype == "tool_error":
            print(f"\n  {_B}[tool_error]{_RST} {turn.get('name', '?')}: "
                  f"{_Y}{turn.get('error', '')}{_RST}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze local benchmark traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--traces-dir", default="traces", metavar="DIR",
                   help="Directory containing trace files (default: traces/)")
    p.add_argument("--run-id", default=None, metavar="ID",
                   help="Filter to a specific run ID")
    p.add_argument("--task", default=None, metavar="ID",
                   help="Show verbose output for a specific task (requires --run-id)")
    p.add_argument("--failed-only", action="store_true",
                   help="Show only failed tasks in the summary")
    args = p.parse_args()

    traces_dir = Path(args.traces_dir)
    if not traces_dir.exists():
        print(f"No traces directory found at '{traces_dir}'. "
              "Run with --trace to generate traces.")
        sys.exit(1)

    traces = load_traces(traces_dir, run_id=args.run_id)
    if not traces:
        print(f"No trace files found under '{traces_dir}'.")
        sys.exit(1)

    if args.task:
        matches = [t for t in traces if t["task_id"] == args.task]
        if not matches:
            print(f"No traces found for task '{args.task}'.")
            sys.exit(1)
        for t in matches:
            print_verbose(t)
    else:
        print_summary(traces, failed_only=args.failed_only)


if __name__ == "__main__":
    main()
