#!/usr/bin/env python3
"""Compare two or more benchmark result JSON files side-by-side.

Usage:
    uv run benchmark/compare.py baseline.json harness.json
    uv run benchmark/compare.py run_a.json run_b.json run_c.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_G = "\033[32m"; _R = "\033[31m"; _Y = "\033[33m"; _B = "\033[1m"; _DIM = "\033[90m"; _RST = "\033[0m"


def load(path: str) -> list[dict]:
    """Return list of run dicts from a JSON file (supports both single-run and ablation output)."""
    data = json.loads(Path(path).read_text())
    if "runs" in data:
        return data["runs"]           # ablation output
    # single-run output: wrap it
    return [{
        "label": data.get("label", Path(path).stem),
        "provider": data.get("provider", "?"),
        "model": data.get("model", "?"),
        "passed": data.get("passed", 0),
        "total": data.get("total", 0),
        "score_pct": data.get("score_pct", 0),
        "results": data.get("results", []),
    }]


def compare(files: list[str]) -> None:
    all_runs: list[dict] = []
    labels: list[str] = []
    for f in files:
        runs = load(f)
        for r in runs:
            tag = f"{Path(f).stem}/{r['label']}" if len(load(f)) > 1 else Path(f).stem
            r["_tag"] = tag
            all_runs.append(r)
            labels.append(tag)

    # Collect all task IDs in order
    task_ids: list[str] = []
    seen: set[str] = set()
    for run in all_runs:
        for res in run.get("results", []):
            tid = res["task_id"]
            if tid not in seen:
                task_ids.append(tid)
                seen.add(tid)

    col_w = 10
    tag_w = max(len(t) for t in labels) + 2

    # Build result lookup: run_idx → task_id → result dict
    lookup: list[dict[str, dict]] = []
    for run in all_runs:
        lookup.append({r["task_id"]: r for r in run.get("results", [])})

    # Header
    header = f"  {'Task':<30}" + "".join(f" {t:>{col_w}}" for t in labels)
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(f"  {_B}Benchmark comparison{_RST}")
    print(sep)
    print(header)
    print(sep)

    for tid in task_ids:
        row = f"  {tid:<30}"
        for i, run in enumerate(all_runs):
            r = lookup[i].get(tid)
            if r is None:
                cell = f"{'—':>{col_w}}"
            elif r.get("error"):
                cell = f"{_Y}{'ERR':>{col_w}}{_RST}"
            elif r.get("passed"):
                cell = f"{_G}{'✓':>{col_w}}{_RST}"
            else:
                cell = f"{_R}{'✗':>{col_w}}{_RST}"
            row += " " + cell
        print(row)

    print(sep)

    # Score row
    score_row = f"  {'SCORE':<30}"
    for run in all_runs:
        pct = f"{run['passed']}/{run['total']}"
        score_row += f" {pct:>{col_w}}"
    print(score_row)

    pct_row = f"  {'':30}"
    for run in all_runs:
        pct_row += f" {run['score_pct']:>{col_w-1}}%"
    print(pct_row)
    print(sep)

    # Delta column: first vs last
    if len(all_runs) >= 2:
        first, last = all_runs[0], all_runs[-1]
        delta = last["score_pct"] - first["score_pct"]
        sign = "+" if delta >= 0 else ""
        color = _G if delta > 0 else (_R if delta < 0 else _DIM)
        print(f"\n  {first['_tag']} → {last['_tag']}:  "
              f"{color}{_B}{sign}{delta}pp{_RST}  "
              f"({first['score_pct']}% → {last['score_pct']}%)\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run benchmark/compare.py file1.json file2.json [file3.json ...]")
        sys.exit(1)
    compare(sys.argv[1:])
