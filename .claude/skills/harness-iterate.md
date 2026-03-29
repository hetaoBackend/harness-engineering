---
name: harness-iterate
description: Automates the harness engineering iteration loop for the harness-engineering project. Runs benchmark, analyzes failure traces, proposes and implements targeted middleware improvements, then verifies the improvement. Use when the user wants to iterate on the harness, improve benchmark scores, or reproduce the LangChain harness engineering workflow.
---

# harness-iterate

Automates the full harness engineering iteration loop:

```
run baseline → run harness → analyze traces → identify patterns →
implement fix → re-run failing tasks → compare → repeat
```

## Arguments (optional, parsed from ARGUMENTS string)

- `--provider NAME`   provider from providers.toml (default: first entry)
- `--tasks ID ...`    restrict to these task IDs for faster iteration
- `--dry-run`         analyze and propose changes, but do not implement them
- `--skip-baseline`   skip baseline run if results already exist

---

## Procedure

When this skill is invoked, follow these steps exactly.

### 0. Locate the project

Check that the current working directory is the harness-engineering project (contains `providers.toml` and `benchmark/run.py`). If not, look for it at `~/workspace/harness-engineering`. `cd` there before proceeding.

### 1. Parse arguments

Read ARGUMENTS to extract `--provider`, `--tasks`, `--dry-run`, `--skip-baseline`.
If `--provider` is not given, read the first provider from `providers.toml`.

### 2. Confirm API key is set

Check that the relevant env var for the chosen provider is exported.
If not, print a clear error and stop.

### 3. Run baseline (no harness)

Unless `--skip-baseline` is set and `baseline_<provider>.json` already exists:

```bash
uv run benchmark/run.py \
  --provider <PROVIDER> \
  [--tasks <TASKS>] \
  --no-harness \
  --trace \
  --trace-dir traces \
  --out baseline_<provider>.json
```

Print a one-line summary: `Baseline: N/M passed (X%)`

### 4. Run current harness

```bash
uv run benchmark/run.py \
  --provider <PROVIDER> \
  [--tasks <TASKS>] \
  --trace \
  --trace-dir traces \
  --out harness_<provider>.json
```

Print: `Harness: N/M passed (X%)`

### 5. Compare baseline vs harness

```bash
uv run benchmark/compare.py baseline_<provider>.json harness_<provider>.json
```

If harness already achieves 100%, congratulate the user and stop — nothing left to improve.

### 6. Analyze failure traces

```bash
uv run benchmark/analyze.py --failed-only --traces-dir traces
```

For every failing task, also run verbose analysis:

```bash
uv run benchmark/analyze.py \
  --traces-dir traces \
  --run-id <most_recent_harness_run_id> \
  --task <failing_task_id> \
  --verbose
```

Read the verbose output carefully. For each failure, record:
- How many LLM turns the agent took
- Which tools it called
- Whether it ran tests
- Whether any middleware (doom-loop warning, pre-completion checklist) fired
- What the final output/error was

### 7. Identify the highest-impact pattern

Group failures by root cause. Prioritise in this order:

| Pattern observed | Likely fix |
|---|---|
| Agent never ran tests | Strengthen `PreCompletionChecklistMiddleware` prompt — make test execution mandatory |
| Agent stopped after 1–2 turns | Add an explicit "minimum turns" hint to system prompt |
| Same file edited 5+ times | Lower `loop_threshold` or add stronger doom-loop wording |
| Wrong file names / missed files | Strengthen `LocalContextMiddleware` — list files more prominently |
| Tool errors (file not found, syntax) | Add error-recovery instruction to system prompt |
| Agent didn't follow 4-phase workflow | Rewrite Plan/Build/Verify/Report section in `HARNESS_SYSTEM_PROMPT` |

Pick the **single most frequent pattern** that affects the most failing tasks.
State clearly: "Root cause: X. Proposed fix: Y in file Z."

If `--dry-run` is set, print the proposal and stop here.

### 8. Implement the fix

Make a **minimal, targeted change** to one of these files:
- `harness/agent.py` — `HARNESS_SYSTEM_PROMPT` for prompt changes
- `harness/middleware/pre_completion.py` — `VERIFICATION_CHECKLIST` text
- `harness/middleware/loop_detection.py` — `DOOM_LOOP_WARNING` text or `LOOP_THRESHOLD`
- `harness/middleware/local_context.py` — context template or `_dir_listing` depth

Do NOT refactor, add new files, or change the middleware architecture unless
the fix genuinely requires it. Change the minimum number of lines.

After editing, briefly explain what you changed and why.

### 9. Re-run only the previously failing tasks

```bash
uv run benchmark/run.py \
  --provider <PROVIDER> \
  --tasks <FAILING_TASK_IDS> \
  --trace \
  --trace-dir traces \
  --out harness_after_fix_<provider>.json
```

### 10. Compare before and after

```bash
uv run benchmark/compare.py \
  harness_<provider>.json \
  harness_after_fix_<provider>.json
```

### 11. Report

Print a concise summary:

```
Iteration complete
──────────────────────────────────────
Baseline (no harness):    X/N  (X%)
Before fix:               Y/N  (Y%)
After fix:                Z/N  (Z%)

Change made: <one sentence description>
Files modified: <list>

Remaining failures: <list of still-failing tasks>
Suggested next: run /harness-iterate again to address the next pattern
```

If the fix made things worse, say so clearly, revert the change (`git checkout --`), and explain why it didn't work.

---

## Notes

- Always use `--trace` so the next iteration has fresh data to analyze.
- Keep each iteration to **one fix at a time** — multiple simultaneous changes make it impossible to attribute improvements.
- The `traces/` directory accumulates runs. Use `--run-id` with `analyze.py` to target the most recent one.
- For the Anthropic provider, `ReasoningSandwichMiddleware` is active and adds extended thinking on planning + verification turns. Don't disable it unless traces show it's causing timeouts.
