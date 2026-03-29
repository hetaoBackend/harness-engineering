# Harness Engineering — Experiment Playbook

Reproduction of the methodology from
**[Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)** (LangChain blog).

The blog improved a coding agent from **52.8% → 66.5%** on Terminal Bench 2.0
by modifying only the system harness — no model changes. This playbook
reproduces that loop with a lightweight local benchmark.

---

## Architecture overview

```
providers.toml          ← configure any number of named model providers
    │
    ▼
create_harness_agent()  ← wires middleware stack around the agent
    │
    ├── LocalContextMiddleware       injects cwd + tool inventory
    ├── LoopDetectionMiddleware      warns on repeated file edits
    ├── PreCompletionChecklistMiddleware  forces verification before exit
    └── ReasoningSandwichMiddleware  extended thinking on plan + verify (Anthropic only)
    │
    ▼
benchmark/run.py        ← runs tasks, collects pass/fail, writes trace files
    │
    ├── traces/<run_id>/<task_id>.json   ← per-task local traces
    │
    ▼
benchmark/analyze.py    ← reads traces, surfaces failure patterns
    │
    ▼
benchmark/compare.py    ← side-by-side diff of two or more result JSON files
```

---

## 1. Setup

```bash
git clone https://github.com/hetaoBackend/harness-engineering
cd harness-engineering
uv sync
```

---

## 2. Configure providers

Edit **`providers.toml`** to add your providers. The `api_key_env` field is the
*name* of an environment variable — the key itself never goes in the file.

```toml
[providers.anthropic]
type        = "anthropic"
model       = "claude-sonnet-4-5-20250929"
api_key_env = "ANTHROPIC_API_KEY"

[providers.deepseek]
type        = "openai"
model       = "deepseek-chat"
base_url    = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"

# [providers.ollama]        ← local, no key needed
# type     = "openai"
# model    = "qwen2.5-coder:32b"
# base_url = "http://localhost:11434/v1"
```

Then export the key(s) you need:

```bash
export DEEPSEEK_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

**Common endpoint reference**

| Provider    | base_url                          | Example model                    |
|-------------|-----------------------------------|----------------------------------|
| Anthropic   | *(built-in)*                      | `claude-sonnet-4-5-20250929`     |
| OpenAI      | *(built-in)*                      | `gpt-4o`                         |
| DeepSeek    | `https://api.deepseek.com/v1`     | `deepseek-chat`                  |
| Together AI | `https://api.together.xyz/v1`     | `meta-llama/Llama-3-70b-chat-hf` |
| Ollama      | `http://localhost:11434/v1`       | `qwen2.5-coder:32b`              |
| LM Studio   | `http://localhost:1234/v1`        | *(whatever you loaded)*          |

---

## 3. The reproduction loop

The blog's workflow was:

```
run baseline → analyze failure traces → add/tune middleware → re-run → compare
```

### Step 1 — Run baseline (no harness)

```bash
uv run benchmark/run.py --provider deepseek --no-harness \
  --trace --out baseline.json
```

`--trace` writes one JSON file per task under `traces/`.

### Step 2 — Run with full harness

```bash
uv run benchmark/run.py --provider deepseek \
  --trace --out harness.json
```

### Step 3 — Compare results

```bash
uv run benchmark/compare.py baseline.json harness.json
```

Output:

```
  Task                           baseline   full_harness
  prime_with_tests                      ✗             ✓
  fix_buggy_bsearch                     ✓             ✓
  rle_encode                            ✓             ✓
  fix_buggy_mergesort                   ✗             ✓
  lru_cache                             ✗             ✓
  word_freq                             ✗             ✗
  retry_decorator                       ✗             ✓
  csv_stats                             ✗             ✓
  SCORE                               2/8           7/8
                                      25%           87%

  baseline → full_harness:  +62pp  (25% → 87%)
```

### Step 4 — Analyze failure traces

```bash
# Summary with auto-detected failure patterns
uv run benchmark/analyze.py

# Only failed tasks
uv run benchmark/analyze.py --failed-only

# Full turn-by-turn detail for one task
uv run benchmark/analyze.py --run-id deepseek_1234567890 \
  --task word_freq --verbose
```

The analyzer detects patterns automatically:

| Pattern | What it means |
|---|---|
| `never ran tests` | Agent wrote code but didn't execute it |
| `doom-loop detected` | LoopDetectionMiddleware fired |
| `pre-completion not triggered` | Agent stopped without hitting the checklist |
| `gave up early (N LLM turns)` | Agent bailed out with minimal work |
| `N tool error(s)` | Tool calls crashed — check sandbox setup |

### Step 5 — Tune and iterate

Based on trace findings, adjust middleware in `harness/agent.py` or tweak
prompts in the middleware files, then re-run steps 1–4.

---

## 4. Ablation experiment

To measure each middleware's individual contribution — exactly as the blog did:

```bash
uv run benchmark/run.py --provider deepseek --ablation \
  --trace --out ablation.json
```

This runs five configurations in sequence and prints a comparison matrix:

```
  Task                    baseline  +local_ctx  +loop_det  +pre_comp  full_harness
  prime_with_tests               ✗           ✗          ✗          ✓             ✓
  fix_buggy_mergesort            ✗           ✗          ✓          ✓             ✓
  word_freq                      ✗           ✓          ✓          ✓             ✓
  ...
  SCORE                        2/8         4/8        5/8        6/8           7/8
                               25%         50%        62%        75%           87%
```

Ablation configurations (cumulative):

| Label | Middleware active |
|---|---|
| `baseline` | none |
| `+local_context` | LocalContext |
| `+loop_detection` | LocalContext + LoopDetection |
| `+pre_completion` | LocalContext + LoopDetection + PreCompletion |
| `full_harness` | all (+ ReasoningSandwich for Anthropic) |

---

## 5. CLI reference

### `benchmark/run.py`

```
uv run benchmark/run.py [OPTIONS] [task]

  --provider NAME        Provider from providers.toml (default: first entry)
  --model MODEL          Override model for this run
  --config PATH          Path to providers.toml (default: ./providers.toml)
  --tasks ID [ID ...]    Run only these tasks (--list to see all)
  --ablation             Run all 5 middleware combinations, print matrix
  --no-harness           Disable all middleware (baseline)
  --trace                Write per-task trace JSON files (no LangSmith needed)
  --trace-dir DIR        Trace output directory (default: traces/)
  --timeout SECS         Per-task agent timeout (default: 180)
  --out FILE             Save results to JSON
  --list                 Print available task IDs and exit
```

### `benchmark/analyze.py`

```
uv run benchmark/analyze.py [OPTIONS]

  --traces-dir DIR       Trace directory (default: traces/)
  --run-id ID            Filter to one run
  --task ID              Show full verbose trace for one task (needs --run-id)
  --failed-only          Show only failed tasks
```

### `benchmark/compare.py`

```
uv run benchmark/compare.py FILE1 FILE2 [FILE3 ...]

  Compares result JSON files side-by-side.
  Accepts both single-run and ablation output.
  Prints delta pp between first and last file.
```

### `main.py` (single task)

```
uv run main.py TASK [OPTIONS]

  --provider NAME        Provider from providers.toml
  --model MODEL          Override model
  --config PATH          Path to providers.toml
  --cwd PATH             Working directory shown to agent
  --thread-id ID         LangGraph thread ID (default: harness-run)
```

---

## 6. Middleware reference

| Class | File | What it does |
|---|---|---|
| `LocalContextMiddleware` | `middleware/local_context.py` | Injects cwd + tool inventory (python, git, node…) into system prompt before every LLM call |
| `LoopDetectionMiddleware` | `middleware/loop_detection.py` | Counts edits per file; injects doom-loop warning after N edits (default 3) |
| `PreCompletionChecklistMiddleware` | `middleware/pre_completion.py` | Intercepts no-tool-call responses; forces one verification pass before agent exits |
| `ReasoningSandwichMiddleware` | `middleware/reasoning_sandwich.py` | Extended thinking on turn 1 (planning) + final verification turn; standard thinking otherwise. **Anthropic only.** |

### Reasoning sandwich detail

```
Turn 1  (planning)     →  extended thinking   budget_tokens = 10 000
Turns 2..N-1 (build)   →  standard thinking   (disabled)
Final   (verification)  →  extended thinking   budget_tokens = 10 000
```

The blog found that maximising thinking on every turn **hurts** (53.9%) due to
timeouts; the sandwich reaches 66.5%.

### Per-middleware toggles (for custom ablations)

```python
from harness.config import get_provider
from harness.agent import create_harness_agent

cfg = get_provider("deepseek")
agent = create_harness_agent(
    provider_config=cfg,
    enable_local_context=True,
    enable_loop_detection=False,   # off for this run
    enable_pre_completion=True,
    enable_reasoning_sandwich=False,
)
```

---

## 7. Benchmark tasks

Eight self-contained coding tasks, each with a programmatic pass/fail verifier.

| Task | Tags | Tests which middleware |
|---|---|---|
| `prime_with_tests` | basic, testing | PreCompletion — must run tests |
| `fix_buggy_bsearch` | bug-fix | general capability floor |
| `rle_encode` | string | general correctness |
| `fix_buggy_mergesort` | bug-fix | LoopDetection — easy to doom-loop |
| `lru_cache` | data-structure | PreCompletion — edge cases |
| `word_freq` | file-io | LocalContext — agent sees corpus.txt |
| `retry_decorator` | decorator | edge case coverage |
| `csv_stats` | csv, file-io | LocalContext + verification |

Verification is fully programmatic (subprocess + asserts). No LLM judge.

---

## 8. Project layout

```
harness-engineering/
├── providers.toml              ← named provider configs (safe to commit)
├── main.py                     ← single-task CLI
├── PLAYBOOK.md                 ← this file
│
├── harness/
│   ├── agent.py                ← create_harness_agent() factory
│   ├── config.py               ← load providers.toml
│   ├── tracer.py               ← LocalTracer (LangChain callback → JSON)
│   └── middleware/
│       ├── local_context.py
│       ├── loop_detection.py
│       ├── pre_completion.py
│       └── reasoning_sandwich.py
│
└── benchmark/
    ├── tasks.py                ← 8 task definitions + verifiers
    ├── run.py                  ← benchmark runner (ablation, tracing, JSON output)
    ├── analyze.py              ← pattern analysis over trace files
    └── compare.py              ← side-by-side diff of result JSON files
```

---

## 9. Extending the harness

### Add a new task

In `benchmark/tasks.py`, define a `Task` with `setup()` and `verify()`:

```python
MY_TASK = Task(
    id="my_task",
    description="Create foo.py with function bar() that ...",
    setup=lambda d: None,                    # plant initial files if needed
    verify=lambda d: _run("python foo.py", d)[0] == 0,   # check result
    tags=["custom"],
)
# Add to ALL_TASKS list at the bottom of the file
```

### Add a new middleware

In `harness/middleware/`, subclass `AgentMiddleware`:

```python
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

class MyMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # modify request before calling the model
        return handler(request)
```

Register it in `create_harness_agent()` in `harness/agent.py`.

### Add a new provider

In `providers.toml`:

```toml
[providers.my_provider]
type        = "openai"
model       = "my-model-name"
base_url    = "https://my.endpoint/v1"
api_key_env = "MY_API_KEY"
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Provider 'x' not found` | Check spelling; run `--list` to see available providers |
| `requires X_API_KEY to be set` | `export X_API_KEY=...` |
| `openai.AuthenticationError` | Set the `api_key_env` var for that provider |
| Wrong endpoint | Check `base_url` in `providers.toml` |
| Agent loops forever | Lower `loop_threshold` in `create_harness_agent()` |
| `thinking` param error | Reasoning sandwich auto-disabled for OpenAI — should not occur |
| No traces written | Pass `--trace` flag; check `--trace-dir` path |
| Empty trace files | The agent may have crashed before any LLM call — check `error` field in JSON |
