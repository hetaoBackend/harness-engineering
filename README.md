# harness-engineering

A reproduction of the harness engineering approach described in:
**[Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)** (LangChain blog)

The blog improved a coding agent from **52.8% → 66.5%** on Terminal Bench 2.0
by changing only the system harness — model unchanged.

Built on the [deepagents](https://github.com/langchain-ai/deepagents) SDK.

---

## Middleware

| Middleware | Purpose |
|---|---|
| `LocalContextMiddleware` | Injects cwd + tool inventory into every model call |
| `LoopDetectionMiddleware` | Warns the model when it edits the same file repeatedly |
| `PreCompletionChecklistMiddleware` | Forces a verification pass before the agent exits |
| `ReasoningSandwichMiddleware` | Extended thinking on planning + verification turns only (Anthropic) |

---

## Setup

```bash
git clone https://github.com/hetaoBackend/harness-engineering
cd harness-engineering
uv sync

cp .env.example .env
# edit .env and fill in your keys
```

---

## Configure providers

All providers are defined in **`providers.toml`**. The `api_key_env` field
is the *name* of an environment variable — the key itself never goes in the file.
Keys are loaded automatically from `.env` at startup (via python-dotenv).

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

[providers.openai]
type        = "openai"
model       = "gpt-4o"
api_key_env = "OPENAI_API_KEY"

# Local Ollama — no key needed
# [providers.ollama]
# type     = "openai"
# model    = "qwen2.5-coder:32b"
# base_url = "http://localhost:11434/v1"
```

Put your keys in `.env` (copied from `.env.example`):

```bash
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Common endpoint URLs:

| Provider    | base_url                          |
|-------------|-----------------------------------|
| DeepSeek    | `https://api.deepseek.com/v1`     |
| Together AI | `https://api.together.xyz/v1`     |
| Ollama      | `http://localhost:11434/v1`       |
| LM Studio   | `http://localhost:1234/v1`        |

---

## Usage

### Run a single task

```bash
# Uses first provider in providers.toml
uv run main.py "Write a Python function that checks if a number is prime, with tests"

# Pick a provider
uv run main.py "Refactor auth.py to use JWT" --provider deepseek

# Point at a specific project directory
uv run main.py "Add docstrings to all public functions" \
  --provider anthropic \
  --cwd /path/to/myproject

# Override the model for this run
uv run main.py "Fix the bug" --provider deepseek --model deepseek-reasoner
```

### Run the benchmark

```bash
# List available tasks
uv run benchmark/run.py --list

# Run all 8 tasks with full harness
uv run benchmark/run.py --provider deepseek

# Run baseline (no harness) for comparison
uv run benchmark/run.py --provider deepseek --no-harness --out baseline.json

# Run only a subset of tasks
uv run benchmark/run.py --provider deepseek --tasks prime_with_tests lru_cache

# Save results to JSON
uv run benchmark/run.py --provider deepseek --out results.json
```

### Ablation experiment

Runs 5 middleware configurations automatically and prints a comparison matrix:

```bash
uv run benchmark/run.py --provider deepseek --ablation
```

Output:

```
  Task                    baseline  +local_ctx  +loop_det  +pre_comp  full_harness
  prime_with_tests               ✗           ✗          ✗          ✓             ✓
  fix_buggy_mergesort            ✗           ✗          ✓          ✓             ✓
  word_freq                      ✗           ✓          ✓          ✓             ✓
  SCORE                        2/8         4/8        5/8        6/8           7/8
```

### Local tracing and analysis

Capture per-task traces locally — no LangSmith needed:

```bash
# Run with tracing enabled
uv run benchmark/run.py --provider deepseek --trace --out results.json

# Analyze failure patterns across all traces
uv run benchmark/analyze.py

# Only show failed tasks
uv run benchmark/analyze.py --failed-only

# Verbose turn-by-turn detail for one task
uv run benchmark/analyze.py --run-id deepseek_1234567890 --task word_freq --verbose
```

### Compare two runs

```bash
uv run benchmark/compare.py baseline.json results.json
```

### Automated iteration (Claude Code skill)

If you have Claude Code, `/harness-iterate` automates the full loop:
run → trace → analyze → implement fix → verify improvement.

```
/harness-iterate --provider deepseek
/harness-iterate --provider deepseek --dry-run   # propose fix without implementing
```

---

## Project layout

```
harness-engineering/
├── providers.toml              ← named provider configs (safe to commit)
├── main.py                     ← single-task CLI
├── PLAYBOOK.md                 ← full experiment guide
│
├── harness/
│   ├── agent.py                ← create_harness_agent() factory
│   ├── config.py               ← load providers.toml
│   ├── tracer.py               ← LocalTracer (writes traces/*.json)
│   └── middleware/
│       ├── local_context.py
│       ├── loop_detection.py
│       ├── pre_completion.py
│       └── reasoning_sandwich.py
│
├── benchmark/
│   ├── tasks.py                ← 8 task definitions + verifiers
│   ├── run.py                  ← benchmark runner
│   ├── analyze.py              ← trace pattern analysis
│   └── compare.py              ← side-by-side result diff
│
└── .claude/skills/
    └── harness-iterate/        ← Claude Code automation skill
```

For the full experiment methodology see **[PLAYBOOK.md](PLAYBOOK.md)**.
