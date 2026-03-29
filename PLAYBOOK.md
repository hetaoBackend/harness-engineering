# Harness Engineering — Experiment Playbook

Reproduction of the techniques from
**[Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)** (LangChain blog).

---

## 1. Setup

```bash
cd harness-engineering
uv sync          # installs langchain-anthropic, langchain-openai, deepagents, etc.
```

---

## 2. Configure your model

### Option A — Anthropic (original paper setup)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Default model: `claude-sonnet-4-5-20250929`

### Option B — OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

Default model: `gpt-4o`

### Option C — OpenAI-compatible endpoint (DeepSeek, Together AI, Ollama, etc.)

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.deepseek.com/v1   # or your endpoint
```

| Provider | Base URL | Model example |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Llama-3-70b-chat-hf` |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5-coder:32b` |
| LM Studio | `http://localhost:1234/v1` | whatever you loaded |

---

## 3. Run the agent

### Basic usage

```bash
# Anthropic (default)
uv run main.py "Write a Python function that checks if a number is prime, with tests"

# OpenAI
uv run main.py "Fix the bug in main.py" --provider openai --model gpt-4o

# DeepSeek via OpenAI-compatible API
uv run main.py "Implement a binary search tree" \
  --provider openai \
  --model deepseek-chat \
  --base-url https://api.deepseek.com/v1

# Point at a specific project directory
uv run main.py "Add docstrings to all public functions" --cwd /path/to/myproject
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `task` | (required) | The task description |
| `--provider` | `anthropic` | `anthropic` or `openai` |
| `--model` | provider default | Model identifier |
| `--base-url` | env `OPENAI_BASE_URL` | Override API endpoint (openai only) |
| `--cwd` | current dir | Working directory shown to agent |
| `--thread-id` | `harness-run` | LangGraph conversation thread ID |

---

## 4. Middleware layers explained

| Layer | What it does | Toggle |
|---|---|---|
| `LocalContextMiddleware` | Injects cwd + tool inventory (python, git, node …) into the system prompt before every model call | Always on |
| `LoopDetectionMiddleware` | Counts edits per file; after `N` edits injects a doom-loop warning | `--loop-threshold` (default 3) |
| `PreCompletionChecklistMiddleware` | Intercepts stop responses, forces one verification pass (run tests, check edge cases, re-read requirements) | Always on |
| `ReasoningSandwichMiddleware` | Uses extended thinking on turn 1 (planning) and verification turn; standard thinking otherwise | Anthropic only; auto-disabled for OpenAI |

### Reasoning sandwich detail (Anthropic)

```
Turn 1  (planning)    → extended thinking (budget_tokens=10 000)
Turns 2..N-1 (build)  → standard thinking (disabled)
Final (verification)  → extended thinking (budget_tokens=10 000)
```

The blog post found that maximising thinking on every turn **hurts** score (53.9 %) due to timeouts, while the sandwich pattern reaches 66.5 %.

---

## 5. Experiment matrix

Use this table to track your own ablation runs.

| Run | Provider | Model | Local­Context | Loop­Detect | Pre­Completion | Reasoning­Sandwich | Score / notes |
|-----|----------|-------|:---:|:---:|:---:|:---:|---|
| baseline | anthropic | claude-sonnet-4-5 | ✓ | ✓ | ✓ | ✓ | |
| no sandwich | anthropic | claude-sonnet-4-5 | ✓ | ✓ | ✓ | ✗ | |
| no pre-completion | anthropic | claude-sonnet-4-5 | ✓ | ✓ | ✗ | ✓ | |
| no local ctx | anthropic | claude-sonnet-4-5 | ✗ | ✓ | ✓ | ✓ | |
| openai baseline | openai | gpt-4o | ✓ | ✓ | ✓ | ✗ | |
| deepseek | openai | deepseek-chat | ✓ | ✓ | ✓ | ✗ | |

Tune `enable_reasoning_sandwich`, `loop_threshold`, and system prompt in
`harness/agent.py` to run ablations programmatically.

---

## 6. Programmatic API

```python
import asyncio
from harness.agent import create_harness_agent

# Anthropic
agent = create_harness_agent(
    model_name="claude-sonnet-4-5-20250929",
    provider="anthropic",
)

# OpenAI-compatible (DeepSeek)
agent = create_harness_agent(
    provider="openai",
    model_name="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="sk-...",
    loop_threshold=5,
    enable_reasoning_sandwich=False,
)

result = asyncio.run(
    agent.ainvoke(
        {"messages": [{"role": "user", "content": "Your task here"}]},
        config={"configurable": {"thread_id": "exp-1"}},
    )
)
```

---

## 7. Extending the harness

All middleware lives in `harness/middleware/`. Each class inherits from
`AgentMiddleware` and can implement any of:

| Hook | When it runs |
|---|---|
| `before_agent` / `abefore_agent` | Once, before the agent loop starts |
| `wrap_model_call` / `awrap_model_call` | Around every LLM call |
| `wrap_tool_call` / `awrap_tool_call` | Around every tool execution |

Register new middleware by appending it to the `middleware` list in
`create_harness_agent()`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ANTHROPIC_API_KEY not set` | `export ANTHROPIC_API_KEY=sk-ant-...` |
| `openai.AuthenticationError` | `export OPENAI_API_KEY=...` |
| Wrong endpoint hit | Pass `--base-url` or set `OPENAI_BASE_URL` |
| Agent loops forever | Lower `--loop-threshold` or increase max turns in `create_deep_agent` |
| `thinking` errors with non-Anthropic model | Use `--provider openai`; sandwich is auto-disabled |
| Tool not found in agent | deepagents injects default shell tools; check `_detect_tools()` in `local_context.py` |
