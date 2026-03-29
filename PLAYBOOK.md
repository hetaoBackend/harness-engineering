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

## 2. Configure providers

All providers live in **`providers.toml`** (project root).
Each entry has a name you use at the CLI. Edit it to add/remove providers.

```toml
[providers.anthropic]
type        = "anthropic"
model       = "claude-sonnet-4-5-20250929"
api_key_env = "ANTHROPIC_API_KEY"   # env var name — not the key itself

[providers.deepseek]
type        = "openai"
model       = "deepseek-chat"
base_url    = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"

[providers.ollama]
type     = "openai"
model    = "qwen2.5-coder:32b"
base_url = "http://localhost:11434/v1"
# no api_key_env for local servers
```

Then export the relevant key(s):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=sk-...
```

### Quick-reference: common endpoint URLs

| Provider | base_url | Model example |
|---|---|---|
| Anthropic | *(built-in)* | `claude-sonnet-4-5-20250929` |
| OpenAI | *(built-in)* | `gpt-4o` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Llama-3-70b-chat-hf` |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5-coder:32b` |
| LM Studio | `http://localhost:1234/v1` | *(whatever you loaded)* |

---

## 3. Run the agent

```bash
# Default provider (first entry in providers.toml)
uv run main.py "Write a prime-checker with tests"

# Pick a named provider
uv run main.py "Fix the bug in auth.py" --provider deepseek
uv run main.py "Refactor to use async" --provider anthropic

# Override the model for one run
uv run main.py "..." --provider openai --model gpt-4o-mini

# Custom config file (e.g. a team-shared one)
uv run main.py "..." --config ~/team-providers.toml --provider staging

# Point at a different project
uv run main.py "Add docstrings" --cwd /path/to/myproject --provider deepseek
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `task` | *(required)* | Task description |
| `--provider NAME` | first in config | Named provider from `providers.toml` |
| `--model MODEL` | from config | Override model for this run only |
| `--config PATH` | `./providers.toml` | Path to providers TOML file |
| `--cwd PATH` | current dir | Working directory shown to agent |
| `--thread-id ID` | `harness-run` | LangGraph conversation thread ID |

---

## 4. Middleware layers explained

| Layer | What it does | Active |
|---|---|---|
| `LocalContextMiddleware` | Injects cwd + tool inventory (python, git, node…) into system prompt every call | Always |
| `LoopDetectionMiddleware` | Counts edits per file; injects doom-loop warning after N edits | Always |
| `PreCompletionChecklistMiddleware` | Intercepts stop responses; forces verification pass before exit | Always |
| `ReasoningSandwichMiddleware` | Extended thinking on planning + verification turns; standard otherwise | Anthropic only |

### Reasoning sandwich (Anthropic)

```
Turn 1  (planning)    → extended thinking  budget_tokens=10 000
Turns 2..N-1 (build)  → standard thinking  (disabled)
Final   (verify)      → extended thinking  budget_tokens=10 000
```

The blog found that maximising thinking on every turn **hurts** (53.9 %) due to timeouts. The sandwich reaches 66.5 %.
Auto-disabled for OpenAI-compatible providers.

---

## 5. Experiment matrix

| Run | Provider | Model | LocalCtx | LoopDet | PreComp | Sandwich | Score / notes |
|-----|----------|-------|:---:|:---:|:---:|:---:|---|
| baseline | anthropic | claude-sonnet-4-5 | ✓ | ✓ | ✓ | ✓ | |
| no sandwich | anthropic | claude-sonnet-4-5 | ✓ | ✓ | ✓ | ✗ | |
| no pre-completion | anthropic | claude-sonnet-4-5 | ✓ | ✓ | ✗ | ✓ | |
| no local ctx | anthropic | claude-sonnet-4-5 | ✗ | ✓ | ✓ | ✓ | |
| deepseek baseline | deepseek | deepseek-chat | ✓ | ✓ | ✓ | ✗ | |
| openai baseline | openai | gpt-4o | ✓ | ✓ | ✓ | ✗ | |

---

## 6. Programmatic API

```python
import asyncio
from harness.config import get_provider
from harness.agent import create_harness_agent

# Load from providers.toml
cfg = get_provider("deepseek")
agent = create_harness_agent(provider_config=cfg, loop_threshold=5)

# Override model for this run
cfg = get_provider("deepseek", model_override="deepseek-reasoner")
agent = create_harness_agent(provider_config=cfg)

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

Register new middleware by appending to the `middleware` list in `create_harness_agent()`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Provider 'x' not found` | Check spelling; run `--help` to see available providers |
| `requires ANTHROPIC_API_KEY to be set` | `export ANTHROPIC_API_KEY=sk-ant-...` |
| `openai.AuthenticationError` | Set the `api_key_env` var for that provider |
| Wrong endpoint used | Check `base_url` in `providers.toml` |
| Agent loops forever | Lower `loop_threshold` in `create_harness_agent()` |
| `thinking` errors with non-Anthropic model | Sandwich is auto-disabled; should not happen |
| Tool not found | See `_detect_tools()` in `harness/middleware/local_context.py` |
