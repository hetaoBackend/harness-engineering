# harness-engineering

A reproduction / skeleton of the harness engineering approach described in:
**[Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)** (LangChain blog)

Built on the [deepagents](https://github.com/langchain-ai/deepagents) SDK.

## What's inside

| Middleware | Purpose |
|---|---|
| `LocalContextMiddleware` | Proactively injects working-directory structure and available tool inventory into every model call |
| `LoopDetectionMiddleware` | Tracks per-file edit counts; warns the model when it appears stuck in a doom loop |
| `PreCompletionChecklistMiddleware` | Intercepts no-tool-call (stop) responses and forces a verification pass before the agent exits |
| `ReasoningSandwichMiddleware` | Allocates extended thinking for planning + verification turns, standard thinking for implementation |

## Setup

```bash
# Install dependencies
uv sync

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
uv run main.py "Write a Python function that checks if a number is prime, with tests"
uv run main.py "Refactor auth.py to use JWT" --cwd /path/to/project
uv run main.py --help
```

## Project layout

```
harness/
├── agent.py          # create_harness_agent() factory
└── middleware/
    ├── local_context.py      # LocalContextMiddleware
    ├── loop_detection.py     # LoopDetectionMiddleware
    ├── pre_completion.py     # PreCompletionChecklistMiddleware
    └── reasoning_sandwich.py # ReasoningSandwichMiddleware
main.py               # CLI entry point
```
