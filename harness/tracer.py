"""LocalTracer: captures agent runs to local JSON without LangSmith.

Each run produces one JSON file under traces/<run_id>/<task_id>.json.

Captured per run:
  - Every LLM turn: input token count, output text, tool calls requested
  - Every tool execution: name, args, result, elapsed ms
  - Middleware injections detected via system prompt diffs
  - Final pass/fail verdict

Usage (via RunnableConfig):
    from harness.tracer import LocalTracer

    tracer = LocalTracer(task_id="prime_with_tests", run_id="exp-001",
                         traces_dir=Path("traces"))
    config = {"configurable": {"thread_id": "..."}, "callbacks": [tracer]}
    await agent.ainvoke({...}, config=config)
    tracer.save()         # writes traces/exp-001/prime_with_tests.json

Or let benchmark/run.py manage it automatically via --trace.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class LocalTracer(BaseCallbackHandler):
    """Lightweight callback handler that logs agent activity to a JSON file."""

    def __init__(
        self,
        task_id: str,
        run_id: str,
        traces_dir: Path | str = Path("traces"),
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.run_id = run_id
        self.traces_dir = Path(traces_dir)
        self._started_at = time.time()
        self._turns: list[dict] = []
        self._tool_starts: dict[str, float] = {}   # run_id str → start time
        self._llm_starts: dict[str, tuple[float, list]] = {}  # run_id → (t, msgs)
        self._verdict: dict = {}

    # ── LLM hooks ─────────────────────────────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        flat = [m for group in messages for m in group]
        self._llm_starts[str(run_id)] = (time.time(), flat)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        t0, input_msgs = self._llm_starts.pop(key, (time.time(), []))
        elapsed_ms = int((time.time() - t0) * 1000)

        # Flatten output text + tool calls
        output_text = ""
        tool_calls: list[dict] = []
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    output_text += getattr(gen, "text", "") or ""
                    continue
                if hasattr(msg, "content"):
                    if isinstance(msg.content, str):
                        output_text += msg.content
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                output_text += block.get("text", "")
                for tc in getattr(msg, "tool_calls", []) or []:
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    })

        # Detect middleware injections via system message
        system_text = ""
        for m in input_msgs:
            if getattr(m, "type", "") == "system" or m.__class__.__name__ == "SystemMessage":
                system_text = str(m.content)
                break

        turn: dict = {
            "type": "llm",
            "elapsed_ms": elapsed_ms,
            "input_messages": [_summarise_msg(m) for m in input_msgs],
            "output_text": output_text[:2000],   # cap to avoid huge files
            "tool_calls_requested": tool_calls,
            "middleware_flags": _detect_middleware(system_text),
        }
        # Token usage if available
        usage = getattr(response, "llm_output", {}) or {}
        if "token_usage" in usage:
            turn["token_usage"] = usage["token_usage"]
        self._turns.append(turn)

    # ── Tool hooks ─────────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_starts[str(run_id)] = time.time()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        t0 = self._tool_starts.pop(str(run_id), time.time())
        elapsed_ms = int((time.time() - t0) * 1000)
        result_str = str(output)[:500]
        self._turns.append({
            "type": "tool",
            "name": name,
            "elapsed_ms": elapsed_ms,
            "result_preview": result_str,
        })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        self._tool_starts.pop(str(run_id), None)
        self._turns.append({
            "type": "tool_error",
            "name": name,
            "error": str(error)[:300],
        })

    # ── Save ──────────────────────────────────────────────────────────────────

    def set_verdict(self, passed: bool, reason: str, error: str = "") -> None:
        self._verdict = {"passed": passed, "reason": reason, "error": error}

    def save(self) -> Path:
        out_dir = self.traces_dir / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.task_id}.json"

        # Compute summary stats
        llm_turns = [t for t in self._turns if t["type"] == "llm"]
        tool_turns = [t for t in self._turns if t["type"] == "tool"]
        tool_errors = [t for t in self._turns if t["type"] == "tool_error"]
        edit_counts: dict[str, int] = {}
        for t in tool_turns:
            if t["name"] in ("edit_file", "write_file"):
                path_arg = ""
                for tc in (llm_turns[-1].get("tool_calls_requested", []) if llm_turns else []):
                    if tc["name"] == t["name"]:
                        path_arg = tc["args"].get("path", tc["args"].get("file_path", "?"))
                edit_counts[path_arg] = edit_counts.get(path_arg, 0) + 1

        payload = {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "started_at": self._started_at,
            "elapsed_s": round(time.time() - self._started_at, 1),
            "verdict": self._verdict,
            "summary": {
                "llm_turns": len(llm_turns),
                "tool_calls": len(tool_turns),
                "tool_errors": len(tool_errors),
                "edit_counts": edit_counts,
            },
            "turns": self._turns,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        return out_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _summarise_msg(m: BaseMessage) -> dict:
    role = getattr(m, "type", m.__class__.__name__)
    content = m.content if isinstance(m.content, str) else str(m.content)
    return {"role": role, "content_preview": content[:300]}


def _detect_middleware(system_text: str) -> list[str]:
    """Heuristically detect which middleware injected content into the system prompt."""
    flags = []
    if "<environment_context>" in system_text:
        flags.append("local_context")
    if "<doom_loop_warning>" in system_text:
        flags.append("loop_detection_triggered")
    if "[[HARNESS:VERIFICATION_CHECKLIST]]" in system_text:
        flags.append("pre_completion")
    return flags
