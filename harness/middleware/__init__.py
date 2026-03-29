"""Harness engineering middleware implementations.

Based on: https://blog.langchain.com/improving-deep-agents-with-harness-engineering/

Three key middleware components:
- LocalContextMiddleware: proactively injects directory + tooling context
- LoopDetectionMiddleware: detects doom loops (repeated edits to same file)
- PreCompletionChecklistMiddleware: forces verification before agent exits
Plus a ReasoningSandwichMiddleware for adaptive thinking budget allocation.
"""

from harness.middleware.local_context import LocalContextMiddleware
from harness.middleware.loop_detection import LoopDetectionMiddleware
from harness.middleware.pre_completion import PreCompletionChecklistMiddleware
from harness.middleware.reasoning_sandwich import ReasoningSandwichMiddleware

__all__ = [
    "LocalContextMiddleware",
    "LoopDetectionMiddleware",
    "PreCompletionChecklistMiddleware",
    "ReasoningSandwichMiddleware",
]
