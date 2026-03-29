"""Lightweight benchmark task definitions.

Each Task has:
  id          - unique slug used in CLI / output table
  description - exact text sent to the agent as the task
  setup(dir)  - creates initial files in the sandbox (may be a no-op)
  verify(dir) - runs after the agent; returns (passed: bool, reason: str)
  tags        - optional labels for filtering

Verification principle: run the agent's output code programmatically and
check results. No LLM judge — pure pass/fail via subprocess.
"""

from __future__ import annotations

import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Task:
    id: str
    description: str
    setup: Callable[[Path], None]
    verify: Callable[[Path], tuple[bool, str]]
    tags: list[str] = field(default_factory=list)


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: str, cwd: Path, timeout: int = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout}s"


def _write(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip())


# ── Task 1: prime checker with tests ─────────────────────────────────────────
# Tests PreCompletionChecklist: agent must run tests, not just write them.

def _setup_prime(d: Path) -> None:
    pass  # blank sandbox


def _verify_prime(d: Path) -> tuple[bool, str]:
    rc, out = _run("python -m pytest . -q --tb=short", d, timeout=20)
    if rc == 0:
        return True, "pytest passed"
    # fallback: maybe agent wrote plain asserts without pytest
    rc2, out2 = _run(
        "python -c \""
        "import importlib, sys, pathlib\n"
        "files = list(pathlib.Path('.').glob('*.py'))\n"
        "for f in files:\n"
        "    spec = importlib.util.spec_from_file_location('m', f)\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    try: spec.loader.exec_module(mod)\n"
        "    except Exception: pass\n"
        "ip = None\n"
        "for f in files:\n"
        "    src = f.read_text()\n"
        "    if 'is_prime' in src:\n"
        "        exec(compile(src, f, 'exec'), globals())\n"
        "        break\n"
        "assert is_prime(2) and is_prime(17) and is_prime(97)\n"
        "assert not is_prime(1) and not is_prime(0) and not is_prime(100)\n"
        "print('ok')\n"
        "\"",
        d, timeout=10,
    )
    if rc2 == 0 and "ok" in out2:
        return True, "assertions passed (no pytest)"
    return False, f"pytest failed:\n{out[:400]}"


PRIME_WITH_TESTS = Task(
    id="prime_with_tests",
    description=(
        "Create `prime.py` with a function `is_prime(n: int) -> bool`.\n"
        "Requirements:\n"
        "  is_prime(2) → True\n"
        "  is_prime(17) → True\n"
        "  is_prime(97) → True\n"
        "  is_prime(1) → False\n"
        "  is_prime(0) → False\n"
        "  is_prime(-5) → False\n"
        "  is_prime(100) → False\n"
        "Write pytest tests in `test_prime.py` covering all cases above.\n"
        "Run the tests with `pytest` and make sure they all pass before finishing."
    ),
    setup=_setup_prime,
    verify=_verify_prime,
    tags=["basic", "testing", "pre-completion"],
)


# ── Task 2: fix a buggy binary search ────────────────────────────────────────
# Tests: can the agent identify and fix an off-by-one? PreCompletion helps.

def _setup_buggy_bsearch(d: Path) -> None:
    _write(d / "bsearch.py", """
        def binary_search(arr, target):
            lo, hi = 0, len(arr)   # bug: should be len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
    """)
    _write(d / "test_bsearch.py", """
        from bsearch import binary_search

        def test_found():
            assert binary_search([1,3,5,7,9], 5) == 2

        def test_first():
            assert binary_search([1,3,5,7,9], 1) == 0

        def test_last():
            assert binary_search([1,3,5,7,9], 9) == 4

        def test_not_found():
            assert binary_search([1,3,5,7,9], 4) == -1

        def test_empty():
            assert binary_search([], 1) == -1
    """)


def _verify_buggy_bsearch(d: Path) -> tuple[bool, str]:
    rc, out = _run("python -m pytest test_bsearch.py -q --tb=short", d, timeout=20)
    if rc == 0:
        return True, "all tests pass after fix"
    return False, f"tests still failing:\n{out[:400]}"


FIX_BUGGY_BSEARCH = Task(
    id="fix_buggy_bsearch",
    description=(
        "The file `bsearch.py` contains a buggy `binary_search(arr, target)` function.\n"
        "`test_bsearch.py` has tests that currently fail.\n"
        "Find and fix the bug in `bsearch.py` so all tests in `test_bsearch.py` pass.\n"
        "Run `pytest test_bsearch.py` to confirm before finishing."
    ),
    setup=_setup_buggy_bsearch,
    verify=_verify_buggy_bsearch,
    tags=["bug-fix", "testing"],
)


# ── Task 3: run-length encoding ──────────────────────────────────────────────

def _setup_rle(d: Path) -> None:
    pass


def _verify_rle(d: Path) -> tuple[bool, str]:
    probe = textwrap.dedent("""
        import sys, pathlib
        for f in pathlib.Path('.').glob('*.py'):
            src = f.read_text()
            if 'rle' in src.lower() or 'encode' in src.lower() or 'compress' in src.lower():
                exec(compile(src, f, 'exec'), globals())
                break
        fn = None
        for name in ('rle_encode', 'encode', 'compress', 'rle'):
            if name in dir():
                fn = eval(name)
                break
        assert fn is not None, "function not found"
        assert fn("aabbbcccc") == "a2b3c4", f"got {fn('aabbbcccc')!r}"
        assert fn("abc") == "a1b1c1" or fn("abc") == "abc", f"got {fn('abc')!r}"
        assert fn("") == "", f"got {fn('')!r}"
        assert fn("aaaa") == "a4", f"got {fn('aaaa')!r}"
        print("ok")
    """)
    (d / "_probe.py").write_text(probe)
    rc, out = _run("python _probe.py", d, timeout=10)
    (d / "_probe.py").unlink(missing_ok=True)
    if rc == 0 and "ok" in out:
        return True, "all assertions passed"
    return False, out[:400]


RLE_ENCODE = Task(
    id="rle_encode",
    description=(
        "Create `rle.py` with a function `rle_encode(s: str) -> str` that performs\n"
        "run-length encoding: each run of identical characters becomes `<char><count>`.\n"
        "Examples:\n"
        "  rle_encode('aabbbcccc') → 'a2b3c4'\n"
        "  rle_encode('aaaa')      → 'a4'\n"
        "  rle_encode('')          → ''\n"
        "Write tests and run them before finishing."
    ),
    setup=_setup_rle,
    verify=_verify_rle,
    tags=["basic", "string"],
)


# ── Task 4: fix a buggy merge sort ───────────────────────────────────────────
# Loop detection: if agent keeps patching the wrong line it'll loop.

def _setup_buggy_mergesort(d: Path) -> None:
    _write(d / "msort.py", """
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            return merge(left, right)

        def merge(left, right):
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            # bug: only appends one of the two remaining tails
            result.extend(left[i:])
            return result   # missing: result.extend(right[j:])
    """)
    _write(d / "test_msort.py", """
        from msort import merge_sort

        def test_basic():
            assert merge_sort([3,1,4,1,5,9,2,6]) == [1,1,2,3,4,5,6,9]

        def test_empty():
            assert merge_sort([]) == []

        def test_single():
            assert merge_sort([42]) == [42]

        def test_already_sorted():
            assert merge_sort([1,2,3,4,5]) == [1,2,3,4,5]

        def test_reversed():
            assert merge_sort([5,4,3,2,1]) == [1,2,3,4,5]
    """)


def _verify_buggy_mergesort(d: Path) -> tuple[bool, str]:
    rc, out = _run("python -m pytest test_msort.py -q --tb=short", d, timeout=20)
    if rc == 0:
        return True, "all tests pass"
    return False, f"tests failing:\n{out[:400]}"


FIX_BUGGY_MERGESORT = Task(
    id="fix_buggy_mergesort",
    description=(
        "`msort.py` contains a buggy merge sort. `test_msort.py` has failing tests.\n"
        "Find and fix the bug in `msort.py`.\n"
        "Run `pytest test_msort.py` to confirm all tests pass before finishing."
    ),
    setup=_setup_buggy_mergesort,
    verify=_verify_buggy_mergesort,
    tags=["bug-fix", "loop-detection"],
)


# ── Task 5: LRU cache ────────────────────────────────────────────────────────

def _setup_lru(d: Path) -> None:
    pass


def _verify_lru(d: Path) -> tuple[bool, str]:
    probe = textwrap.dedent("""
        import sys, pathlib
        for f in pathlib.Path('.').glob('*.py'):
            src = f.read_text()
            if 'LRU' in src or 'lru' in src.lower():
                exec(compile(src, f, 'exec'), globals())
                break

        # find the class
        cls = None
        for name in ('LRUCache', 'LruCache', 'lru_cache'):
            if name in dir():
                cls = eval(name)
                break
        assert cls is not None, "LRUCache class not found"

        c = cls(2)
        c.put(1, 1)
        c.put(2, 2)
        assert c.get(1) == 1
        c.put(3, 3)           # evicts key 2
        assert c.get(2) == -1, f"key 2 should be evicted, got {c.get(2)}"
        c.put(4, 4)           # evicts key 1
        assert c.get(1) == -1, f"key 1 should be evicted, got {c.get(1)}"
        assert c.get(3) == 3
        assert c.get(4) == 4
        print("ok")
    """)
    (d / "_probe.py").write_text(probe)
    rc, out = _run("python _probe.py", d, timeout=10)
    (d / "_probe.py").unlink(missing_ok=True)
    if rc == 0 and "ok" in out:
        return True, "LRU eviction correct"
    return False, out[:400]


LRU_CACHE = Task(
    id="lru_cache",
    description=(
        "Create `lru.py` with class `LRUCache(capacity: int)` implementing:\n"
        "  get(key) → value or -1 if not present\n"
        "  put(key, value) → insert; evict least-recently-used when over capacity\n"
        "Example (capacity=2):\n"
        "  put(1,1), put(2,2), get(1)→1\n"
        "  put(3,3)  ← evicts key 2\n"
        "  get(2) → -1\n"
        "Write tests, run them, confirm they pass before finishing."
    ),
    setup=_setup_lru,
    verify=_verify_lru,
    tags=["data-structure", "testing"],
)


# ── Task 6: word frequency from file ─────────────────────────────────────────

def _setup_word_freq(d: Path) -> None:
    _write(d / "corpus.txt", """
        the quick brown fox jumps over the lazy dog
        the dog barked at the fox
        the fox ran away quickly
        a quick brown dog outpaced a lazy fox
    """)


def _verify_word_freq(d: Path) -> tuple[bool, str]:
    probe = textwrap.dedent("""
        import pathlib, sys
        for f in pathlib.Path('.').glob('*.py'):
            if f.name.startswith('_'): continue
            src = f.read_text()
            if 'freq' in src.lower() or 'count' in src.lower() or 'word' in src.lower():
                exec(compile(src, f, 'exec'), globals())
                break

        # try calling top_words / word_freq / word_frequency / count_words
        fn = None
        for name in ('top_words', 'word_freq', 'word_frequency', 'count_words', 'top_n_words'):
            if name in dir():
                fn = eval(name)
                break
        assert fn is not None, "word frequency function not found"

        # call with file path, get top 3
        result = fn('corpus.txt', 3) if 'n' in fn.__code__.co_varnames or fn.__code__.co_argcount > 1 else fn('corpus.txt')
        if isinstance(result, dict):
            top = sorted(result.items(), key=lambda x: -x[1])[:3]
            words = [w for w, _ in top]
        elif isinstance(result, list):
            words = [w if isinstance(w, str) else w[0] for w in result[:3]]
        else:
            words = list(result)[:3]

        assert 'the' in words, f"'the' should be top word, got {words}"
        print("ok")
    """)
    (d / "_probe.py").write_text(probe)
    rc, out = _run("python _probe.py", d, timeout=10)
    (d / "_probe.py").unlink(missing_ok=True)
    if rc == 0 and "ok" in out:
        return True, "'the' correctly identified as top word"
    return False, out[:400]


WORD_FREQ = Task(
    id="word_freq",
    description=(
        "The file `corpus.txt` contains text.\n"
        "Create `wordfreq.py` with function `top_words(filepath: str, n: int) -> list`\n"
        "that returns the top-n most frequent words (case-insensitive, ignoring punctuation).\n"
        "Example: top_words('corpus.txt', 3) should return 'the' as #1.\n"
        "Write and run tests before finishing."
    ),
    setup=_setup_word_freq,
    verify=_verify_word_freq,
    tags=["file-io", "string"],
)


# ── Task 7: retry decorator ───────────────────────────────────────────────────

def _setup_retry(d: Path) -> None:
    pass


def _verify_retry(d: Path) -> tuple[bool, str]:
    probe = textwrap.dedent("""
        import pathlib
        for f in pathlib.Path('.').glob('*.py'):
            if f.name.startswith('_'): continue
            src = f.read_text()
            if 'retry' in src.lower():
                exec(compile(src, f, 'exec'), globals())
                break

        fn = None
        for name in ('retry', 'Retry', 'with_retry'):
            if name in dir():
                fn = eval(name)
                break
        assert fn is not None, "retry decorator not found"

        # test: function fails twice then succeeds
        call_count = [0]
        @fn(max_attempts=3)
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "ok"

        result = flaky()
        assert result == "ok", f"expected 'ok', got {result!r}"
        assert call_count[0] == 3, f"expected 3 calls, got {call_count[0]}"

        # test: exhausted retries raises
        @fn(max_attempts=2)
        def always_fails():
            raise RuntimeError("boom")

        try:
            always_fails()
            assert False, "should have raised"
        except RuntimeError:
            pass

        print("ok")
    """)
    (d / "_probe.py").write_text(probe)
    rc, out = _run("python _probe.py", d, timeout=10)
    (d / "_probe.py").unlink(missing_ok=True)
    if rc == 0 and "ok" in out:
        return True, "retry decorator works correctly"
    return False, out[:400]


RETRY_DECORATOR = Task(
    id="retry_decorator",
    description=(
        "Create `retry.py` with a decorator `retry(max_attempts: int)`.\n"
        "Behaviour:\n"
        "  - On exception, retry the function up to max_attempts times total.\n"
        "  - If all attempts fail, re-raise the last exception.\n"
        "  - If an attempt succeeds, return its result immediately.\n"
        "Example:\n"
        "  @retry(max_attempts=3)\n"
        "  def flaky(): ...  # fails twice, succeeds on attempt 3\n"
        "Write tests (including the exhausted-retry case) and run them."
    ),
    setup=_setup_retry,
    verify=_verify_retry,
    tags=["decorator", "testing"],
)


# ── Task 8: CSV column stats ─────────────────────────────────────────────────
# Tests local-context awareness: agent sees corpus.csv in the listing.

def _setup_csv_stats(d: Path) -> None:
    _write(d / "data.csv", """
        name,age,score
        Alice,30,88.5
        Bob,25,92.0
        Carol,35,76.0
        Dave,28,95.5
        Eve,32,81.0
    """)


def _verify_csv_stats(d: Path) -> tuple[bool, str]:
    probe = textwrap.dedent("""
        import pathlib, importlib.util
        for f in pathlib.Path('.').glob('*.py'):
            if f.name.startswith('_'): continue
            src = f.read_text()
            if 'csv' in src.lower() or 'stat' in src.lower():
                exec(compile(src, f, 'exec'), globals())
                break

        fn = None
        for name in ('col_stats', 'column_stats', 'csv_stats', 'stats', 'compute_stats'):
            if name in dir():
                fn = eval(name)
                break
        assert fn is not None, "stats function not found"

        result = fn('data.csv')
        # result should be a dict with 'age' and 'score' keys
        if isinstance(result, dict) and 'age' in result:
            age = result['age']
            if isinstance(age, dict):
                mean_age = age.get('mean') or age.get('avg') or age.get('average')
            else:
                mean_age = age
            assert abs(float(mean_age) - 30.0) < 0.1, f"mean age should be 30, got {mean_age}"
        print("ok")
    """)
    (d / "_probe.py").write_text(probe)
    rc, out = _run("python _probe.py", d, timeout=10)
    (d / "_probe.py").unlink(missing_ok=True)
    if rc == 0 and "ok" in out:
        return True, "column stats correct"
    return False, out[:400]


CSV_STATS = Task(
    id="csv_stats",
    description=(
        "The file `data.csv` has columns: name, age, score.\n"
        "Create `csvstats.py` with function `col_stats(filepath: str) -> dict`\n"
        "that returns mean, min, max for each numeric column.\n"
        "Example output:\n"
        "  {'age': {'mean': 30.0, 'min': 25, 'max': 35},\n"
        "   'score': {'mean': 86.6, 'min': 76.0, 'max': 95.5}}\n"
        "Write tests and run them before finishing."
    ),
    setup=_setup_csv_stats,
    verify=_verify_csv_stats,
    tags=["file-io", "csv", "local-context"],
)


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_TASKS: list[Task] = [
    PRIME_WITH_TESTS,
    FIX_BUGGY_BSEARCH,
    RLE_ENCODE,
    FIX_BUGGY_MERGESORT,
    LRU_CACHE,
    WORD_FREQ,
    RETRY_DECORATOR,
    CSV_STATS,
]

TASK_MAP: dict[str, Task] = {t.id: t for t in ALL_TASKS}
