"""Deterministic offline baseline for a small program-repair benchmark."""
import shutil
import tempfile
import time
from pathlib import Path
from .benchmarks import CASES, write_case
from .patching import safe_candidate, unified_diff
from .validation import run_tests

def propose_patch(case_id: str, source: str) -> str:
    """Known-case baseline. Replace this provider seam with an LLM for model experiments."""
    replacements = {
        "average_off_by_one": ("len(values) - 1", "len(values)"),
        "empty_list": ("def first_item(items):\n    return items[0]",
                       "def first_item(items):\n    if not items:\n        return None\n    return items[0]"),
        "discount_boundary": ("if price > threshold:", "if price >= threshold:"),
        "safe_division": ("    return 0", "    return None"),
    }
    if case_id not in replacements:
        return source
    old, new = replacements[case_id]
    return source.replace(old, new, 1) if old in source else source

def repair_case(case_id: str, max_attempts: int = 2, timeout_seconds: int = 4) -> dict:
    started = time.perf_counter()
    def result(status, message, **kwargs):
        return {"status": status, "message": message, "patch": "", "test_output": "",
                "elapsed_ms": (time.perf_counter()-started)*1000, "attempts": 0,
                "case_id": case_id, **kwargs}
    if case_id not in CASES:
        return result("abstained", f"Unknown benchmark case: {case_id}")
    if not 1 <= max_attempts <= 3:
        return result("abstained", "max_attempts must be between 1 and 3")
    workspace = Path(tempfile.mkdtemp(prefix="flowfix-"))
    try:
        write_case(case_id, workspace)
        source_path = workspace / "solution.py"
        original = source_path.read_text(encoding="utf-8")
        baseline_passed, baseline_output = run_tests(workspace, timeout_seconds)
        if baseline_passed:
            return result("abstained", "Baseline tests already pass; no reproducible failure.",
                          test_output=baseline_output)
        candidate = propose_patch(case_id, original)
        ok, reason = safe_candidate(original, candidate)
        if not ok or candidate == original:
            return result("abstained", reason if not ok else "No safe patch candidate generated",
                          test_output=baseline_output)
        patch = unified_diff(original, candidate)
        source_path.write_text(candidate, encoding="utf-8")
        passed, output = run_tests(workspace, timeout_seconds)
        if passed:
            return result("verified_suggestion",
                          "Candidate passed the benchmark tests. Human review is still required.",
                          patch=patch, test_output=output, attempts=1)
        return result("abstained", "Candidate did not pass validation; no patch returned.",
                      test_output=output, attempts=1)
    except Exception as exc:
        return result("error", f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
