from dataclasses import dataclass
from typing import Callable, Dict, Any
import random
import time

@dataclass
class Task:
    task_id: str
    name: str
    description: str
    baseline_ms: float
    candidate_factories: Dict[str, Callable[[], str]]
    test_fn: Callable[[str], tuple[bool, int, int, str | None]]

def _topk_heap():
    return """def solve(nums, k):
    import heapq
    return heapq.nlargest(k, nums)
"""

def _topk_sort():
    return """def solve(nums, k):
    return sorted(nums, reverse=True)[:k]
"""

def _topk_set():
    return """def solve(nums, k):
    # Intentionally poor baseline for exploration diversity.
    return sorted(set(nums), reverse=True)[:k]
"""

def _topk_numpy():
    return """def solve(nums, k):
    # Optional dependency is deliberately avoided; use Python partition-like logic.
    nums = list(nums)
    nums.sort(reverse=True)
    return nums[:k]
"""

def _tests_topk(code):
    scope = {}
    try:
        exec(code, {"__builtins__": __builtins__}, scope)
        solve = scope["solve"]
        tests = [
            ([3,1,5,2,4], 2),
            ([-1,-5,2,8,3], 3),
            ([7], 1),
            ([4,4,1,2,4], 3),
            (list(range(100, 0, -1)), 10),
        ]
        passed = 0
        for nums, k in tests:
            got = solve(nums, k)
            expected = sorted(nums, reverse=True)[:k]
            if got == expected:
                passed += 1
        return passed == len(tests), passed, len(tests), None
    except Exception as e:
        return False, 0, len(tests), f"{type(e).__name__}: {e}"

def make_registry():
    return {
        "topk": Task(
            "topk",
            "Top-K Elements",
            "Return the k largest elements of an integer array in descending order.",
            1.0,
            {
                "heap": _topk_heap,
                "sort": _topk_sort,
                "set_sort": _topk_set,
                "numpy_style": _topk_numpy,
            },
            _tests_topk,
        )
    }
