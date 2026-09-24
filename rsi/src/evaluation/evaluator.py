import time
from .sandbox import evaluate_in_subprocess
from src.models import Evaluation

def evaluate_candidate(task, code):
    # Correctness is deterministic and task-defined.
    start = time.perf_counter()
    correct, passed, total, error = task.test_fn(code)
    runtime_ms = (time.perf_counter() - start) * 1000

    if not correct:
        failure = "incorrect" if error is None else "runtime"
        return Evaluation(False, runtime_ms, 0.0, 0.0, passed, total, error, failure)

    # Benchmark score is deliberately simple and reproducible for the prototype.
    score = max(0.01, task.baseline_ms / max(runtime_ms, 0.01))
    return Evaluation(True, runtime_ms, 0.0, score, passed, total)
