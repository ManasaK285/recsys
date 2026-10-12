from __future__ import annotations
import numpy as np

def failure_metrics(y_true: np.ndarray, queried: np.ndarray, threshold: float = .65) -> dict:
    y = np.asarray(y_true, dtype=float)
    queried = np.asarray(queried, dtype=int)
    truth_failures = y >= threshold
    found_failures = queried[ y[queried] >= threshold ] if len(queried) else np.array([], dtype=int)
    tp = len(found_failures)
    precision = tp / len(queried) if len(queried) else 0.0
    recall = tp / int(truth_failures.sum()) if truth_failures.sum() else 0.0
    # Category coverage is calculated by caller when categories are available.
    return {"failure_precision": float(precision), "failure_recall": float(recall), "failures_found": int(tp)}
