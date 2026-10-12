from __future__ import annotations
import numpy as np

def uniform_sample_estimate(observed_y: np.ndarray) -> float:
    """Sample mean; unbiased for the finite population under uniform random sampling."""
    if len(observed_y) == 0:
        return float("nan")
    return float(np.mean(observed_y))

def gp_plugin_estimate(observed_indices: np.ndarray, observed_y: np.ndarray,
                       all_predictions: np.ndarray) -> float:
    """Plug-in population mean: observed labels plus surrogate predictions for remaining cases."""
    n = len(all_predictions)
    if n == 0:
        return float("nan")
    estimate = np.asarray(all_predictions, dtype=float).copy()
    estimate[np.asarray(observed_indices, dtype=int)] = np.asarray(observed_y, dtype=float)
    return float(np.mean(np.clip(estimate, 0, 1)))
