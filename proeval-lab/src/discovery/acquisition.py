from __future__ import annotations
import numpy as np

STRATEGIES = ["random", "uncertainty", "risk", "hybrid"]

def choose_next(
    strategy: str, candidate_indices: np.ndarray, mean: np.ndarray | None,
    std: np.ndarray | None, y_observed: np.ndarray, budget: int,
    rng: np.random.Generator, threshold: float = .65, batch_size: int = 1
) -> np.ndarray:
    """Select candidate positions; returned values are indices into the full benchmark."""
    candidates = np.asarray(candidate_indices, dtype=int)
    if len(candidates) == 0:
        return np.array([], dtype=int)
    k = min(int(batch_size), len(candidates))
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if strategy == "random" or mean is None or std is None:
        return rng.choice(candidates, size=k, replace=False)
    mu = np.asarray(mean)[candidates]
    sigma = np.asarray(std)[candidates]
    if strategy == "uncertainty":
        score = sigma
    elif strategy == "risk":
        score = mu
    else:
        # Mix predicted risk, uncertainty, and proximity to the failure boundary.
        boundary = np.exp(-np.abs(mu - threshold) / .15)
        score = .50 * mu + .30 * sigma + .20 * boundary
    # Random tie-break avoids a deterministic index bias.
    score = score + rng.uniform(0, 1e-8, size=len(score))
    return candidates[np.argsort(score)[-k:]]
