import numpy as np


def bootstrap(values, statistic=np.mean, n_iterations=5000, confidence=0.95, seed=42):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty collection.")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_iterations)

    for i in range(n_iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[i] = statistic(sample)

    alpha = 1 - confidence

    return {
        "estimate": float(statistic(values)),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
        "samples": estimates,
    }
