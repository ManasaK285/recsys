import numpy as np


def permutation_test(group_a, group_b, n_iterations=10000, seed=42):
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError("Both groups must contain observations.")

    observed = group_a.mean() - group_b.mean()
    combined = np.concatenate([group_a, group_b])

    rng = np.random.default_rng(seed)
    null = np.empty(n_iterations)
    n_a = len(group_a)

    for i in range(n_iterations):
        shuffled = rng.permutation(combined)
        null[i] = shuffled[:n_a].mean() - shuffled[n_a:].mean()

    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_iterations + 1)

    return {
        "observed_difference": float(observed),
        "p_value": float(p_value),
        "null_distribution": null,
    }
