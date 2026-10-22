import numpy as np
from src.data.preference_dataset import PreferenceData

def make_synthetic_data(n_users=80, n_prompts=500, feature_dim=8, heterogeneity=0.8, noise=0.15, seed=42):
    """Create noisy pairwise preferences from simulated users with known reward weights."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_prompts, feature_dim))
    b = rng.normal(size=(n_prompts, feature_dim))
    center = rng.normal(0, 0.7, size=feature_dim)
    weights = center + heterogeneity * rng.normal(size=(n_users, feature_dim))
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-8)
    user_ids = rng.integers(0, n_users, size=n_prompts)
    logits = np.sum((a-b) * weights[user_ids], axis=1) + rng.normal(0, noise, n_prompts)
    probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    labels = rng.binomial(1, probs).astype(np.float32)
    return PreferenceData(a.astype(np.float32), b.astype(np.float32), user_ids,
                          labels, np.arange(n_prompts)), weights.astype(np.float32)

def split_by_prompt(data, test_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    n_test = max(1, int(len(idx) * test_fraction))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    mt = np.zeros(len(data), dtype=bool); mt[train_idx] = True
    mv = np.zeros(len(data), dtype=bool); mv[test_idx] = True
    return data.subset(mt), data.subset(mv)
