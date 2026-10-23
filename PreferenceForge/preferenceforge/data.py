from pathlib import Path
import numpy as np
import pandas as pd
from preferenceforge.config import FEATURES, SEED

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def generate_dataset(users=30, pairs_per_user=80, seed=SEED, heterogeneity=0.8):
    """Create a reproducible synthetic pairwise dataset with user-specific preferences."""
    rng = np.random.default_rng(seed)
    rows = []
    user_weights = {}
    for u in range(users):
        weights = rng.normal(0, heterogeneity, len(FEATURES))
        weights += np.array([0.8, 0.3, 0.0, 0.0, 0.8])
        user_weights[f"user_{u:03d}"] = weights.tolist()
        for i in range(pairs_per_user):
            a = rng.uniform(0, 1, len(FEATURES))
            b = rng.uniform(0, 1, len(FEATURES))
            # Add structured prompt-family variation to avoid a purely identical dataset.
            prompt = rng.choice([
                "Explain a technical concept.",
                "Help debug a Python program.",
                "Summarize a long document.",
                "Give safety-aware practical advice.",
                "Compare two possible approaches."
            ])
            p_a = sigmoid(np.dot(weights, a - b))
            label = int(rng.random() < p_a)
            row = {"user_id": f"user_{u:03d}", "prompt": prompt,
                   "response_a": f"Candidate A for comparison {i}",
                   "response_b": f"Candidate B for comparison {i}", "label": label}
            for j, f in enumerate(FEATURES):
                row[f"a_{f}"] = float(a[j])
                row[f"b_{f}"] = float(b[j])
            rows.append(row)
    return pd.DataFrame(rows), user_weights

def save_dataset(df, path="data/preferences.csv"):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p

def load_dataset(path="data/preferences.csv"):
    return pd.read_csv(path)
