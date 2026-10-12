from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    "difficulty", "ambiguity", "compositionality",
    "instruction_complexity", "safety_sensitivity", "distribution_shift"
]
CATEGORIES = ["reasoning", "instruction_following", "safety", "factuality", "formatting"]

def make_benchmark(n_cases: int = 500, seed: int = 42, task_shift: float = 0.0) -> pd.DataFrame:
    """Create a deterministic, labeled synthetic evaluation benchmark."""
    if n_cases < 20:
        raise ValueError("n_cases must be at least 20")
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n_cases, len(FEATURES)))
    cat_idx = rng.choice(len(CATEGORIES), size=n_cases, p=[.25, .20, .18, .22, .15])
    cat = np.array(CATEGORIES)[cat_idx]
    difficulty, ambiguity, composition, instruction, safety, shift = x.T

    # Nonlinear, category-dependent latent severity with interactions.
    latent = (
        -2.45 + 1.35 * difficulty + .85 * ambiguity + .85 * composition
        + .55 * instruction + .75 * safety + .75 * shift
        + 1.15 * difficulty * composition
        + .95 * safety * shift
        + .55 * ambiguity * instruction
    )
    latent += np.select(
        [cat == "reasoning", cat == "instruction_following", cat == "safety",
         cat == "factuality", cat == "formatting"],
        [.40 * composition, .45 * instruction, .85 * safety, .50 * shift, .55 * ambiguity],
        default=0.0
    )
    latent += task_shift
    probability = 1.0 / (1.0 + np.exp(-latent))
    # Severity is continuous, with a small amount of observation noise.
    severity = np.clip(probability + rng.normal(0, .055, n_cases), 0, 1)
    texts = [
        f"Synthetic {c} test case #{i}: difficulty={difficulty[i]:.2f}, "
        f"ambiguity={ambiguity[i]:.2f}, shift={shift[i]:.2f}"
        for i, c in enumerate(cat)
    ]
    df = pd.DataFrame(x, columns=FEATURES)
    df.insert(0, "case_id", [f"case_{i:05d}" for i in range(n_cases)])
    df["category"] = cat
    df["text"] = texts
    df["error_severity"] = severity
    df["is_failure"] = (severity >= .65).astype(int)
    return df

def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix with one-hot category indicators."""
    numeric = df[FEATURES].to_numpy(dtype=float)
    categories = pd.get_dummies(df["category"]).reindex(columns=CATEGORIES, fill_value=0).to_numpy(dtype=float)
    return np.concatenate([numeric, categories], axis=1)
