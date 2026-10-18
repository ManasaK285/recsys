"""Data generation, loading, and schema validation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"item_id", "rater_id", "group", "severity", "rating"}


def generate_synthetic_data(
    n_items: int = 180,
    raters_per_group: int = 18,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic ordinal ratings with intentionally different response styles."""
    rng = np.random.default_rng(seed)
    groups = {
        # Noise and scale use are illustrative, not demographic assumptions.
        "Group_A": {"slope": 0.95, "noise": 0.45, "offset": -0.15},
        "Group_B": {"slope": 0.62, "noise": 0.72, "offset": 0.15},
        "Group_C": {"slope": 1.25, "noise": 0.55, "offset": 0.00},
    }
    rows = []
    severity_by_item = rng.integers(0, 5, size=n_items)
    violation_types = np.array(["harassment", "privacy", "violence", "misinformation"])
    item_types = rng.choice(violation_types, size=n_items)
    for group, params in groups.items():
        for ri in range(raters_per_group):
            rater_id = f"{group}_r{ri:03d}"
            personal_bias = rng.normal(0, 0.22)
            personal_noise = max(0.2, params["noise"] + rng.normal(0, 0.12))
            for item_idx, severity in enumerate(severity_by_item):
                latent = (
                    params["slope"] * (severity - 2)
                    + params["offset"]
                    + personal_bias
                    + rng.normal(0, personal_noise)
                )
                # Convert latent value to a 0..4 ordinal scale.
                rating = int(np.clip(np.rint(latent + 2), 0, 4))
                rows.append({
                    "item_id": f"item_{item_idx:04d}",
                    "rater_id": rater_id,
                    "group": group,
                    "severity": int(severity),
                    "rating": rating,
                    "violation_type": str(item_types[item_idx]),
                })
    df = pd.DataFrame(rows)
    validate_ratings(df)
    return df


def validate_ratings(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Ratings dataset is empty.")
    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Required columns contain missing values.")
    if not pd.api.types.is_numeric_dtype(df["rating"]):
        raise ValueError("'rating' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["severity"]):
        raise ValueError("'severity' must be numeric.")
    if not df["rating"].between(0, 4).all():
        raise ValueError("'rating' values must be in the inclusive range 0..4.")
    if not np.allclose(df["rating"], np.round(df["rating"])):
        raise ValueError("'rating' values must be integers.")
    if df["item_id"].nunique() < 2:
        raise ValueError("At least two unique items are required.")
    if df["rater_id"].nunique() < 2:
        raise ValueError("At least two unique raters are required.")


def load_ratings(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_ratings(df)
    df["rating"] = df["rating"].astype(int)
    return df
