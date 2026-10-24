from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, chi2_contingency

def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "protocol", "condition"], as_index=False)
            .agg(trials=("trial", "count"),
                 mean_optimality_gap=("optimality_gap", "mean"),
                 median_optimality_gap=("optimality_gap", "median"),
                 mean_winner_qualification=("winner_qualification", "mean"),
                 group_a_selection_rate=("winner_group", lambda x: (x == "A").mean())))

def paired_identity_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of matched trials where the winner changes across identity conditions."""
    keys = ["trial", "model", "protocol"]
    wide = df.pivot_table(index=keys, columns="condition", values="winner_id", aggfunc="first").dropna()
    if not {"anonymous", "visible"}.issubset(wide.columns):
        return pd.DataFrame(columns=["model", "protocol", "paired_trials", "identity_sensitivity"])
    wide["changed"] = wide["anonymous"] != wide["visible"]
    return wide.reset_index().groupby(["model", "protocol"], as_index=False).agg(
        paired_trials=("changed", "size"),
        identity_sensitivity=("changed", "mean")
    )

def bootstrap_mean_ci(values, confidence_level=0.95, seed=42):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return (float(np.mean(arr)) if len(arr) else float("nan"), float("nan"), float("nan"))
    res = bootstrap((arr,), np.mean, confidence_level=confidence_level,
                    n_resamples=2000, random_state=np.random.default_rng(seed), method="percentile")
    return float(np.mean(arr)), float(res.confidence_interval.low), float(res.confidence_interval.high)

def selection_disparity(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute difference in winner rates between profile groups A and B."""
    rows = []
    for keys, part in df.groupby(["model", "protocol", "condition"]):
        rate_a = (part["winner_group"] == "A").mean()
        rate_b = (part["winner_group"] == "B").mean()
        rows.append({"model": keys[0], "protocol": keys[1], "condition": keys[2],
                     "rate_a": rate_a, "rate_b": rate_b,
                     "absolute_selection_disparity": abs(rate_a-rate_b),
                     "trials": len(part)})
    return pd.DataFrame(rows)

def write_analysis(df: pd.DataFrame, out_dir):
    from pathlib import Path
    out = Path(out_dir)
    summarize_results(df).to_csv(out / "summary_metrics.csv", index=False)
    paired_identity_sensitivity(df).to_csv(out / "identity_sensitivity.csv", index=False)
    selection_disparity(df).to_csv(out / "selection_disparity.csv", index=False)
