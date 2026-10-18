"""Transparent, non-parametric summaries for ordinal safety ratings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def normalized_entropy(values: pd.Series, scale_min: int = 0, scale_max: int = 4) -> float:
    counts = values.value_counts().reindex(range(scale_min, scale_max + 1), fill_value=0)
    probs = counts.to_numpy(dtype=float)
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    probs /= probs.sum()
    entropy = -(probs * np.log(probs)).sum()
    max_entropy = np.log(scale_max - scale_min + 1)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def rater_responsiveness(df: pd.DataFrame) -> dict:
    """Summarize how one rater maps increasing reference severity to ratings."""
    per_severity = df.groupby("severity", observed=True)["rating"].median().sort_index()
    if len(per_severity) >= 2:
        vals = per_severity.to_numpy(dtype=float)
        monotonicity = float(np.mean(np.diff(vals) >= 0))
        rho_result = spearmanr(df["severity"], df["rating"])
        rho = float(rho_result.statistic) if np.isfinite(rho_result.statistic) else 0.0
    else:
        monotonicity, rho = np.nan, np.nan
    means = df.groupby("severity", observed=True)["rating"].mean()
    span = float(means.max() - means.min()) if len(means) else np.nan
    distinct_levels = float(df["rating"].nunique())
    return {
        "scale_span": span,
        "distinct_rating_levels": distinct_levels,
        "monotonicity": monotonicity,
        "severity_spearman": rho,
        "rating_entropy": normalized_entropy(df["rating"]),
        "n_ratings": int(len(df)),
        "n_items": int(df["item_id"].nunique()),
    }


def compute_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics per group, averaging rater-level summaries to avoid prolific raters dominating."""
    records = []
    for group, group_df in df.groupby("group", dropna=False):
        rater_rows = []
        for rater_id, rater_df in group_df.groupby("rater_id"):
            result = rater_responsiveness(rater_df)
            result["rater_id"] = rater_id
            rater_rows.append(result)
        per_rater = pd.DataFrame(rater_rows)
        row = {"group": group, "n_raters": int(group_df["rater_id"].nunique()),
               "n_ratings": int(len(group_df))}
        for metric in ["scale_span", "distinct_rating_levels", "monotonicity",
                       "severity_spearman", "rating_entropy"]:
            row[metric] = float(per_rater[metric].mean())
            row[f"{metric}_std"] = float(per_rater[metric].std(ddof=1)) if len(per_rater) > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records).sort_values("group").reset_index(drop=True)


def compute_item_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute disagreement for each item; preserve the distribution rather than only the mean."""
    rows = []
    for item_id, item_df in df.groupby("item_id"):
        counts = item_df["rating"].value_counts(normalize=True).reindex(range(5), fill_value=0.0)
        p = counts.to_numpy(dtype=float)
        nonzero = p[p > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(5)) if len(nonzero) > 1 else 0.0
        rows.append({
            "item_id": item_id,
            "severity": float(item_df["severity"].median()),
            "mean_rating": float(item_df["rating"].mean()),
            "median_rating": float(item_df["rating"].median()),
            "rating_std": float(item_df["rating"].std(ddof=0)),
            "rating_range": int(item_df["rating"].max() - item_df["rating"].min()),
            "normalized_entropy": entropy,
            "n_ratings": int(len(item_df)),
            **{f"p_rating_{i}": float(counts.loc[i]) for i in range(5)},
        })
    return pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)


def compute_item_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Contrast simple aggregation with distribution-preserving summaries."""
    result = compute_item_disagreement(df)
    result["majority_rating"] = result[[f"p_rating_{i}" for i in range(5)]].to_numpy().argmax(axis=1)
    result["mean_rounded_rating"] = np.rint(result["mean_rating"]).astype(int)
    result["pluralistic_flag"] = (
        (result["normalized_entropy"] >= result["normalized_entropy"].quantile(0.75))
        | (result["rating_range"] >= 3)
    )
    return result
