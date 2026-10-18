"""Headless-safe matplotlib figures for the report."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_plots(df: pd.DataFrame, groups: pd.DataFrame, items: pd.DataFrame, out: Path) -> None:
    # 1. Responsiveness comparison.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(groups))
    width = 0.25
    for j, metric in enumerate(["scale_span", "distinct_rating_levels", "rating_entropy"]):
        vals = groups[metric].to_numpy(dtype=float)
        # Normalize each metric only for visual comparison; raw values remain in CSV.
        denom = np.nanmax(vals) if len(vals) and np.nanmax(vals) > 0 else 1
        ax.bar(x + (j - 1) * width, vals / denom, width, label=f"{metric} (normalized)")
    ax.set_xticks(x, groups["group"].astype(str))
    ax.set_ylabel("Within-metric normalized value")
    ax.set_title("Ordinal rating behavior by group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_group_responsiveness.png", dpi=150)
    plt.close(fig)

    # 2. Rating distribution by group.
    pivot = df.groupby(["group", "rating"]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    ax = pivot.reindex(columns=range(5), fill_value=0).plot(
        kind="bar", stacked=True, figsize=(9, 5), colormap="viridis"
    )
    ax.set_ylabel("Share of ratings")
    ax.set_xlabel("Group")
    ax.set_title("Ordinal rating distributions by group")
    ax.legend(title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "fig_group_rating_distributions.png", dpi=150)
    plt.close(ax.figure)

    # 3. Items with largest disagreement.
    top = items.nlargest(min(20, len(items)), "normalized_entropy").sort_values("normalized_entropy")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["item_id"].astype(str), top["normalized_entropy"])
    ax.set_xlabel("Normalized rating entropy")
    ax.set_title("Items with the most disagreement")
    fig.tight_layout()
    fig.savefig(out / "fig_item_disagreement.png", dpi=150)
    plt.close(fig)
