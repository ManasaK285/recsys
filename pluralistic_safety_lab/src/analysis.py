"""Pipeline orchestration and JSON-serializable summary."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from .data import generate_synthetic_data, load_ratings, validate_ratings
from .metrics import compute_group_metrics, compute_item_disagreement, compute_item_aggregates
from .plots import create_plots


def run_analysis(input_path: str | None = None, output_dir: str = "reports",
                 no_synthetic: bool = False, seed: int = 42) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if input_path and Path(input_path).exists():
        df = load_ratings(input_path)
        data_source = f"file:{input_path}"
    elif no_synthetic:
        raise FileNotFoundError(f"Input file not found: {input_path or 'not provided'}")
    else:
        df = generate_synthetic_data(seed=seed)
        data_source = "synthetic_demo_data"
    validate_ratings(df)

    groups = compute_group_metrics(df)
    items = compute_item_disagreement(df)
    aggregates = compute_item_aggregates(df)

    groups.to_csv(out / "group_metrics.csv", index=False)
    items.to_csv(out / "item_disagreement.csv", index=False)
    aggregates.to_csv(out / "item_aggregates.csv", index=False)
    create_plots(df, groups, items, out)

    summary = {
        "project": "Pluralistic Safety Lens",
        "data_source": data_source,
        "n_ratings": int(len(df)),
        "n_items": int(df["item_id"].nunique()),
        "n_raters": int(df["rater_id"].nunique()),
        "n_groups": int(df["group"].nunique()),
        "rating_scale": {"min": int(df["rating"].min()), "max": int(df["rating"].max())},
        "metrics_note": "Transparent baseline summaries; not asserted to be exact reproductions of the paper's metrics.",
        "group_metrics": groups.round(5).to_dict(orient="records"),
        "high_disagreement_items": items.nlargest(min(10, len(items)), "normalized_entropy")[
            ["item_id", "severity", "mean_rating", "normalized_entropy", "rating_range", "n_ratings"]
        ].round(5).to_dict(orient="records"),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
