import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def synthetic(out, seed=42):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    n_users, n_items = 250, 500
    categories = ["running", "fitness", "gaming", "kitchen", "outdoor", "books"]
    rows = []
    item_rows = []

    for i in range(n_items):
        cat = categories[i % len(categories)]
        item_rows.append({
            "item_id": f"item_{i}",
            "title": f"{cat} product {i}",
            "brand": f"brand_{i % 25}",
            "category": cat,
            "description": f"A {cat} item with feature group {i % 15}."
        })

    ts = 0
    for u in range(n_users):
        length = int(rng.integers(8, 20))
        cat = u % len(categories)
        base = np.arange(cat, n_items, len(categories))
        seq = rng.choice(base, size=length, replace=False)
        for it in seq:
            rows.append({
                "user_id": f"user_{u}",
                "item_id": f"item_{int(it)}",
                "timestamp": ts
            })
            ts += 1

    pd.DataFrame(rows).to_csv(out / "raw_interactions.csv", index=False)
    pd.DataFrame(item_rows).to_csv(out / "raw_items.csv", index=False)
    process(out)


def process(out):
    inter = pd.read_csv(out / "raw_interactions.csv")
    items = pd.read_csv(out / "raw_items.csv")

    counts = inter["item_id"].value_counts()
    keep = set(counts[counts >= 2].index)
    inter = inter[inter.item_id.isin(keep)].copy()

    item_ids = sorted(inter.item_id.unique())
    user_ids = sorted(inter.user_id.unique())
    imap = {x: i for i, x in enumerate(item_ids)}
    umap = {x: i for i, x in enumerate(user_ids)}

    inter["item_idx"] = inter.item_id.map(imap)
    inter["user_idx"] = inter.user_id.map(umap)
    items = items[items.item_id.isin(item_ids)].copy()
    items["item_idx"] = items.item_id.map(imap)
    items = items.sort_values("item_idx")

    inter[["user_idx", "item_idx", "timestamp"]].to_csv(
        out / "interactions.csv", index=False
    )
    items.to_csv(out / "items.csv", index=False)

    with open(out / "mappings.json", "w") as f:
        json.dump({"item_to_idx": imap, "user_to_idx": umap}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["synthetic", "amazon"], default="synthetic")
    p.add_argument("--output", default="data/synthetic")
    p.add_argument("--reviews")
    p.add_argument("--items")
    args = p.parse_args()

    out = Path(args.output)
    if args.mode == "synthetic":
        synthetic(out)
    else:
        out.mkdir(parents=True, exist_ok=True)
        pd.read_csv(args.reviews).to_csv(out / "raw_interactions.csv", index=False)
        pd.read_csv(args.items).to_csv(out / "raw_items.csv", index=False)
        process(out)
