from pathlib import Path
import json
import pandas as pd
from src.utils.config import load_config, ROOT

def load_ratings(raw_dir):
    path = raw_dir / "ml-1m" / "ratings.dat"
    df = pd.read_csv(
        path, sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int}
    )
    return df

def build_mappings(df):
    users = sorted(df.user_id.unique())
    items = sorted(df.item_id.unique())
    user2idx = {u: i + 1 for i, u in enumerate(users)}  # 0 = padding
    item2idx = {it: i + 1 for i, it in enumerate(items)}
    return user2idx, item2idx

def chronological_sequences(df, user2idx, item2idx, min_interactions):
    df = df.sort_values(["user_id", "timestamp"])
    records = []
    for uid, g in df.groupby("user_id"):
        if len(g) < min_interactions:
            continue
        items = [item2idx[x] for x in g.item_id.tolist()]
        records.append({"user_id": user2idx[uid], "items": items})
    return records

def main():
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    out = ROOT / cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)

    df = load_ratings(raw_dir)
    user2idx, item2idx = build_mappings(df)
    sequences = chronological_sequences(
        df, user2idx, item2idx, cfg["data"]["min_interactions"]
    )

    popularity = df["item_id"].value_counts().to_dict()
    mapped_popularity = {item2idx[k]: int(v) for k, v in popularity.items()}

    n = len(mapped_popularity)
    sorted_items = sorted(mapped_popularity, key=mapped_popularity.get, reverse=True)
    head_n = max(1, int(n * cfg["data"]["head_fraction"]))
    tail_n = max(1, int(n * cfg["data"]["tail_fraction"]))

    buckets = {}
    for rank, item in enumerate(sorted_items):
        if rank < head_n:
            buckets[item] = "head"
        elif rank >= n - tail_n:
            buckets[item] = "tail"
        else:
            buckets[item] = "mid"

    with open(out / "sequences.json", "w") as f:
        json.dump(sequences, f)
    with open(out / "item_popularity.json", "w") as f:
        json.dump({str(k): v for k, v in mapped_popularity.items()}, f)
    with open(out / "item_buckets.json", "w") as f:
        json.dump({str(k): v for k, v in buckets.items()}, f)
    with open(out / "mappings.json", "w") as f:
        json.dump({
            "user2idx": {str(k): v for k, v in user2idx.items()},
            "item2idx": {str(k): v for k, v in item2idx.items()},
            "num_users": len(user2idx),
            "num_items": len(item2idx),
        }, f)

    # Keep movie metadata for the UI.
    movies_path = raw_dir / "ml-1m" / "movies.dat"
    movies = pd.read_csv(
        movies_path, sep="::", engine="python",
        names=["item_id", "title", "genres"], encoding="latin-1"
    )
    movies["mapped_item_id"] = movies["item_id"].map(item2idx)
    movies.to_csv(out / "movies.csv", index=False)

    print(f"Users: {len(user2idx):,}")
    print(f"Items: {len(item2idx):,}")
    print(f"Sequences: {len(sequences):,}")
    print(f"Wrote processed data to {out}")

if __name__ == "__main__":
    main()
