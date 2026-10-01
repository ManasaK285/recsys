from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "interactions.csv"

def main():
    rng = np.random.default_rng(42)
    n_users, n_items, n_rows = 1000, 2000, 60000
    users = rng.integers(0, n_users, n_rows)
    items = rng.integers(0, n_items, n_rows)
    up = rng.normal(0, 1, n_users)
    iq = rng.normal(0, 1, n_items)
    ip = rng.normal(0, 0.7, n_items)
    base = 0.75*up[users] + 0.90*iq[items] + 0.35*ip[items] + rng.normal(0,1,n_rows)
    sig = lambda x: 1/(1+np.exp(-np.clip(x,-20,20)))
    liked = rng.binomial(1, sig(base-0.15))
    strong = rng.binomial(1, sig(0.85*base-0.55))
    repeat = rng.binomial(1, sig(0.60*base+0.55*up[users]-0.75))
    engage = rng.binomial(1, sig(0.75*base+0.35*ip[items]-0.25))
    strong = np.maximum(strong, (liked & (rng.random(n_rows)<0.10)).astype(int))
    repeat = np.maximum(repeat, (strong & (rng.random(n_rows)<0.08)).astype(int))
    rating = np.clip(3 + 0.65*base + rng.normal(0,0.65,n_rows), 1, 5)
    df = pd.DataFrame({
        "user_id": users, "item_id": items, "rating": rating,
        "liked": liked, "strong_preference": strong,
        "repeat_interest": repeat, "high_engagement": engage
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT}")

if __name__ == "__main__":
    main()
