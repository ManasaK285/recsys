import argparse
from pathlib import Path
import torch
import pandas as pd

from src.data import load_processed, build_splits
from src.model import SCRec
from src.utils import device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/best.pt")
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--user", type=int, default=0)
    args = p.parse_args()

    dev = device()
    ckpt = torch.load(args.checkpoint, map_location=dev)
    cfg = ckpt["config"]
    interactions, items, _ = load_processed(args.data)
    train, val, test = build_splits(
        interactions, cfg["data"]["min_user_interactions"]
    )
    tok = torch.load(Path(args.checkpoint).parent / "tokenizer.pt", map_location="cpu")
    sem = tok["semantic"].numpy()
    codes = tok["codes"].numpy()

    model = SCRec(
        ckpt["n_items"], cfg["model"]["codebook_size"], cfg["model"]["code_length"],
        ckpt["semantic_dim"], cfg["model"]["hidden_dim"], cfg["model"]["n_heads"],
        cfg["model"]["n_layers"], cfg["model"]["dropout"], cfg["model"]["max_history"],
        cfg["model"]["manifold_dim"]
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    g = pd.concat([train, val]).query("user_id == @args.user").sort_values("timestamp")
    hist = g.item_id.tolist()[-cfg["model"]["max_history"]:]
    H = torch.tensor([hist], dtype=torch.long, device=dev)
    M = torch.ones_like(H, dtype=torch.bool, device=dev)

    recs = model.recommend(H, M, sem, codes, top_k=10)[0].cpu().tolist()
    history_ids = set(hist)
    print("\nHistory:")
    for idx in hist:
        row = items[items.item_idx == idx].iloc[0]
        print(f"  - {row['title']}")

    print("\nTop recommendations:")
    shown = 0
    for idx in recs:
        if idx in history_ids:
            continue
        row = items[items.item_idx == idx].iloc[0]
        print(f"  {shown+1}. {row['title']} | category={row.get('category','')}")
        shown += 1
        if shown == 5:
            break


if __name__ == "__main__":
    main()
