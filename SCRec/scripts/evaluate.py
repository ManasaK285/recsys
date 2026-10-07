import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.data import load_processed, build_splits, collate_batch
from src.metrics import evaluate_rankings
from src.model import SCRec
from src.utils import device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/best.pt")
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    dev = device()
    ckpt = torch.load(args.checkpoint, map_location=dev)
    cfg = ckpt["config"]

    interactions, items, _ = load_processed(args.data)
    train_df, val_df, test_df = build_splits(
        interactions, cfg["data"]["min_user_interactions"]
    )
    tok = torch.load(Path(args.checkpoint).parent / "tokenizer.pt", map_location="cpu")
    sem = tok["semantic"].numpy()
    codes = tok["codes"].numpy()

    model = SCRec(
        ckpt["n_items"],
        cfg["model"]["codebook_size"],
        cfg["model"]["code_length"],
        ckpt["semantic_dim"],
        cfg["model"]["hidden_dim"],
        cfg["model"]["n_heads"],
        cfg["model"]["n_layers"],
        cfg["model"]["dropout"],
        cfg["model"]["max_history"],
        cfg["model"]["manifold_dim"],
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # One test example per user: all interactions before the held-out item.
    histories, targets = [], []
    if "user_idx" in interactions.columns:
        interactions = interactions.rename(columns={"user_idx": "user_id"})

    if "item_idx" in interactions.columns:
        interactions = interactions.rename(columns={"item_idx": "item_id"})
    merged = (
        __import__("pandas").concat([train_df, val_df], ignore_index=True)
        .sort_values(["user_id", "timestamp"])
    )
    for _, row in test_df.iterrows():
        hist = merged[merged.user_id == row.user_id].sort_values("timestamp").item_id.tolist()
        if hist:
            histories.append(hist[-cfg["model"]["max_history"]:])
            targets.append(int(row.item_id))

    max_len = max(len(x) for x in histories)
    H = torch.zeros(len(histories), max_len, dtype=torch.long)
    M = torch.zeros_like(H, dtype=torch.bool)
    for i, h in enumerate(histories):
        H[i, -len(h):] = torch.tensor(h)
        M[i, -len(h):] = True

    H, M = H.to(dev), M.to(dev)
    rankings = model.recommend(H, M, sem, codes, top_k=args.topk).cpu().tolist()
    metrics = evaluate_rankings(rankings, targets, ks=(5, 10))
    print("\nSCRec test metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
