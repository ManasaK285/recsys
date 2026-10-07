import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import seed_everything, device
from src.data import load_processed, build_splits, collaborative_neighbors, SequentialDataset, collate_batch
from src.tokenizer import CollaborativeSemanticTokenizer
from src.model import SCRec
from src.losses import info_nce, manifold_alignment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])
    dev = device()
    print("device:", dev)

    interactions, items, _ = load_processed(args.data)
    train_df, val_df, test_df = build_splits(
        interactions, cfg["data"]["min_user_interactions"]
    )

    n_items = len(items)
    neighbors = collaborative_neighbors(
        train_df, n_items, cfg["tokenizer"]["top_k_neighbors"]
    )

    tokenizer = CollaborativeSemanticTokenizer(
        cfg["tokenizer"]["text_model"],
        cfg["tokenizer"]["top_k_neighbors"],
        cfg["model"]["codebook_size"],
        cfg["model"]["code_length"],
    )
    sem, coll, fused, codes = tokenizer.build(items, neighbors)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    torch.save({
        "semantic": torch.tensor(sem),
        "collaborative": torch.tensor(coll),
        "fused": torch.tensor(fused),
        "codes": torch.tensor(codes, dtype=torch.long),
        "codebooks": torch.tensor(tokenizer.rq.codebooks),
        "neighbors": neighbors,
    }, Path(args.output) / "tokenizer.pt")
    Path(args.output).mkdir(exist_ok=True)

    train_ds = SequentialDataset(train_df, codes, cfg["model"]["max_history"])
    val_ds = SequentialDataset(
        pd.concat([train_df, val_df], ignore_index=True),
        codes,
        cfg["model"]["max_history"],
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_batch,
    )

    model = SCRec(
        n_items,
        cfg["model"]["codebook_size"],
        cfg["model"]["code_length"],
        sem.shape[1],
        cfg["model"]["hidden_dim"],
        cfg["model"]["n_heads"],
        cfg["model"]["n_layers"],
        cfg["model"]["dropout"],
        cfg["model"]["max_history"],
        cfg["model"]["manifold_dim"],
    ).to(dev)

    with torch.no_grad():
        model.item_semantic.weight.copy_(torch.tensor(sem))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best = float("inf")
    patience = 0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total = 0.0
        for histories, mask, targets in tqdm(loader, desc=f"epoch {epoch}"):
            histories, mask, targets = histories.to(dev), mask.to(dev), targets.to(dev)
            target_sem = model.item_semantic(targets)
            target_codes = torch.tensor(codes[targets.cpu().numpy()], device=dev)

            logits, user_h, semantic_h, code_repr = model(
                histories, mask, target_sem, target_codes
            )
            gen = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_codes.reshape(-1),
            )
            con = info_nce(semantic_h, code_repr, cfg["model"]["temperature"])
            geo = manifold_alignment(
                target_sem, code_repr, model.manifold_s, model.manifold_c
            )
            loss = (
                gen
                + cfg["train"]["alpha_contrastive"] * con
                + cfg["train"]["beta_manifold"] * geo
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            total += loss.item()

        avg = total / max(1, len(loader))
        print(f"epoch={epoch} loss={avg:.4f} gen={gen.item():.4f} con={con.item():.4f} geo={geo.item():.4f}")

        if avg < best:
            best = avg
            patience = 0
            torch.save({
                "model": model.state_dict(),
                "semantic_dim": sem.shape[1],
                "n_items": n_items,
                "config": cfg,
            }, out / "best.pt")
        else:
            patience += 1
            if patience >= cfg["train"]["patience"]:
                break

    print("saved:", out / "best.pt")


if __name__ == "__main__":
    import pandas as pd
    main()
