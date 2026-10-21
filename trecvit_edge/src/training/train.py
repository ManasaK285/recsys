
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.video_dataset import VideoFolderDataset
from src.models.trecvit_inspired import TRecViTInspired
from src.models.baselines import CNNGRUBaseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/videos")
    p.add_argument("--model", choices=["trecvit", "gru"], default="trecvit")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = VideoFolderDataset(
        args.data_dir, args.frames, args.image_size
    )

    labels = base.label_to_idx
    targets = np.array([label for _, label in base.samples])
    indices = np.arange(len(base))

    if len(base) < 4 or len(labels) < 2:
        raise SystemExit(
            "Need at least 4 videos across at least 2 class folders."
        )

    # Prefer stratification so each class appears in both splits.
    class_counts = np.bincount(targets, minlength=len(labels))
    can_stratify = (
        len(class_counts) > 0
        and np.all(class_counts >= 2)
        and int(np.ceil(len(base) * 0.25)) >= len(labels)
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.5 if len(base) <= 6 else 0.25,
        random_state=args.seed,
        stratify=targets if can_stratify else None,
    )

    train_ds = Subset(base, train_idx.tolist())
    val_ds = Subset(base, val_idx.tolist())

    print(f"Total videos: {len(base)}")
    print(f"Classes: {labels}")
    print(f"Training videos: {len(train_ds)}")
    print(f"Validation videos: {len(val_ds)}")
    print(
        "Training labels:",
        {name: sum(targets[i] == idx for i in train_idx)
         for name, idx in labels.items()},
    )
    print(
        "Validation labels:",
        {name: sum(targets[i] == idx for i in val_idx)
         for name, idx in labels.items()},
    )

    if len(base) <= 6:
        print(
            "WARNING: Tiny dataset. Validation scores are diagnostic only; "
            "use more videos per class for reliable evaluation."
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model_cls = (
        TRecViTInspired if args.model == "trecvit" else CNNGRUBaseline
    )
    model = model_cls(num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for clips, y in tqdm(
            train_loader, desc=f"epoch {epoch}/{args.epochs}"
        ):
            clips, y = clips.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(clips).mean(dim=1)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        ys, preds = [], []

        with torch.no_grad():
            for clips, y in val_loader:
                logits = model(clips.to(device)).mean(dim=1)
                preds.extend(logits.argmax(-1).cpu().tolist())
                ys.extend(y.tolist())

        acc = accuracy_score(ys, preds)
        f1 = f1_score(
            ys, preds, average="macro", zero_division=0
        )

        print(
            f"epoch={epoch} train_loss={np.mean(losses):.4f} "
            f"val_accuracy={acc:.4f} val_macro_f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model,
                    "num_classes": len(labels),
                    "labels": labels,
                    "frames": args.frames,
                    "image_size": args.image_size,
                    "dim": 128,
                    "heads": 4,
                    "epoch": epoch,
                    "val_accuracy": acc,
                    "val_macro_f1": f1,
                },
                out / "best.pt",
            )

    print(f"Best checkpoint: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
