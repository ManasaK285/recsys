import os

import torch

from src.experiments.common import load_data_and_config, make_loaders
from src.models.sasrec import SASRec
from src.training.trainer import train_sasrec
from src.utils.config import ROOT


def main():
    cfg, mappings, train, val, test = load_data_and_config()

    quick = os.getenv("TAILTUNE_QUICK", "0") == "1"

    if quick:
        print("\nQUICK MODE ENABLED")
        print("Using a tiny SASRec configuration for smoke testing.\n")

        # IMPORTANT:
        # The loader creates sequences using cfg["model"]["max_seq_len"].
        # Therefore we must keep the same sequence length as the loader.
        max_seq_len = int(cfg["model"]["max_seq_len"])

        # Tiny model, but compatible with the loader.
        embedding_dim = 32
        num_heads = 2
        num_layers = 1
        dropout = 0.1

        epochs = 1
        max_batches = 20
        patience = 1

    else:
        max_seq_len = int(cfg["model"]["max_seq_len"])
        embedding_dim = int(cfg["model"]["embedding_dim"])
        num_heads = int(cfg["model"]["num_heads"])
        num_layers = int(cfg["model"]["num_layers"])
        dropout = float(cfg["model"]["dropout"])

        epochs = int(cfg["training"]["epochs"])
        max_batches = None
        patience = int(cfg["training"]["patience"])

    # ---------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------
    train_loader, val_loader = make_loaders(train, val, cfg)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("SASRec TRAINING")
    print("=" * 70)
    print(f"device       : {device}")
    print(f"epochs       : {epochs}")
    print(f"batches/epoch: {len(train_loader)}")
    print(f"max batches  : {max_batches}")
    print(f"quick mode   : {quick}")
    print(f"seq length   : {max_seq_len}")
    print(f"embedding    : {embedding_dim}")
    print(f"heads        : {num_heads}")
    print(f"layers       : {num_layers}")
    print(f"dropout      : {dropout}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = SASRec(
        num_items=int(mappings["num_items"]),
        max_seq_len=max_seq_len,
        d_model=embedding_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------
    path = (
        ROOT
        / cfg["paths"]["checkpoint_dir"]
        / "sasrec.pt"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    train_sasrec(
        model=model,
        loader=train_loader,
        epochs=epochs,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        device=device,
        checkpoint_path=path,
        val_loader=val_loader,
        patience=patience,
        max_batches=max_batches,
    )

    print(f"\nSaved checkpoint: {path}")


if __name__ == "__main__":
    main()