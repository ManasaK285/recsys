import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.experiments.common import load_data_and_config
from src.data.dataset import BTTrainDataset
from src.models.bt_sr import BTSR
from src.training.trainer import train_btsr
from src.utils.config import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    cfg, mappings, train, val, test = load_data_and_config()

    alpha = (
        cfg["barlow_twins"]["alpha"]
        if args.alpha is None
        else args.alpha
    )

    quick = os.getenv("TAILTUNE_QUICK", "0") == "1"

    # -------------------------------------------------------------
    # Controlled experiment architecture
    # -------------------------------------------------------------
    if quick:
        embedding_dim = 32
        num_heads = 2
        num_layers = 1
        dropout = 0.1

        epochs = 1
        max_batches = 20
    else:
        embedding_dim = int(cfg["model"]["embedding_dim"])
        num_heads = int(cfg["model"]["num_heads"])
        num_layers = int(cfg["model"]["num_layers"])
        dropout = float(cfg["model"]["dropout"])

        epochs = int(cfg["training"]["epochs"])
        max_batches = None

    max_seq_len = int(cfg["data"]["max_seq_len"])

    # -------------------------------------------------------------
    # BT-SR training data
    # -------------------------------------------------------------
    ds = BTTrainDataset(
        train,
        max_seq_len,
        cfg["seed"],
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model = BTSR(
        num_items=int(mappings["num_items"]),
        max_seq_len=max_seq_len,
        d_model=embedding_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        alpha=alpha,
        lambda_offdiag=cfg["barlow_twins"]["lambda_offdiag"],
    ).to(device)

    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------
    print("=" * 70)
    print("BT-SR TRAINING")
    print("=" * 70)
    print(f"device       : {device}")
    print(f"quick mode   : {quick}")
    print(f"epochs       : {epochs}")
    print(f"max batches  : {max_batches}")
    print(f"seq length   : {max_seq_len}")
    print(f"embedding    : {embedding_dim}")
    print(f"heads        : {num_heads}")
    print(f"layers       : {num_layers}")
    print(f"dropout      : {dropout}")
    print(f"alpha        : {alpha}")
    print(
        f"lambda_offdiag: "
        f"{cfg['barlow_twins']['lambda_offdiag']}"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------
    path = (
        ROOT
        / cfg["paths"]["checkpoint_dir"]
        / f"bt_sr_alpha_{alpha}.pt"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------
    train_btsr(
        model=model,
        loader=loader,
        epochs=epochs,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        device=device,
        checkpoint_path=path,
        patience=cfg["training"]["patience"],
        max_batches=max_batches,
    )

    print(f"Saved {path}")


if __name__ == "__main__":
    main()