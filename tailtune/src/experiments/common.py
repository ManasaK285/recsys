import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, ROOT
from src.utils.seed import set_seed
from src.data.dataset import load_sequences, chronological_split, SequentialTrainDataset

def load_data_and_config():
    cfg = load_config()
    set_seed(cfg["seed"])
    proc = ROOT / cfg["paths"]["processed_dir"]
    with open(proc / "mappings.json") as f:
        mappings = json.load(f)
    sequences = load_sequences(proc / "sequences.json")
    train, val, test = chronological_split(sequences)
    return cfg, mappings, train, val, test

def make_loaders(train, val, cfg):
    max_len = cfg["data"]["max_seq_len"]
    train_ds = SequentialTrainDataset(train, max_len)
    val_ds = SequentialTrainDataset(val, max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False)
    return train_loader, val_loader
