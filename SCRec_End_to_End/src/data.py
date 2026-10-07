from pathlib import Path
from collections import defaultdict, Counter
import json

import pandas as pd
import torch
from torch.utils.data import Dataset


SPECIAL_PAD = 0
SPECIAL_BOS = 1


def load_processed(root):
    root = Path(root)

    interactions = pd.read_csv(root / "interactions.csv")
    items = pd.read_csv(root / "items.csv")

    with open(root / "mappings.json", "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # Normalize synthetic dataset column names
    if "user_idx" in interactions.columns:
        interactions = interactions.rename(columns={"user_idx": "user_id"})

    if "item_idx" in interactions.columns:
        interactions = interactions.rename(columns={"item_idx": "item_id"})

    return interactions, items, mappings


def build_splits(interactions, min_user_interactions=5):
    interactions = interactions.copy()

    # Normalize possible column names
    if "user_idx" in interactions.columns:
        interactions = interactions.rename(columns={"user_idx": "user_id"})

    if "item_idx" in interactions.columns:
        interactions = interactions.rename(columns={"item_idx": "item_id"})

    if "user" in interactions.columns and "user_id" not in interactions.columns:
        interactions = interactions.rename(columns={"user": "user_id"})

    if "item" in interactions.columns and "item_id" not in interactions.columns:
        interactions = interactions.rename(columns={"item": "item_id"})

    if "time" in interactions.columns and "timestamp" not in interactions.columns:
        interactions = interactions.rename(columns={"time": "timestamp"})

    interactions = interactions.sort_values(
        ["user_id", "timestamp"]
    )

    groups = []

    for uid, g in interactions.groupby("user_id"):
        if len(g) >= min_user_interactions:
            groups.append((uid, g))

    train = []
    val = []
    test = []

    for uid, g in groups:
        rows = g.to_dict("records")

        train.extend(rows[:-2])
        val.append(rows[-2])
        test.append(rows[-1])

    return (
        pd.DataFrame(train),
        pd.DataFrame(val),
        pd.DataFrame(test),
    )


def collaborative_neighbors(train_df, n_items, top_k=5):
    """
    Build item-item collaborative neighbors using
    user interaction co-occurrence.
    """

    co = defaultdict(Counter)

    for _, g in train_df.groupby("user_id"):

        seq = list(
            g.sort_values("timestamp")["item_id"]
            .astype(int)
        )

        # Remove duplicate interactions by the same user
        unique_items = list(dict.fromkeys(seq))

        for i in unique_items:
            for j in unique_items:
                if i != j:
                    co[i][j] += 1

    neighbors = [[] for _ in range(n_items)]

    for i in range(n_items):
        ranked = sorted(
            co[i].items(),
            key=lambda x: (-x[1], x[0])
        )

        neighbors[i] = [
            j for j, _ in ranked[:top_k]
        ]

    return neighbors


class SequentialDataset(Dataset):

    def __init__(self, df, item_codes, max_history):

        self.examples = []
        self.item_codes = item_codes
        self.max_history = max_history

        for _, g in df.groupby("user_id"):

            seq = list(
                g.sort_values("timestamp")["item_id"]
                .astype(int)
            )

            for t in range(1, len(seq)):

                hist = seq[
                    max(0, t - max_history):t
                ]

                target = seq[t]

                self.examples.append(
                    (hist, target)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        hist, target = self.examples[idx]

        return hist, target


def collate_batch(batch):

    max_len = max(
        len(x[0])
        for x in batch
    )

    histories = torch.zeros(
        len(batch),
        max_len,
        dtype=torch.long,
    )

    mask = torch.zeros(
        len(batch),
        max_len,
        dtype=torch.bool,
    )

    targets = torch.tensor(
        [x[1] for x in batch],
        dtype=torch.long,
    )

    for i, (hist, _) in enumerate(batch):

        histories[
            i,
            -len(hist):
        ] = torch.tensor(
            hist,
            dtype=torch.long,
        )

        mask[
            i,
            -len(hist):
        ] = True

    return histories, mask, targets