import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class SequentialTrainDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.examples = []
        self.max_len = max_len
        for rec in sequences:
            items = rec["items"]
            # Every prefix predicts its next item.
            for t in range(1, len(items)):
                hist = items[max(0, t-max_len):t]
                target = items[t]
                self.examples.append((hist, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        hist, target = self.examples[idx]
        x = torch.zeros(self.max_len, dtype=torch.long)
        x[-len(hist):] = torch.tensor(hist, dtype=torch.long)
        return x, torch.tensor(target, dtype=torch.long)

class BTTrainDataset(Dataset):
    """
    Same-target positive pairs. For each target item, keep histories that
    precede that target and sample deterministic pairs.
    """
    def __init__(self, sequences, max_len, seed=42):
        import random
        self.max_len = max_len
        rng = random.Random(seed)
        by_target = {}

        for rec in sequences:
            items = rec["items"]
            for t in range(1, len(items)):
                hist = items[max(0, t-max_len):t]
                target = items[t]
                by_target.setdefault(target, []).append(hist)

        self.examples = []
        for target, histories in by_target.items():
            if len(histories) >= 2:
                for i, h1 in enumerate(histories):
                    j = rng.randrange(len(histories))
                    if len(histories) > 1 and j == i:
                        j = (j + 1) % len(histories)
                    self.examples.append((h1, histories[j], target))

    def __len__(self):
        return len(self.examples)

    def _pad(self, hist):
        x = torch.zeros(self.max_len, dtype=torch.long)
        x[-len(hist):] = torch.tensor(hist, dtype=torch.long)
        return x

    def __getitem__(self, idx):
        h1, h2, target = self.examples[idx]
        return self._pad(h1), self._pad(h2), torch.tensor(target, dtype=torch.long)

def load_sequences(path):
    with open(path, "r") as f:
        return json.load(f)

def chronological_split(records):
    """
    Leave the last interaction for test and the previous one for validation.
    Training uses all earlier interactions.
    """
    train, val, test = [], [], []
    for rec in records:
        items = rec["items"]
        if len(items) < 3:
            continue
        train.append({"user_id": rec["user_id"], "items": items[:-2]})
        val.append({"user_id": rec["user_id"], "items": items[:-1]})
        test.append(rec)
    return train, val, test
