import json
from pathlib import Path
import pandas as pd
import torch

from src.models.sasrec import SASRec
from src.models.bt_sr import BTSR

ROOT = Path(__file__).resolve().parents[1]

class Recommender:
    def __init__(self, checkpoint, model_type="bt", alpha=0.2):
        with open(ROOT / "data/processed/mappings.json") as f:
            mappings = json.load(f)
        self.mappings = mappings
        self.max_len = 50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = dict(
            num_items=int(mappings["num_items"]),
            max_seq_len=50,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.2
        )

        if model_type == "bt":
            self.model = BTSR(**kwargs, alpha=alpha, lambda_offdiag=0.005)
        else:
            self.model = SASRec(**kwargs)

        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.to(self.device).eval()

        self.movies = pd.read_csv(ROOT / "data/processed/movies.csv")
        with open(ROOT / "data/processed/item_buckets.json") as f:
            self.buckets = {int(k): v for k, v in json.load(f).items()}

        self.user_sequences = {}
        with open(ROOT / "data/processed/sequences.json") as f:
            for rec in json.load(f):
                self.user_sequences[int(rec["user_id"])] = rec["items"]

    def recommend(self, user_id, k=10):
        seq = self.user_sequences[int(user_id)]
        x = [0] * (self.max_len - min(len(seq), self.max_len)) + seq[-self.max_len:]
        x = torch.tensor([x], dtype=torch.long, device=self.device)

        with torch.no_grad():
            scores = self.model(x)
            scores[0, 0] = -float("inf")
            vals, inds = torch.topk(scores, k=k)

        rows = []
        for score, item in zip(vals[0].cpu().tolist(), inds[0].cpu().tolist()):
            row = self.movies[self.movies.mapped_item_id == item]
            title = row.iloc[0]["title"] if len(row) else f"Item {item}"
            rows.append({
                "item_id": item,
                "title": title,
                "score": score,
                "bucket": self.buckets.get(item, "unknown")
            })
        return rows

    def history(self, user_id):
        seq = self.user_sequences[int(user_id)][-10:]
        rows = []
        for item in seq:
            row = self.movies[self.movies.mapped_item_id == item]
            rows.append(row.iloc[0]["title"] if len(row) else f"Item {item}")
        return rows
