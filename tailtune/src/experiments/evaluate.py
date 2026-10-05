from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.experiments.common import load_data_and_config
from src.models.sasrec import SASRec
from src.utils.config import ROOT


class TestDataset(Dataset):
    """One leave-one-out next-item example per user."""

    def __init__(self, records, max_len):
        self.examples = []
        for rec in records:
            items = rec["items"]
            if len(items) < 2:
                continue
            hist = items[:-1][-max_len:]
            x = torch.zeros(max_len, dtype=torch.long)
            x[-len(hist):] = torch.tensor(hist, dtype=torch.long)
            self.examples.append((x, int(items[-1]), rec["user_id"]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def build_popularity(records):
    pop = {}
    for rec in records:
        for item in rec["items"]:
            item = int(item)
            pop[item] = pop.get(item, 0) + 1
    return pop


def metric_rank(ranked, target, k):
    try:
        return ranked[:k].index(target) + 1
    except ValueError:
        return 0


def ndcg(rank):
    return 0.0 if rank == 0 else 1.0 / math.log2(rank + 1)


def load_model(path, mappings, cfg, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    state = ckpt.get(
        "model_state_dict",
        ckpt.get("state_dict", ckpt)
    )

    if not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    # Infer architecture directly from checkpoint
    d_model = int(state["item_embedding.weight"].shape[1])
    max_seq_len = int(state["position_embedding.weight"].shape[0])

    # Infer number of Transformer layers from checkpoint keys.
    layer_ids = set()

    for key in state.keys():
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))

    num_layers = max(layer_ids) + 1 if layer_ids else 1

    # Both models were trained with 2 attention heads.
    # Quick mode also uses 2 heads.
    quick = os.getenv("TAILTUNE_QUICK", "0") == "1"

    nhead = 2 if quick else int(cfg["model"]["num_heads"])
    dropout = 0.1 if quick else float(cfg["model"]["dropout"])

    print(
        f"  architecture: d_model={d_model}, "
        f"max_seq_len={max_seq_len}, "
        f"heads={nhead}, "
        f"layers={num_layers}"
    )

    model = SASRec(
        num_items=int(mappings["num_items"]),
        max_seq_len=max_seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    return model

@torch.no_grad()
def evaluate(model, loader, num_items, popularity, device):
    ks = (5, 10, 20)
    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({"MRR@10": 0.0, "HitRate@10": 0.0})

    all_recs = []
    redundancies = []
    pop_ranks = []
    long_tail_hits = 0
    rec_count = 0
    latency_ms = []
    seen_count = 0
    users = 0

    # Bottom 20% of training-popularity distribution = long tail.
    vals = sorted(popularity.values())
    threshold = vals[max(0, int(0.20 * len(vals)) - 1)] if vals else 0
    ordered = sorted(popularity, key=lambda x: (-popularity[x], x))
    pop_rank = {item: i + 1 for i, item in enumerate(ordered)}

    for batch_i, (x, targets, _) in enumerate(loader):
        x = x.to(device)
        targets = targets.to(device)

        t0 = time.perf_counter()
        scores = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0 / x.size(0)
        if batch_i >= 2:
            latency_ms.append(elapsed)

        # Never recommend padding id 0.
        scores[:, 0] = -float("inf")

        # Keep evaluation comparable and standard: recommend the highest
        # scoring catalog items. Do not use target information to filter.
        top = torch.topk(scores, k=min(20, num_items), dim=1).indices.cpu().tolist()
        targets_cpu = targets.cpu().tolist()

        for row, target in zip(top, targets_cpu):
            target = int(target)
            users += 1
            all_recs.append(row)

            for k in ks:
                rank = metric_rank(row, target, k)
                sums[f"Recall@{k}"] += float(rank > 0)
                sums[f"NDCG@{k}"] += ndcg(rank)

            rank10 = metric_rank(row, target, 10)
            sums["MRR@10"] += 0.0 if rank10 == 0 else 1.0 / rank10
            sums["HitRate@10"] += float(rank10 > 0)

            # Recommendation-list popularity statistics.
            for item in row:
                item = int(item)
                rec_count += 1
                pop = popularity.get(item, 0)
                if pop <= threshold:
                    long_tail_hits += 1
                if item in pop_rank:
                    pop_ranks.append(pop_rank[item])

            emb = torch.nn.functional.normalize(model.item_embedding.weight[row], dim=1)
            if len(row) > 1:
                sim = emb @ emb.T
                n = sim.shape[0]
                vals_upper = sim[torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)]
                redundancies.append(float(vals_upper.mean().item()) if vals_upper.numel() else 0.0)

    if users == 0:
        raise RuntimeError("No test examples were available for evaluation.")

    result = {k: v / users for k, v in sums.items()}
    result["Catalog Coverage"] = len({i for row in all_recs for i in row}) / max(1, num_items)
    result["Long-tail Rate"] = long_tail_hits / max(1, rec_count)
    result["Avg Popularity Rank"] = sum(pop_ranks) / max(1, len(pop_ranks))

    total_interactions = sum(popularity.values())
    novelty_values = []
    for row in all_recs:
        for item in row:
            p = popularity.get(int(item), 0) / max(1, total_interactions)
            novelty_values.append(-math.log2(max(p, 1e-12)))
    result["Novelty"] = sum(novelty_values) / max(1, len(novelty_values))
    result["Redundancy"] = sum(redundancies) / max(1, len(redundancies))
    result["ILD Diversity"] = 1.0 - result["Redundancy"]
    result["Latency (ms/user)"] = sum(latency_ms) / max(1, len(latency_ms))
    result["Users Evaluated"] = users
    return result


METRICS = [
    "Recall@5", "Recall@10", "Recall@20",
    "NDCG@5", "NDCG@10", "NDCG@20",
    "MRR@10", "HitRate@10", "Catalog Coverage",
    "Long-tail Rate", "Novelty", "ILD Diversity",
    "Redundancy", "Avg Popularity Rank", "Latency (ms/user)",
]


def write_outputs(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "SASRec", "BT-SR", "Delta"])
        for m in METRICS:
            a, b = results["SASRec"][m], results["BT-SR"][m]
            w.writerow([m, a, b, b - a])

    payload = {
        "results": results,
        "delta": {m: results["BT-SR"][m] - results["SASRec"][m] for m in METRICS},
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "TAILTUNE EVALUATION",
        "=" * 78,
        "",
        f"{'Metric':<25}{'SASRec':>15}{'BT-SR':>15}{'Delta':>15}",
        "-" * 70,
    ]
    for m in METRICS:
        a, b = results["SASRec"][m], results["BT-SR"][m]
        lines.append(f"{m:<25}{a:>15.4f}{b:>15.4f}{b-a:>15.4f}")
    lines += ["-" * 70, "", f"Users evaluated: {results['SASRec']['Users Evaluated']}"]
    (out_dir / "comparison.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 78)
    print("TAILTUNE EVALUATION")
    print("=" * 78)

    cfg, mappings, train, val, test = load_data_and_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = int(cfg["data"]["max_seq_len"])
    ds = TestDataset(test, max_len)
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    popularity = build_popularity(train)
    num_items = int(mappings["num_items"])
    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]

    print(f"device       : {device}")
    print(f"test users   : {len(ds)}")
    print(f"items        : {num_items}")
    print()

    results = {}
    for name, filename in [("SASRec", "sasrec.pt"), ("BT-SR", "bt_sr_alpha_0.2.pt")]:
        path = ckpt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        print(f"Evaluating {name}...")
        model = load_model(path, mappings, cfg, device)
        results[name] = evaluate(model, loader, num_items, popularity, device)
        print(f"  users={results[name]['Users Evaluated']}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print(f"{'Metric':<25}{'SASRec':>15}{'BT-SR':>15}{'Delta':>15}")
    print("-" * 70)
    for m in METRICS:
        a, b = results["SASRec"][m], results["BT-SR"][m]
        print(f"{m:<25}{a:>15.4f}{b:>15.4f}{b-a:>15.4f}")

    out_dir = ROOT / "artifacts" / "evaluation"
    write_outputs(results, out_dir)
    print("\nArtifacts:")
    print(out_dir / "comparison.csv")
    print(out_dir / "comparison.json")
    print(out_dir / "comparison.txt")


if __name__ == "__main__":
    main()