import math
import torch

def _get_recommendations(model, histories, k, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(histories, dtype=torch.long, device=device)
        scores = model(x)
        # Padding item is never recommended.
        scores[:, 0] = -float("inf")
        vals, inds = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
    return inds.cpu().tolist()

def hit_rate_at_k(recs, targets, k):
    return sum(int(t in r[:k]) for r, t in zip(recs, targets)) / len(targets)

def ndcg_at_k(recs, targets, k):
    total = 0.0
    for r, t in zip(recs, targets):
        try:
            rank = r[:k].index(t)
            total += 1.0 / math.log2(rank + 2)
        except ValueError:
            pass
    return total / len(targets)

def mrr_at_k(recs, targets, k):
    total = 0.0
    for r, t in zip(recs, targets):
        try:
            rank = r[:k].index(t)
            total += 1.0 / (rank + 1)
        except ValueError:
            pass
    return total / len(targets)

def evaluate_model(model, examples, max_len, device, ks=(5, 10, 20)):
    histories, targets = [], []
    for hist, target in examples:
        padded = [0] * (max_len - len(hist)) + hist[-max_len:]
        histories.append(padded)
        targets.append(target)

    recs = _get_recommendations(model, histories, max(ks), device)
    out = {}
    for k in ks:
        out[f"HR@{k}"] = hit_rate_at_k(recs, targets, k)
        out[f"NDCG@{k}"] = ndcg_at_k(recs, targets, k)
        out[f"MRR@{k}"] = mrr_at_k(recs, targets, k)
    return out, recs, targets
