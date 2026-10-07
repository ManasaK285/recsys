import math


def recall_at_k(ranked, target, k):
    return float(target in ranked[:k])


def ndcg_at_k(ranked, target, k):
    try:
        rank = list(ranked[:k]).index(target)
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def evaluate_rankings(rankings, targets, ks=(5, 10)):
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = sum(
            recall_at_k(r, t, k) for r, t in zip(rankings, targets)
        ) / max(1, len(targets))
        out[f"NDCG@{k}"] = sum(
            ndcg_at_k(r, t, k) for r, t in zip(rankings, targets)
        ) / max(1, len(targets))
    return out
