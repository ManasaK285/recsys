import numpy as np

def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    if len(rels) == 0:
        return 0.0
    return float(np.sum((2**rels - 1) / np.log2(np.arange(2, len(rels)+2))))

def ndcg_at_k(frame, score_col, k=10):
    vals = []
    for _, g in frame.groupby("impression_id"):
        pred = g.sort_values(score_col, ascending=False).head(k)["relevance"].to_numpy()
        ideal = np.sort(g["relevance"].to_numpy())[::-1][:k]
        d = dcg(ideal)
        vals.append(dcg(pred)/d if d else 0.0)
    return float(np.mean(vals))

def mrr_at_k(frame, score_col, k=10, threshold=0.7):
    vals = []
    for _, g in frame.groupby("impression_id"):
        rel = g.sort_values(score_col, ascending=False).head(k)["relevance"].to_numpy()
        hits = np.flatnonzero(rel >= threshold)
        vals.append(1/(hits[0]+1) if len(hits) else 0.0)
    return float(np.mean(vals))

def hit_rate_at_k(frame, score_col, k=10, threshold=0.7):
    hits = 0
    total = 0
    for _, g in frame.groupby("impression_id"):
        top = g.sort_values(score_col, ascending=False).head(k)
        hits += int((top["relevance"] >= threshold).any())
        total += 1
    return hits / max(total, 1)

def pairwise_accuracy(frame, score_col):
    vals = []
    for _, g in frame.groupby("impression_id"):
        rel = g["relevance"].to_numpy()
        score = g[score_col].to_numpy()
        diff_rel = rel[:,None] - rel[None,:]
        diff_score = score[:,None] - score[None,:]
        mask = np.triu(diff_rel != 0, 1)
        if mask.any():
            vals.append(float(
                (np.sign(diff_rel[mask]) == np.sign(diff_score[mask])).mean()
            ))
    return float(np.mean(vals)) if vals else 0.0

def evaluate(frame, score_col):
    return {
        "ndcg@10": ndcg_at_k(frame, score_col),
        "mrr@10": mrr_at_k(frame, score_col),
        "hit_rate@10": hit_rate_at_k(frame, score_col),
        "pairwise_accuracy": pairwise_accuracy(frame, score_col),
    }

def evaluate_models(test, models):
    out = {}
    for name, model in [
        ("naive", models.naive),
        ("ips", models.ips),
        ("control_function", models.control),
    ]:
        scored = test.copy()
        scored["score"] = model.score(scored)
        out[name] = evaluate(scored, "score")
    return out
