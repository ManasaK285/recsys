from __future__ import annotations
import numpy as np

def recall_at_k(retrieved_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum((1.0 / np.log2(rank + 1)) for rank, doc_id in enumerate(retrieved_ids[:k], start=1) if doc_id in relevant)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return float(dcg / idcg) if idcg else 0.0

def evaluate_rankings(rankings, queries, k=10):
    rows = []
    for ranking, query in zip(rankings, queries):
        rel = query["relevant_doc_ids"]
        rows.append({
            "query_id": query["query_id"],
            "recall_at_k": recall_at_k(ranking, rel, k),
            "mrr": reciprocal_rank(ranking, rel),
            "ndcg_at_k": ndcg_at_k(ranking, rel, k),
        })
    return rows
