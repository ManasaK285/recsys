from __future__ import annotations
import pandas as pd

def choose_configuration(results: pd.DataFrame, min_ndcg=0.5, max_storage_bytes=None,
                         max_query_ms=None):
    """Choose the lowest-storage feasible tested configuration; return None if none qualify."""
    candidates = results[results["ndcg_at_k"] >= min_ndcg].copy()
    if max_storage_bytes is not None:
        candidates = candidates[candidates["vector_storage_bytes"] <= max_storage_bytes]
    if max_query_ms is not None:
        candidates = candidates[candidates["query_search_ms_mean"] <= max_query_ms]
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["vector_storage_bytes", "query_search_ms_mean", "ndcg_at_k"],
        ascending=[True, True, False]
    ).iloc[0].to_dict()
