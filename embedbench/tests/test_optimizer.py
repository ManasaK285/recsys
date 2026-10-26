import pandas as pd
from embedbench.optimizer import choose_configuration
def test_choose_lowest_storage_feasible():
    df = pd.DataFrame([
        {"ndcg_at_k": .9, "vector_storage_bytes": 100, "query_search_ms_mean": 1},
        {"ndcg_at_k": .8, "vector_storage_bytes": 50, "query_search_ms_mean": 2},
        {"ndcg_at_k": .4, "vector_storage_bytes": 10, "query_search_ms_mean": 1},
    ])
    result = choose_configuration(df, min_ndcg=.7)
    assert result["vector_storage_bytes"] == 50
