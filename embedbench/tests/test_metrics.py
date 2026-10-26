from embedbench.metrics import recall_at_k, reciprocal_rank, ndcg_at_k
def test_recall():
    assert recall_at_k(["a","b"], ["a","c"], 2) == 0.5
def test_mrr():
    assert reciprocal_rank(["x","b"], ["b"]) == 0.5
def test_ndcg_perfect():
    assert abs(ndcg_at_k(["a","b"], ["a","b"], 2) - 1.0) < 1e-9
