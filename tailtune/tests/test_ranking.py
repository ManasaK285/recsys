from src.evaluation.ranking import hit_rate_at_k, ndcg_at_k, mrr_at_k

def test_ranking_metrics():
    recs = [[1,2,3], [4,5,6]]
    targets = [2,7]
    assert hit_rate_at_k(recs, targets, 3) == 0.5
    assert ndcg_at_k(recs, targets, 3) > 0
    assert mrr_at_k(recs, targets, 3) > 0
