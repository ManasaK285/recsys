from src.evaluation.metrics import recall,ndcg,coverage
def test_metrics():assert recall([1,2],2)==1 and ndcg([1,2],2)>0 and coverage([[1,2],[2,3]],4)==.75
