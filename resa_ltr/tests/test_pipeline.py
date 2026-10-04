import numpy as np
from src.data import Environment
from src.models import train_all
from src.evaluation import evaluate_models

def test_data_generation():
    log, truth = Environment(
        users=8, candidates_per_impression=6,
        impressions_per_user=3, seed=1
    ).generate()
    assert len(log) == 8*6*3
    assert len(truth) == len(log)
    assert {"click","position","instrument","relevance"}.issubset(log.columns)

def test_models():
    log, _ = Environment(
        users=12, candidates_per_impression=8,
        impressions_per_user=3, seed=2
    ).generate()
    models = train_all(log, seed=2)
    for model in [models.naive, models.ips, models.control]:
        scores = model.score(log.head(20))
        assert len(scores) == 20
        assert np.isfinite(scores).all()

def test_evaluation():
    log, _ = Environment(
        users=10, candidates_per_impression=8,
        impressions_per_user=3, seed=3
    ).generate()
    result = evaluate_models(log, train_all(log, seed=3))
    assert set(result) == {"naive","ips","control_function"}
    for m in result.values():
        assert "ndcg@10" in m
