from src.evaluation.metrics import evaluate_probabilities

def test_perfect_predictions():
    m=evaluate_probabilities([0,1,0,1],[.01,.99,.02,.98])
    assert m["accuracy"]==1.0
    assert m["log_loss"]<.1
