import numpy as np
import pytest
from src.data.benchmark import make_benchmark, feature_matrix
from src.models.gp_surrogate import GPSurrogate
from src.discovery.acquisition import choose_next
from src.estimation.estimators import uniform_sample_estimate, gp_plugin_estimate
from src.pipeline import run_experiment

def test_benchmark_reproducible():
    a = make_benchmark(50, seed=7)
    b = make_benchmark(50, seed=7)
    assert a.equals(b)
    assert a["error_severity"].between(0, 1).all()

def test_feature_matrix_shape():
    df = make_benchmark(40, seed=1)
    assert feature_matrix(df).shape == (40, 11)

def test_gp_fit_predict():
    df = make_benchmark(40, seed=2)
    X = feature_matrix(df)
    gp = GPSurrogate(max_train_points=30, random_state=2).fit(X[:20], df.error_severity.to_numpy()[:20])
    mean, std = gp.predict(X[20:])
    assert len(mean) == 20 and len(std) == 20
    assert np.all((mean >= 0) & (mean <= 1))
    assert np.all(std > 0)

def test_acquisition_returns_unique_candidates():
    rng = np.random.default_rng(0)
    candidates = np.arange(10, 30)
    out = choose_next("hybrid", candidates, np.linspace(0,1,30), np.ones(30), np.array([.1]), 5, rng, batch_size=4)
    assert len(out) == 4
    assert len(set(out)) == 4
    assert set(out).issubset(set(candidates))

def test_estimators():
    assert uniform_sample_estimate(np.array([0.2, 0.4])) == pytest.approx(0.3)
    assert gp_plugin_estimate(np.array([0]), np.array([1.0]), np.array([0.0, 0.5])) == pytest.approx(0.75)

def test_pipeline_smoke():
    result, curve, cases = run_experiment(n_cases=60, budget=15, seed=5, strategy="random", checkpoints=3)
    assert result["evaluations_used"] == 15
    assert 0 <= result["gp_estimate"] <= 1
    assert len(cases) == 15
    assert not curve.empty
