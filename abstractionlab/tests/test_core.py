import numpy as np
import pytest
from abstractionlab.core import (
    discretize,
    policy_from_symbols,
    simulate_continuous_world,
    run_map_experiment,
    run_substrate_comparison,
)


def test_discretize_uses_thresholds():
    assert discretize([-1, 0, 1], [0]).tolist() == [0, 1, 1]


def test_discretize_rejects_unsorted_thresholds():
    with pytest.raises(ValueError):
        discretize([0.0], [1.0, -1.0])


def test_discretize_rejects_non_finite_values():
    with pytest.raises(ValueError):
        discretize([float("nan")], [0.0])


def test_world_is_reproducible():
    a = simulate_continuous_world(steps=40, seed=7)
    b = simulate_continuous_world(steps=40, seed=7)
    assert np.allclose(a["state"], b["state"])


def test_world_moves_toward_reachable_target():
    world = simulate_continuous_world(steps=500, seed=42, target=0.75)
    assert abs(float(world["state"].iloc[-1]) - 0.75) < 0.05


def test_policy_outputs_allowed_actions():
    assert policy_from_symbols([0, 1, 2], target_symbol=1).tolist() == [1, 0, -1]


def test_map_experiment_shape_and_metrics():
    world = simulate_continuous_world(steps=80, seed=2)
    frame, summary = run_map_experiment(world)
    assert len(frame) == 80
    assert 0 <= summary["symbol_disagreement_coarse_vs_balanced"] <= 1
    assert 0 <= summary["action_agreement_coarse_vs_balanced"] <= 1
    assert isinstance(summary["task_success"], bool)


def test_substrate_comparison_metrics():
    frame, summary = run_substrate_comparison(steps=80, seed=3)
    assert len(frame) == 80
    assert 0 <= summary["output_agreement"] <= 1
    assert summary["mean_absolute_state_difference"] >= 0
