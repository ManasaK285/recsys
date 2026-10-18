import numpy as np
import pandas as pd
import pytest

from src.data import generate_synthetic_data, validate_ratings
from src.metrics import normalized_entropy, rater_responsiveness, compute_group_metrics, compute_item_disagreement


def test_synthetic_data_is_valid_and_reproducible():
    a = generate_synthetic_data(n_items=12, raters_per_group=3, seed=7)
    b = generate_synthetic_data(n_items=12, raters_per_group=3, seed=7)
    validate_ratings(a)
    pd.testing.assert_frame_equal(a, b)


def test_entropy_edges():
    assert normalized_entropy(pd.Series([2, 2, 2])) == 0.0
    assert normalized_entropy(pd.Series([0, 1, 2, 3, 4])) == pytest.approx(1.0)


def test_responsiveness_increases_for_monotonic_ratings():
    df = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(5)],
        "rater_id": ["r1"] * 5,
        "group": ["A"] * 5,
        "severity": [0, 1, 2, 3, 4],
        "rating": [0, 1, 2, 3, 4],
    })
    result = rater_responsiveness(df)
    assert result["monotonicity"] == 1.0
    assert result["severity_spearman"] == pytest.approx(1.0)
    assert result["scale_span"] == 4.0


def test_group_and_item_outputs():
    df = generate_synthetic_data(n_items=10, raters_per_group=3, seed=2)
    groups = compute_group_metrics(df)
    items = compute_item_disagreement(df)
    assert len(groups) == 3
    assert len(items) == 10
    assert items["normalized_entropy"].between(0, 1).all()


def test_invalid_rating_fails():
    df = generate_synthetic_data(n_items=5, raters_per_group=2)
    df.loc[0, "rating"] = 8
    with pytest.raises(ValueError):
        validate_ratings(df)
