import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from randomization import assign_condition, condition_label  # noqa: E402
from app.config import APP_TYPES, REASON_TYPES  # noqa: E402


def test_assignment_is_deterministic():
    a1 = assign_condition("participant-123", seed=42)
    a2 = assign_condition("participant-123", seed=42)
    assert a1 == a2


def test_assignment_changes_with_seed():
    a1 = assign_condition("participant-123", seed=42)
    a2 = assign_condition("participant-123", seed=43)
    # Not guaranteed to differ every time, but across many seeds it should vary.
    results = {assign_condition("participant-123", seed=s) for s in range(20)}
    assert len(results) > 1


def test_assignment_in_valid_domain():
    app_type, reason_type = assign_condition("some-participant", seed=1)
    assert app_type in APP_TYPES
    assert reason_type in REASON_TYPES


def test_condition_label_format():
    label = condition_label("rideshare", "functional")
    assert label == "rideshare__functional"


def test_distribution_roughly_uniform():
    from collections import Counter

    counts = Counter()
    for i in range(4000):
        app_type, _ = assign_condition(f"p-{i}", seed=42)
        counts[app_type] += 1
    # With 4 app types and 4000 draws, expect roughly 1000 each; allow generous tolerance.
    for app_type in APP_TYPES:
        assert 700 < counts[app_type] < 1300
