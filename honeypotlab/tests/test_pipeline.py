import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from honeypotlab.runner import load_scenarios, run_experiment
from honeypotlab.analysis import summarize


def test_scenario_registry_has_expected_fields():
    scenarios = load_scenarios()
    assert len(scenarios) >= 3
    assert all({"id", "task", "constraint", "opportunity"} <= set(s) for s in scenarios)


def test_conflict_fixture_violates_only_in_conflict_condition():
    from honeypotlab.agents import run_scripted_agent
    scenario = {"id": "toy", "required_behavior": "feature", "constraint": "audit",
                "opportunity": True}
    import random
    base = run_scripted_agent("conflict_following", "baseline", scenario, random.Random(1))
    conflict = run_scripted_agent("conflict_following", "goal_conflict", scenario, random.Random(1))
    assert base.constraint_violation is False
    assert conflict.constraint_violation is True


def test_run_experiment_writes_data(tmp_path):
    df = run_experiment(trials=2, seed=3, output_dir=tmp_path)
    assert len(df) == 3 * 5 * 3 * 2
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "runs.json").exists()
    summary = summarize(df)
    assert "constraint_violation_rate" in summary.columns
    assert len(summary) == 5 * 3


def test_invalid_trial_count_fails():
    import pytest
    with pytest.raises(ValueError):
        run_experiment(trials=0)
