import json
from pathlib import Path
from gramlab.scenarios import SCENARIOS
from gramlab.simulator import run_trajectory
from gramlab.scoring import score_trajectory
from gramlab.investigator import make_static_fixture, replay_fixture, run_ablations
from gramlab.pipeline import run_experiment


def test_scenario_registry_has_17_unique_scenarios():
    assert len(SCENARIOS) == 17
    assert len({s.scenario_id for s in SCENARIOS}) == 17


def test_same_seed_reproduces_same_trajectory():
    a = run_trajectory(SCENARIOS[0], "pressure", "shortcut_seeking", 7, 2)
    b = run_trajectory(SCENARIOS[0], "pressure", "shortcut_seeking", 7, 2)
    assert a == b


def test_compliant_policy_does_not_record_shortcut():
    for scenario in SCENARIOS:
        row = score_trajectory(run_trajectory(scenario, "pressure", "compliant", 1, 0))
        assert not row["constraint_violation"]
        assert not row["sabotage_like_action"]


def test_pipeline_writes_expected_artifacts(tmp_path):
    meta = run_experiment(tmp_path, runs_per_cell=1, seed=3)
    assert meta["trajectory_count"] == 17 * 2 * 3
    for filename in [
        "trajectories.jsonl", "summary_by_condition.csv", "summary_by_policy.csv",
        "summary_by_scenario.csv", "summary_by_condition_policy.csv",
        "investigator_report.json", "run_metadata.json"
    ]:
        assert (tmp_path / filename).exists()


def test_static_replay_and_ablations():
    raw = run_trajectory(SCENARIOS[0], "pressure", "shortcut_seeking", 42, 0)
    scored = score_trajectory(raw)
    fixture = make_static_fixture(scored)
    replay = replay_fixture(fixture)
    assert replay["replay_matches"]
    variants = run_ablations(fixture)
    assert {v["variant"] for v in variants} == {"original", "remove_pressure_cue", "compliant_policy_control"}
