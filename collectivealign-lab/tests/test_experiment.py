from collectivealign.experiment import run_experiment
from collectivealign.metrics import paired_identity_sensitivity

def test_small_experiment_writes_results(tmp_path):
    config = {"experiment": {
        "name": "test", "seed": 1, "trials": 3, "group_size": 3,
        "candidate_count": 3, "identity_conditions": ["anonymous", "visible"],
        "protocols": ["independent_vote"], "models": ["competence"],
        "human_reference": {"mode": "synthetic"}
    }}
    df = run_experiment(config, tmp_path)
    assert len(df) == 3 * 2 * 1 * 1
    assert (tmp_path / "trial_results.csv").exists()
    assert df["synthetic"].all()
    sens = paired_identity_sensitivity(df)
    assert len(sens) == 1
