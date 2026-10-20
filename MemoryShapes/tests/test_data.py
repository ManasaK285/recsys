from src.data import simulate_bandit_data


def test_simulator_schema_and_bounds():
    df = simulate_bandit_data(n_participants=3, n_trials=10, seed=1)
    assert len(df) == 30
    assert {"participant_id", "trial", "choice", "reward"}.issubset(df.columns)
    assert df["choice"].between(0, 3).all()
    assert df["reward"].isin([0, 1]).all()
    assert df.groupby("participant_id").size().eq(10).all()
