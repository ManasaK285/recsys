import os
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from analysis import descriptive, statistics as stats_mod, regression  # noqa: E402


@pytest.fixture
def toy_df():
    data = {
        "participant_id": [f"p{i}" for i in range(40)],
        "app_type": (["rideshare"] * 20 + ["wallpaper"] * 20),
        "reason_type": (["functional"] * 10 + ["none"] * 10) * 2,
        "decision": (["granted"] * 8 + ["denied"] * 2 + ["granted"] * 3 + ["denied"] * 7) * 2,
        "precision": (["precise"] * 5 + ["approximate"] * 3 + [None] * 2 + ["approximate"] * 3 + [None] * 7) * 2,
        "response_time_ms": [1000 + (i * 37) % 500 for i in range(40)],
        "app_trust": [3 + (i % 3) for i in range(40)],
        "android_trust": [3 for _ in range(40)],
        "necessity": [3 for _ in range(40)],
        "privacy_concern": [2 for _ in range(40)],
    }
    df = pd.DataFrame(data)
    df["granted"] = (df["decision"] == "granted").astype(int)
    df["precise"] = (df["precision"] == "precise").astype(int)
    df["precise"] = df["precise"].where(df["decision"] == "granted", other=pd.NA)
    return df


def test_overall_rates_sum_to_one(toy_df):
    rates = descriptive.overall_rates(toy_df)
    assert abs(rates["grant_rate"] + rates["denial_rate"] - 1.0) < 1e-9


def test_rate_by_group_has_all_groups(toy_df):
    g = descriptive.rate_by_group(toy_df, "reason_type")
    assert set(g["reason_type"]) == {"functional", "none"}


def test_chi_square_runs(toy_df):
    result = stats_mod.chi_square_test(toy_df, "reason_type")
    assert "p_value" in result
    assert 0 <= result["p_value"] <= 1


def test_logistic_regression_runs(toy_df):
    result = regression.logistic_main_effects(toy_df)
    assert result["n_obs"] == len(toy_df)
    assert len(result["coefficients"]) > 0
    for coef in result["coefficients"]:
        assert "odds_ratio" in coef
