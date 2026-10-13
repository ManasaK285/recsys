import pandas as pd
import pytest
from bargaining_lab.simulation import simulate_game, run_experiment, STRATEGIES


def test_game_is_seed_reproducible():
    rounds1, agents1 = simulate_game(0, n_agents=4, max_rounds=10, seed=123)
    rounds2, agents2 = simulate_game(0, n_agents=4, max_rounds=10, seed=123)
    assert [r.__dict__ for r in rounds1] == [r.__dict__ for r in rounds2]
    assert agents1 == agents2


def test_payoffs_are_nonnegative_and_total_surplus_bounded():
    rounds, agents = simulate_game(0, n_agents=4, max_rounds=10, seed=8)
    assert all(a["payoff"] >= 0 for a in agents)
    assert all(0 <= r.total_surplus <= 1 for r in rounds)
    assert sum(a["payoff"] for a in agents) <= 1.0000001


def test_game_stops_after_acceptance():
    rounds, _ = simulate_game(0, n_agents=3, max_rounds=20, seed=4)
    assert len(rounds) <= 20
    if any(r.offer_accepted for r in rounds):
        assert rounds[-1].offer_accepted


def test_experiment_shapes_and_strategy_coverage():
    rounds, agents, summary = run_experiment(n_games=25, seed=11)
    assert len(agents) == 25 * 4
    assert set(summary["strategy"]).issubset(set(STRATEGIES))
    assert not rounds.empty
    assert {"mean_payoff", "agreement_rate", "mean_rejections"}.issubset(summary.columns)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simulate_game(0, n_agents=1)
    with pytest.raises(ValueError):
        run_experiment(n_games=0)
