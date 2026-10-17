import numpy as np
import pytest

from optisimplicity.encoding import all_bitstrings, compressed_length_bits, int_to_bits
from optisimplicity.problems import make_problem
from optisimplicity.complexity import summarize_optimum_complexity
from optisimplicity.search import run_search
from optisimplicity.experiments.run_all import run_extrema_experiment


def test_binary_encoding_round_trip():
    assert int_to_bits(5, 4) == "0101"
    assert all_bitstrings(3) == [format(i, "03b") for i in range(8)]


def test_compression_proxy_deterministic():
    assert compressed_length_bits("00000000") == compressed_length_bits("00000000")
    assert compressed_length_bits("01010101") > 0


def test_invalid_bits_rejected():
    with pytest.raises(ValueError):
        compressed_length_bits("012")
    with pytest.raises(ValueError):
        int_to_bits(8, 3)


def test_problem_has_exact_search_space_and_optima():
    p = make_problem("ones", 4)
    assert p.n_candidates == 16
    assert np.isclose(p.scores[p.optimal_indices()].max(), 4.0)
    assert len(p.optimal_indices()) == 1


def test_complexity_summary_contains_proxy_caveat():
    result = summarize_optimum_complexity(make_problem("ones", 4))
    assert result["n_candidates"] == 16
    assert "not Kolmogorov complexity" in result["proxy_note"]


@pytest.mark.parametrize("algorithm", [
    "uniform", "simplicity_biased", "hill_climbing", "simulated_annealing"
])
def test_search_returns_valid_result(algorithm):
    p = make_problem("ones", 4)
    summary, curve = run_search(p, algorithm, budget=20, seed=7)
    assert summary["objective_evaluations"] == 20
    assert len(curve) == 20
    assert summary["optimality_gap"] >= 0
    assert summary["best_candidate_bits"] in all_bitstrings(4)


def test_search_rejects_bad_budget():
    with pytest.raises(ValueError):
        run_search(make_problem("ones", 4), "uniform", budget=0)


def test_extrema_null_preserves_expected_columns():
    observed, null = run_extrema_experiment(n_bits=4, pairs=3, null_samples=5, seed=3)
    assert len(observed) == 3
    assert len(null) == 15
    assert "jaccard_overlap" in observed.columns
    assert "intersection_size" in null.columns
