"""Description-length proxy analysis; never presented as true Kolmogorov complexity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .encoding import compressed_length_bits, int_to_bits
from .problems import BinaryProblem


def complexity_table(problem: BinaryProblem, maximize: bool = True) -> pd.DataFrame:
    rows = []
    optimal = set(problem.optimal_indices(maximize=maximize).tolist())
    for idx, score in enumerate(problem.scores):
        bits = int_to_bits(idx, problem.n_bits)
        rows.append({
            "candidate_int": idx,
            "bits": bits,
            "objective": float(score),
            "description_length_bits": compressed_length_bits(bits),
            "is_optimal": idx in optimal,
        })
    return pd.DataFrame(rows)


def summarize_optimum_complexity(problem: BinaryProblem, maximize: bool = True) -> dict:
    table = complexity_table(problem, maximize=maximize)
    optimum = table[table["is_optimal"]]
    typical = table["description_length_bits"]
    return {
        "problem": problem.name,
        "n_bits": problem.n_bits,
        "n_candidates": problem.n_candidates,
        "best_objective": float(optimum["objective"].iloc[0]),
        "number_of_optima": int(len(optimum)),
        "mean_optimum_description_length_bits": float(optimum["description_length_bits"].mean()),
        "median_all_description_length_bits": float(typical.median()),
        "mean_all_description_length_bits": float(typical.mean()),
        "optimum_minus_population_median_bits": float(
            optimum["description_length_bits"].mean() - typical.median()
        ),
        "proxy_note": "zlib compressed length of canonical fixed-width ASCII bitstrings; not Kolmogorov complexity",
    }
