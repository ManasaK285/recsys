"""Optimization algorithms over finite binary spaces."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from .encoding import compressed_length_bits, int_to_bits
from .problems import BinaryProblem


def _flip_neighbors(x: int, n_bits: int) -> list[int]:
    return [x ^ (1 << bit) for bit in range(n_bits)]


def run_search(
    problem: BinaryProblem,
    algorithm: str,
    budget: int = 250,
    seed: int = 42,
    beta: float = 0.25,
    maximize: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Run one search and return summary plus best-so-far curve."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and >= 0")
    allowed = {"uniform", "simplicity_biased", "hill_climbing", "simulated_annealing"}
    if algorithm not in allowed:
        raise ValueError(f"algorithm must be one of {sorted(allowed)}")

    rng = np.random.default_rng(seed)
    n = problem.n_candidates
    scores = problem.scores
    sign = 1.0 if maximize else -1.0
    start = time.perf_counter()
    best_idx = None
    best_value = -np.inf
    curve = []
    seen = set()
    current = int(rng.integers(n))
    current_value = sign * float(scores[current])
    temperature = max(float(np.std(scores)), 1e-9)

    # Exact candidate weights are precomputed for the finite search space.
    if algorithm == "simplicity_biased":
        lengths = np.array([compressed_length_bits(int_to_bits(i, problem.n_bits)) for i in range(n)], dtype=float)
        # Centering avoids numerical overflow; probabilities are normalized stably.
        logw = -beta * (lengths - lengths.min())
        weights = np.exp(logw - logw.max())
        probabilities = weights / weights.sum()
    else:
        probabilities = None

    for eval_no in range(1, budget + 1):
        if algorithm == "uniform":
            candidate = int(rng.integers(n))
        elif algorithm == "simplicity_biased":
            candidate = int(rng.choice(n, p=probabilities))
        elif algorithm == "hill_climbing":
            if eval_no == 1:
                candidate = current
            else:
                neighbors = _flip_neighbors(current, problem.n_bits)
                candidate = neighbors[int(rng.integers(len(neighbors)))]
        else:  # simulated annealing
            if eval_no == 1:
                candidate = current
            else:
                neighbors = _flip_neighbors(current, problem.n_bits)
                candidate = neighbors[int(rng.integers(len(neighbors)))]

        value = sign * float(scores[candidate])
        seen.add(candidate)
        if value > best_value:
            best_value, best_idx = value, candidate

        if algorithm == "hill_climbing":
            if value >= current_value:
                current, current_value = candidate, value
        elif algorithm == "simulated_annealing":
            delta = value - current_value
            temp = temperature * max(0.02, 1.0 - eval_no / max(budget, 1))
            if delta >= 0 or rng.random() < np.exp(delta / temp):
                current, current_value = candidate, value

        curve.append({
            "evaluation": eval_no,
            "best_objective": float(best_value * sign),
            "candidate_objective": float(scores[candidate]),
            "candidate_int": candidate,
        })

    elapsed = time.perf_counter() - start
    true_best = np.max(scores) if maximize else np.min(scores)
    best_reported = float(best_value * sign)
    # Gap is expressed in objective units and is zero for exact optimum.
    gap = float(true_best - best_reported) if maximize else float(best_reported - true_best)
    summary = {
        "algorithm": algorithm,
        "seed": int(seed),
        "problem": problem.name,
        "n_bits": problem.n_bits,
        "budget": budget,
        "best_candidate_int": int(best_idx),
        "best_candidate_bits": int_to_bits(int(best_idx), problem.n_bits),
        "best_objective": best_reported,
        "known_optimum": float(true_best),
        "optimality_gap": max(0.0, gap),
        "reached_optimum": bool(np.isclose(best_reported, true_best)),
        "unique_candidates_seen": len(seen),
        "objective_evaluations": budget,
        "elapsed_seconds": elapsed,
        "beta": beta if algorithm == "simplicity_biased" else None,
    }
    return summary, pd.DataFrame(curve)
