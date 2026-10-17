"""Run all three research-inspired experiments and export reproducible artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..complexity import complexity_table, summarize_optimum_complexity
from ..problems import PROBLEM_FAMILIES, make_problem
from ..search import run_search


def run_complexity_experiment(n_bits: int, seed: int) -> pd.DataFrame:
    rows = []
    for family in PROBLEM_FAMILIES:
        problem = make_problem(family, n_bits, seed)
        rows.append(summarize_optimum_complexity(problem))
    return pd.DataFrame(rows)


def run_search_experiment(n_bits: int, seeds: int, budget: int, beta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, curves = [], []
    # Use multiple objective families so findings are not tied to a single score landscape.
    for family in PROBLEM_FAMILIES:
        for seed in range(seeds):
            problem = make_problem(family, n_bits, seed=seed)
            for algorithm in ("uniform", "simplicity_biased", "hill_climbing", "simulated_annealing"):
                summary, curve = run_search(problem, algorithm, budget=budget, seed=seed + 1000, beta=beta)
                summary["run_seed"] = seed
                summaries.append(summary)
                curve["algorithm"] = algorithm
                curve["problem"] = family
                curve["run_seed"] = seed
                curves.append(curve)
    return pd.DataFrame(summaries), pd.concat(curves, ignore_index=True)


def run_extrema_experiment(n_bits: int, pairs: int, null_samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare structured objective pairs with shuffled-score null pairs.

    Each objective's score multiset is preserved under permutation, so the number
    of ties/optima is preserved. This is a transparent null, not a universal one.
    """
    if pairs < 1 or null_samples < 1:
        raise ValueError("pairs and null_samples must be >= 1")
    rng = np.random.default_rng(seed)
    families = ("ones", "target_similarity", "structured_blocks")
    observed_rows, null_rows = [], []
    count = 1 << n_bits
    for pair_idx in range(pairs):
        fam_a = families[pair_idx % len(families)]
        fam_b = families[(pair_idx + 1) % len(families)]
        a = make_problem(fam_a, n_bits, seed=seed + pair_idx).scores
        b = make_problem(fam_b, n_bits, seed=seed + 10000 + pair_idx).scores
        opt_a = set(np.flatnonzero(np.isclose(a, a.max())).tolist())
        opt_b = set(np.flatnonzero(np.isclose(b, b.max())).tolist())
        intersection = len(opt_a & opt_b)
        union = len(opt_a | opt_b)
        observed_rate = intersection / max(1, min(len(opt_a), len(opt_b)))
        observed_rows.append({
            "pair_id": pair_idx, "objective_a": fam_a, "objective_b": fam_b,
            "n_candidates": count, "optima_a": len(opt_a), "optima_b": len(opt_b),
            "intersection_size": intersection, "jaccard_overlap": intersection / union if union else 0.0,
            "intersection_over_smaller_optimum_set": observed_rate,
        })
        for null_id in range(null_samples):
            # Independently permute the score-to-candidate assignments. Score multisets,
            # and therefore tie counts, are preserved.
            pa = rng.permutation(a)
            pb = rng.permutation(b)
            na = set(np.flatnonzero(np.isclose(pa, pa.max())).tolist())
            nb = set(np.flatnonzero(np.isclose(pb, pb.max())).tolist())
            ni = len(na & nb)
            nu = len(na | nb)
            null_rows.append({
                "pair_id": pair_idx, "null_sample": null_id,
                "intersection_size": ni,
                "jaccard_overlap": ni / nu if nu else 0.0,
                "intersection_over_smaller_optimum_set": ni / max(1, min(len(na), len(nb))),
            })
    return pd.DataFrame(observed_rows), pd.DataFrame(null_rows)


def run_all(out: Path, n_bits: int = 10, seeds: int = 5, budget: int = 250, beta: float = 0.25,
            extrema_pairs: int = 30, null_samples: int = 200, seed: int = 42) -> dict:
    if not 1 <= n_bits <= 16:
        raise ValueError("n_bits must be between 1 and 16")
    if seeds < 1 or budget < 1:
        raise ValueError("seeds and budget must be >= 1")
    out.mkdir(parents=True, exist_ok=True)

    complexity = run_complexity_experiment(n_bits, seed)
    complexity.to_csv(out / "complexity_optima.csv", index=False)

    search_runs, search_curves = run_search_experiment(n_bits, seeds, budget, beta)
    search_runs.to_csv(out / "search_runs.csv", index=False)
    search_curves.to_csv(out / "search_curves.csv", index=False)

    observed, null = run_extrema_experiment(n_bits, extrema_pairs, null_samples, seed)
    observed.to_csv(out / "extrema_pairs.csv", index=False)
    null.to_csv(out / "extrema_null_samples.csv", index=False)

    algo_aggregate = (search_runs.groupby(["problem", "algorithm"], as_index=False)
                      .agg(mean_best_objective=("best_objective", "mean"),
                           mean_optimality_gap=("optimality_gap", "mean"),
                           optimum_success_rate=("reached_optimum", "mean"),
                           mean_unique_candidates=("unique_candidates_seen", "mean"),
                           mean_runtime_seconds=("elapsed_seconds", "mean")))
    aggregate_path = out / "search_aggregates.csv"
    algo_aggregate.to_csv(aggregate_path, index=False)

    # Pairwise null summaries: descriptive comparisons, not automatic significance claims.
    null_agg = null.groupby("pair_id")["jaccard_overlap"].agg(["mean", "std"]).reset_index()
    merged = observed.merge(null_agg, on="pair_id", how="left")
    merged = merged.rename(columns={"mean": "null_mean_jaccard", "std": "null_std_jaccard"})
    merged["observed_minus_null_mean"] = merged["jaccard_overlap"] - merged["null_mean_jaccard"]
    merged.to_csv(out / "extrema_pairs.csv", index=False)

    summary = {
        "project": "OptiSimplicity Lab",
        "seed": seed,
        "configuration": {"n_bits": n_bits, "candidate_space_size": 1 << n_bits, "seeds_per_search_family": seeds,
                          "evaluations_per_run": budget, "simplicity_beta": beta,
                          "extrema_objective_pairs": extrema_pairs, "null_samples_per_pair": null_samples},
        "experiments": {
            "A_complexity_of_optima": {
                "output": "complexity_optima.csv",
                "interpretation": "Compares zlib description-length proxy of exact optima with the full candidate population."
            },
            "B_search_benchmark": {
                "outputs": ["search_runs.csv", "search_curves.csv", "search_aggregates.csv"],
                "algorithms": ["uniform", "simplicity_biased", "hill_climbing", "simulated_annealing"],
                "interpretation": "Compare quality, optimality gap, success rate, unique candidates, and runtime under fixed objective-evaluation budgets."
            },
            "C_coincidence_of_extrema": {
                "outputs": ["extrema_pairs.csv", "extrema_null_samples.csv"],
                "interpretation": "Compares observed optimal-set overlap against independently shuffled score assignments preserving score multisets and tie counts."
            }
        },
        "scientific_caveats": [
            "Compressed length is an encoding-dependent proxy, not Kolmogorov complexity.",
            "The simplicity-biased sampler is not true universal algorithmic probability.",
            "Synthetic objective families and small exact search spaces limit generalization.",
            "The shuffled-score null model is one explicit baseline, not the only possible null.",
            "No result is hard-coded; conclusions must be drawn from generated output."
        ]
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--n-bits", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--extrema-pairs", type=int, default=30)
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_all(Path(args.out), args.n_bits, args.seeds, args.budget, args.beta,
                      args.extrema_pairs, args.null_samples, args.seed)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote experiment artifacts to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
