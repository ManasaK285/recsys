from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.benchmark import make_benchmark, feature_matrix
from src.models.gp_surrogate import GPSurrogate
from src.discovery.acquisition import choose_next, STRATEGIES
from src.estimation.estimators import uniform_sample_estimate, gp_plugin_estimate
from src.evaluation.metrics import failure_metrics

def run_experiment(n_cases=500, budget=80, seed=42, strategy="hybrid",
                   checkpoints=10, threshold=.65, target_mae=.03, max_train_points=250):
    if budget < 5:
        raise ValueError("budget must be at least 5")
    if budget >= n_cases:
        budget = n_cases - 1
    rng = np.random.default_rng(seed)
    df = make_benchmark(n_cases=n_cases, seed=seed)
    X = feature_matrix(df)
    y = df["error_severity"].to_numpy()
    categories = df["category"].to_numpy()
    ground_truth = float(y.mean())

    initial_size = min(8, max(3, budget // 8))
    observed = list(rng.choice(n_cases, size=initial_size, replace=False))
    history = []
    curve = []
    checkpoint_budgets = sorted(set(np.linspace(initial_size, budget, num=max(2, checkpoints), dtype=int).tolist()))
    target_reached_at = None

    while len(observed) <= budget:
        obs = np.array(observed, dtype=int)
        gp = GPSurrogate(max_train_points=max_train_points, random_state=seed)
        gp.fit(X[obs], y[obs])
        pred_mean, pred_std = gp.predict(X)
        # GP plug-in estimate uses known labels only at queried indices.
        gp_est = gp_plugin_estimate(obs, y[obs], pred_mean)
        # A simple uniform sample mean is reported as a comparison, but is unbiased only
        # for random sampling; under adaptive acquisition it is descriptive, not unbiased.
        sample_est = uniform_sample_estimate(y[obs])
        mae = abs(gp_est - ground_truth)
        if len(observed) in checkpoint_budgets or len(observed) == budget:
            curve.append({
                "seed": seed, "strategy": strategy, "budget_used": len(observed),
                "ground_truth_mean_severity": ground_truth,
                "gp_estimate": gp_est, "sample_mean": sample_est,
                "gp_estimation_abs_error": mae,
            })
            if target_reached_at is None and mae <= target_mae:
                target_reached_at = len(observed)
        if len(observed) >= budget:
            break
        remaining = np.setdiff1d(np.arange(n_cases), obs, assume_unique=False)
        next_idx = choose_next(
            strategy, remaining, pred_mean, pred_std, y[obs], len(observed), rng,
            threshold=threshold, batch_size=1
        )
        if len(next_idx) == 0:
            break
        observed.extend(next_idx.tolist())

    queried = np.array(observed, dtype=int)
    fm = failure_metrics(y, queried, threshold)
    found = queried[y[queried] >= threshold]
    category_coverage = len(set(categories[found])) if len(found) else 0
    # final GP estimate
    obs = np.array(queried, dtype=int)
    gp = GPSurrogate(max_train_points=max_train_points, random_state=seed).fit(X[obs], y[obs])
    final_mean, final_std = gp.predict(X)
    gp_est = gp_plugin_estimate(obs, y[obs], final_mean)
    result = {
        "seed": int(seed), "strategy": strategy, "n_cases": int(n_cases),
        "budget": int(budget), "evaluations_used": int(len(queried)),
        "ground_truth_mean_severity": ground_truth,
        "gp_estimate": gp_est, "gp_estimation_abs_error": abs(gp_est-ground_truth),
        "sample_mean": uniform_sample_estimate(y[queried]),
        "failure_threshold": threshold,
        **fm, "failure_category_coverage": int(category_coverage),
        "evaluations_to_tolerance": target_reached_at,
        "mean_predictive_std": float(np.mean(final_std)),
        "queried_indices": queried.tolist(),
    }
    return result, pd.DataFrame(curve), df.iloc[queried].copy()

def run_suite(n_cases=500, budget=80, seeds=3, checkpoints=10, threshold=.65, target_mae=.03):
    results, curves = [], []
    for seed in range(seeds):
        for strategy in STRATEGIES:
            result, curve, _ = run_experiment(
                n_cases=n_cases, budget=budget, seed=seed+42, strategy=strategy,
                checkpoints=checkpoints, threshold=threshold, target_mae=target_mae
            )
            results.append(result)
            curves.append(curve)
    raw = pd.DataFrame(results)
    summary = raw.groupby("strategy").agg(
        mean_gp_abs_error=("gp_estimation_abs_error", "mean"),
        std_gp_abs_error=("gp_estimation_abs_error", "std"),
        mean_failure_precision=("failure_precision", "mean"),
        mean_failure_recall=("failure_recall", "mean"),
        mean_failures_found=("failures_found", "mean"),
        mean_category_coverage=("failure_category_coverage", "mean"),
        mean_predictive_std=("mean_predictive_std", "mean"),
    ).reset_index()
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return raw, summary, curve_df

def main():
    parser = argparse.ArgumentParser(description="Run ProEval-Lab synthetic evaluation experiments")
    parser.add_argument("--n-cases", type=int, default=500)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=.65)
    parser.add_argument("--target-mae", type=float, default=.03)
    args = parser.parse_args()
    Path("reports").mkdir(exist_ok=True)
    if args.strategy == "all":
        raw, summary, curve = run_suite(args.n_cases, args.budget, args.seeds, args.checkpoints, args.threshold, args.target_mae)
        raw.to_csv("reports/seed_results.csv", index=False)
        summary.to_csv("reports/summary.csv", index=False)
        curve.to_csv("reports/estimation_curves.csv", index=False)
        payload = {"config": vars(args), "summary": summary.to_dict(orient="records")}
    else:
        results, curves = [], []
        for seed in range(args.seeds):
            r, c, _ = run_experiment(args.n_cases, args.budget, seed+42, args.strategy,
                                     args.checkpoints, args.threshold, args.target_mae)
            results.append(r); curves.append(c)
        raw = pd.DataFrame(results)
        curve = pd.concat(curves, ignore_index=True)
        raw.to_csv("reports/seed_results.csv", index=False)
        curve.to_csv("reports/estimation_curves.csv", index=False)
        payload = {"config": vars(args), "results": raw.drop(columns=["queried_indices"]).to_dict(orient="records")}
    with open("reports/latest_run.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print("\nSaved experiment outputs to reports/")

if __name__ == "__main__":
    main()
