"""
ReasonLens end-to-end reproducible pipeline.

Usage:
    python -m experiments.run_all               # simulation mode (default)
    python -m experiments.run_all --mode pilot --data-dir path/to/export

Steps:
    1. Generate/replay experiment data (simulation) or load a pilot export
    2. Validate data
    3. Calculate descriptive statistics
    4. Run hypothesis tests
    5. Train ML models (reason-quality classifier + regressor)
    6. Generate plots
    7. Generate dashboard datasets (tidy processed CSV)
    8. Generate final results JSON
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, HERE)

import simulation  # noqa: E402
from analysis import preprocessing, descriptive, statistics as stats_mod, regression, visualization, results as results_mod  # noqa: E402
from ml import train as ml_train, evaluate as ml_evaluate, explain as ml_explain  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the full ReasonLens pipeline.")
    parser.add_argument("--mode", choices=["simulation", "pilot"], default="simulation")
    parser.add_argument("--data-dir", default=None, help="Required for --mode pilot: directory with the four CSV exports")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = simulation.load_config(args.config)
    synthetic_dir = os.path.join(ROOT, cfg["output"]["synthetic_dir"])
    processed_dir = os.path.join(ROOT, cfg["output"]["processed_dir"])
    results_dir = os.path.join(ROOT, cfg["output"]["results_dir"])
    figures_dir = os.path.join(ROOT, cfg["output"]["figures_dir"])

    # --- Step 1: generate or load data ---
    if args.mode == "simulation":
        print("== Step 1/8: generating synthetic data (simulation mode) ==")
        simulation.run(config_path=args.config, out_dir=synthetic_dir)
        data_dir = synthetic_dir
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir is required in --mode pilot")
        print(f"== Step 1/8: loading pilot export from {args.data_dir} ==")
        data_dir = args.data_dir

    # --- Step 2: validate ---
    print("== Step 2/8: validating data ==")
    df = preprocessing.build_tidy_dataframe(data_dir)
    warnings = preprocessing.validate(df)
    for w in warnings:
        print(f"  [warning] {w}")
    if not warnings:
        print("  no validation issues found")

    # --- Step 7 (dataset persisted early so later steps + dashboard can use it) ---
    tidy_path = preprocessing.save_processed(df, processed_dir)
    print(f"  tidy dataset saved to {tidy_path}")

    # --- Step 3: descriptive statistics ---
    print("== Step 3/8: descriptive statistics ==")
    descriptive_results = descriptive.summarize(df)
    print(f"  overall grant rate: {descriptive_results['overall']['grant_rate']:.3f}")

    # --- Step 4: hypothesis tests ---
    print("== Step 4/8: hypothesis tests ==")
    hypothesis_results = stats_mod.run_all_tests(df)
    regression_results = regression.run_all_regressions(df)

    # --- Step 5: ML models ---
    print("== Step 5/8: training ML reason-quality models ==")
    clf, reg, split = ml_train.train_all(out_dir=results_dir, seed=cfg["seed"])
    ml_metrics = ml_evaluate.run_evaluation(clf, reg, split)
    ml_metrics["explainability"] = {
        "regressor_top_terms": ml_explain.top_terms_for_regressor(reg),
        "classifier_top_terms": ml_explain.top_terms_for_classifier(clf),
        "shap": ml_explain.shap_summary(reg, split["X_test"]),
    }

    # --- Step 6: plots ---
    print("== Step 6/8: generating plots ==")
    figures = visualization.generate_all(df, figures_dir)

    # --- Step 7: dashboard dataset already written above ---
    print("== Step 7/8: dashboard dataset ready ==")

    # --- Step 8: final results JSON ---
    print("== Step 8/8: compiling results.json ==")
    compiled = results_mod.compile_results(
        descriptive=descriptive_results,
        hypothesis_tests=hypothesis_results,
        regressions=regression_results,
        ml_metrics=ml_metrics,
        figures={k: os.path.relpath(v, ROOT) for k, v in figures.items()},
        mode=args.mode,
    )
    results_path = results_mod.save_results(compiled, results_dir)
    print(f"  results written to {results_path}")

    print("\nDone. Launch the dashboard with:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
