"""Compiles descriptive stats, hypothesis tests, and regression output
into a single results.json consumed by the Streamlit dashboard."""
import json
import os
from datetime import datetime, timezone


def compile_results(descriptive: dict, hypothesis_tests: dict, regressions: dict,
                     ml_metrics: dict, figures: dict, mode: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "descriptive": descriptive,
        "hypothesis_tests": hypothesis_tests,
        "regressions": regressions,
        "ml_metrics": ml_metrics,
        "figures": figures,
    }


def save_results(results: dict, results_dir: str, name: str = "results.json") -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path
