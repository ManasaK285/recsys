from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .config import ARTIFACT_DIR, TARGET_DEFAULT

def evaluate(target: str = TARGET_DEFAULT):
    metrics_path = ARTIFACT_DIR / f"metrics_{target}.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Metrics not found. Run `python -m src.diveguard.train` first.")
    payload = json.loads(metrics_path.read_text())
    print(json.dumps(payload, indent=2))
    group_path = ARTIFACT_DIR / f"group_metrics_{target}.csv"
    if group_path.exists():
        groups = pd.read_csv(group_path)
        if not groups.empty:
            print("\\nLargest subgroup MAE values:")
            print(groups.sort_values("mae", ascending=False).head(15).to_string(index=False))
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET_DEFAULT)
    args = parser.parse_args()
    evaluate(args.target)
