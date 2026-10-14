from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from honeypotlab.analysis import summarize


def main():
    output = ROOT / "outputs"
    run_file = output / "runs.csv"
    if not run_file.exists():
        raise SystemExit("No outputs/runs.csv found. Run scripts/run_experiments.py first.")
    df = pd.read_csv(run_file)
    summary = summarize(df)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
