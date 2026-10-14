import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from honeypotlab.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the HoneypotLab synthetic benchmark.")
    parser.add_argument("--trials", type=int, default=20, help="Trials per scenario-condition-agent combination")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"), help="Output directory")
    args = parser.parse_args()
    df = run_experiment(args.trials, args.seed, args.output_dir)
    print(f"Completed {len(df)} runs.")
    print(f"Run-level data: {Path(args.output_dir) / 'runs.csv'}")
    print("Reminder: scripted fixtures validate the pipeline; these are not real-model results.")


if __name__ == "__main__":
    main()
