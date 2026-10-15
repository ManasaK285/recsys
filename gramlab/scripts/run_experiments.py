import argparse
import json
import sys
from pathlib import Path

# Allow running directly from a source checkout without `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from gramlab.pipeline import run_experiment

parser = argparse.ArgumentParser(description="Run the GramLab synthetic audit.")
parser.add_argument("--runs-per-cell", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", default="outputs")
args = parser.parse_args()

result = run_experiment(args.output_dir, args.runs_per_cell, args.seed)
print(json.dumps(result, indent=2))
print(f"\nArtifacts written to: {Path(args.output_dir).resolve()}")
