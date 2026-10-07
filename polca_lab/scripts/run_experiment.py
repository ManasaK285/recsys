import argparse, json, csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polca_lab.benchmarks.support_benchmark import SupportBenchmark
from polca_lab.core.evaluator import StochasticEvaluator
from polca_lab.core.optimizer import POLCAOptimizer
from polca_lab.providers.local_oracle import LocalProposalOracle
from polca_lab.providers.openai_oracle import OpenAICompatibleOracle

SEEDS = [
    "Answer customer requests directly and helpfully.",
    "Be professional and provide useful information to the customer.",
    "Resolve the customer's issue using the available policy information.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    benchmark = SupportBenchmark()
    evaluator = StochasticEvaluator(benchmark, config.get("noise_std", 0.08), config.get("seed", 7))
    oracle = LocalProposalOracle(config.get("seed", 7)) if config.get("oracle") == "local" else OpenAICompatibleOracle()
    opt = POLCAOptimizer(evaluator, oracle, config)
    opt.initialize(SEEDS[:config.get("initial_candidates", 3)])
    log = opt.run()
    prefix = config.get("output_prefix", "polca")
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    json_path = outdir / f"{prefix}.json"
    csv_path = outdir / f"{prefix}_trajectory.csv"
    opt.export(str(json_path))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader(); writer.writerows(log)
    best = opt.memory.best()
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Evaluations: {evaluator.total_evals}")
    print(f"Memory: {len(opt.memory.items)} candidates")
    print(f"Best train mean: {best.mean:.4f}")
    print(f"Held-out test score: {benchmark.full_test(best.program):.4f}")
    print("\nBEST POLICY\n" + best.program)

if __name__ == "__main__":
    main()
