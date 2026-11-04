import argparse, json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paretosaddler.optimizer import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run ParetoSaddler synthetic experiment")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bootstrap-draws", type=int, default=1200)
    parser.add_argument("--gamma", type=float, default=0.70)
    parser.add_argument("--quality-loss", type=float, default=0.10)
    parser.add_argument("--mode", choices=["pareto", "min_cost", "weighted"], default="pareto")
    parser.add_argument("--quality-floor", type=float, default=0.60)
    parser.add_argument("--cost-weight", type=float, default=0.15)
    args = parser.parse_args()

    result = run_experiment(
        iterations=args.iterations, seed=args.seed, repeats=args.repeats,
        bootstrap_draws=args.bootstrap_draws, gamma=args.gamma,
        quality_loss=args.quality_loss, mode=args.mode,
        quality_floor=args.quality_floor, cost_weight=args.cost_weight,
    )
    out = Path("artifacts") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    result["history"].to_csv(out / "candidate_history.csv", index=False)
    result["test_results"].to_csv(out / "test_results.csv", index=False)

    summary = {
        "config": result["config"],
        "development_evaluated_count": len(result["records"]),
        "development_pareto_front": result["pareto_front"],
        "selected_harnesses": result["selected"],
        "held_out_test_results": result["test_summary"],
        "note": "Synthetic benchmark results; not ClawMark or GAIA2 reproduction."
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(9, 6))
    records = result["records"]
    ax.scatter([r["cost"] for r in records], [r["quality"] for r in records],
               alpha=0.45, label="Development-evaluated")
    front = sorted(result["pareto_front"], key=lambda x: x["cost"])
    if front:
        ax.plot([r["cost"] for r in front], [r["quality"] for r in front],
                marker="o", linewidth=2, label="Pareto frontier")
    ax.set_xlabel("Mean cost per task (USD-equivalent)")
    ax.set_ylabel("Mean task quality")
    ax.set_title("Development quality–cost trade-off")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "pareto_frontier.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    hist = result["history"]
    ax.plot(hist["iteration"], hist["promotion_support"], marker=".")
    ax.axhline(args.gamma, linestyle="--", label="Promotion threshold")
    ax.set(xlabel="Optimization iteration", ylabel="Bayesian promotion support",
           title="Candidate screening trace", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "optimization_trace.png", dpi=160); plt.close(fig)

    print(f"\nExperiment complete. Results: {out.resolve()}")
    print(f"Development-evaluated harnesses: {len(records)}")
    print(f"Pareto-optimal harnesses: {len(result['pareto_front'])}")
    print("\nHeld-out test results:")
    for row in result["test_summary"]:
        print(f"  {row['name']}: quality={row['quality']:.3f}, cost=${row['cost']:.5f}")
    print("\nSee summary.json, CSV files and PNG plots in the artifacts folder.")

if __name__ == "__main__":
    main()
