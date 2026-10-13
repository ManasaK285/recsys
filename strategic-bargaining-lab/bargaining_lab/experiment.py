from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .simulation import run_experiment, summarize_pairwise, STRATEGIES


def main():
    parser = argparse.ArgumentParser(description="Run reproducible bargaining simulation experiments.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rounds, agents, summary = run_experiment(args.games, args.agents, args.rounds, args.seed)
    pairwise = summarize_pairwise(rounds)
    rounds.to_csv(out / "rounds.csv", index=False)
    agents.to_csv(out / "agent_outcomes.csv", index=False)
    summary.to_csv(out / "strategy_summary.csv", index=False)
    pairwise.to_csv(out / "offer_acceptance_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["strategy"], summary["mean_payoff"])
    ax.set_ylabel("Mean accumulated payoff per agent")
    ax.set_title("Bargaining outcomes by policy archetype")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "mean_payoff_by_strategy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(pairwise["proposer_strategy"], pairwise["acceptance_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Offer acceptance rate")
    ax.set_title("Acceptance rate by proposer policy")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "acceptance_by_strategy.png", dpi=160)
    plt.close(fig)

    metadata = {
        "games": args.games, "agents_per_game": args.agents, "max_rounds": args.rounds,
        "seed": args.seed, "strategies": list(STRATEGIES),
        "note": "Synthetic toy-policy simulation; not a reproduction of the paper's participant study."
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Experiment complete.")
    print("\nStrategy outcomes:\n", summary.to_string(index=False))
    print("\nOffer behavior:\n", pairwise.to_string(index=False))
    print(f"\nArtifacts written to: {out.resolve()}")


if __name__ == "__main__":
    main()
