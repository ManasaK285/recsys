import argparse
import json

from config import Config
from environment.evaluator import StrictEvaluator, VulnerableEvaluator
from environment.swarm import Swarm
from analysis.metrics import aggregate, write_csv
from analysis.plots import plot_governance, plot_tradeoff
from llm.client import LLMClient


POLICIES = [
    "baseline",
    "peer_audit",
    "audit_quarantine",
    "audit_quarantine_recovery",
]

EVALUATORS = {
    "vulnerable": VulnerableEvaluator,
    "strict": StrictEvaluator,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--evaluator",
        choices=["vulnerable", "strict", "both"],
        default="both",
    )

    args = parser.parse_args()

    cfg = Config(
        agents=args.agents,
        rounds=args.rounds,
        llm_enabled=args.llm,
    )

    runs = args.runs if args.runs is not None else cfg.seeds

    if args.provider:
        cfg.llm_provider = args.provider

    if args.model:
        cfg.llm_model = args.model

    if args.evaluator == "both":
        evaluator_names = list(EVALUATORS.keys())
    else:
        evaluator_names = [args.evaluator]

    all_rows = []
    summaries = []

    for evaluator_index, evaluator_name in enumerate(evaluator_names):
        evaluator_cls = EVALUATORS[evaluator_name]

        for policy_index, policy_name in enumerate(POLICIES):
            rows = []

            for seed in range(runs):

                def factory(i):
                    if (
                        not cfg.llm_enabled
                        or i >= int(cfg.agents * cfg.llm_fraction)
                    ):
                        return None

                    return LLMClient(
                        cfg.llm_provider,
                        cfg.llm_model,
                        cfg.llm_base_url,
                        cfg.llm_api_key,
                        cfg.temperature,
                    )

                condition_seed = (
                    seed
                    + 1000 * evaluator_index
                    + 100 * policy_index
                )

                result = Swarm(
                    cfg,
                    condition_seed,
                    policy_name,
                    factory if cfg.llm_enabled else None,
                    evaluator=evaluator_cls(),
                ).run()

                rows.append(result)
                all_rows.append(result)

            summaries.append(aggregate(rows))

    write_csv(
        all_rows,
        "results/runs/v2_results.csv",
    )

    with open(
        "results/runs/v2_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summaries, f, indent=2)

    plot_governance(
        summaries,
        "results/plots/governance_adoption.png",
    )

    plot_tradeoff(
        summaries,
        "results/plots/governance_tradeoff.png",
    )

    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()