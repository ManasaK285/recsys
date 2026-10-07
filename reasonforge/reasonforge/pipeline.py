from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .taxonomy import build_taxonomy
from .sampler import sample_factor_combinations
from .generator import LocalGenerator, OpenAIGenerator
from .critics import run_critics
from .complexity import score_complexity
from .metrics import dataset_report
from .downstream import train_and_evaluate


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(cfg: Config):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    taxonomy = build_taxonomy(cfg.domain, cfg.task, cfg.provider)
    (out / "taxonomy.json").write_text(
        taxonomy.model_dump_json(indent=2),
        encoding="utf-8",
    )

    combinations = sample_factor_combinations(
        taxonomy,
        cfg.n_samples,
        cfg.seed,
    )

    generator = (
        OpenAIGenerator()
        if cfg.provider == "openai"
        else LocalGenerator()
    )

    scenarios = []
    for i, factors in enumerate(combinations):
        s = generator.generate(f"rf-{i:05d}", factors)
        s.metadata["complexity_score"] = score_complexity(s)
        s = run_critics(s)
        scenarios.append(s)

    report = dataset_report(scenarios)
    (out / "intrinsic_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    save_jsonl(
        out / "synthetic_dataset.jsonl",
        [s.model_dump() for s in scenarios if s.accepted],
    )

    # Evaluate against the bundled held-out gold set.
    gold_path = Path("data/gold_validation.jsonl")
    if gold_path.exists():
        gold = [
            json.loads(line)
            for line in gold_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        synthetic_train = [
            {
                "instruction": s.instruction,
                "label": s.expected_label,
            }
            for s in scenarios
            if s.accepted
        ]

        if synthetic_train and gold:
            downstream = train_and_evaluate(synthetic_train, gold)
            report["downstream_accuracy"] = downstream.accuracy
            report["downstream_macro_f1"] = downstream.macro_f1

    (out / "final_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nArtifacts written to: {out.resolve()}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--provider", choices=["local", "openai"], default="local")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        Config(
            n_samples=args.n,
            provider=args.provider,
            seed=args.seed,
        )
    )
