"""Experiment orchestration and artifact writing."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from .scenarios import SCENARIOS
from .simulator import CONDITIONS, POLICIES, run_trajectory
from .scoring import score_trajectory, aggregate, aggregate_multi
from .investigator import investigate


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(output_dir="outputs", runs_per_cell=30, seed=42):
    if runs_per_cell < 1:
        raise ValueError("runs_per_cell must be >= 1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for scenario in SCENARIOS:
        for condition in CONDITIONS:
            for policy in POLICIES:
                for run_index in range(runs_per_cell):
                    raw = run_trajectory(scenario, condition, policy, seed, run_index)
                    records.append(score_trajectory(raw))

    with (out / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(out / "summary_by_condition.csv", aggregate(records, "condition"))
    write_csv(out / "summary_by_policy.csv", aggregate(records, "policy"))
    write_csv(out / "summary_by_scenario.csv", aggregate(records, "scenario_id"))
    write_csv(out / "summary_by_condition_policy.csv", aggregate_multi(records, ["condition", "policy"]))

    flagged = [r for r in records if r["flagged"]]
    report = investigate(flagged)
    (out / "investigator_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "runs_per_cell": runs_per_cell,
        "scenario_count": len(SCENARIOS),
        "condition_count": len(CONDITIONS),
        "policy_count": len(POLICIES),
        "trajectory_count": len(records),
        "flagged_count": len(flagged),
        "output_dir": str(out),
        "prototype_disclaimer": "Synthetic scripted-policy experiment; not an LLM evaluation and not the paper's reproduction.",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
