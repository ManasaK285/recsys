from __future__ import annotations
import json
from pathlib import Path
import yaml
import pandas as pd
from .scenarios import matched_scenarios
from .protocols import run_group

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_experiment(config: dict, out_dir: str | Path) -> pd.DataFrame:
    exp = config["experiment"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in range(int(exp["trials"])):
        trial_seed = int(exp["seed"]) + trial * 101
        anon, visible = matched_scenarios(trial_seed, int(exp["candidate_count"]))
        for scenario in (anon, visible):
            for model_idx, model in enumerate(exp["models"]):
                for protocol_idx, protocol in enumerate(exp["protocols"]):
                    result = run_group(
                        scenario=scenario,
                        model=model,
                        protocol=protocol,
                        group_size=int(exp["group_size"]),
                        seed=trial_seed + model_idx * 1009 + protocol_idx * 313,
                    )
                    rows.append({
                        "trial": trial,
                        "scenario_id": scenario.scenario_id,
                        "condition": scenario.condition,
                        "model": model,
                        "protocol": protocol,
                        "winner_id": result["winner_id"],
                        "winner_group": result["winner_group"],
                        "winner_qualification": result["winner_qualification"],
                        "optimal_qualification": result["optimal_qualification"],
                        "optimality_gap": result["optimality_gap"],
                        "group_size": int(exp["group_size"]),
                        "seed": trial_seed,
                        "synthetic": True,
                        "votes_json": json.dumps(result["votes"]),
                        "scores_json": json.dumps(result["aggregated_scores"], sort_keys=True),
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out / "trial_results.csv", index=False)
    df.to_json(out / "trial_results.jsonl", orient="records", lines=True)
    (out / "run_metadata.json").write_text(json.dumps({
        "experiment": exp,
        "data_mode": "synthetic",
        "warning": "These are simulated results, not observations from the paper or real people."
    }, indent=2), encoding="utf-8")
    return df
