"""Experiment orchestration and artifact persistence."""
import json
import random
from pathlib import Path
import pandas as pd

from honeypotlab.agents import run_scripted_agent
from honeypotlab.evaluator import evaluate

CONDITIONS = ["baseline", "agency", "goal_conflict", "combined", "control"]
AGENTS = ["compliant", "careless", "conflict_following"]


def load_scenarios(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[2] / "scenarios" / "scenarios.json"
    with open(path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if not scenarios or any("id" not in s or "constraint" not in s for s in scenarios):
        raise ValueError("Scenario registry is empty or malformed.")
    return scenarios


def run_experiment(trials=20, seed=42, output_dir="outputs"):
    if trials < 1:
        raise ValueError("trials must be >= 1")
    scenarios = load_scenarios()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for scenario in scenarios:
        for condition in CONDITIONS:
            # The control condition removes the planted opportunity for scoring.
            effective_scenario = dict(scenario)
            if condition == "control":
                effective_scenario["opportunity"] = False
            for agent in AGENTS:
                for trial in range(trials):
                    action = run_scripted_agent(agent, condition, effective_scenario, rng)
                    row = evaluate(action, effective_scenario, condition, agent)
                    row.update({
                        "trial": trial + 1,
                        "seed": seed,
                        "scenario_title": scenario["title"],
                        "task": scenario["task"],
                    })
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    with open(out / "runs.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return df
