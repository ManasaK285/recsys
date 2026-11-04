from dataclasses import asdict
import random
import pandas as pd
from .benchmark import make_benchmark, by_split
from .harness import Harness, mutate
from .evaluator import evaluate, summarize
from .bayesian import bayesian_promotion_support
from .pareto import pareto_front, select_deployment

def run_experiment(iterations=24, seed=42, repeats=3, screening_tasks=8,
                   bootstrap_draws=1200, gamma=0.70, quality_loss=0.10,
                   mode="pareto", quality_floor=0.60, cost_weight=0.15):
    tasks = make_benchmark(seed)
    train = by_split(tasks, "train")
    dev = by_split(tasks, "dev")
    test = by_split(tasks, "test")
    rng = __import__("numpy").random.default_rng(seed)
    py_rng = random.Random(seed)

    baseline = Harness()
    baseline_record = {"name": baseline.name, **summarize(
        evaluate(baseline, dev, repeats, seed + 100))}
    evaluated = [{"harness": baseline, **baseline_record}]
    history = [{
        "iteration": 0, "candidate": baseline.name, "parent": None,
        "promoted": True, "promotion_support": 1.0,
        "quality": baseline_record["quality"], "cost": baseline_record["cost"],
        "reason": "initial baseline",
    }]

    # Candidate parents are drawn from all dev-evaluated harnesses, not only archive.
    for iteration in range(1, iterations + 1):
        parent_entry = evaluated[int(rng.integers(0, len(evaluated)))]
        parent = parent_entry["harness"]
        candidate = mutate(parent, rng, iteration)

        chosen = rng.choice(len(train), size=min(screening_tasks, len(train)), replace=False)
        screen = [train[int(i)] for i in chosen]
        parent_screen = evaluate(parent, screen, repeats, seed + iteration * 1000 + 1)
        candidate_screen = evaluate(candidate, screen, repeats, seed + iteration * 1000 + 2)

        decision = bayesian_promotion_support(
            parent_screen, candidate_screen, draws=bootstrap_draws,
            gamma=gamma, quality_loss=quality_loss, seed=seed + iteration
        )
        if decision["promoted"]:
            dev_frame = evaluate(candidate, dev, repeats, seed + iteration * 10000 + 3)
            metrics = summarize(dev_frame)
            record = {"harness": candidate, "name": candidate.name, **metrics}
            evaluated.append(record)
            reason = "promoted and evaluated on development split"
        else:
            metrics = summarize(candidate_screen)
            reason = "rejected by Bayesian Pareto screening"
        history.append({
            "iteration": iteration, "candidate": candidate.name,
            "parent": parent.name, "promoted": decision["promoted"],
            "promotion_support": decision["support"],
            "quality": metrics["quality"], "cost": metrics["cost"],
            "reason": reason,
        })

    records = [{"name": x["name"], "quality": x["quality"], "cost": x["cost"],
                "harness": x["harness"].to_dict()} for x in evaluated]
    dev_front = pareto_front(records)
    selected = select_deployment(records, mode, quality_floor, cost_weight)

    # Evaluate only after development-based selection; test domains never guide search.
    selected_test = []
    test_frames = []
    for item in selected:
        h = Harness.from_dict(item["harness"])
        frame = evaluate(h, test, repeats, seed + 900000 + len(test_frames))
        metrics = summarize(frame)
        selected_test.append({"name": h.name, **metrics})
        test_frames.append(frame)

    history_df = pd.DataFrame(history)
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    return {
        "history": history_df, "records": records, "pareto_front": dev_front,
        "selected": selected, "test_summary": selected_test,
        "test_results": test_df, "config": {
            "iterations": iterations, "seed": seed, "repeats": repeats,
            "screening_tasks": screening_tasks, "bootstrap_draws": bootstrap_draws,
            "gamma": gamma, "quality_loss": quality_loss, "mode": mode,
            "quality_floor": quality_floor, "cost_weight": cost_weight,
        }
    }
