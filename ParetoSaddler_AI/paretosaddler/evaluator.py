import numpy as np
import pandas as pd
from .harness import Harness
from .benchmark import Task

# Illustrative token prices per million tokens; not tied to a live provider.
PRICE_INPUT = 2.50
PRICE_CACHED = 0.25
PRICE_OUTPUT = 15.00

def rollout(harness: Harness, task: Task, rng: np.random.Generator):
    """One stochastic agent trajectory with paired quality and token cost."""
    r, td, v, ce, rr = harness.vector()
    # Reasoning helps difficult/ambiguous tasks but consumes tokens.
    capability = (
        0.18 + 0.31 * r + 0.22 * td * task.tool_need
        + 0.18 * v * task.ambiguity + 0.13 * rr
    )
    difficulty_penalty = 0.30 * task.difficulty
    efficiency_bonus = 0.15 * ce
    expected_quality = capability - difficulty_penalty + efficiency_bonus
    quality = float(np.clip(expected_quality + rng.normal(0, 0.075), 0, 1))

    calls = 1 + int(round(2.5 * task.tool_need * td + 1.5 * rr))
    input_tokens = int(650 + 850 * r + 500 * task.ambiguity)
    cached_tokens = int(input_tokens * (0.15 + 0.55 * ce))
    output_tokens = int(180 + 430 * r + 250 * v + 120 * rr)
    # More disciplined tools can reduce unnecessary calls.
    calls = max(1, calls - int(td > 0.78))
    total_input = calls * input_tokens
    total_cached = min(total_input, calls * cached_tokens)
    total_output = calls * output_tokens
    cost = (
        (total_input - total_cached) * PRICE_INPUT
        + total_cached * PRICE_CACHED
        + total_output * PRICE_OUTPUT
    ) / 1_000_000
    cost = float(max(cost, 0.00001))
    return quality, cost, calls, total_input, total_output

def evaluate(harness: Harness, tasks: list[Task], repeats: int, seed: int):
    """Return per-task repeated observations; task weighting is equal downstream."""
    rng = np.random.default_rng(seed)
    rows = []
    for task in tasks:
        for rep in range(repeats):
            q, c, calls, tin, tout = rollout(harness, task, rng)
            rows.append({
                "harness": harness.name, "task_id": task.task_id,
                "domain": task.domain, "split": task.split, "repeat": rep,
                "quality": q, "cost": c, "tool_calls": calls,
                "input_tokens": tin, "output_tokens": tout,
            })
    return pd.DataFrame(rows)

def summarize(results: pd.DataFrame):
    per_task = results.groupby("task_id", as_index=False).agg(
        quality=("quality", "mean"), cost=("cost", "mean")
    )
    return {
        "quality": float(per_task.quality.mean()),
        "cost": float(per_task.cost.mean()),
        "tasks": int(len(per_task)),
    }
