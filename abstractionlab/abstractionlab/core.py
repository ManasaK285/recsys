from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int = 500
    seed: int = 42
    noise: float = 0.025
    target: float = 0.75

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if self.noise < 0:
            raise ValueError("noise must be non-negative")
        if not -1.0 <= self.target <= 1.0:
            raise ValueError("target must be between -1 and 1")


def simulate_continuous_world(
    steps: int = 500,
    seed: int = 42,
    noise: float = 0.025,
    target: float = 0.75,
) -> pd.DataFrame:
    """Simulate a target-seeking continuous toy system, reproducible for a seed.

    The target attraction is explicit: this environment is designed to test
    observer-selected representations around a reachable target. It is not a
    model of consciousness or a model of a real physical substrate.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if not -1.0 <= target <= 1.0:
        raise ValueError("target must be between -1 and 1")

    rng = np.random.default_rng(seed)
    x = np.zeros(steps, dtype=float)
    external = np.zeros(steps, dtype=float)
    x[0] = 0.15
    for t in range(1, steps):
        # Small bounded perturbations around a reachable target.
        external[t] = 0.04 * np.sin(t / 17.0) + rng.normal(0.0, noise)
        x[t] = np.clip(
            0.92 * x[t - 1] + 0.08 * target + 0.08 * external[t],
            -1.0,
            1.0,
        )
    return pd.DataFrame({"step": np.arange(steps), "state": x, "external_input": external})


def discretize(values: Iterable[float], thresholds: Iterable[float]) -> np.ndarray:
    """Map continuous values to integer labels using observer-selected thresholds."""
    values_array = np.asarray(list(values), dtype=float)
    thresholds_array = np.asarray(list(thresholds), dtype=float)
    if thresholds_array.ndim != 1 or np.any(np.diff(thresholds_array) <= 0):
        raise ValueError("thresholds must be a strictly increasing 1D sequence")
    if not np.all(np.isfinite(thresholds_array)):
        raise ValueError("thresholds must be finite")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("values must be finite")
    return np.digitize(values_array, thresholds_array)


def policy_from_symbols(symbols: Iterable[int], target_symbol: int) -> np.ndarray:
    """Toy symbolic policy: move up below the target symbol, down above, else hold."""
    symbols_array = np.asarray(list(symbols), dtype=int)
    return np.where(
        symbols_array < target_symbol,
        1,
        np.where(symbols_array > target_symbol, -1, 0),
    )


def run_map_experiment(world: pd.DataFrame, target: float = 0.75) -> tuple[pd.DataFrame, dict]:
    state = world["state"].to_numpy(dtype=float)
    maps = {
        "coarse_map": [-0.25, 0.25, 0.65],
        "balanced_map": [-0.1, 0.2, 0.45, 0.7],
        "fine_map": [-0.4, -0.1, 0.1, 0.3, 0.5, 0.7, 0.85],
    }
    rows = world.copy()
    symbol_columns: dict[str, np.ndarray] = {}
    action_columns: dict[str, np.ndarray] = {}
    for name, thresholds in maps.items():
        symbols = discretize(state, thresholds)
        target_symbol = int(discretize([target], thresholds)[0])
        actions = policy_from_symbols(symbols, target_symbol)
        rows[f"{name}_symbol"] = symbols
        rows[f"{name}_action"] = actions
        symbol_columns[name] = symbols
        action_columns[name] = actions

    symbol_disagreement = float(np.mean(symbol_columns["coarse_map"] != symbol_columns["balanced_map"]))
    action_agreement = float(np.mean(action_columns["coarse_map"] == action_columns["balanced_map"]))
    tolerance = 0.05
    distances = np.abs(state - target)
    task_success = bool(np.any(distances <= tolerance))
    summary = {
        "steps": int(len(world)),
        "symbol_disagreement_coarse_vs_balanced": symbol_disagreement,
        "action_agreement_coarse_vs_balanced": action_agreement,
        "task_success": task_success,
        "target_tolerance": tolerance,
        "minimum_target_distance": float(np.min(distances)),
        "final_state": float(state[-1]),
        "target": float(target),
        "interpretation": "Representation sensitivity in this target-seeking toy environment; not a consciousness measure.",
    }
    return rows, summary


def run_substrate_comparison(steps: int = 500, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Compare quantized and continuous toy dynamics under identical inputs."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    rng = np.random.default_rng(seed + 1)
    inputs = 0.4 * np.sin(np.arange(steps) / 13.0) + rng.normal(0, 0.015, steps)
    discrete_state = np.zeros(steps)
    continuous_state = np.zeros(steps)
    discrete_output = np.zeros(steps, dtype=int)
    continuous_output = np.zeros(steps, dtype=int)

    for t in range(1, steps):
        q_prev = np.round(discrete_state[t - 1] * 10.0) / 10.0
        discrete_state[t] = np.clip(0.86 * q_prev + 0.14 * inputs[t], -1, 1)
        discrete_state[t] = np.round(discrete_state[t] * 10.0) / 10.0
        continuous_state[t] = np.clip(0.86 * continuous_state[t - 1] + 0.14 * inputs[t], -1, 1)
        discrete_output[t] = int(discrete_state[t] >= 0)
        continuous_output[t] = int(continuous_state[t] >= 0)

    frame = pd.DataFrame({
        "step": np.arange(steps),
        "input": inputs,
        "discrete_state": discrete_state,
        "continuous_state": continuous_state,
        "discrete_output": discrete_output,
        "continuous_output": continuous_output,
    })
    summary = {
        "steps": int(steps),
        "output_agreement": float(np.mean(discrete_output == continuous_output)),
        "mean_absolute_state_difference": float(np.mean(np.abs(discrete_state - continuous_state))),
        "interpretation": "Compares two mathematical toy implementations; not a claim about consciousness or actual hardware.",
    }
    return frame, summary
