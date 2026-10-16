from __future__ import annotations
import json
from pathlib import Path
from .core import simulate_continuous_world, run_map_experiment, run_substrate_comparison


def run_all(steps: int = 500, seed: int = 42, noise: float = 0.025, target: float = 0.75, out: str = "outputs") -> dict:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    world = simulate_continuous_world(steps=steps, seed=seed, noise=noise, target=target)
    trajectory, map_summary = run_map_experiment(world, target=target)
    substrate, substrate_summary = run_substrate_comparison(steps=steps, seed=seed)

    trajectory.to_csv(outdir / "trajectory.csv", index=False)
    substrate.to_csv(outdir / "substrate_comparison.csv", index=False)
    summary = {
        "project": "AbstractionLab",
        "seed": seed,
        "map_experiment": map_summary,
        "substrate_comparison": substrate_summary,
        "scientific_note": "These experiments operationalize narrow representation/dynamics questions. They cannot establish or rule out consciousness.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
