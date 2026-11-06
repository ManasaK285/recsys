from dataclasses import dataclass
import numpy as np

@dataclass
class Workcell:
    width: int = 14
    height: int = 14
    n_robots: int = 4
    n_tasks: int = 12
    obstacle_density: float = 0.12
    seed: int = 7

def generate_workcell(cfg: Workcell):
    """Generate a connected-enough grid scene with unique starts and targets."""
    rng = np.random.default_rng(cfg.seed)
    cells = [(x, y) for x in range(cfg.width) for y in range(cfg.height)]
    # Rectangular obstacle clusters create less toy-like geometry.
    obstacles = set()
    for _ in range(max(1, int(cfg.width * cfg.height * cfg.obstacle_density / 5))):
        x, y = int(rng.integers(1, cfg.width-1)), int(rng.integers(1, cfg.height-1))
        w, h = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        for xx in range(x, min(cfg.width-1, x+w)):
            for yy in range(y, min(cfg.height-1, y+h)):
                if rng.random() < 0.75:
                    obstacles.add((xx, yy))
    free = [p for p in cells if p not in obstacles]
    needed = cfg.n_robots + cfg.n_tasks
    if len(free) < needed:
        obstacles.clear()
        free = cells[:]
    chosen = rng.choice(len(free), size=needed, replace=False)
    points = [free[int(i)] for i in chosen]
    starts = points[:cfg.n_robots]
    tasks = points[cfg.n_robots:]
    return {"starts": starts, "tasks": tasks, "obstacles": obstacles,
            "width": cfg.width, "height": cfg.height}
