from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    difficulty: float
    tool_need: float
    ambiguity: float
    split: str

def make_benchmark(seed: int = 42):
    """Create deterministic, domain-disjoint train/dev/test task groups."""
    rng = np.random.default_rng(seed)
    domain_groups = {
        "train": ["research", "customer_support", "data_analysis", "planning"],
        "dev": ["legal_ops", "content_ops"],
        "test": ["health_admin", "commerce_ops"],
    }
    tasks = []
    idx = 0
    for split, domains in domain_groups.items():
        for domain in domains:
            for _ in range(18):
                idx += 1
                tasks.append(Task(
                    task_id=f"T{idx:04d}",
                    domain=domain,
                    difficulty=float(rng.uniform(0.15, 0.95)),
                    tool_need=float(rng.uniform(0.0, 1.0)),
                    ambiguity=float(rng.uniform(0.0, 1.0)),
                    split=split,
                ))
    return tasks

def by_split(tasks, split):
    return [t for t in tasks if t.split == split]
