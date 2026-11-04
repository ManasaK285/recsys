from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class Harness:
    name: str = "baseline"
    reasoning_depth: float = 0.50
    tool_discipline: float = 0.50
    verification: float = 0.50
    context_efficiency: float = 0.50
    retry_rate: float = 0.50

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def vector(self):
        return np.array([
            self.reasoning_depth, self.tool_discipline,
            self.verification, self.context_efficiency, self.retry_rate
        ], dtype=float)

def mutate(parent: Harness, rng: np.random.Generator, iteration: int) -> Harness:
    """Create a small policy change; no model weights are modified."""
    values = parent.vector().copy()
    index = int(rng.integers(0, len(values)))
    values[index] += float(rng.choice([-1, 1]) * rng.uniform(0.06, 0.22))
    values = np.clip(values, 0.05, 0.98)
    return Harness(
        name=f"candidate_{iteration:03d}",
        reasoning_depth=float(values[0]),
        tool_discipline=float(values[1]),
        verification=float(values[2]),
        context_efficiency=float(values[3]),
        retry_rate=float(values[4]),
    )
