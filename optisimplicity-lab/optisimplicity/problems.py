"""Small, exactly enumerable binary combinatorial optimization problems."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BinaryProblem:
    name: str
    n_bits: int
    scores: np.ndarray
    description: str

    def __post_init__(self):
        expected = 1 << self.n_bits
        if self.n_bits < 1 or self.scores.shape != (expected,):
            raise ValueError("scores must have one value for every binary candidate")

    @property
    def n_candidates(self) -> int:
        return 1 << self.n_bits

    def score_int(self, candidate: int) -> float:
        if not 0 <= candidate < self.n_candidates:
            raise ValueError("candidate outside search space")
        return float(self.scores[candidate])

    def optimal_indices(self, maximize: bool = True) -> np.ndarray:
        best = np.max(self.scores) if maximize else np.min(self.scores)
        return np.flatnonzero(np.isclose(self.scores, best))


def make_problem(name: str, n_bits: int, seed: int = 42) -> BinaryProblem:
    """Create a deterministic objective over every n-bit string."""
    if not 1 <= n_bits <= 16:
        raise ValueError("n_bits must be between 1 and 16 for exact experiments")
    count = 1 << n_bits
    ids = np.arange(count, dtype=np.uint32)
    bits = ((ids[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1).astype(np.int8)

    if name == "ones":
        scores = bits.sum(axis=1).astype(float)
        desc = "Maximize the number of one bits."
    elif name == "target_similarity":
        rng = np.random.default_rng(seed)
        target = rng.integers(0, 2, size=n_bits, dtype=np.int8)
        scores = (bits == target).sum(axis=1).astype(float)
        desc = f"Maximize bitwise similarity to seeded target {''.join(map(str, target.tolist()))}."
    elif name == "structured_blocks":
        # Reward alternating two-bit blocks (00 or 11) and reward matching a
        # deterministic repeating 0011 pattern, making the objective structured.
        pattern = np.array(([0, 0, 1, 1] * ((n_bits + 3) // 4))[:n_bits], dtype=np.int8)
        matches = (bits == pattern).sum(axis=1)
        pairs = n_bits // 2
        pair_bonus = np.zeros(count, dtype=float)
        if pairs:
            pair_bonus = (bits[:, :2 * pairs:2] == bits[:, 1:2 * pairs:2]).sum(axis=1)
        scores = matches.astype(float) + 0.25 * pair_bonus
        desc = "Reward a repeating 0011 target pattern and equal adjacent bit pairs."
    elif name == "random_table":
        rng = np.random.default_rng(seed)
        scores = rng.normal(size=count)
        desc = "Seeded random score assigned independently to every candidate."
    else:
        raise ValueError(f"unknown problem family: {name}")
    return BinaryProblem(name=name, n_bits=n_bits, scores=np.asarray(scores, dtype=float), description=desc)


PROBLEM_FAMILIES = ("ones", "target_similarity", "structured_blocks", "random_table")
