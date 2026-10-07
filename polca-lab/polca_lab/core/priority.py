from typing import Iterable
from .candidate import Candidate


def priority(candidate: Candidate, total_evals: int, mode: str = "mean") -> float:
    if mode == "ucb":
        return candidate.ucb(total_evals)
    return candidate.mean


def rank(candidates: Iterable[Candidate], total_evals: int, mode: str = "mean"):
    return sorted(candidates, key=lambda c: priority(c, total_evals, mode), reverse=True)
