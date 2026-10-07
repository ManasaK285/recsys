from typing import Dict, List, Optional
from .candidate import Candidate
from .priority import rank

class CandidateMemory:
    def __init__(self, capacity: int, priority_mode: str = "mean"):
        self.capacity = capacity
        self.priority_mode = priority_mode
        self.items: Dict[str, Candidate] = {}
        self.total_evals = 0
        self.rejections = 0

    def add(self, candidate: Candidate) -> None:
        self.items[candidate.cid] = candidate
        self._trim()

    def _trim(self):
        if len(self.items) <= self.capacity:
            return
        ordered = rank(self.items.values(), self.total_evals, self.priority_mode)
        keep = ordered[: self.capacity]
        self.items = {c.cid: c for c in keep}

    def record_eval(self, cid: str, score: float, feedback: str) -> None:
        self.items[cid].record(score, feedback)
        self.total_evals += 1

    def select(self, k: int) -> List[Candidate]:
        return rank(self.items.values(), self.total_evals, self.priority_mode)[:k]

    def best(self) -> Optional[Candidate]:
        if not self.items:
            return None
        return max(self.items.values(), key=lambda c: c.mean)

    def texts(self) -> List[str]:
        return [c.program for c in self.items.values()]

    def summary_rows(self, limit: int = 8):
        return [
            {"cid": c.cid, "mean": round(c.mean, 4), "n": c.count, "program": c.program}
            for c in rank(self.items.values(), self.total_evals, self.priority_mode)[:limit]
        ]
