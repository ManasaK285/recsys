from dataclasses import dataclass, field
from typing import List, Dict
import math

@dataclass
class Candidate:
    cid: str
    program: str
    scores: List[float] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    created_at: int = 0

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def variance(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        m = self.mean
        return sum((x - m) ** 2 for x in self.scores) / (len(self.scores) - 1)

    @property
    def count(self) -> int:
        return len(self.scores)

    @property
    def latest_feedback(self) -> str:
        return self.feedback[-1] if self.feedback else "No evaluation yet."

    def ucb(self, total_evals: int, c: float = 1.0) -> float:
        if self.count == 0:
            return float("inf")
        bonus = c * math.sqrt(math.log(max(total_evals, 2)) / self.count)
        return self.mean + bonus

    def record(self, score: float, feedback: str) -> None:
        self.scores.append(float(score))
        self.feedback.append(feedback)

    def to_dict(self) -> Dict:
        return {
            "cid": self.cid,
            "program": self.program,
            "scores": self.scores,
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
            "feedback": self.feedback,
            "created_at": self.created_at,
        }
