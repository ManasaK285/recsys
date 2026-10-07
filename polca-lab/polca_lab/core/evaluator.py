from dataclasses import dataclass
from typing import Dict
import random

@dataclass
class Evaluation:
    score: float
    feedback: str
    details: Dict[str, float]

class StochasticEvaluator:
    """Evaluator wrapper with reproducible stochastic observations.

    The benchmark returns a latent quality vector. Noise is sampled per metric,
    so the same candidate can receive different observed rewards on repeated
    evaluations while its underlying quality remains stable.
    """
    def __init__(self, benchmark, noise_std: float = 0.08, seed: int = 7):
        self.benchmark=benchmark; self.noise_std=noise_std; self.rng=random.Random(seed); self.total_evals=0
    def evaluate(self, program: str, batch_indices=None) -> Evaluation:
        self.total_evals += 1
        score, feedback, details=self.benchmark.evaluate(program,batch_indices,self.rng,self.noise_std)
        return Evaluation(score,feedback,details)
