from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .scenarios import Candidate, Scenario

@dataclass
class AgentVote:
    agent_id: str
    candidate_id: str
    scores: dict[str, float]
    rationale: str

class BaseAgent:
    name = "base"
    def __init__(self, seed: int = 0, bias_strength: float = 0.18):
        self.rng = np.random.default_rng(seed)
        self.bias_strength = bias_strength

    def score_candidate(self, candidate: Candidate, scenario: Scenario) -> float:
        raise NotImplementedError

    def vote(self, scenario: Scenario, agent_id: str) -> AgentVote:
        scores = {c.candidate_id: float(self.score_candidate(c, scenario)) for c in scenario.candidates}
        winner = max(scores, key=scores.get)
        return AgentVote(agent_id, winner, scores, f"Selected {winner} using {self.name} scoring rule.")

class CompetenceAgent(BaseAgent):
    name = "competence"
    def score_candidate(self, candidate, scenario):
        return candidate.qualification + self.rng.normal(0, 0.025)

class StatusBiasAgent(BaseAgent):
    """Toy bias baseline: favors profile group A regardless of condition."""
    name = "status_bias"
    def score_candidate(self, candidate, scenario):
        group_effect = self.bias_strength if candidate.profile_group == "A" else 0.0
        return candidate.qualification + group_effect + self.rng.normal(0, 0.025)

class IdentitySensitiveAgent(BaseAgent):
    """Toy baseline whose visible-condition behavior depends on the cue.
    This is an intentionally artificial stress-test, not a claim about real models.
    """
    name = "identity_sensitive"
    def score_candidate(self, candidate, scenario):
        cue_effect = 0.0
        if scenario.condition == "visible":
            cue_effect = self.bias_strength if candidate.identity_cue == "masculine" else 0.0
        return candidate.qualification + cue_effect + self.rng.normal(0, 0.025)

class RandomAgent(BaseAgent):
    name = "random"
    def score_candidate(self, candidate, scenario):
        return float(self.rng.random())

AGENT_CLASSES = {
    "competence": CompetenceAgent,
    "status_bias": StatusBiasAgent,
    "identity_sensitive": IdentitySensitiveAgent,
    "random": RandomAgent,
}

def build_agent(name: str, seed: int, bias_strength: float = 0.18) -> BaseAgent:
    if name not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent {name!r}; choose from {sorted(AGENT_CLASSES)}")
    return AGENT_CLASSES[name](seed=seed, bias_strength=bias_strength)
