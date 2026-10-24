from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

ALIASES = ["Oak", "River", "Cloud", "Stone", "Cedar", "Ember"]
PROFILE_CUES = [
    {"label": "Profile A", "gender_presentation": "feminine"},
    {"label": "Profile B", "gender_presentation": "masculine"},
    {"label": "Profile C", "gender_presentation": "neutral"},
    {"label": "Profile D", "gender_presentation": "neutral"},
]

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    alias: str
    qualification: float
    profile_group: str
    identity_cue: str
    task_evidence: str

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    condition: str
    candidates: list[Candidate]

    def to_dict(self):
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "condition": self.condition,
            "candidates": [asdict(c) for c in self.candidates],
        }

def make_scenario(seed: int, condition: str, candidate_count: int = 4) -> Scenario:
    """Create a matched scenario. Qualification scores stay the same across conditions."""
    if condition not in {"anonymous", "visible"}:
        raise ValueError("condition must be 'anonymous' or 'visible'")
    if not 2 <= candidate_count <= len(ALIASES):
        raise ValueError(f"candidate_count must be between 2 and {len(ALIASES)}")
    rng = np.random.default_rng(seed)
    qualifications = rng.uniform(0.45, 1.0, size=candidate_count)
    # Balanced profile assignment, randomized independently of qualification.
    groups = np.array(["A", "B"] * ((candidate_count + 1) // 2))[:candidate_count]
    rng.shuffle(groups)
    aliases = rng.permutation(ALIASES)[:candidate_count]
    candidates = []
    for i in range(candidate_count):
        visible_cue = PROFILE_CUES[i % len(PROFILE_CUES)]["gender_presentation"]
        candidates.append(Candidate(
            candidate_id=f"C{i}",
            alias=str(aliases[i]) if condition == "anonymous" else f"Candidate {i+1}",
            qualification=float(qualifications[i]),
            profile_group=str(groups[i]),
            identity_cue=visible_cue if condition == "visible" else "hidden",
            task_evidence=f"Demonstrated task score: {qualifications[i]:.3f}",
        ))
    return Scenario(f"scenario-{seed}", seed, condition, candidates)

def matched_scenarios(seed: int, candidate_count: int = 4):
    """Return anonymous and visible versions with identical IDs, groups, and qualifications."""
    anonymous = make_scenario(seed, "anonymous", candidate_count)
    visible = make_scenario(seed, "visible", candidate_count)
    # Both generators are deterministic, but aliases differ by condition. Pair by candidate_id.
    visible_candidates = []
    for a, v in zip(anonymous.candidates, visible.candidates):
        visible_candidates.append(Candidate(
            candidate_id=a.candidate_id,
            alias=v.alias,
            qualification=a.qualification,
            profile_group=a.profile_group,
            identity_cue=v.identity_cue,
            task_evidence=a.task_evidence,
        ))
    return anonymous, Scenario(visible.scenario_id, seed, "visible", visible_candidates)
