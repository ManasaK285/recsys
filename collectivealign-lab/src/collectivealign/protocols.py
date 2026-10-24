from __future__ import annotations
from collections import Counter
from .agents import build_agent
from .scenarios import Scenario

PROTOCOLS = {"independent_vote", "free_discussion", "evidence_first"}

def run_group(scenario: Scenario, model: str, protocol: str, group_size: int, seed: int):
    """Run a group of simulated agents and return a transparent decision record."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}")
    votes = []
    for i in range(group_size):
        agent = build_agent(model, seed + i * 7919)
        vote = agent.vote(scenario, f"A{i}")
        votes.append(vote)
    # In this toy simulator, deliberation protocols aggregate each agent's score.
    # Evidence-first downweights random noise by blending scores with task evidence.
    score_by_candidate = {}
    for candidate in scenario.candidates:
        vals = [v.scores[candidate.candidate_id] for v in votes]
        mean_score = sum(vals) / len(vals)
        if protocol == "evidence_first":
            mean_score = 0.35 * mean_score + 0.65 * candidate.qualification
        elif protocol == "free_discussion":
            # A modest group-consensus effect, explicitly modeled rather than LLM discussion.
            mean_score = 0.85 * mean_score + 0.15 * candidate.qualification
        score_by_candidate[candidate.candidate_id] = float(mean_score)
    winner = max(score_by_candidate, key=score_by_candidate.get)
    winner_obj = next(c for c in scenario.candidates if c.candidate_id == winner)
    optimal = max(c.qualification for c in scenario.candidates)
    return {
        "winner_id": winner,
        "winner_group": winner_obj.profile_group,
        "winner_qualification": winner_obj.qualification,
        "optimal_qualification": float(optimal),
        "optimality_gap": float(optimal - winner_obj.qualification),
        "votes": [{"agent_id": v.agent_id, "candidate_id": v.candidate_id, "rationale": v.rationale} for v in votes],
        "aggregated_scores": score_by_candidate,
    }
