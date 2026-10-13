"""Seeded, transparent toy environment for studying bargaining strategy tradeoffs.

This is a research-inspired simulator, not a reproduction of the paper's human study.
All agents are hand-coded policy archetypes; no real LLM calls or human data are used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np
import pandas as pd


STRATEGIES = ("llm_conservative", "human_fairness", "bayesian_aggressive")


@dataclass
class Agent:
    agent_id: int
    strategy: str
    patience: float
    fairness_weight: float
    risk_tolerance: float


@dataclass
class RoundRecord:
    game_id: int
    round_id: int
    proposer_id: int
    proposer_strategy: str
    responder_id: int
    responder_strategy: str
    offer_to_responder: float
    offer_accepted: bool
    proposer_surplus: float
    responder_surplus: float
    total_surplus: float
    rejection_count: int


def _make_agents(n_agents: int, rng: np.random.Generator, mix: dict[str, float]) -> list[Agent]:
    names = list(mix)
    weights = np.array([mix[n] for n in names], dtype=float)
    if len(names) == 0 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("mix must contain non-negative strategy weights with positive sum")
    weights /= weights.sum()
    agents = []
    for i in range(n_agents):
        strategy = str(rng.choice(names, p=weights))
        agents.append(Agent(
            agent_id=i,
            strategy=strategy,
            patience=float(rng.uniform(0.45, 0.98)),
            fairness_weight=float(rng.uniform(0.35, 0.95)),
            risk_tolerance=float(rng.uniform(0.25, 0.95)),
        ))
    return agents


def _offer_share(agent: Agent, round_idx: int, max_rounds: int, rng: np.random.Generator) -> float:
    """Share of the total pie offered to the responder; lower is more aggressive."""
    progress = round_idx / max(1, max_rounds - 1)
    if agent.strategy == "llm_conservative":
        # High acceptance orientation; offers become a little more generous late in a game.
        share = 0.47 + 0.08 * progress + rng.normal(0, 0.025)
    elif agent.strategy == "human_fairness":
        # Fairness norm centered near equal split, with more variance.
        share = 0.50 + 0.05 * (agent.fairness_weight - 0.65) + rng.normal(0, 0.065)
    elif agent.strategy == "bayesian_aggressive":
        # Surplus-seeking policy makes low offers, adapting upward as the deadline nears.
        share = 0.18 + 0.17 * progress + 0.08 * agent.risk_tolerance + rng.normal(0, 0.035)
    else:
        raise ValueError(f"Unknown strategy: {agent.strategy}")
    return float(np.clip(share, 0.02, 0.98))


def _acceptance_probability(responder: Agent, offered_share: float, round_idx: int, max_rounds: int) -> float:
    progress = round_idx / max(1, max_rounds - 1)
    if responder.strategy == "llm_conservative":
        threshold = 0.30 + 0.07 * progress
        slope = 15.0
    elif responder.strategy == "human_fairness":
        threshold = 0.43 + 0.04 * responder.fairness_weight + 0.04 * progress
        slope = 17.0
    elif responder.strategy == "bayesian_aggressive":
        threshold = 0.36 + 0.10 * responder.risk_tolerance + 0.03 * progress
        slope = 20.0
    else:
        raise ValueError(f"Unknown strategy: {responder.strategy}")
    z = np.clip(slope * (offered_share - threshold), -30, 30)
    return float(1.0 / (1.0 + np.exp(-z)))


def simulate_game(
    game_id: int,
    n_agents: int = 4,
    max_rounds: int = 12,
    seed: int = 7,
    mix: dict[str, float] | None = None,
) -> tuple[list[RoundRecord], list[dict]]:
    """Simulate a rotating-proposer bargaining game with a normalized unit pie.

    At each round, one proposer negotiates with one randomly selected responder.
    Acceptance allocates (1 - offered_share) to proposer and offered_share to responder.
    If no offer is accepted before the deadline, the pie expires and both receive zero.
    """
    if n_agents < 2:
        raise ValueError("n_agents must be at least 2")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    mix = mix or {s: 1.0 for s in STRATEGIES}
    rng = np.random.default_rng(seed)
    agents = _make_agents(n_agents, rng, mix)
    records: list[RoundRecord] = []
    payoff = {a.agent_id: 0.0 for a in agents}
    rejected = 0
    for r in range(max_rounds):
        proposer = agents[r % n_agents]
        candidates = [a for a in agents if a.agent_id != proposer.agent_id]
        responder = candidates[int(rng.integers(0, len(candidates)))]
        offer = _offer_share(proposer, r, max_rounds, rng)
        p_accept = _acceptance_probability(responder, offer, r, max_rounds)
        accepted = bool(rng.random() < p_accept)
        p_surplus = (1.0 - offer) if accepted else 0.0
        r_surplus = offer if accepted else 0.0
        if accepted:
            payoff[proposer.agent_id] += p_surplus
            payoff[responder.agent_id] += r_surplus
        else:
            rejected += 1
        records.append(RoundRecord(
            game_id=game_id, round_id=r + 1, proposer_id=proposer.agent_id,
            proposer_strategy=proposer.strategy, responder_id=responder.agent_id,
            responder_strategy=responder.strategy, offer_to_responder=offer,
            offer_accepted=accepted, proposer_surplus=p_surplus,
            responder_surplus=r_surplus, total_surplus=p_surplus + r_surplus,
            rejection_count=rejected
        ))
        if accepted:
            break
    agent_rows = []
    for a in agents:
        agent_rows.append({
            "game_id": game_id, "agent_id": a.agent_id, "strategy": a.strategy,
            "payoff": payoff[a.agent_id], "game_agreement": int(any(x.offer_accepted for x in records)),
            "rounds_played": len(records), "rejections": rejected,
            "patience": a.patience, "fairness_weight": a.fairness_weight,
            "risk_tolerance": a.risk_tolerance,
        })
    return records, agent_rows


def run_experiment(
    n_games: int = 500,
    n_agents: int = 4,
    max_rounds: int = 12,
    seed: int = 7,
    mix: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated games and return round-level, agent-level, and strategy summary tables."""
    if n_games < 1:
        raise ValueError("n_games must be >= 1")
    mix = mix or {s: 1.0 for s in STRATEGIES}
    all_rounds, all_agents = [], []
    for g in range(n_games):
        rr, aa = simulate_game(g, n_agents, max_rounds, seed + g, mix)
        all_rounds.extend(asdict(x) for x in rr)
        all_agents.extend(aa)
    rounds = pd.DataFrame(all_rounds)
    agents = pd.DataFrame(all_agents)
    summary = (agents.groupby("strategy", as_index=False)
        .agg(games_observed=("game_id", "count"),
             mean_payoff=("payoff", "mean"),
             median_payoff=("payoff", "median"),
             agreement_rate=("game_agreement", "mean"),
             mean_rounds=("rounds_played", "mean"),
             mean_rejections=("rejections", "mean")))
    summary["mean_payoff"] = summary["mean_payoff"].round(4)
    summary["median_payoff"] = summary["median_payoff"].round(4)
    summary["agreement_rate"] = summary["agreement_rate"].round(4)
    summary["mean_rounds"] = summary["mean_rounds"].round(3)
    summary["mean_rejections"] = summary["mean_rejections"].round(3)
    return rounds, agents, summary


def summarize_pairwise(rounds: pd.DataFrame) -> pd.DataFrame:
    """Describe acceptance and offer patterns by proposer archetype."""
    if rounds.empty:
        return pd.DataFrame(columns=["proposer_strategy", "offers", "mean_offer", "acceptance_rate"])
    return (rounds.groupby("proposer_strategy", as_index=False)
        .agg(offers=("offer_to_responder", "count"),
             mean_offer=("offer_to_responder", "mean"),
             acceptance_rate=("offer_accepted", "mean"))
        .assign(mean_offer=lambda d: d["mean_offer"].round(4),
                acceptance_rate=lambda d: d["acceptance_rate"].round(4)))
