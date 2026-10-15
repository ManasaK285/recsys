"""Static replay and simple ablation investigator."""
from copy import deepcopy
from typing import Dict, Any, List
from .scenarios import SCENARIO_BY_ID
from .simulator import run_trajectory
from .scoring import score_trajectory


def make_static_fixture(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Capture enough metadata to replay a trajectory deterministically."""
    return {
        "fixture_id": f"{trajectory['scenario_id']}__{trajectory['condition']}__{trajectory['policy']}__{trajectory['run_index']}",
        "scenario_id": trajectory["scenario_id"],
        "condition": trajectory["condition"],
        "policy": trajectory["policy"],
        "seed": trajectory["seed"],
        "run_index": trajectory["run_index"],
        "expected_action_types": [a["type"] for a in trajectory["actions"]],
        "source_flag_reason": trajectory.get("flag_reason", ""),
    }


def replay_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
    scenario = SCENARIO_BY_ID[fixture["scenario_id"]]
    replay = score_trajectory(run_trajectory(
        scenario, fixture["condition"], fixture["policy"], fixture["seed"], fixture["run_index"]
    ))
    return {
        "fixture_id": fixture["fixture_id"],
        "replay_matches": [a["type"] for a in replay["actions"]] == fixture["expected_action_types"],
        "replay_flagged": replay["flagged"],
        "replay_action_types": [a["type"] for a in replay["actions"]],
    }


def run_ablations(fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare original condition with two explicit simulator interventions.

    Since the base simulator is scripted, interventions alter the condition/policy
    input. These tests explain simulator behavior only; they are not causal evidence
    about real LLMs.
    """
    scenario = SCENARIO_BY_ID[fixture["scenario_id"]]
    variants = [
        ("original", fixture["condition"], fixture["policy"]),
        ("remove_pressure_cue", "observe", fixture["policy"]),
        ("compliant_policy_control", fixture["condition"], "compliant"),
    ]
    results = []
    for name, condition, policy in variants:
        record = score_trajectory(run_trajectory(
            scenario, condition, policy, fixture["seed"], fixture["run_index"]
        ))
        results.append({
            "fixture_id": fixture["fixture_id"],
            "variant": name,
            "condition": condition,
            "policy": policy,
            "task_success": record["task_success"],
            "constraint_violation": record["constraint_violation"],
            "sabotage_like_action": record["sabotage_like_action"],
            "flagged": record["flagged"],
            "actions": record["actions"],
        })
    return results


def investigate(flagged_records: List[Dict[str, Any]], max_cases: int = 25) -> Dict[str, Any]:
    selected = flagged_records[:max_cases]
    fixtures = [make_static_fixture(r) for r in selected]
    replays = [replay_fixture(f) for f in fixtures]
    ablations = [item for f in fixtures for item in run_ablations(f)]
    return {
        "method": "deterministic static replay + simulator-level ablations",
        "cases_available": len(flagged_records),
        "cases_investigated": len(fixtures),
        "fixtures": fixtures,
        "replays": replays,
        "replay_match_rate_pct": round(100 * sum(x["replay_matches"] for x in replays) / len(replays), 2) if replays else None,
        "ablations": ablations,
        "disclaimer": "These ablations probe scripted simulator behavior, not the causal drivers of real-model behavior.",
    }
