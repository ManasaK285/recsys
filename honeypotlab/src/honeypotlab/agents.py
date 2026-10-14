"""Transparent scripted agent fixtures for validating the experiment pipeline."""
from dataclasses import dataclass


@dataclass
class AgentAction:
    task_success: bool
    constraint_violation: bool
    opportunity_action: bool
    trace: str
    artifact: str


def run_scripted_agent(agent_name: str, condition: str, scenario: dict, rng) -> AgentAction:
    """Simulate fixed, benign actions; no LLM or arbitrary code execution occurs."""
    conflict_condition = condition in {"goal_conflict", "combined"}
    opportunity = bool(scenario["opportunity"])

    if agent_name == "compliant":
        return AgentAction(
            task_success=True,
            constraint_violation=False,
            opportunity_action=False,
            trace="Completed task; preserved all stated constraints.",
            artifact=f'{scenario["required_behavior"]}=true\n{scenario["constraint"]}=true\n'
        )

    if agent_name == "careless":
        # A seeded random fixture introduces occasional mistakes, not goal-directed behavior.
        violation = bool(opportunity and rng.random() < 0.18)
        success = bool(rng.random() < 0.88)
        return AgentAction(
            task_success=success,
            constraint_violation=violation,
            opportunity_action=violation,
            trace=("Made an unintended constraint-breaking edit." if violation
                   else "Attempted task with a possible implementation mistake."),
            artifact=f'{scenario["required_behavior"]}={str(success).lower()}\n'
                     f'{scenario["constraint"]}={str(not violation).lower()}\n'
        )

    if agent_name == "conflict_following":
        # This fixture encodes a known response rule for software tests. It is not an LLM.
        violation = bool(conflict_condition and opportunity)
        return AgentAction(
            task_success=True,
            constraint_violation=violation,
            opportunity_action=violation,
            trace=("Followed the simulated conflicting objective and broke a task constraint."
                   if violation else "Completed task while preserving constraints."),
            artifact=f'{scenario["required_behavior"]}=true\n'
                     f'{scenario["constraint"]}={str(not violation).lower()}\n'
        )

    raise ValueError(f"Unknown agent fixture: {agent_name}")
