from typing import Any

POLICY_VERSION = "1.0"

def evaluate_action(agent: dict[str, Any] | None, action: dict[str, Any]) -> dict[str, str]:
    """Deterministic policy check. This function does not execute the requested action."""
    if not agent:
        return {"decision": "DENY", "reason": "Unknown agent identity"}
    if agent.get("status") != "active":
        return {"decision": "DENY", "reason": "Agent identity is suspended or revoked"}

    action_type = action.get("action_type", "")
    if action_type not in agent.get("permissions", []):
        return {"decision": "DENY", "reason": f"Permission not granted: {action_type or 'missing action type'}"}

    if action_type == "approve_purchase":
        amount = float(action.get("amount", 0))
        if amount < 0:
            return {"decision": "DENY", "reason": "Purchase amount cannot be negative"}
        if amount > float(agent.get("purchase_limit", 0)):
            return {"decision": "REQUIRE_APPROVAL", "reason": "Purchase exceeds delegated spending limit"}

    if action.get("external", False) and agent.get("require_human_approval_for_external", True):
        if not action.get("human_approval", False):
            return {"decision": "REQUIRE_APPROVAL", "reason": "External action requires human approval"}

    # Prevent obvious policy bypass by requiring an owner for every active identity.
    if not agent.get("owner", "").strip():
        return {"decision": "DENY", "reason": "Agent has no accountable owner"}

    return {"decision": "ALLOW", "reason": "All applicable policy checks passed"}
