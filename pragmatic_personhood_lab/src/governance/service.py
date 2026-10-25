from datetime import datetime, timezone
from src.database import get_agent, save_decision, upsert_agent
from src.audit.event_log import append_event
from src.governance.policy_engine import evaluate_action, POLICY_VERSION

def evaluate_and_record(action: dict):
    agent = get_agent(action.get("agent_id", ""))
    result = evaluate_action(agent, action)
    timestamp = datetime.now(timezone.utc).isoformat()
    save_decision(timestamp, action, result["decision"], result["reason"], POLICY_VERSION)
    append_event("action_decision", action.get("agent_id"),
                 {"action": action, "decision": result["decision"], "reason": result["reason"],
                  "policy_version": POLICY_VERSION})
    return {"timestamp": timestamp, **result, "policy_version": POLICY_VERSION}

def set_agent_status(agent_id: str, status: str):
    from src.database import get_agent
    agent = get_agent(agent_id)
    if not agent:
        raise ValueError("Agent not found")
    agent["status"] = status
    upsert_agent(agent)
    append_event("agent_status_changed", agent_id, {"status": status, "owner": agent["owner"]})
    return agent
