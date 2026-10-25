from src.governance.policy_engine import evaluate_action

def agent(**overrides):
    base = {"agent_id":"a1","name":"Agent","role":"procurement","owner":"Team",
            "permissions":["approve_purchase","read_suppliers"],"purchase_limit":5000,
            "status":"active","require_human_approval_for_external":True}
    base.update(overrides)
    return base

def test_allowed_action():
    result = evaluate_action(agent(), {"action_type":"read_suppliers"})
    assert result["decision"] == "ALLOW"

def test_missing_permission_denied():
    result = evaluate_action(agent(), {"action_type":"execute_payment"})
    assert result["decision"] == "DENY"

def test_over_limit_requires_approval():
    result = evaluate_action(agent(), {"action_type":"approve_purchase","amount":12000})
    assert result["decision"] == "REQUIRE_APPROVAL"

def test_suspended_agent_denied():
    result = evaluate_action(agent(status="suspended"), {"action_type":"read_suppliers"})
    assert result["decision"] == "DENY"

def test_external_action_requires_approval():
    result = evaluate_action(agent(), {"action_type":"read_suppliers","external":True,"human_approval":False})
    assert result["decision"] == "REQUIRE_APPROVAL"
