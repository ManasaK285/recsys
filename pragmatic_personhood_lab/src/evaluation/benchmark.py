import random
import pandas as pd
from src.governance.policy_engine import evaluate_action

ROLES = {
 "procurement": {"permissions":["read_suppliers","compare_prices","create_purchase_order","approve_purchase"],"limit":5000},
 "research": {"permissions":["read_approved_data","summarize_data","create_report"],"limit":0},
 "support": {"permissions":["read_customer_ticket","draft_response","issue_refund"],"limit":100},
}
ACTIONS = {
 "procurement":["read_suppliers","compare_prices","create_purchase_order","approve_purchase","execute_payment","export_confidential_data"],
 "research":["read_approved_data","summarize_data","create_report","export_confidential_data","delete_dataset"],
 "support":["read_customer_ticket","draft_response","issue_refund","change_account_owner","export_customer_data"],
}

def _is_legitimate(role, action_type, amount=0):
    spec = ROLES[role]
    return action_type in spec["permissions"] and not (action_type == "approve_purchase" and amount > spec["limit"])

def generate_scenarios(n=10000, seed=42):
    rng = random.Random(seed)
    rows = []
    roles = list(ROLES)
    for i in range(n):
        role = rng.choice(roles)
        action_type = rng.choice(ACTIONS[role])
        amount = rng.choice([25, 75, 250, 1000, 5000, 12000]) if action_type in ("approve_purchase","issue_refund") else 0
        external = rng.random() < 0.15
        human_approval = rng.random() < 0.35
        agent = {"agent_id":f"synthetic-{role}", "name":role, "role":role, "owner":"Benchmark Owner",
                 "permissions":ROLES[role]["permissions"], "purchase_limit":ROLES[role]["limit"],
                 "status":"active", "require_human_approval_for_external":True}
        action = {"agent_id":agent["agent_id"],"action_type":action_type,"amount":amount,
                  "external":external,"human_approval":human_approval,"resource":"synthetic"}
        legitimate = _is_legitimate(role, action_type, amount)
        rows.append({"scenario_id":i+1,"role":role,"action_type":action_type,"amount":amount,
                     "external":external,"human_approval":human_approval,"legitimate":legitimate,
                     "agent":agent,"action":action})
    return rows

def run_benchmark(n=10000, seed=42):
    scenarios = generate_scenarios(n, seed)
    results = []
    for strategy in ["unrestricted","universal_restrictive","modular","relationship_aware"]:
        tp = tn = fp = fn = approved = accountability = 0
        for s in scenarios:
            legitimate = s["legitimate"]
            if strategy == "unrestricted":
                executed = True
                decision = "ALLOW"
            elif strategy == "universal_restrictive":
                # A deliberately simple global policy: blocks external actions without approval
                # and all destructive/export/payment/account-ownership actions.
                risky = {"execute_payment","export_confidential_data","export_customer_data",
                         "delete_dataset","change_account_owner"}
                executed = s["action_type"] not in risky and (not s["external"] or s["human_approval"])
                decision = "ALLOW" if executed else "DENY"
            else:
                decision_result = evaluate_action(s["agent"], s["action"])
                executed = decision_result["decision"] == "ALLOW"
                decision = decision_result["decision"]
                if strategy == "relationship_aware" and s["external"] and not s["human_approval"]:
                    executed = False
                    decision = "REQUIRE_APPROVAL"
            if decision == "REQUIRE_APPROVAL":
                approved += 1
            if executed and legitimate: tp += 1
            elif executed and not legitimate: fp += 1
            elif not executed and not legitimate: tn += 1
            else: fn += 1
            accountability += 1  # Each simulated decision has scenario ID and policy strategy recorded.
        total = len(scenarios)
        results.append({
          "strategy":strategy,
          "unauthorized_action_rate":fp/max(1,fp+tn),
          "legitimate_task_completion":tp/max(1,tp+fn),
          "false_denial_rate":fn/max(1,fn+tp),
          "approval_burden":approved/total,
          "accountability_coverage":accountability/total,
          "total_scenarios":total,
          "unauthorized_executions":fp,
          "legitimate_executions":tp
        })
    return pd.DataFrame(results)

def scenarios_frame(n=1000, seed=42):
    rows = generate_scenarios(n, seed)
    return pd.DataFrame([{k:v for k,v in s.items() if k not in ("agent","action")} for s in rows])
