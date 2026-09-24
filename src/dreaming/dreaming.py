from datetime import datetime, timezone
import uuid
from src.db import insert_policy, insert_policy_eval
from src.policy.developer import PolicyDeveloper
from src.replay.simulator import evaluate_policy_on_world

def now():
    return datetime.now(timezone.utc).isoformat()

def dream(nodes):
    results = []
    for policy in PolicyDeveloper().candidates():
        result = evaluate_policy_on_world(policy, nodes, world_id="current-history")
        policy_id = str(uuid.uuid4())
        insert_policy(policy_id, policy.name, repr(policy), now())
        insert_policy_eval(
            policy_id, "current-history", result.reward, result.work,
            result.details, now()
        )
        results.append((policy, result))
    results.sort(key=lambda x: x[1].reward, reverse=True)
    return results
