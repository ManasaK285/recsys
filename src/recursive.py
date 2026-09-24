from src.orchestrator import run_discovery
from src.dreaming import dream
from src.policy.base import GreedyPolicy

def recursive_improvement(task_id="topk", cycles=3):
    policy = GreedyPolicy()
    history = []
    for cycle in range(cycles):
        run_id, tree = run_discovery(task_id, policy=policy, width=2, cycles=3)
        results = dream(list(tree.nodes.values()))
        best_policy, best_result = results[0]
        history.append({
            "cycle": cycle,
            "run_id": run_id,
            "policy_before": policy.name,
            "policy_after": best_policy.name,
            "replay_reward": best_result.reward,
            "replay_work": best_result.work,
        })
        policy = best_policy
    return history
