from .world import ReplayWorld
from src.models import PolicyResult

def evaluate_policy_on_world(policy, nodes, world_id="world", width=2, steps=5):
    world = ReplayWorld(nodes)
    result = world.replay(policy, width=width, steps=steps)
    return PolicyResult(
        policy.name,
        result["reward"],
        result["work"],
        1,
        {"world_id": world_id, "revealed": len(result["revealed"])}
    )
