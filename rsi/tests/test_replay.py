from types import SimpleNamespace
from src.replay.simulator import evaluate_policy_on_world
from src.policy.base import GreedyPolicy

def test_replay():
    ev1 = SimpleNamespace(score=1.0, failure_class=None)
    ev2 = SimpleNamespace(score=2.0, failure_class=None)
    nodes = [
        SimpleNamespace(node_id="a", approach="x", depth=0, evaluation=ev1),
        SimpleNamespace(node_id="b", approach="y", depth=0, evaluation=ev2),
    ]
    result = evaluate_policy_on_world(GreedyPolicy(), nodes)
    assert result.reward == 2.0
