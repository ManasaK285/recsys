from src.tasks.registry import make_registry
from src.evaluator.evaluator import evaluate_candidate

def test_topk_candidates():
    task = make_registry()["topk"]
    for factory in task.candidate_factories.values():
        result = evaluate_candidate(task, factory())
        assert result.tests_total == 5
