from src.evaluation.benchmark import run_benchmark

def test_benchmark_has_four_strategies():
    result = run_benchmark(100, 42)
    assert set(result["strategy"]) == {"unrestricted","universal_restrictive","modular","relationship_aware"}
    assert len(result) == 4
    for metric in ["unauthorized_action_rate","legitimate_task_completion","false_denial_rate"]:
        assert result[metric].between(0, 1).all()
