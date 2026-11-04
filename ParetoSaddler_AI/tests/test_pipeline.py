from paretosaddler.optimizer import run_experiment

def test_small_pipeline_runs():
    result = run_experiment(iterations=3, seed=7, repeats=2,
                             bootstrap_draws=100, mode="pareto")
    assert len(result["history"]) == 4
    assert len(result["test_summary"]) >= 1
    assert result["test_results"]["split"].eq("test").all()
