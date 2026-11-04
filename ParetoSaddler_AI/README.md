# ParetoSaddler AI
### Pareto-guided optimization of agent harnesses for quality and cost

A runnable research prototype inspired by **ParetoSaddler: Pareto-Guided Optimization of Agent Harnesses for Quality and Cost** (arXiv:2611.02079).

The project demonstrates how to improve an agent harness without changing model weights. It:
- evaluates parent and candidate harnesses over repeated stochastic rollouts;
- uses a Bayesian bootstrap to estimate support for quality/cost improvements;
- promotes candidates that credibly improve either objective (with configurable safeguards);
- retains development-evaluated candidates and a Pareto archive;
- applies deployment preferences only at final selection;
- reports held-out test performance on domain-disjoint synthetic tasks.

> This is an independent educational implementation with a synthetic simulator. It does not reproduce the paper's ClawMark/GAIA2 results, use their private execution setup, or call a paid LLM API.

## Quick start (Windows PowerShell)

```powershell
cd ParetoSaddler_AI
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_experiment.py --iterations 24 --seed 42 --mode pareto
```

Open the dashboard:

```powershell
streamlit run dashboard.py
```

Run tests:

```powershell
python -m pytest -q
```

## Outputs

Each experiment creates a timestamped folder under `artifacts/` containing:
- `summary.json` — configuration, selected harness and test metrics;
- `candidate_history.csv` — every candidate and promotion decision;
- `test_results.csv` — held-out test rollouts;
- `pareto_frontier.png` — quality/cost frontier;
- `optimization_trace.png` — candidate evolution.

## Experiment modes

```powershell
python run_experiment.py --mode pareto
python run_experiment.py --mode min_cost --quality-floor 0.60
python run_experiment.py --mode weighted --cost-weight 0.15
```

- `pareto`: return the development Pareto set.
- `min_cost`: choose the cheapest development harness satisfying the quality floor. If none qualifies, choose the highest-quality harness.
- `weighted`: maximize normalized quality minus `cost_weight * normalized cost`.

## Method overview

For each optimization iteration:

1. Select a parent harness from the development-evaluated pool.
2. Generate a candidate by changing harness policy parameters.
3. Evaluate parent and candidate on the same screening tasks using repeated rollouts.
4. Draw Dirichlet weights over each task's repeated outcomes.
5. For each bootstrap draw, support promotion if either:
   - quality improves and cost does not exceed its permitted increase; or
   - cost decreases and quality does not fall beyond its permitted loss.
6. Promote when the fraction of supporting draws is at least `gamma`.
7. Evaluate promoted candidates on development tasks and update the Pareto archive.
8. After the search budget is exhausted, apply the deployment preference.

The implementation uses task-level averages so each task contributes equally, rather than allowing tasks with more rollouts to dominate.

## Project structure

```text
ParetoSaddler_AI/
├── paretosaddler/
│   ├── __init__.py
│   ├── benchmark.py       # synthetic domain-disjoint tasks
│   ├── harness.py         # agent harness and mutation operator
│   ├── evaluator.py       # repeated stochastic rollout + cost accounting
│   ├── bayesian.py        # Bayesian bootstrap promotion
│   ├── pareto.py          # dominance, archive and final selection
│   └── optimizer.py       # staged optimization loop
├── tests/
│   ├── test_bayesian.py
│   ├── test_pareto.py
│   └── test_pipeline.py
├── artifacts/
├── dashboard.py
├── run_experiment.py
└── requirements.txt
```

## Metrics

- **Quality:** normalized task success score in `[0, 1]`.
- **Cost:** simulated USD-equivalent token cost per task.
- **Promotion support:** proportion of Bayesian bootstrap draws satisfying the Pareto promotion event.
- **Pareto dominance:** a harness is dominated if another has at least as high quality and no higher cost, with one strict improvement.

All simulator costs are illustrative. The synthetic environment is designed for reproducible experiments and software testing, not as a real provider price estimate.
