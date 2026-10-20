# MemoryShapes: Hybrid Neural–Cognitive Models for Human Reward Learning

An end-to-end, reproducible research project inspired by:

Eckstein, M. K., Summerfield, C., Daw, N. D., & Miller, K. J. (2026).
*Hybrid neural–cognitive models reveal how memory shapes human reward learning.*
Nature Human Behaviour, 10, 972–987.
https://doi.org/10.1038/s41562-025-02324-0

Official paper: https://deepmind.google/research/publications/94006/  
Official implementation: https://github.com/google-deepmind/hybrid_rnns_reward_learning  
Study dataset: https://osf.io/8xz3w/

## Research question

Do flexible representations of past choices and outcomes improve prediction of human reward-guided choices beyond classical reinforcement learning (RL) models that incrementally update scalar action values?

This repository is an independent educational/research implementation inspired by the paper. It is **not** an exact reproduction of the authors' architecture, dataset preprocessing, or reported results. Use the official code and data for a faithful reproduction.

## What is included

- Synthetic four-armed non-stationary bandit task with participant-level variation.
- Classical Q-learning baseline.
- Recency-weighted episodic-memory baseline.
- Neural history encoder (GRU).
- Hybrid model combining Q-values and learned history representations.
- Participant-level train/test split to reduce leakage.
- Cross-entropy/log loss, accuracy, Brier score, and parameter counts.
- Ablation-friendly model registry and saved metrics.
- Streamlit dashboard for results exploration.
- Unit tests and a single command-line training/evaluation pipeline.

## Architecture

1. Simulate or load trial-level observations.
2. Encode each trial's chosen action and reward.
3. Predict the next choice using Q-learning, episodic memory, a GRU, or a hybrid model.
4. Train neural models with cross-entropy on observed choices.
5. Evaluate on held-out participants and compare predictive performance.
6. Inspect results in the dashboard.

## Expected input schema for real data

The project simulator writes `data/processed/synthetic_trials.csv` with columns:

- `participant_id`: participant identifier
- `trial`: trial index within participant
- `choice`: chosen action, integer 0–3
- `reward`: reward observed after the choice
- `reward_0` ... `reward_3`: reward schedule for each action on that trial (optional for fitting, useful for simulation analysis)

For a real dataset, map its columns to this schema and carefully follow the original dataset's license, consent, inclusion criteria, and preprocessing documentation. Do not treat synthetic results as human-subject findings.

## Quickstart (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.run_experiment --participants 120 --trials 150 --epochs 12
streamlit run app.py
pytest -q
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again. Python 3.10–3.12 is recommended.

## Quickstart (macOS/Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.run_experiment --participants 120 --trials 150 --epochs 12
streamlit run app.py
pytest -q
```

## Outputs

The experiment creates:
- `data/processed/synthetic_trials.csv`
- `results/metrics.csv`
- `results/predictions.csv`
- `results/learning_curves.csv`
- `results/summary.json`
- `results/model_comparison.png`

Neural training is stochastic; results can vary slightly by seed and hardware. The small default experiment is intended for a laptop. Increase participants/epochs only after confirming the pipeline works.

## Model descriptions

- **Q-learning**: incrementally updates the selected action's value using a learning rate; choices follow a softmax over action values.
- **Episodic memory**: predicts values using similarity/recency-weighted past outcomes for each action.
- **GRU history model**: processes previous choice/reward events and predicts the next choice.
- **Hybrid Q+GRU**: combines interpretable Q-values with a learned representation of recent history.

The GRU and hybrid models are simplified project models, not the exact Memory-ANN/Context-ANN architecture in the publication.

## Evaluation protocol

- Split by participant, not by randomly splitting individual trials.
- Optimize neural models using training-participant sequences.
- Report held-out choice log loss, accuracy, Brier score, and parameter count.
- Compare models on the same test participants.
- Run multiple seeds and report mean ± standard deviation before making claims.
- Include ablations: remove memory, shuffle history, shorten history, and vary synthetic reward drift.
- Distinguish predicting human-like choices from maximizing task reward.

## Recommended next experiments

1. Run 5 seeds and aggregate test metrics.
2. Compare stationary and drifting reward schedules.
3. Measure performance as a function of time since an outcome.
4. Test whether history shuffling degrades the GRU/hybrid model.
5. Download the public dataset and official implementation for a paper-faithful replication.

## Citation

```bibtex
@article{eckstein2026hybrid,
  title={Hybrid neural-cognitive models reveal how memory shapes human reward learning},
  author={Eckstein, Maria K. and Summerfield, Christopher and Daw, Nathaniel D. and Miller, Kevin J.},
  journal={Nature Human Behaviour},
  volume={10},
  pages={972--987},
  year={2026},
  doi={10.1038/s41562-025-02324-0}
}
```

## Ethics and scope

The included data are synthetic. This is a modeling project, not a clinical or psychological diagnostic tool. Any use of human behavioral data must follow the dataset's terms and applicable ethical requirements.
