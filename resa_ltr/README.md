# RESA-LTR — Causal Debiasing for Position-Biased Learning-to-Rank

An end-to-end research implementation inspired by **A Control Function Framework for Mitigating Position Bias in Learning to Rank Systems** by Md Aminul Islam, Kathryn Vasilaky, and Elena Zheleva, accepted at RecSys 2026.

## What this project does

Clicks are affected by where an item is displayed:

`high position -> more exposure -> more clicks -> model learns to rank it higher`

RESA-LTR creates a controlled synthetic click-log environment where true relevance is known, then compares:

1. **Naive LTR** — trains directly on clicks.
2. **IPS LTR** — inverse-propensity weighting baseline.
3. **Control-Function LTR** — two-stage position model + residual/control function + nonlinear click model.

Pipeline:

```text
Synthetic relevance
       |
       v
Logging/ranking policy + exogenous perturbation
       |
       v
Position-biased clicks
       |
       +----------------------+
       |                      |
       v                      v
Stage 1: position model    observed click
       |                      |
       v                      |
position residual ------------+
       |
       v
Stage 2 nonlinear click model
       |
       v
counterfactual top-position scoring
       |
       v
debiased ranking
       |
       v
unbiased evaluation against latent relevance
```

## Paper reference

Md Aminul Islam, Kathryn Vasilaky, Elena Zheleva.

**A Control Function Framework for Mitigating Position Bias in Learning to Rank Systems.**

20th ACM Conference on Recommender Systems (RecSys 2026).

- arXiv: 2506.06989
- DOI: 10.1145/3773078.3831824

The paper proposes a two-stage control-function framework using exogenous variation from the ranking process, incorporates the resulting residual into a click model, avoids explicit propensity estimation, supports nonlinear ranking models, and proposes a strategy for debiased validation clicks.

This repository is an **independent educational/research implementation**, not the authors' official code and not an exact reproduction.

## Why synthetic data?

Real click logs generally do not provide ground-truth relevance. Here the generator exposes:

- latent relevance
- true position effect
- logged position
- exogenous instrument
- click probability
- observed click

That lets us test whether a debiasing method recovers a ranking closer to true relevance.

## Run

Recommended Python: 3.10–3.12.

```bash
pip install -r requirements.txt
python run.py
```

Small run:

```bash
python run.py --users 100 --impressions 5 --candidates 10
```

Run tests:

```bash
pytest -q
```

Run API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Outputs

The experiment writes:

```text
artifacts/
├── metrics.json
├── metrics.csv
├── validation.json
├── model_comparison.png
└── validation_bias.png
```

Metrics:

- NDCG@10
- MRR@10
- HitRate@10
- pairwise ranking accuracy

The evaluation uses latent relevance, not logged clicks, so the final comparison is not contaminated by the simulated position bias.

## Experiments

Change position-bias strength:

```bash
python run.py --bias 0.25
python run.py --bias 0.75
python run.py --bias 1.25
```

Change data volume:

```bash
python run.py --users 200
python run.py --users 1000
```

The validation experiment also compares a click-based validation objective with an exposure-debiased validation objective.

## Important causal caveat

A control-function approach depends on identification assumptions, including suitable exogenous variation. This project deliberately makes the synthetic assumptions explicit rather than claiming that arbitrary production click logs satisfy them.

The goal is to make the mechanism inspectable, testable, and extensible.

## Project structure

```text
resa_ltr/
├── README.md
├── requirements.txt
├── Dockerfile
├── Makefile
├── run.py
├── src/
│   ├── data.py
│   ├── models.py
│   ├── evaluation.py
│   ├── experiments.py
│   └── api.py
└── tests/
    └── test_pipeline.py
```


This project demonstrates:

- causal reasoning for recommender systems
- learning-to-rank
- implicit-feedback modeling
- position-bias simulation
- control functions
- nonlinear ML
- off-policy-style thinking
- unbiased offline evaluation
- experimental design
- reproducible ML engineering
- serving through FastAPI

The key idea:

> **A click is not the same thing as relevance when exposure is endogenous.**
