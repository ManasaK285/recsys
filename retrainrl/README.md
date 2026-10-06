# ReTrainRL — Adaptive Retraining of Recommender Systems via Reinforcement Learning

> **An RL-based research prototype for learning when and how to retrain recommendation models under preference drift and limited retraining budgets.**

Recommender systems become stale as user preferences, item popularity, and interaction distributions evolve. Retraining continuously can improve freshness, but unrestricted retraining is computationally expensive.

**ReTrainRL** formulates retraining as a sequential decision problem. At each deployment period, an RL agent observes the current recommendation quality, distribution drift, model age, catalog statistics, and remaining retraining budget, then chooses whether and how to update the recommender.

The project is inspired by the paper:

**“Adaptive Retraining of Recommender Systems via Reinforcement Learning”**
Diego Russo, Valerio La Gatta, Claudio Spasiano, Vincenzo Moscato

This repository is an **independent research-inspired implementation**, not the authors' official implementation.

---

## 🎯 Research Question

> Can a reinforcement-learning policy learn **when and how much to retrain** a recommender system under distribution drift and a constrained compute budget?

Instead of using a fixed schedule such as:

```text
Retrain every N periods
```

ReTrainRL treats retraining as a decision:

```text
                  Interaction Stream
                         │
                         ▼
                  Drift Monitoring
                         │
                         ▼
                Current Model State
                         │
                         ▼
                    PPO Agent
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        SKIP         FINE-TUNE      FULL RETRAIN
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Updated Recommender
                         │
                         ▼
                 Recommendation Quality
```

---

## 🧠 Core Idea

At every deployment period, the environment provides the RL agent with a state containing signals such as:

* Recall@K
* NDCG@K
* Catalog coverage
* Recommendation diversity
* Quality change
* Jensen-Shannon distribution drift
* Data volume
* Catalog size
* Model age
* Remaining retraining budget
* Previous retraining action

The agent chooses one of four actions:

| Action          | Description                        | Relative Cost |
| --------------- | ---------------------------------- | ------------: |
| `SKIP`          | Keep the current model             |             0 |
| `FINE-TUNE`     | Update using recent interactions   |             1 |
| `SAMPLE UPDATE` | Retrain using a sampled subset     |             2 |
| `FULL RETRAIN`  | Retrain using the complete history |             4 |

The objective combines recommendation quality with retraining cost and model staleness.

---

## 🏗️ Architecture

```text
MovieLens-1M
     │
     ▼
Preprocessing
     │
     ▼
Temporal Interaction Stream
     │
     ├── Initial Historical Data
     │
     └── Deployment Periods
             │
             ▼
       Drift Simulation
             │
             ▼
    ┌────────────────────┐
    │ Retraining         │
    │ Environment        │
    │                    │
    │ State → Action     │
    │        ↓           │
    │ Reward ← Model     │
    └─────────┬──────────┘
              │
              ▼
          PPO / RTagent
              │
              ▼
       Policy Evaluation
              │
       ┌──────┴───────┐
       ▼              ▼
   Baselines       Learned Policy
       │              │
       └──────┬───────┘
              ▼
        Metric Comparison
```

---

## 📊 Dataset

The default experiment uses **MovieLens-1M** from GroupLens.

The current CPU-friendly configuration uses:

* 1,500 users
* 3,544 items
* Minimum 10 interactions per user
* 45% initial historical data
* 12 deployment periods

The deployment stream introduces controlled distribution drift by modifying a small fraction of interactions toward historically less-frequent items.

> **Important:** The drift mechanism is a controlled synthetic simulation designed for experimentation. It is not an exact reproduction of the paper's dataset protocol.

---

## 🤖 Recommendation Models

The project provides a common recommender interface supporting:

### Matrix Factorization

A lightweight user/item embedding model with:

* User embeddings
* Item embeddings
* User bias
* Item bias
* Dot-product scoring

### NeuMF

A Neural Matrix Factorization implementation combining:

* Generalized Matrix Factorization
* MLP interaction modeling

The default CPU experiment uses **Matrix Factorization** for faster iteration.

---

## 🔄 Retraining Strategies

The environment supports multiple retraining mechanisms.

### 1. Skip

No model update.

```text
Cost = 0
```

### 2. Fine-Tune

Update the existing model using recent interactions.

```text
Cost = 1
```

### 3. Sample Update

Update using a subset of historical and recent interactions.

```text
Cost = 2
```

### 4. Full Retraining

Train from the complete available interaction history.

```text
Cost = 4
```

A global retraining budget constrains the agent's decisions across the deployment horizon.

---

## 🧪 Baselines

ReTrainRL compares the learned policy against several scheduling strategies:

* **Never retrain**
* **Periodic fine-tuning**
* **Periodic sample update**
* **Random budget allocation**
* **Drift-triggered retraining**
* **PPO RTagent**

Evaluation includes:

* Recall@10
* NDCG@10
* Catalog coverage
* Diversity
* Total retraining cost
* Total reward

---

## 📈 Current Experimental Results

The current CPU experiment successfully completed the entire pipeline.

### Strategy Comparison

| Strategy           | Mean Recall@10 | Mean NDCG@10 |    Coverage |   Diversity |  Cost |
| ------------------ | -------------: | -----------: | ----------: | ----------: | ----: |
| Never              |        0.00439 |      0.00254 |     0.12728 |     0.51088 |     0 |
| Drift Trigger      |        0.00439 |      0.00254 |     0.12728 |     0.51088 |     0 |
| Periodic Fine-Tune |        0.00577 |  **0.00300** |     0.07271 |     0.48510 |     4 |
| Periodic Sample    |        0.00462 |      0.00265 |     0.11040 | **0.52440** |     2 |
| Random             |    **0.00653** |      0.00285 |     0.09862 |     0.52142 |     4 |
| PPO RTagent        |        0.00439 |      0.00254 | **0.12728** |     0.51088 | **0** |

### Learned Policy

In the current CPU experiment, PPO selected:

```text
Period     Action
-----------------
1          SKIP
2          SKIP
3          SKIP
4          SKIP
5          SKIP
6          SKIP
7          SKIP
8          SKIP
9          SKIP
10         SKIP
11         SKIP
12         SKIP
```

This means the current PPO policy converged to the **zero-cost / never-retrain strategy**.

Therefore, the current results **do not demonstrate that PPO outperforms the retraining baselines**.

This is an intentional limitation of the current research prototype rather than a claim of reproduction of the paper's reported results.

---

## 🔍 What the Experiment Demonstrates

Even though the current RTagent does not yet learn a useful retraining schedule, the implementation demonstrates the complete research workflow:

1. Construct a temporal recommendation environment.
2. Simulate evolving interaction distributions.
3. Monitor recommendation quality and distribution drift.
4. Define retraining actions with different computational costs.
5. Constrain decisions using a global retraining budget.
6. Train an RL policy with PPO.
7. Compare learned decisions against fixed and heuristic schedules.
8. Analyze the resulting policy and deployment trajectory.

The experiment also highlights an important research issue:

> **Reward design strongly determines whether an RL retraining agent considers model updates worthwhile.**

The current policy's preference for `SKIP` indicates that the reward/cost trade-off needs further calibration before meaningful adaptive retraining behavior emerges.

---

## 📁 Project Structure

```text
retrainrl/
│
├── README.md
├── requirements.txt
├── config.yaml
├── run.py
│
├── app/
│   └── app.py
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── stream.py
│   │
│   ├── models/
│   │   ├── mf.py
│   │   ├── neumf.py
│   │   └── factory.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── monitoring/
│   │   └── drift.py
│   │
│   ├── retraining/
│   │   └── strategies.py
│   │
│   ├── environment/
│   │   └── retraining_env.py
│   │
│   ├── baselines/
│   │   └── schedules.py
│   │
│   ├── agents/
│   │   └── train.py
│   │
│   ├── experiments/
│   │   ├── train_ppo.py
│   │   ├── evaluate.py
│   │   └── policy_analysis.py
│   │
│   └── utils/
│       └── config.py
│
├── tests/
│   ├── test_metrics.py
│   ├── test_drift.py
│   └── test_schedule.py
│
├── data/
│   ├── raw/
│   └── processed/
│
└── artifacts/
    ├── checkpoints/
    ├── evaluation/
    └── figures/
```

---

## 🚀 Quick Start

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Run the complete experiment

```powershell
python run.py
```

The pipeline performs:

```text
Download
   ↓
Preprocess
   ↓
Temporal Stream Construction
   ↓
PPO Training
   ↓
Baseline Evaluation
   ↓
RTagent Evaluation
   ↓
Policy Analysis
```

### 3. Launch the dashboard

```powershell
python -m streamlit run app/app.py
```

The dashboard provides:

* Strategy comparison
* Recall/NDCG plots
* Drift visualization
* Policy actions
* Retraining cost
* Budget usage

---

## ⚙️ Configuration

The main configuration is in `config.yaml`.

Example:

```yaml
environment:
  retraining_budget: 4
  sample_fraction: 0.35
  recent_fraction: 0.20
  drift_strength: 0.75
  top_k: 10
  reward_quality_weight: 1.0
  reward_diversity_weight: 0.15
  reward_cost_weight: 0.12
  reward_staleness_weight: 0.08
```

PPO configuration:

```yaml
rl:
  total_timesteps: 6000
  learning_rate: 0.0003
  n_steps: 256
  batch_size: 64
  gamma: 0.98
  gae_lambda: 0.95
  ent_coef: 0.01
```

---

## 📦 Outputs

After execution, artifacts are written to:

```text
artifacts/
├── checkpoints/
│   └── ppo.zip
│
├── evaluation/
│   ├── results.csv
│   └── period_results.csv
│
└── figures/
    ├── drift_curve.png
    └── policy_actions.png
```

---

## 🧪 Testing

Run:

```powershell
pytest
```

The tests cover:

* Recommendation metrics
* Distribution drift
* Retraining schedules

---

## ⚠️ Current Limitations

This project is a **research prototype**, not a production recommender system.

Current limitations include:

* MovieLens is limited to 1,500 users for the default CPU experiment.
* Distribution drift is synthetically controlled.
* Matrix Factorization is the default model.
* Diversity uses a simple item-ID-based proxy.
* PPO training is computationally expensive on CPU.
* The current reward configuration can lead PPO to favor `SKIP`.
* The current experiment does not reproduce the paper's exact experimental protocol or reported numbers.
* The current RTagent result should therefore be interpreted as an implementation baseline rather than a reproduction claim.

---

## 🔬 Future Work

Natural extensions include:

### Better RL Environment

Precompute model states and transitions to make PPO training substantially faster.

### Reward Calibration

Better balance:

```text
Recommendation Quality
        +
Diversity
        +
Freshness
        -
Retraining Cost
```

so that the agent learns meaningful adaptive schedules.

### Real Temporal Drift

Evaluate on naturally occurring temporal splits rather than synthetic drift.

### More Recommendation Models

Compare:

* MF
* NeuMF
* SVD
* Sequential recommenders
* Transformer-based recommenders

### Stronger Evaluation

Add:

* MRR
* Hit Rate
* Long-term user utility
* Freshness
* Tail exposure
* Retraining latency
* Compute consumption

### Multi-Environment Training

Train one policy across multiple recommender architectures and datasets to investigate whether retraining policies generalize.

---

## 📚 Reference

**Russo, D., La Gatta, V., Spasiano, C., & Moscato, V.**

*Adaptive Retraining of Recommender Systems via Reinforcement Learning.*

This project is an independent implementation inspired by the research direction.

---

