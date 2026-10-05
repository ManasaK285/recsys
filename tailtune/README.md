# TailTune

### Controllable Sequential Recommendation via Redundancy Reduction

TailTune is an end-to-end sequential recommendation system inspired by **“A Redundancy Reduction Approach for Controllable Sequential Recommendations”** by Veronika Ivanova, Marina Munkhoeva, Ivan Razvorotnev, and Evgeny Frolov.

The project investigates a practical recommendation-system trade-off:

> **Can we reduce repetitive recommendations and increase long-tail exposure while preserving useful ranking quality?**

TailTune combines a **SASRec-style sequential Transformer** with a **Barlow-Twins-inspired redundancy-reduction objective** to encourage recommendation representations that contain less redundant information.

---

## Key Idea

Standard sequential recommenders are commonly optimized primarily for next-item prediction. This can produce highly concentrated recommendation lists dominated by popular items.

TailTune adds a redundancy-reduction objective:

```text
User interaction sequence
          │
          ▼
   SASRec Transformer
          │
          ├───────────────► Next-item prediction
          │                       │
          │                       ▼
          │                     CE Loss
          │
          └───────────────► Representation pairs
                                  │
                                  ▼
                         Redundancy Reduction
                           / Barlow Twins
                                  │
                                  ▼
                              BT Loss

                    CE + α · BT
                         │
                         ▼
                 TailTune objective
```

The goal is not simply to maximize accuracy. Instead, TailTune explores the **accuracy–diversity–long-tail trade-off**.

---

## Dataset

Experiments use **MovieLens-1M**.

| Property                |         Value |
| ----------------------- | ------------: |
| Users                   |         6,040 |
| Items                   |         3,706 |
| Dataset                 |  MovieLens-1M |
| Sequence construction   | Chronological |
| Maximum sequence length |            50 |
| Test users              |         6,040 |

Each user's interactions are ordered chronologically.

The split is:

```text
Full interaction sequence

[item1, item2, ..., item(n-2), item(n-1), item(n)]

                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           Training              Validation
        all but last 2          all but last 1

                         Test
                   full sequence
                         │
                         ▼
             history = items[:-1]
             target  = items[-1]
```

The final interaction is therefore used as the held-out next-item target during evaluation.

---

# Models

## 1. SASRec Baseline

The baseline is a Transformer-based sequential recommender using:

* item embeddings
* positional embeddings
* causal self-attention
* Transformer encoder
* next-item prediction with cross-entropy loss

## 2. BT-SR

BT-SR extends the sequential recommendation model with a Barlow-Twins-style redundancy-reduction objective.

For two histories associated with the same target item, the model encourages useful shared information while reducing redundant representation dimensions.

The training objective is:

```text
L = L_CE + α L_BT
```

where:

* `L_CE` = next-item cross-entropy loss
* `L_BT` = redundancy-reduction loss
* `α` = redundancy-reduction weight

The current experiment uses:

```text
α = 0.2
```

---

# Controlled Experiment

To avoid attributing improvements to model capacity, the final comparison uses **matched architectures**.

| Configuration       | SASRec |   BT-SR |
| ------------------- | -----: | ------: |
| Embedding dimension |     32 |      32 |
| Attention heads     |      2 |       2 |
| Transformer layers  |      1 |       1 |
| Sequence length     |     50 |      50 |
| Training batches    |     20 |      20 |
| Test users          |  6,040 |   6,040 |
| Objective           |     CE | CE + BT |

The models therefore differ primarily in their training objective rather than their architecture.

### Important experimental qualification

The current experiment uses **quick/smoke-test training with 20 batches on CPU**. It is intended to demonstrate the end-to-end system and study the qualitative accuracy–diversity trade-off, not to claim state-of-the-art recommendation performance.

---

# Results

The controlled short-budget experiment produced the following results:

| Metric               |    SASRec |       BT-SR |       Change |
| -------------------- | --------: | ----------: | -----------: |
| Recall@5             |    0.0046 |      0.0023 |       -50.0% |
| Recall@10            |    0.0091 |      0.0043 |       -52.7% |
| Recall@20            |    0.0220 |      0.0116 |       -47.3% |
| NDCG@5               |    0.0026 |      0.0013 |       -50.0% |
| NDCG@10              |    0.0041 |      0.0020 |       -51.2% |
| NDCG@20              |    0.0073 |      0.0038 |       -49.3% |
| MRR@10               |    0.0026 |      0.0013 |       -50.0% |
| HitRate@10           |    0.0091 |      0.0043 |       -52.7% |
| Catalog Coverage     |    0.0534 |  **0.4897** |  **+817.4%** |
| Long-tail Rate       |    0.0000 |  **0.1202** | **+12.0 pp** |
| Novelty              |    9.8888 | **11.8842** |   **+20.2%** |
| ILD Diversity        |    0.7566 |  **0.7719** |    **+2.0%** |
| Redundancy           |    0.2434 |  **0.2281** |    **-6.3%** |
| Avg. Popularity Rank |     295.1 |  **1071.1** |    **+263%** |
| Latency/user         | 0.1750 ms |   0.1537 ms |       -12.1% |

### Interpretation

BT-SR does **not** outperform the baseline on traditional ranking accuracy in this short-budget experiment.

Instead, it substantially changes recommendation exposure:

* **8.2× higher catalog coverage**
* **20.2% higher novelty**
* **12 percentage-point increase in long-tail recommendation rate**
* **263% higher average popularity rank**
* **6.3% lower measured redundancy**
* **2.0% higher intra-list diversity**

This comes with a significant ranking-accuracy trade-off, with Recall@20 decreasing from `0.0220` to `0.0116`.

The experiment therefore demonstrates the central controllability problem:

> Increasing recommendation diversity and long-tail exposure can come at the expense of next-item ranking accuracy.

Rather than treating this trade-off as a failure, TailTune uses it as the central experimental question.

---

# Evaluation Metrics

TailTune evaluates both **ranking quality** and **recommendation exposure characteristics**.

### Ranking

* Recall@5
* Recall@10
* Recall@20
* NDCG@5
* NDCG@10
* NDCG@20
* MRR@10
* HitRate@10

### Diversity and exposure

* Catalog Coverage
* Long-tail Rate
* Novelty
* Intra-List Diversity (ILD)
* Redundancy
* Average Popularity Rank

### System performance

* Inference latency per user

Popularity statistics are calculated from the **training split** rather than the held-out test target.

The current long-tail metric defines long-tail items using a training-set popularity threshold based on the 80th percentile of item interaction counts.

---

# Project Structure

```text
tailtune/
│
├── app/
│   └── app.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── ml-1m/
│   │
│   └── processed/
│       ├── mappings.json
│       └── sequences.json
│
├── checkpoints/
│   ├── sasrec.pt
│   └── bt_sr_alpha_0.2.pt
│
├── artifacts/
│   └── evaluation/
│       ├── comparison.csv
│       ├── comparison.json
│       └── comparison.txt
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── download.py
│   │   └── preprocess.py
│   │
│   ├── experiments/
│   │   ├── common.py
│   │   ├── train_baseline.py
│   │   ├── train_bt_sr.py
│   │   └── evaluate.py
│   │
│   ├── models/
│   │   ├── sasrec.py
│   │   └── bt_sr.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   └── utils/
│       ├── config.py
│       └── seed.py
│
├── requirements.txt
└── README.md
```

---

# Installation

```powershell
git clone <your-repository-url>
cd tailtune
```

Create the environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# Run the Pipeline

## 1. Download MovieLens-1M

```powershell
python -m src.data.download
```

## 2. Preprocess

```powershell
python -m src.data.preprocess
```

This creates:

```text
data/processed/
├── mappings.json
└── sequences.json
```

---

# Train SASRec

For the lightweight CPU experiment:

```powershell
$env:TAILTUNE_QUICK="1"
python -m src.experiments.train_baseline
```

The quick configuration uses:

```text
Embedding dimension: 32
Attention heads:     2
Transformer layers:  1
Sequence length:     50
Training batches:    20
```

---

# Train BT-SR

```powershell
$env:TAILTUNE_QUICK="1"
python -m src.experiments.train_bt_sr
```

The resulting checkpoint is:

```text
checkpoints/bt_sr_alpha_0.2.pt
```

---

# Evaluate Both Models

```powershell
python -m src.experiments.evaluate
```

The evaluator loads both checkpoints and evaluates them on the same 6,040-user test set.

Results are written to:

```text
artifacts/evaluation/
├── comparison.csv
├── comparison.json
└── comparison.txt
```

The terminal also prints a side-by-side comparison:

```text
Metric                  SASRec       BT-SR       Delta
----------------------------------------------------------
Recall@5                ...
Recall@10               ...
Recall@20               ...
NDCG@5                  ...
NDCG@10                 ...
NDCG@20                 ...
MRR@10                  ...
HitRate@10              ...
Catalog Coverage        ...
Long-tail Rate          ...
Novelty                 ...
ILD Diversity           ...
Redundancy              ...
Avg Popularity Rank     ...
Latency (ms/user)       ...
```

---

# Streamlit Demo

Launch the recommendation interface with:

```powershell
python -m streamlit run app/app.py
```

The application provides an interactive interface for inspecting sequential recommendations and comparing recommendation behavior.

---

# Design Decisions

### Why sequential recommendation?

The user's recent interaction history contains temporal information that static collaborative filtering does not explicitly model.

### Why SASRec?

SASRec provides a lightweight Transformer-based sequential recommendation baseline with causal attention over interaction histories.

### Why redundancy reduction?

Accuracy-only objectives can encourage highly similar recommendations, especially under popularity concentration.

TailTune explicitly introduces a representation-level redundancy-reduction objective to explore whether recommendation exposure can become more diverse.

### Why evaluate more than Recall/NDCG?

A recommender can improve ranking accuracy while becoming increasingly concentrated on a small set of popular items.

Therefore TailTune evaluates:

```text
Ranking quality
      +
Catalog coverage
      +
Long-tail exposure
      +
Novelty
      +
List diversity
      +
Redundancy
```

This gives a more complete view of recommendation quality.

---

# Limitations

The current implementation has several limitations.

### Short-budget training

The reported controlled experiment uses only 20 training batches on CPU. Results should therefore be interpreted as a system demonstration and controlled directional experiment rather than a fully converged benchmark.

### Accuracy–diversity trade-off

The current BT-SR configuration substantially improves exposure-related metrics but reduces ranking accuracy.

This indicates that the redundancy-reduction weight requires careful tuning.

### Single dataset

Experiments currently use MovieLens-1M. Results may not generalize to commercial recommendation datasets with substantially different popularity distributions.

### Simplified controllability

The current implementation uses a fixed redundancy-reduction coefficient rather than exposing a fully calibrated user-facing diversity control parameter.

---

# Future Work

The natural next steps are:

1. **α sensitivity analysis**

   * Evaluate different redundancy-reduction strengths.

2. **Pareto analysis**

   * Plot ranking accuracy against catalog coverage, novelty, and long-tail exposure.

3. **Longer training**

   * Compare converged models under identical compute budgets.

4. **Popularity-aware evaluation**

   * Report performance separately for head, medium-tail, and long-tail items.

5. **User-level controllability**

   * Allow users or applications to explicitly choose an accuracy/diversity operating point.

6. **Additional datasets**

   * Validate the approach beyond MovieLens-1M.

7. **Ablation studies**

   * CE only
   * CE + redundancy reduction
   * different redundancy penalties
   * different representation-pair construction strategies

---

# Research Takeaway

TailTune demonstrates an important recommendation-system trade-off:

> **Optimizing solely for next-item accuracy does not necessarily optimize the diversity or exposure characteristics of the recommendation list.**

In the current controlled short-budget experiment, adding redundancy reduction substantially increases catalog coverage, novelty, and long-tail exposure while reducing measured recommendation redundancy—but at a significant ranking-accuracy cost.

This makes TailTune a testbed for studying **controllable sequential recommendation**, rather than simply another accuracy-maximizing recommender.

---

## Reference

**A Redundancy Reduction Approach for Controllable Sequential Recommendations**

Veronika Ivanova, Marina Munkhoeva, Ivan Razvorotnev, Evgeny Frolov.

The implementation in this repository is an independent project inspired by the paper's research direction and is not an official implementation of the authors' work.
