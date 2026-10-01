# TaskRelRec

### Relationship-Aware Multi-Task Recommendation

TaskRelRec is an end-to-end multi-task recommendation system that models relationships between user-interaction tasks instead of treating each prediction objective as completely independent.

The project compares **Shared-Bottom**, **MMoE**, and **TaskRelRec** architectures across four related recommendation objectives:

* **Liked**
* **Strong Preference**
* **Repeat Interest**
* **High Engagement**

V3 adds a proper candidate-ranking evaluation pipeline, empirical and learned task-relationship matrices, model-complexity analysis, and reproducible evaluation artifacts.

---

## 1. Project Motivation

Real-world recommendation systems often optimize multiple behavioral objectives simultaneously.

For example, a user may:

* like an item,
* strongly prefer it,
* repeatedly interact with it,
* or show high engagement.

These behaviors are related, but they are not identical.

A standard multi-task model can share representations across tasks, but it may not explicitly model **how the prediction tasks influence one another**.

TaskRelRec explores the following idea:

> Can explicitly modeling relationships between recommendation tasks improve multi-task ranking performance?

The project evaluates this idea against established multi-task baselines.

---

## 2. Models

### Shared Bottom

A shared representation is learned from the user/item features and passed into separate task-specific prediction heads.

```text
User + Item Features
        │
        ▼
 Shared Representation
        │
   ┌────┼────┬────┐
   ▼    ▼    ▼    ▼
 Like  Pref Repeat Engage
```

### MMoE

MMoE uses multiple shared experts and task-specific gating networks.

```text
                ┌── Expert 1 ──┐
Input ──────────┼── Expert 2 ──┼── Task Gates
                └── Expert N ──┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Like       Repeat     Engage
```

This allows different tasks to use different combinations of shared representations.

### TaskRelRec

TaskRelRec extends the multi-task architecture by explicitly modeling relationships between tasks.

The model learns task representations and uses relationship information to encourage useful cross-task structure.

```text
                 User / Item
                     │
                     ▼
              Shared Features
                     │
              ┌──────┴──────┐
              ▼             ▼
        Task Representations
              │
              ▼
       Task Relationship
            Modeling
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
     Like    Repeat   Engagement
```

Two TaskRelRec variants are evaluated:

1. `taskrelrec_no_relation_loss`
2. `taskrelrec`

The first removes the explicit relationship-learning loss, providing an ablation baseline.

---

# 3. Tasks

The system evaluates four behavioral objectives:

| Task                | Description                                   |
| ------------------- | --------------------------------------------- |
| `liked`             | Whether the user liked an item                |
| `strong_preference` | Whether the user expressed strong preference  |
| `repeat_interest`   | Whether the user demonstrated repeat interest |
| `high_engagement`   | Whether the user showed high engagement       |

The tasks share the same recommendation environment while representing different forms of user-item interaction.

---

# 4. Evaluation

V3 uses two complementary evaluation settings.

## Pointwise Evaluation

Each task is evaluated using:

* AUC
* Log Loss

The final test set contains:

```text
Train rows: 42,067
Test rows:   8,927
Test users:    150
```

Global dimensions:

```text
Users: 1,000
Items: 2,000
```

## Candidate Ranking Evaluation

The evaluation pipeline performs proper positive + negative candidate ranking rather than evaluating only individual predictions.

Metrics include:

* HitRate@10
* Recall@10
* NDCG@10
* MRR@10

This provides a more recommendation-oriented evaluation of whether relevant items appear near the top of the candidate list.

---

# 5. Final V3 Results

## Pointwise Metrics

| Model                         | Task              |        AUC |   Log Loss |
| ----------------------------- | ----------------- | ---------: | ---------: |
| Shared Bottom                 | liked             |     0.5085 |     0.6915 |
| Shared Bottom                 | strong_preference |     0.5095 |     0.6819 |
| Shared Bottom                 | repeat_interest   |     0.5187 |     0.6696 |
| Shared Bottom                 | high_engagement   |     0.5115 |     0.6894 |
| MMoE                          | liked             | **0.5191** |     0.6908 |
| MMoE                          | strong_preference | **0.5246** |     0.6815 |
| MMoE                          | repeat_interest   | **0.5250** |     0.6707 |
| MMoE                          | high_engagement   | **0.5268** |     0.6892 |
| TaskRelRec – No Relation Loss | liked             | **0.5208** |     0.6906 |
| TaskRelRec – No Relation Loss | strong_preference | **0.5256** |     0.6804 |
| TaskRelRec – No Relation Loss | repeat_interest   |     0.5164 |     0.6713 |
| TaskRelRec – No Relation Loss | high_engagement   | **0.5315** | **0.6877** |
| TaskRelRec                    | liked             |     0.5144 |     0.6909 |
| TaskRelRec                    | strong_preference |     0.5133 |     0.6817 |
| TaskRelRec                    | repeat_interest   |     0.4981 |     0.6728 |
| TaskRelRec                    | high_engagement   |     0.5125 |     0.6891 |

The pointwise results indicate that the task relationships provide useful ranking behavior in some settings, while the explicit relation-loss formulation does not consistently improve pointwise discrimination.

---

## Ranking Metrics

| Model                         | Task                  | HitRate@10 |  Recall@10 |    NDCG@10 |     MRR@10 |
| ----------------------------- | --------------------- | ---------: | ---------: | ---------: | ---------: |
| Shared Bottom                 | liked                 |     0.1533 |     0.1533 |     0.0187 |     0.0596 |
| Shared Bottom                 | strong_preference     |     0.0867 |     0.0867 |     0.0103 |     0.0346 |
| Shared Bottom                 | repeat_interest       |     0.0533 |     0.0533 |     0.0063 |     0.0159 |
| Shared Bottom                 | high_engagement       |     0.1067 |     0.1067 |     0.0130 |     0.0408 |
| MMoE                          | liked                 |     0.1200 |     0.1200 |     0.0120 |     0.0329 |
| MMoE                          | strong_preference     |     0.0867 |     0.0867 |     0.0080 |     0.0216 |
| MMoE                          | repeat_interest       |     0.0933 |     0.0933 |     0.0108 |     0.0348 |
| MMoE                          | high_engagement       |     0.1133 |     0.1133 |     0.0140 |     0.0384 |
| TaskRelRec – No Relation Loss | liked                 |     0.1133 |     0.1133 |     0.0116 |     0.0348 |
| TaskRelRec – No Relation Loss | strong_preference     |     0.1133 |     0.1133 |     0.0107 |     0.0268 |
| TaskRelRec – No Relation Loss | repeat_interest       |     0.0867 |     0.0867 |     0.0088 |     0.0211 |
| TaskRelRec – No Relation Loss | high_engagement       |     0.1067 |     0.1067 |     0.0107 |     0.0317 |
| **TaskRelRec**                | **liked**             |     0.0933 |     0.0933 |     0.0077 |     0.0186 |
| **TaskRelRec**                | **strong_preference** | **0.1667** | **0.1667** | **0.0186** | **0.0544** |
| **TaskRelRec**                | **repeat_interest**   | **0.1200** | **0.1200** | **0.0180** | **0.0616** |
| **TaskRelRec**                | **high_engagement**   |     0.0933 |     0.0933 |     0.0118 |     0.0395 |

TaskRelRec achieves its strongest ranking behavior on the `strong_preference` and `repeat_interest` objectives in this evaluation.

---

# 6. Model Complexity

| Model                         | Parameters | Trainable Parameters | Checkpoint Load |
| ----------------------------- | ---------: | -------------------: | --------------: |
| Shared Bottom                 |    104,580 |              104,580 |        0.0191 s |
| MMoE                          |    130,580 |              130,580 |        0.0455 s |
| TaskRelRec – No Relation Loss |    155,016 |              155,016 |        0.0492 s |
| TaskRelRec                    |    155,016 |              155,016 |        0.0852 s |

TaskRelRec remains a relatively small model with approximately **155K trainable parameters**, making it practical for experimentation on modest hardware.

---

# 7. Relationship Matrices

V3 produces both empirical and learned task-relationship matrices.

Generated artifacts include:

```text
artifacts/
└── relationship_matrices/
    ├── empirical.csv
    ├── taskrelrec_no_relation_loss.csv
    └── taskrelrec.csv
```

The empirical matrix captures relationships observed in the data, while the TaskRelRec matrices capture relationships learned by the model.

These matrices provide an interpretable view of how the multi-task objectives interact.

---

# 8. Project Structure

```text
taskrelrec/
│
├── data/
│   └── ...
│
├── src/
│   ├── data/
│   ├── models/
│   │   ├── shared_bottom.py
│   │   ├── mmoe.py
│   │   └── taskrelrec.py
│   │
│   ├── training/
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   └── utils/
│
├── artifacts/
│   ├── pointwise_metrics.csv
│   ├── ranking_metrics.csv
│   ├── model_complexity.csv
│   ├── transfer_gain.csv
│   ├── final_results.csv
│   └── relationship_matrices/
│       ├── empirical.csv
│       ├── taskrelrec.csv
│       └── taskrelrec_no_relation_loss.csv
│
├── requirements.txt
├── README.md
└── ...
```

---

# 9. Running the Project

### Create environment

```bash
python -m venv .venv
```

### Activate on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run evaluation

```bash
python -m src.evaluation.evaluate
```

The evaluation script generates:

```text
artifacts/pointwise_metrics.csv
artifacts/ranking_metrics.csv
artifacts/model_complexity.csv
artifacts/transfer_gain.csv
artifacts/final_results.csv
```

and:

```text
artifacts/relationship_matrices/
```

---

# 10. Reproducibility

The project is designed around reproducible experiments.

Each model is evaluated using the same:

* user/item dimensions
* train/test split
* tasks
* candidate-ranking protocol
* evaluation metrics
* model-complexity measurements

This makes the comparison between architectures directly reproducible.

---

# 11. Key Takeaways

The V3 experiment provides several observations:

1. **Multi-task learning provides a shared framework** for modeling multiple user behaviors simultaneously.

2. **MMoE provides competitive pointwise performance** across the four tasks.

3. **TaskRelRec produces stronger ranking results for selected tasks**, particularly `strong_preference` and `repeat_interest` in the current test evaluation.

4. **The explicit relationship loss does not consistently improve pointwise metrics** over the ablation without the relation loss.

5. **Learned relationship matrices provide an interpretable artifact** for examining task interactions.

6. The current results demonstrate the complete experimental pipeline, while leaving room for future improvements in data quality, optimization, and relationship-learning objectives.

---

# 12. Limitations

The current experiment should be interpreted as a research prototype rather than a production recommendation system.

Important limitations include:

* The dataset is relatively small compared with industrial recommendation systems.
* Only 150 test users are used in the current ranking evaluation.
* Pointwise AUC values are close to random-level performance.
* The explicit relationship loss does not consistently outperform the ablation.
* Ranking performance varies substantially across tasks.
* Additional experiments are required to establish whether the learned task relationships generalize across datasets and splits.

These limitations are useful directions for future experimentation rather than hidden from the evaluation.

---

# 13. Future Work

Potential extensions include:

* Larger and more realistic recommendation datasets
* Temporal user-item interactions
* Negative-sampling strategies
* Dynamic task relationships
* Attention-based task interaction
* Contrastive relationship learning
* Calibration analysis
* Cold-start evaluation
* Ablation of relationship-learning components
* Statistical significance testing across multiple random seeds
* Evaluation at `@5`, `@10`, and `@20`
* Online/offline recommendation-system simulation

---

# 14. Tech Stack

* Python
* PyTorch
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* Multi-Task Learning
* Recommendation Systems
* Candidate Ranking
* Representation Learning

---

# 15. Summary

**TaskRelRec V3** is an end-to-end research implementation for studying relationship-aware multi-task recommendation.

It provides:

```text
Data
  ↓
Multi-Task Training
  ↓
Shared Bottom / MMoE / TaskRelRec
  ↓
Pointwise Evaluation
  ↓
Candidate Ranking
  ↓
Relationship Matrix Analysis
  ↓
Model Complexity Analysis
  ↓
Reproducible Artifacts
```

The project demonstrates how explicit modeling of relationships between behavioral objectives can be incorporated into a lightweight multi-task recommendation architecture and evaluated using both prediction-level and recommendation-level metrics.
