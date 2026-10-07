# SCRec: End-to-End Sequential Recommendation

An independent engineering reproduction of the core ideas behind **SCRec**, a generative sequential recommendation approach that incorporates collaborative signals into semantic item tokenization, semantic-guided generation, and manifold alignment.
## Citation

This project is an independent engineering reproduction inspired by:

> Ivanova, V., Munkhoeva, M., Razvorotnev, I., & Frolov, E.
> *A Redundancy Reduction Approach for Controllable Sequential Recommendations.*

The implementation in this repository is not the authors' official implementation and should not be interpreted as an exact reproduction of the reported experimental results.
---

## 1. Overview

Sequential recommendation models predict the next item from a user's interaction history.

SCRec addresses three limitations of semantic-ID-based generative recommendation:

1. Item tokenization can rely too heavily on textual semantics while underusing collaborative behavior.
2. Generation may not sufficiently exploit the original semantic representation of an item.
3. Discrete semantic codebook indices and continuous semantic representations can have a geometric mismatch.

This implementation reproduces these ideas through three components:

### Collaborative-Enhanced Tokenization

Behavioral co-occurrence information is incorporated into item representations before semantic tokenization.

```text
Item text
   +
Collaborative item relationships
   ↓
Fused item representation
   ↓
Semantic codebook
   ↓
Semantic IDs
```

### Semantic-Guided Generation

The sequential Transformer uses learned semantic/code representations to guide next-item generation.

```text
User interaction history
          ↓
Sequential Transformer
          +
Semantic/code representation
          ↓
Next-item semantic code prediction
```

### Manifold Alignment

A geometric alignment objective encourages continuous semantic representations and discrete code representations to remain compatible.

The training objective is:

```text
L = L_generation
    + α L_contrastive
    + β L_manifold
```

---

## 2. Project Structure

```text
SCRec_End_to_End/
│
├── configs/
│   └── default.yaml
│
├── data/
│   └── synthetic/
│       ├── interactions.csv
│       └── items.csv
│
├── outputs/
│   └── best.pt
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
│
├── src/
│   ├── data.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

## 3. Dataset

The project currently supports a small synthetic sequential-recommendation dataset for validating the complete pipeline.

### Interactions

`interactions.csv`

```text
user_idx,item_idx,timestamp
0,117,0
0,273,1
0,204,2
...
```

### Item metadata

`items.csv`

```text
item_id,title,brand,category,description,item_idx
item_0,running product 0,brand_0,running,...,0
item_1,fitness product 1,brand_1,fitness,...,1
...
```

The pipeline performs chronological splitting so that future interactions are not used to predict earlier interactions.

---

## 4. Installation

Create a Python environment and install the dependencies:

```powershell
pip install -r requirements.txt
```

---

## 5. Prepare the Data

For the included synthetic dataset:

```powershell
python scripts/prepare_data.py --data data/synthetic
```

If using an Amazon-format dataset, provide the corresponding interaction and item metadata files according to the expected schema.

---

## 6. Training

Train the complete SCRec model using:

```powershell
python scripts/train.py --config configs/default.yaml --data data/synthetic
```

The trained checkpoint is saved to:

```text
outputs/best.pt
```

---

## 7. SCRec Configuration

The default configuration enables all three SCRec components.

In `configs/default.yaml`:

```yaml
ablation:
  collaborative_tokenization: true
  semantic_guided_generation: true
  manifold_alignment: true
```

These switches control the three proposed components:

| Configuration                | Component                                |
| ---------------------------- | ---------------------------------------- |
| `collaborative_tokenization` | Collaborative-enhanced item tokenization |
| `semantic_guided_generation` | Semantic guidance during generation      |
| `manifold_alignment`         | Semantic/code manifold alignment         |

When all three are enabled, the model corresponds to the **full SCRec configuration**.

The switches are read by `SCRec` during initialization:

```python
ablation = cfg.get("ablation", {})

self.use_collaborative = ablation.get(
    "collaborative_tokenization", True
)

self.use_semantic_guidance = ablation.get(
    "semantic_guided_generation", True
)

self.use_manifold_alignment = ablation.get(
    "manifold_alignment", True
)
```

At the current stage, these configuration switches establish the experiment interface. The individual ablation variants should only be evaluated after their corresponding components are conditionally disabled throughout the relevant training/model pipeline.

---

## 8. Evaluation

Evaluate a trained checkpoint with:

```powershell
python scripts/evaluate.py --checkpoint outputs/best.pt --data data/synthetic
```

The evaluation reports:

* Recall@5
* NDCG@5
* Recall@10
* NDCG@10

These metrics evaluate whether the held-out next item appears near the top of the recommendation list.

---

## 9. Current Full SCRec Result

Using the current synthetic-data experiment, the trained model produced:

```text
Recall@5:  0.0040
NDCG@5:    0.0040
Recall@10: 0.0120
NDCG@10:   0.0064
```

These results demonstrate that the end-to-end training and evaluation pipeline is functioning.

They should **not** be interpreted as paper-level SCRec performance because the current experiment uses a small synthetic dataset and a lightweight independent implementation.

---

## 10. Recommendation Demo

Run the trained model for a specific user:

```powershell
python scripts/demo.py --checkpoint outputs/best.pt --data data/synthetic --user 0
```

Example:

```text
History:
  - running product 204
  - running product 348
  - running product 300
  - running product 342
  - running product 468
  - running product 42
  - running product 36
  - running product 96

Top recommendations:
  1. gaming product 374 | category=gaming
  2. gaming product 464 | category=gaming
  3. gaming product 308 | category=gaming
  4. gaming product 488 | category=gaming
  5. gaming product 194 | category=gaming
```

This verifies the complete inference path:

```text
User history
     ↓
SCRec model
     ↓
Candidate scoring
     ↓
Top-K recommendations
     ↓
Item metadata
```

---

## 11. Ablation Study

The project is structured to evaluate the contribution of each SCRec component independently.

The intended four configurations are:

| Variant                      | Collaborative Tokenization | Semantic-Guided Generation | Manifold Alignment |
| ---------------------------- | -------------------------: | -------------------------: | -----------------: |
| Sequential Baseline          |                          ❌ |                          ❌ |                  ❌ |
| + Collaborative Tokenization |                          ✅ |                          ❌ |                  ❌ |
| + Semantic-Guided Generation |                          ✅ |                          ✅ |                  ❌ |
| Full SCRec                   |                          ✅ |                          ✅ |                  ✅ |

The purpose of the ablation is to determine whether each component provides an incremental improvement in sequential recommendation quality.

The final experiment should report:

| Model                        | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| ---------------------------- | -------: | -----: | --------: | ------: |
| Sequential Baseline          |        — |      — |         — |       — |
| + Collaborative Tokenization |        — |      — |         — |       — |
| + Semantic-Guided Generation |        — |      — |         — |       — |
| Full SCRec                   |        — |      — |         — |       — |

All variants should use the same:

* Dataset
* Train/validation/test split
* Random seed
* Number of epochs
* Batch size
* Learning rate
* Evaluation protocol

Only the corresponding SCRec component should be changed.

---

## 12. Reproducibility

The experiments are designed to be reproducible using fixed configuration files and deterministic data splitting.

For meaningful ablation comparisons, do not change the dataset split or training hyperparameters between variants.

Recommended experiment structure:

```text
outputs/
├── baseline/
├── collaborative/
├── semantic/
└── full_screc/
```

Each directory should contain the checkpoint and evaluation results for that configuration.

---

## 13. End-to-End Pipeline

```text
                    ┌──────────────────────┐
                    │ Interaction History  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sequential Dataset   │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ Collaborative-Enhanced         │
              │ Item Tokenization               │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Semantic Codebook    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sequential           │
                    │ Transformer         │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          Semantic-Guided          Manifold
             Generation            Alignment
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Next-Item Prediction │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Recall / NDCG        │
                    └──────────────────────┘
```

---

## 14. Limitations

This implementation is an independent engineering reproduction intended to demonstrate the SCRec methodology and provide a reproducible end-to-end pipeline.

Current limitations include:

* Synthetic data is used for the initial experiment.
* The implementation is lightweight compared with a production-scale recommender.
* Current recommendation metrics should not be compared directly with published paper results.
* The implementation does not claim to reproduce the authors' exact training infrastructure or hyperparameters.
* The ablation switches currently define the experiment interface; complete component-specific disabling must be implemented before reporting ablation results.

---

## 15. Next Research Experiments

The next experiments are:

1. Complete the four ablation variants.
2. Train each variant from scratch.
3. Evaluate Recall@5, Recall@10, NDCG@5, and NDCG@10.
4. Compare incremental gains from each SCRec component.
5. Repeat on a real sequential recommendation dataset.
6. Compare against a standard sequential baseline such as SASRec.
7. Analyze recommendation diversity and catalog coverage.
8. Perform hyperparameter sensitivity experiments.

---

## 16. Summary

This project provides an end-to-end implementation of a lightweight SCRec-style sequential recommender:

```text
Data
 ↓
Collaborative + Semantic Item Representation
 ↓
Semantic Tokenization
 ↓
Sequential Transformer
 ↓
Semantic-Guided Generation
 ↓
Manifold Alignment
 ↓
Top-K Recommendation
 ↓
Recall / NDCG Evaluation
```

The implementation is designed to make the individual architectural contributions explicit and experimentally testable through controlled ablations.
