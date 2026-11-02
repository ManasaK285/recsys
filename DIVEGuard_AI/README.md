# 🧭 DIVEGuard AI
### Pluralistic Safety Intelligence — Understand Disagreement, Not Just Averages.

DIVEGuard AI is a research-oriented machine learning platform for studying pluralistic human safety judgments in text-to-image systems.

Instead of treating safety as a single universal score, DIVEGuard AI investigates how harmfulness perceptions vary across prompts, topics, and annotator contexts.

The project uses the **DIVE dataset** to build and compare global and pluralistic machine learning models, explore annotation disagreements, and audit prediction errors across demographic groups.

> Understand disagreement, not just averages.

---

## 1. Project Overview

Traditional safety evaluation often relies on aggregated judgments or a single shared safety score. However, people can interpret the same prompt differently depending on their perspectives and contexts.

DIVEGuard AI explores this challenge through two predictive approaches:

- **Global Safety Model:** Learns a shared relationship between prompt text, topic, and human harmfulness ratings.
- **Pluralistic Safety Model:** Incorporates available annotator context and context-topic interactions to better represent differences in individual judgments.

The objective is to investigate whether modeling pluralistic perspectives improves prediction of human safety judgments.

## 2. Dataset Overview

DIVEGuard AI uses real annotation data from the DIVE dataset.

| Dataset statistic | Value |
|---|---:|
| Total annotations | 30,770 |
| Unique prompts/images | 1,000 |
| Mean harmfulness rating | 0.96 / 4 |
| Number of topics | 19 |
| Rating scale | 0–4 |

**Important:** Ratings represent human judgments in the dataset, not objective ground truth.

The project uses annotation records and associated prompt/image identifiers. It does not claim to perform visual inference on the original generated images.

## 3. Key Features

### Dataset Intelligence
- Real DIVE annotation data ingestion.
- Data cleaning and preprocessing.
- Topic-level analysis.
- Human harmfulness rating distributions.
- Prompt-level aggregation and disagreement analysis.

### Pluralistic Safety Modeling
- Global baseline model.
- Pluralistic context-aware model.
- TF-IDF text representations.
- Categorical feature encoding.
- Logistic Regression classification.

### Pluralism Audit
- Demographic subgroup evaluation.
- Group-level prediction error analysis.
- Comparison of global and pluralistic approaches.
- Identification of groups requiring further investigation.

### Disagreement Explorer
- Prompt-level rating distributions.
- Mean and standard deviation of ratings.
- Identification of high-disagreement prompts.
- Exploration of differences in human safety judgments.

### Interactive Research Dashboard
The Streamlit application provides:

1. Overview
2. Pluralism Audit
3. Model Lab
4. Perspective Explorer
5. Disagreement Explorer

## 4. Machine Learning Architecture

```text
                 DIVE Annotation Dataset
                          |
                          v
                  Data Ingestion
                          |
                          v
                  Data Cleaning
                          |
                          v
               Feature Engineering
                          |
             +------------+------------+
             |                         |
             v                         v
       Global Model              Pluralistic Model
             |                         |
       Prompt Text              Prompt Text
       Topic Features           Topic Features
                                 Rater Context
                                 Context-Topic
             |                         |
             +------------+------------+
                          |
                          v
                  Model Evaluation
                          |
             +------------+------------+
             |            |             |
             v            v             v
          Accuracy     Macro F1        MAE
                          |
                          v
                Quadratic Weighted Kappa
                          |
                          v
                 Pluralism & Error Audit
                          |
                          v
                 Streamlit Dashboard
```

## 5. Experimental Setup

### Data Split

The dataset was divided using an 80/20 `GroupShuffleSplit` based on `image_id`.

| Split | Annotations | Unique images |
|---|---:|---:|
| Training | 24,694 | 800 |
| Testing | 6,076 | 200 |
| Total | 30,770 | 1,000 |

Grouping by image ensures that annotations belonging to the same image are not distributed across training and testing.

This reduces image-level data leakage and provides a more meaningful evaluation of generalization to unseen images.

### Models

**Global Safety Model**

Uses:
- TF-IDF prompt representations.
- Topic information.
- Logistic Regression classifier.

**Pluralistic Safety Model**

Uses:
- TF-IDF prompt representations.
- Topic information.
- Available annotator demographic/contextual features.
- Context-topic categorical interactions.
- Logistic Regression classifier.

The pluralistic model is designed to capture contextual variation in human safety judgments rather than assume that every annotator shares the same perspective.

## 6. Experimental Results

Both models were successfully trained and evaluated on 6,076 held-out annotations.

### Global vs. Pluralistic Model

| Metric | Global Model | Pluralistic Model |
|---|---:|---:|
| Accuracy | 0.3436 | **0.3896** |
| Macro F1 | 0.2409 | **0.2805** |
| MAE | 1.3015 | **1.1702** |
| Quadratic Weighted Kappa | 0.1617 | **0.2390** |

### Key Observations

**1. Improved prediction accuracy**

The pluralistic model achieved 38.96% accuracy compared with 34.36% for the global baseline.

**2. Improved class-level performance**

Macro F1 increased from 0.2409 to 0.2805, indicating improved performance across rating classes.

**3. Reduced prediction error**

Mean Absolute Error decreased from 1.3015 to 1.1702, approximately a 10.1% reduction.

**4. Improved rating agreement**

Quadratic weighted kappa increased from 0.1617 to 0.2390.

These results suggest that incorporating available annotator context can improve the prediction of individual human harmfulness ratings.

However, this is an initial experimental result, not proof that the model has learned a universally fair or unbiased safety representation.

## 7. Subgroup Error Analysis

The evaluation pipeline also examines prediction errors across annotator groups.

Examples of groups with relatively high MAE in the reported evaluation:

| Attribute | Group | Sample Count | MAE |
|---|---|---:|---:|
| Country of birth | Kenya | 20 | 2.150 |
| Country of birth | Pakistan | 104 | 1.663 |
| Nationality | Pakistan | 49 | 1.612 |
| Nationality | Malaysia | 38 | 1.605 |
| Ethnicity | Black | 1,195 | 1.466 |
| Country of birth | Nigeria | 442 | 1.418 |

These results identify groups that warrant additional investigation.

A higher subgroup MAE does not independently establish demographic bias. Differences can reflect annotation variability, sample sizes, prompt distributions, and other factors.

Future experiments should compare subgroup errors across both models and include uncertainty estimates.

## 8. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Text Representation | TF-IDF |
| Feature Encoding | OneHotEncoder |
| Classification | Logistic Regression |
| Evaluation | Scikit-learn Metrics |
| Dashboard | Streamlit |
| Visualization | Plotly / Streamlit |
| Model Persistence | Joblib |
| Testing | Pytest |

## 9. Project Structure

```text
DIVEGuard_AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   └── diveguard/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── train.py
│       ├── evaluate.py
│       ├── inference.py
│       └── demo.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── global_model.joblib
│   ├── pluralistic_model.joblib
│   └── metrics.json
│
└── tests/
    ├── test_data.py
    └── test_features.py
```

Generated datasets and model artifacts may be excluded from version control depending on their size and licensing requirements.

## 10. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DIVEGuard_AI.git

cd DIVEGuard_AI
```

### Step 2: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Place the cleaned DIVE dataset in the expected project data location:

```text
data/processed/
```

The project supports the cleaned dataset generated by its preprocessing pipeline.

### Step 5: Train the models

```powershell
python -m src.diveguard.train
```

### Step 6: Evaluate model performance

```powershell
python -m src.diveguard.evaluate
```

### Step 7: Launch the dashboard

```powershell
streamlit run app.py
```

The dashboard will be available at the local Streamlit URL printed in your terminal.

### Step 8: Run tests

```powershell
python -m pytest -q
```

## 11. Research Limitations

DIVEGuard AI is an experimental research prototype.

- Human ratings are subjective judgments, not objective safety labels.
- Demographic attributes are contextual proxies, not deterministic representations of individual identities or beliefs.
- The current model predicts annotation ratings using text and available annotator context; it does not directly analyze generated image pixels.
- The current evaluation holds out images, not necessarily annotators.
- The initial results use a single grouped train/test split.
- Subgroup results can be affected by unequal sample sizes and annotation distributions.
- Better predictive performance does not automatically imply fairness, representativeness, or improved real-world safety.

The system is not a production moderation system and should not be used as an automated decision-making authority.

## 12. Future Research Directions

Potential extensions include:

1. **Hierarchical Pluralistic Modeling:** Learn shared and individual-level patterns in human judgments.
2. **Multi-Task Learning:** Jointly predict different safety-related rating dimensions.
3. **Uncertainty Quantification:** Estimate prediction confidence and identify ambiguous judgments.
4. **Robustness Evaluation:** Repeat experiments across multiple image-grouped splits.
5. **Fairness Auditing:** Compare subgroup errors with confidence intervals and minimum sample-size requirements.
6. **Disagreement-Aware Learning:** Explicitly model rating distributions instead of predicting only a single rating.
7. **Visual Safety Evaluation:** Integrate separately licensed text-to-image models and image-level human evaluation.
8. **Human-in-the-Loop Review:** Support expert review of ambiguous or high-disagreement cases.

## 13. Research Motivation

The central motivation behind DIVEGuard AI is that safety judgments are not always universal.

A system that optimizes only for an average rating can overlook meaningful differences in human perspectives.

By modeling annotator context, exploring disagreement, and auditing prediction errors, DIVEGuard AI investigates how machine learning systems can better represent pluralistic human judgments.

**The goal is not to eliminate disagreement. It is to understand it.**

---

## Acknowledgments

This project is inspired by research on pluralistic alignment and the DIVE dataset.

- Dataset: DIVE — Deep DIVE Dataset for Pluralistic Alignment of Text-to-Image Models.
- Research reference: [DIVE — arXiv:2507.13383](https://arxiv.org/abs/2507.13383).

Please consult the original dataset repository and accompanying documentation for data access, attribution, and licensing requirements.

## License

Project source code: MIT License (if distributed with the repository's MIT license).

Dataset licensing and attribution remain subject to the original DIVE dataset terms (CC-BY-4.0).

---

**DIVEGuard AI**

*Pluralistic safety intelligence. Understand disagreement, not just averages.*