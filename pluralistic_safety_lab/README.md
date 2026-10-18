# Pluralistic Safety Lens

An end-to-end, runnable research-inspired project exploring how different groups of raters use an ordinal safety-rating scale and how much ratings vary across items. The project is inspired by the paper **“Decoding Safety Feedback from Diverse Raters: A Data-driven Lens on Responsiveness to Severity”** by Pushkar Mishra et al. (*Transactions on Machine Learning Research*, 2026).

> **Important research distinction:** The current results below come from this project's **synthetic demonstration dataset**, not from the paper's empirical data. This is an independent educational implementation using transparent baseline statistics; it is not an exact reproduction of the paper's methods, metrics, or findings. The paper motivates studying differences in how people respond to severity, while this project demonstrates one practical analysis workflow.

## Research question

How do rating patterns vary across simulated rater groups as item severity changes, and which items receive the most varied ratings?

The workflow retains group-level and item-level summaries instead of relying only on one overall average. This helps surface potential variation that can be hidden when ordinal ratings are collapsed into a single score.

## Current run: data and setup

The pipeline ran successfully and generated the following synthetic dataset:

| Quantity | Value |
|---|---:|
| Ratings | 9,720 |
| Items | 180 |
| Raters | 54 |
| Simulated groups | 3 |
| Raters per group | 18 |
| Ratings per item | 54 |
| Ordinal rating scale | 0–4 |

The data are generated deterministically for repeatable demonstrations. The three simulated groups are deliberately given different rating-response styles. Consequently, the observed differences validate that the analysis pipeline can detect differences in the data it was designed to generate; they are **not evidence about real populations**.

## Findings from the current synthetic run

### 1. Group-level responsiveness summaries

| Group | Mean scale span | Distinct levels used (mean) | Monotonicity | Spearman correlation with severity | Rating entropy |
|---|---:|---:|---:|---:|---:|
| Group A | 3.539 | 5.000 | 1.000 | 0.926 | 0.981 |
| Group B | 2.362 | 4.944 | 1.000 | 0.749 | 0.904 |
| Group C | 3.889 | 5.000 | 1.000 | 0.933 | 0.981 |

**Interpretation:**

- **Group C** has the largest mean scale span (3.889) and strongest severity association (Spearman 0.933) in this generated run. Its ratings tend to differentiate more across severity levels.
- **Group A** also shows a strong severity association (0.926) and uses all five rating categories on average.
- **Group B** has a smaller mean scale span (2.362), lower severity association (0.749), and lower rating entropy (0.904) than the other simulated groups. In this dataset, its ratings show less differentiation with severity and a somewhat less even distribution across rating categories.
- All groups have monotonicity of 1.0 under this implementation's definition: the group-level median rating did not decrease across adjacent ordered severity levels. This does not mean every individual rating is monotonic or that groups agree with each other.

These values describe the synthetic dataset and the project's own metric definitions. They should not be attributed to the paper.

### 2. Items with the highest rating entropy

The dashboard/export identified the following 10 items with the highest normalized entropy across the 54 ratings per item:

| Rank | Item | Severity | Mean rating | Normalized entropy | Rating range |
|---:|---|---:|---:|---:|---:|
| 1 | `item_0128` | 2 | 1.926 | 0.793 | 4 |
| 2 | `item_0098` | 3 | 2.685 | 0.770 | 3 |
| 3 | `item_0143` | 2 | 2.019 | 0.761 | 4 |
| 4 | `item_0004` | 2 | 2.037 | 0.757 | 4 |
| 5 | `item_0135` | 1 | 1.074 | 0.734 | 4 |
| 6 | `item_0168` | 1 | 1.111 | 0.734 | 3 |
| 7 | `item_0130` | 2 | 2.037 | 0.732 | 4 |
| 8 | `item_0067` | 1 | 1.000 | 0.731 | 3 |
| 9 | `item_0141` | 1 | 1.056 | 0.728 | 3 |
| 10 | `item_0021` | 1 | 1.037 | 0.727 | 3 |

**Interpretation:** `item_0128` has the highest normalized entropy (0.793), and its ratings span the full 0–4 scale. `item_0098` is also high entropy (0.770), with a severity of 3 and mean rating of 2.685. Items `item_0143` and `item_0004` each span the full rating scale as well.

Normalized entropy measures how spread out the rating distribution is across categories. It is **not, by itself, a direct measure of disagreement between particular groups**. To assess group differences for these items, inspect the group-specific rating distributions and compare them directly.

## Metrics implemented

These are transparent baseline metrics implemented in this repository, not claims about the paper's exact named metrics:

- **Scale span:** average within-rater rating range across ordered severity levels.
- **Distinct rating levels:** average number of distinct rating categories used across ordered severity levels.
- **Monotonicity:** fraction of adjacent severity-level pairs for which the group median rating does not decrease.
- **Severity association:** Spearman correlation between item severity and ratings.
- **Normalized rating entropy:** entropy of the item/group rating distribution normalized by the maximum possible entropy for five categories.
- **Item disagreement summaries:** mean, median, standard deviation, rating range, and rating-category proportions per item.

Metric values depend on the implementation and on the data and should be interpreted together rather than as standalone rankings of groups.

## What the project does

- Generates deterministic synthetic ordinal ratings so the pipeline runs without a download or API key.
- Validates a CSV input schema.
- Computes group-level responsiveness summaries and item-level rating distributions.
- Writes CSV and JSON analysis artifacts and PNG plots.
- Provides an interactive Streamlit dashboard.
- Includes unit tests for key metrics.

## Quick start (Windows PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.run_pipeline
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. The existing run also completed in the user's Python 3.14 environment, but Python 3.11 is the documented setup for a predictable environment.

### Launch the dashboard

The dashboard file is `src/dashboard.py` (there is no root-level `app.py` in this project):

```powershell
python -m streamlit run src/dashboard.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Run tests

```powershell
python -m pytest -q
```

## Input data schema

Place a compatible CSV at `data/ratings.csv`, with one row per rating:

| Column | Meaning |
|---|---|
| `item_id` | Content or prompt identifier |
| `rater_id` | Pseudonymous rater identifier |
| `group` | Analysis group label, where available and ethically appropriate |
| `severity` | Ordered reference severity, numeric |
| `rating` | Ordinal safety/harm rating, integer from 0 to 4 for the current implementation |

The reference severity should come from an expert label, controlled scenario design, or a defensible external measure. Map the source dataset's scale carefully before running the pipeline; do not assume that all safety datasets use the same rating scale or schema.

Run on an input file with:

```powershell
python -m src.run_pipeline --input data/ratings.csv
```

If the input is missing, the pipeline generates synthetic data automatically. Use `--no-synthetic` to fail instead of generating demo data.

## Outputs

The pipeline writes these artifacts under `reports/`:

- `group_metrics.csv` — group-level summary metrics.
- `item_disagreement.csv` — per-item distributions and disagreement summaries.
- `item_aggregates.csv` — aggregated item ratings.
- `summary.json` — run metadata and selected summaries.
- `fig_group_responsiveness.png` — group metric comparison.
- `fig_group_rating_distributions.png` — rating distributions by group.
- `fig_item_disagreement.png` — item-level entropy/disagreement visualization.

The CSV exported from the dashboard is a filtered table of high-entropy items; it is not the full raw rating-level dataset.

## Limitations and responsible use

- Synthetic data is useful for checking that code executes and illustrating analysis, but it cannot support claims about real rater populations or real safety behavior.
- These baseline metrics are not a faithful reproduction of every method in the referenced paper.
- Group-level differences are descriptive and can reflect item mix, sampling, context, or measurement effects; they do not establish that group identity causes a rating pattern.
- Preserve pseudonymity and avoid re-identifying raters.
- Report sample sizes and uncertainty, especially when working with real or small groups.
- Treat disagreement as potentially informative; do not automatically discard it as noise.

## Project structure

```text
pluralistic_safety_lab/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    # Add compatible input data here
├── reports/                 # Generated CSV, JSON, and plots
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── metrics.py
│   ├── analysis.py
│   ├── plots.py
│   ├── run_pipeline.py
│   └── dashboard.py
└── tests/
    └── test_metrics.py
```

## Research reference

Mishra, P. et al. **“Decoding Safety Feedback from Diverse Raters: A Data-driven Lens on Responsiveness to Severity.”** *Transactions on Machine Learning Research* (2026).

- [Google DeepMind publication page](https://deepmind.google/research/publications/137741/)
- [Paper PDF](https://openreview.net/pdf?id=vx6qrM7VB5)
- Related dataset resource: [DICES dataset repository](https://github.com/google-research-datasets/dices-dataset)

The reference paper motivates studying how safety feedback varies across raters and severity. This repository uses that motivation to build a small, inspectable workflow; use the paper itself for its formal definitions, empirical findings, and methodological claims.
