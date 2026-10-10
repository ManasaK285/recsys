# TrustLens

## Synthetic Study of AI-vs-Human Source Detection, Agreement, and Robustness

TrustLens is a controlled research pipeline investigating how machine-learning models distinguish AI-generated from human-generated responses and how sensitive that detection is to synthetic linguistic artifacts.

The project focuses on an important methodological question:
 
> **When a classifier achieves high AI-vs-human source-detection performance on synthetic data, how much of that performance reflects genuine source-related signal versus artifacts introduced by the data-generation process?**

The pipeline evaluates source detection, agreement prediction, humanization, feature ablation, scenario generalization, style-template generalization, style neutralization, and source-signal ablation.

---

## Research Questions

TrustLens investigates five primary questions:

1. Can AI-generated and human-generated responses be distinguished in a controlled synthetic dataset?
2. Can response characteristics predict whether a participant agrees with a judgment?
3. Does humanizing an AI response change its detectability?
4. Does source-detection performance generalize to unseen scenarios and unseen stylistic templates?
5. How much of source-detection performance can be attributed to explicit generator-specific stylistic artifacts?

---

## Dataset

The current dataset contains:

* **2,400 responses**
* **1,200 human**
* **1,200 AI**
* **2,400 unique response texts**
* **7 ethical scenarios**
* **15 opening templates**
* **12 transition templates**
* **12 caveat templates**
* **12 ending templates**
* **6 framing templates**
* **10 source-specific style templates**

  * 5 human
  * 5 AI
* **6 decision phrases**

Scenarios include:

* Trolley
* Bridge
* Medicine
* Privacy
* Lying
* Autonomy
* Fairness

The dataset also contains perceived-source judgments and agreement labels for downstream experiments.

---

# Experimental Pipeline

## Experiment 01 — Source Detection

A TF-IDF + Logistic Regression classifier was used to distinguish AI from human responses.

The original synthetic dataset produced approximately:

* Accuracy: **93.8%**
* Macro F1: **93.7%**
* ROC-AUC: **99.0%**

Additional comparisons included sentence embeddings and linguistic features.

A label-shuffling sanity check produced performance near chance, indicating that the observed signal was associated with the actual source labels.

---

## Experiment 02 — Agreement Prediction

Agreement prediction was evaluated using progressively richer feature sets:

* Response representation
* Scenario + response
* Scenario + response + linguistic features
* All features + perceived source

Performance remained substantially more difficult than source detection, with accuracy in the approximate **56–58%** range.

This suggests that identifying response source and predicting agreement are distinct tasks.

---

## Experiment 03 — Humanization

AI responses were transformed toward a more human-like linguistic style and evaluated for source detectability.

The experiment compared source-detection recall before and after humanization.

The results showed that stylistic transformation alone did not reliably eliminate source-detection signal.

---

## Experiment 04 — Feature Ablation

Different feature combinations were evaluated for agreement prediction.

The experiment showed that adding additional linguistic and perceived-source features did not automatically produce large improvements, reinforcing the difficulty of predicting agreement from response-level information alone.

---

## Experiment 05 — Scenario Generalization

Source detection was evaluated using leave-one-scenario-out validation.

The model maintained relatively strong performance on unseen scenarios, but this does not by itself establish generalization beyond the synthetic generator.

This experiment tests scenario transfer rather than real-world human/LLM transfer.

---

# Experiment 06 — Paired Style-Template Generalization

Experiment 06 was designed to test whether the detector could generalize to stylistic templates that were completely absent from training.

Five paired folds were created:

```text
Fold 0: human_style_0 + ai_style_0
Fold 1: human_style_1 + ai_style_1
Fold 2: human_style_2 + ai_style_2
Fold 3: human_style_3 + ai_style_3
Fold 4: human_style_4 + ai_style_4
```

Results:

| Metric    |          Mean ± Std |
| --------- | ------------------: |
| Accuracy  | **82.67% ± 16.23%** |
| Macro F1  | **80.95% ± 18.42%** |
| Precision |  **99.07% ± 1.28%** |
| Recall    | **66.00% ± 32.11%** |
| ROC-AUC   |  **97.60% ± 3.22%** |

Per-fold accuracy ranged from **63.15% to 98.92%**.

This large variation is important: high source-detection performance does not transfer uniformly across unseen synthetic stylistic constructions.

---

# Experiment 07 — Source-Style Neutralization

The explicit human-vs-AI style phrases were replaced with shared neutral phrases.

Results:

| Condition         |   Accuracy |   Macro F1 |    ROC-AUC |
| ----------------- | ---------: | ---------: | ---------: |
| Style neutralized | **81.87%** | **81.87%** | **93.66%** |
| Label shuffle     | **50.83%** | **50.83%** | **49.82%** |

The classifier therefore retained substantial source-detection performance even after explicit style phrases were neutralized.

This indicates that the explicit style vocabulary is an important source of signal, but it is **not the only source-dependent signal in the synthetic generator**.

---

# Experiment 08 — Source-Signal Ablation

The final diagnostic experiment progressively neutralized source-related response components.

| Condition                            |    Accuracy |     ROC-AUC |
| ------------------------------------ | ----------: | ----------: |
| Full response                        | **100.00%** | **100.00%** |
| Style neutralized                    |  **81.67%** |  **93.56%** |
| Frame neutralized                    | **100.00%** | **100.00%** |
| Decision neutralized                 | **100.00%** | **100.00%** |
| Style + frame neutralized            |  **81.46%** |  **93.65%** |
| Style + frame + decision neutralized |  **81.46%** |  **93.60%** |

The ablation indicates that explicit source-specific style is the dominant component among the tested generator components.

Neutralizing the frame or decision components alone had essentially no effect.

After style neutralization, approximately **81% accuracy** remained, indicating additional source-dependent information elsewhere in the generated response structure.

---

# Main Findings

The experiments support four main observations.

### 1. Synthetic source detection can be very strong

TF-IDF models can achieve high performance when trained and evaluated on responses generated from the same controlled synthetic process.

### 2. Performance is sensitive to stylistic construction

When source-specific style templates are held out, performance varies substantially across unseen templates.

### 3. Explicit style contributes substantially to detection

Neutralizing the human-vs-AI style phrases reduces performance from perfect detection in the final ablation setup to approximately **82% accuracy**.

### 4. Residual generator-specific signal remains

Even after explicit style neutralization, source detection remains well above chance.

Therefore, the generator likely contains additional correlations between source and response characteristics.

---

# Interpretation and Limitations

The results should **not** be interpreted as demonstrating that AI-generated text can generally be detected with approximately 82–94% accuracy in real-world settings.

The experiments use a controlled synthetic generator in which source labels are directly associated with the response-generation process.

Consequently, the classifier may learn:

* generator-specific vocabulary
* stylistic patterns
* phrase combinations
* structural regularities
* source-dependent lexical distributions

rather than universal properties of human versus AI writing.

The scenario and template generalization experiments partially test robustness to changes in the synthetic distribution, but they do not replace evaluation on independently collected human and LLM-generated responses.

---

# Overall Conclusion

TrustLens demonstrates a broader methodological point:

> **High AI-vs-human source-detection performance on synthetic data should not automatically be interpreted as robust detection of AI-generated language.**

In this controlled setting, explicit source-specific style accounts for a substantial portion of the detectable signal, while additional source-dependent structure remains after style neutralization.

The substantial variation across unseen style templates further demonstrates that detector performance can depend strongly on how the synthetic responses are constructed.

The results therefore motivate evaluation protocols that explicitly control for:

1. Template leakage
2. Stylistic artifacts
3. Scenario overlap
4. Generator-specific vocabulary
5. Unseen response constructions
6. Independent real-world human and LLM data

---

# Reproducibility

From the project root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Generate the dataset:

```powershell
python -m trustlens.data.make_dataset
```

Run individual experiments:

```powershell
python experiments\01_source_detection.py
python experiments\02_agreement_prediction.py
python experiments\03_humanization.py
python experiments\04_ablation.py
python experiments\05_scenario_generalization.py
python experiments\06_template_generalization.py
python experiments\07_style_neutralization.py
python experiments\08_source_signal_ablation.py
```

The resulting metrics are stored under:

```text
results/
```

---

# Project Structure

```text
trustlens/
│
├── data/
│   └── raw/
│       └── trustlens.csv
│
├── experiments/
│   ├── 01_source_detection.py
│   ├── 02_agreement_prediction.py
│   ├── 03_humanization.py
│   ├── 04_ablation.py
│   ├── 05_scenario_generalization.py
│   ├── 06_template_generalization.py
│   ├── 07_style_neutralization.py
│   └── 08_source_signal_ablation.py
│
├── results/
│   ├── 06_paired_style_generalization.csv
│   ├── 07_style_neutralization.csv
│   └── 08_source_signal_ablation.csv
│
├── trustlens/
│   ├── data/
│   ├── explainability/
│   └── visualization/
│
└── README.md
```

---

## Status

**Research pipeline complete.**

The current version is intended as a controlled synthetic study and methodological demonstration rather than a production AI-text detector.
