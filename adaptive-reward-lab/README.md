# AdaptiveReward Lab: Learning Personalized Reward Models from Human Preferences

A research-inspired implementation of **Capturing Human Preferences with Reward Features** by Andre Barreto et al., presented at NeurIPS 2025.

The project investigates whether learning shared reward features combined with user-specific preferences can outperform a single reward function when different people disagree about which responses are best.

## Research Reference

**Paper:** *Capturing Human Preferences with Reward Features*

- Publication: https://deepmind.google/research/publications/141313/
- OpenReview paper: https://openreview.net/pdf?id=TgCkj4uEPl
- NeurIPS 2025 poster: https://neurips.cc/virtual/2025/loc/mexico-city/poster/117832

## 1. Problem Statement

Conventional reinforcement learning from human feedback (RLHF) often learns a shared reward model that assumes all users have similar preferences. In practice, users may disagree about qualities such as helpfulness, detail, concision, tone, and creativity.

A single reward function can obscure these differences.

This project compares three approaches:

1. **Global Reward Model:** One reward function shared by all users.
2. **Independent User Models:** Separate reward functions for users with sufficient training data.
3. **Adaptive Reward Model:** Shared learned reward features combined with user-specific weights.

The central research question is:

> Can shared reward features improve personalized preference prediction, particularly when users have heterogeneous preferences and only limited feedback is available?

## 2. Project Objectives

- Generate controlled pairwise preference data for simulated users.
- Train global, independent-user, and adaptive reward models.
- Evaluate pairwise preference prediction on held-out examples.
- Investigate the effect of user preference heterogeneity.
- Explore adaptation to new users using limited feedback.
- Compare models using accuracy, log loss, and Brier score.
- Provide an interactive Streamlit dashboard.
- Establish a reproducible experimental framework for further research.

## 3. Methodology

### Pairwise preference learning

Given two candidate responses, A and B, the model predicts the probability that a user prefers A.

For user \(u\):

\[
r_u(x)=w_u^\top \phi(x)
\]

where:

- \(r_u(x)\) is the reward assigned to response \(x\).
- \(\phi(x)\) represents response features.
- \(w_u\) represents the user's preference weights.

The pairwise preference probability is:

\[
P(A\succ B\mid u)=
\sigma\left(r_u(A)-r_u(B)\right)
\]

where \(\sigma\) is the sigmoid function.

### Model architectures

**Global model**

Learns a single linear reward vector shared by all users.

**Independent-user model**

Learns individual linear reward vectors for users with sufficient examples. A global model provides a fallback for users with sparse data.

**Adaptive model**

Learns a shared nonlinear feature transformation and user-specific reward weights:

\[
\phi(x)=\tanh(xW+b)
\]

\[
r_u(x)=\phi(x)^\top v_u
\]

The shared representation is intended to capture reusable reward features, while each user's vector represents their preferences over those features.

This adaptive implementation is a simplified research-inspired architecture, not an exact reproduction of the paper's complete method.

## 4. End-to-End Pipeline

1. **Data generation:** Generate synthetic response feature vectors and preference labels using simulated user reward weights.
2. **Data splitting:** Separate training and held-out comparisons.
3. **Model training:** Fit global, independent-user, and adaptive models.
4. **Prediction:** Estimate pairwise preference probabilities.
5. **Evaluation:** Calculate accuracy, log loss, and Brier score.
6. **Heterogeneity analysis:** Compare model performance under different simulated levels of preference disagreement.
7. **Cold-start adaptation:** Fit user-specific weights using a small support set and evaluate on separate comparisons.
8. **Visualization:** Generate comparison tables, plots, and downloadable CSV results through the dashboard.

## 5. Initial Experimental Results

The first quick-mode experiment produced the following results on 180 held-out comparisons per model.

| Model | Accuracy | Log Loss | Brier Score | Test Samples |
|---|---:|---:|---:|---:|
| Global | 0.622222 | 0.664221 | 0.236154 | 180 |
| Independent | **0.650000** | **0.622159** | **0.216878** | 180 |
| Adaptive | 0.516667 | 0.684899 | 0.245910 | 180 |

### Results interpretation

**Independent-user model**

The independent model achieved the best results in the initial run:

- Accuracy: 65.00%
- Log loss: 0.622159
- Brier score: 0.216878

It improved accuracy by approximately 2.78 percentage points over the global baseline. This suggests that modeling user-specific preferences may help in the current synthetic setting.

**Global model**

The global model achieved 62.22% accuracy. It provides a useful baseline but cannot directly represent individual differences in reward weights.

**Adaptive model**

The adaptive model achieved 51.67% accuracy, with a log loss of 0.684899 and Brier score of 0.245910. Its performance was weaker than both baselines in this run.

Potential explanations include limited feedback per user, optimization difficulties, feature-learning instability, and model complexity relative to the available data. These are hypotheses to test, not established causes.

### Initial conclusion

The current experiment does **not** demonstrate that the adaptive architecture outperforms the baselines. Instead, it establishes a reproducible starting point and identifies the adaptive model as the main target for further investigation.

The experiment uses a relatively small test set and a single run. The observed differences should not be interpreted as statistically significant without repeated trials and uncertainty estimates.

## 6. Planned Experiments

### Experiment A: Preference heterogeneity

Vary the degree of disagreement between simulated users and compare the global and adaptive approaches.

**Hypothesis:** Personalization may become more beneficial as user preferences become more heterogeneous.

### Experiment B: Feedback quantity

Evaluate adaptation using different numbers of user preference comparisons:

- 1 example
- 2 examples
- 5 examples
- 10 examples
- 20 examples

Measure how quickly performance improves as additional feedback becomes available.

### Experiment C: Number of raters

Vary the number of users contributing preference data and investigate whether broader coverage improves shared feature learning.

### Experiment D: Optimization and regularization

Investigate learning rates, regularization strengths, feature dimensions, initialization, and training stability for the adaptive model.

### Experiment E: Repeated trials

Run multiple random seeds and report mean and standard deviation for each metric. Use appropriate uncertainty estimates before drawing conclusions.

### Experiment F: Real preference data

Explore the optional Anthropic HH-RLHF dataset to test text-based preference modeling.

The dataset's chosen/rejected pairs do not, by themselves, guarantee identifiable rater identities or multiple ratings of the same comparison. Therefore, this track cannot automatically establish user-specific generalization.

## 7. Evaluation Metrics

**Accuracy:** Fraction of pairwise preferences predicted correctly. Higher is better.

**Log loss:** Measures the quality of predicted preference probabilities. Lower is better.

**Brier score:** Measures the mean squared difference between predicted probabilities and binary outcomes. Lower is better.

**Cold-start accuracy:** Measures preference prediction for a user after a limited number of adaptation examples.

For meaningful comparisons, all models should use equivalent test data and consistent data-splitting rules.

## 8. Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn ecosystem
- Matplotlib
- Plotly
- Streamlit
- Pytest
- Hugging Face Datasets for the optional HH-RLHF track

The core synthetic experiment runs locally without downloading an external dataset.

## 9. Repository Structure

```text
adaptive-reward-lab/
├── app.py
├── configs/
│   ├── synthetic.yaml
│   └── hh_rlhf.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── preference_dataset.py
│   │   ├── synthetic_raters.py
│   │   └── load_hh_rlhf.py
│   ├── features/
│   │   └── reward_features.py
│   ├── models/
│   │   ├── global_reward.py
│   │   ├── independent_reward.py
│   │   └── adaptive_reward.py
│   ├── training/
│   │   ├── train.py
│   │   └── adapt_user.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cold_start.py
│   │   └── disagreement.py
│   └── experiments/
│       ├── run_all.py
│       └── plot_results.py
├── tests/
├── artifacts/
│   ├── results/
│   ├── figures/
│   └── checkpoints/
├── requirements.txt
└── README.md
```

## 10. Installation and Execution

### Windows PowerShell

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the quick experiment:

```powershell
python -m src.experiments.run_all --quick
```

Run the larger experiment configuration:

```powershell
python -m src.experiments.run_all
```

Launch the interactive dashboard:

```powershell
streamlit run app.py
```

Run tests:

```powershell
python -m pytest
```

The experiment pipeline writes model comparison metrics and heterogeneity results under `artifacts/results/`, with figures under `artifacts/figures/`.

## 11. Limitations

- Synthetic response features simplify the complexity of real language-model responses.
- The current adaptive model is a simplified shared-feature architecture.
- Sparse user feedback may make user-specific weights difficult to estimate.
- The initial results come from one small experiment.
- Real human preferences may be inconsistent, contextual, and non-linear.
- The optional HH-RLHF dataset does not necessarily identify individual raters.
- The project has not established that its adaptive implementation reproduces the paper's theoretical guarantees or reported experimental results.

## 12. Future Improvements

1. Improve the adaptive model's optimization and training stability.
2. Increase and balance the number of comparisons available per user.
3. Add a rigorous unseen-user evaluation protocol.
4. Run repeated experiments and report confidence intervals.
5. Add controlled ablations of shared features and personalization.
6. Explore richer response representations beyond handcrafted features.
7. Compare results against the paper's method and baselines wherever the published methodology and available data permit a fair comparison.

## 13. Research Significance

This project provides an experimental framework for studying the relationship between human preference disagreement, shared reward representations, and personalized reward modeling.

Its current results favor the independent-user baseline, but the main research opportunity is to determine whether shared reward features can offer better generalization when user feedback is limited.

**The goal is not to assume that personalization wins, but to identify when it helps, how much feedback it requires, and whether shared reward features improve adaptation to users whose preferences differ from those represented during training.**

## References

1. Barreto, A., Dumoulin, V., Mao, Y., Rowland, M., Perez-Nieves, N., Shahriari, B., Dauphin, Y., Precup, D., and Larochelle, H. *Capturing Human Preferences with Reward Features*. NeurIPS 2025.
2. https://deepmind.google/research/publications/141313/
3. https://openreview.net/pdf?id=TgCkj4uEPl
4. https://neurips.cc/virtual/2025/loc/mexico-city/poster/117832
5. https://huggingface.co/datasets/Anthropic/hh-rlhf
