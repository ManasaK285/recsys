# Experiment plan

## Primary hypothesis
A shared reward-feature model that adapts to an individual from a small support set can outperform a single global reward model when preferences are heterogeneous.

## Controlled variables
- number of users / raters
- preference heterogeneity
- support examples per user (1, 2, 4, 8, 16)
- training comparisons per user
- random seed

## Baselines
- global reward model
- independent per-user model
- adaptive shared-feature model
- support-summary baseline

## Metrics
- pairwise accuracy
- log loss
- Brier score
- sample efficiency as support size varies

## Suggested experiment matrix
Run `--heterogeneity 0.1`, `0.5`, `1.0`, `1.5` and compare adaptive-vs-global results. Repeat with user counts 5, 15, 30, 60. Record seeds and use multiple seeds before drawing conclusions.

## Interpretation cautions
Synthetic preferences are generated from a known simulator and can favor methods aligned to that simulator. The model implementation is educational and not an exact reproduction of the paper's architecture or theoretical proof. For real claims, collect consented human preference data and use user-level holdouts where appropriate.
