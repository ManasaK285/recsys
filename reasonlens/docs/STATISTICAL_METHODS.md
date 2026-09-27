# ReasonLens — Statistical Methods

## Descriptive statistics (`analysis/descriptive.py`)

- **Overall rates:** grant rate, denial rate, and (among granted
  decisions) the precise vs. approximate split.
- **By-group rates:** the same, broken out by `reason_type`, `app_type`,
  and their cross-tabulation, along with mean response time and mean
  Likert scores per group.

## Hypothesis tests (`analysis/statistics.py`)

- **Chi-square test of independence** — tests whether permission
  decision (granted/denied) is independent of `reason_type` (and
  separately, `app_type`). Reports χ², p-value, degrees of freedom, and
  the full contingency table.
- **One-way ANOVA** — tests whether mean response time, app trust, or
  privacy concern differ across `reason_type` groups.

These are appropriate for the initial, unadjusted look at each factor in
isolation; the regression models below are the primary analysis because
they let both factors and their interaction be modeled jointly.

## Regression (`analysis/regression.py`) — primary analysis

Two logistic regression models are fit with `statsmodels`:

1. **Main effects:** `granted ~ C(reason_type) + C(app_type)`
2. **Interaction:** `granted ~ C(reason_type) * C(app_type)`

Both report, per coefficient: the raw logit coefficient, p-value, odds
ratio (`exp(coef)`), and the 95% CI on the odds ratio. Model fit is
summarized with McFadden's pseudo-R², log-likelihood, and AIC, so the
interaction model can be compared against the main-effects model (e.g.
via a likelihood-ratio test computed as
`2 * (interaction.llf - main.llf)` against a χ² distribution with the
difference in degrees of freedom, if a formal comparison is needed).

Categorical reference levels are the alphabetically-first category
(statsmodels' default `C()` treatment coding) — read the coefficient
table's `term` column to see exactly which levels are being contrasted.

## Reason-quality ML models (`ml/`, RQ4)

- **Classifier:** TF-IDF (unigrams+bigrams, 500 features) →
  `LogisticRegression`, predicting `reason_type` from `reason_text`, as
  a sanity check that the text carries the intended category signal.
- **Regressor:** the same TF-IDF representation → `Ridge` regression,
  predicting the continuous `quality_score` composite (specificity,
  necessity, transparency, privacy language, minus an advertising
  penalty). This model's *predictions* are the operationalization of
  "reason quality" used to test RQ4 — regress `granted` or trust items
  on `quality_score` instead of the categorical `reason_type` and
  compare model fit.

Reported metrics: accuracy, macro-precision/recall/F1, macro ROC-AUC
(classifier); MAE, RMSE, R², and a rank-based calibration correlation
(regressor).

## Explainability (`ml/explain.py`)

Primary method is direct inspection of the linear models' TF-IDF
coefficients (which n-grams push the quality score up or down, and
which n-grams are most associated with each reason-type class) — fast,
deterministic, and easy to sanity-check by eye. When the `shap` package
is installed, a best-effort SHAP explanation (`LinearExplainer` on a
sampled subset) is added as a secondary, model-agnostic cross-check;
this step degrades gracefully (returns a note, not an error) if `shap`
is unavailable or fails on a given model.

## A note on the simulated-data metrics

Because `experiments/simulation.py` generates reason text from a small,
fixed set of templates per `reason_type`, the ML classifier and
regressor will appear to perform near-perfectly on simulated data (this
is expected and confirms the pipeline works end-to-end). Once real
pilot reason text is substituted in, expect meaningfully noisier,
more realistic metrics — that is the actual research signal.
