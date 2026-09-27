# ReasonLens — Results

**These are results from a simulation-mode run of the pipeline
(n = 2,000 synthetic participants, seed = 42), generated to validate
that the full pipeline — randomization → data → stats → ML → dashboard
— works end to end. They are NOT findings about real user behavior.**
Re-run `python -m experiments.run_all --mode pilot --data-dir <export>`
once real participant data is collected, and this file (or, better, the
dashboard's "Statistical Analysis" tab) should be regenerated from that.

## Overall permission behavior (simulated)

- N = 2,000
- Grant rate: 55.2%
- Denial rate: 44.8%
- Of granted decisions: 45.1% precise, 54.9% approximate

## Hypothesis tests (simulated)

- Grant decision × reason_type: χ² test p = 0.014 (associated, as
  designed into the simulation's behavioral model)
- Grant decision × app_type: χ² test p = 0.213 (not significant at this
  sample size, in this particular simulated draw)

## Logistic regression — main effects (simulated)

Reference levels: `reason_type = advertising`, `app_type = food_delivery`
(alphabetically-first categories under statsmodels' default coding).
Pseudo R² = 0.0062, AIC = 2749.7.

| Term | Odds ratio | p-value |
|---|---|---|
| Intercept | 1.496 | 0.002 |
| reason_type: functional | 1.103 | 0.496 |
| reason_type: none | 0.799 | 0.114 |
| reason_type: privacy_preserving | 1.044 | 0.760 |
| reason_type: vague | 0.728 | 0.028 |
| app_type: local_news | 0.895 | 0.395 |
| app_type: rideshare | 0.921 | 0.520 |
| app_type: wallpaper | 0.763 | 0.037 |

Full coefficient tables (including the interaction model) are in
`data/results/results.json` and rendered in the dashboard's
"Statistical Analysis" tab.

## ML reason-quality model (simulated)

- **Reason-type classifier** (TF-IDF + LogisticRegression): accuracy =
  1.00, macro-F1 = 1.00, macro ROC-AUC = 1.00.
- **Quality-score regressor** (TF-IDF + Ridge): MAE = 0.042, RMSE =
  0.052, R² = 0.969.

These near-perfect scores are expected on simulated data, since
`experiments/simulation.py` draws reason text from a small fixed set of
templates per `reason_type` — the model is essentially memorizing the
templates. This confirms the ML pipeline runs correctly; it says
nothing about how well reason quality can be predicted from the more
varied, noisier text a real pilot would produce (see
`docs/STATISTICAL_METHODS.md`, "A note on the simulated-data metrics").

## Figures

Generated at `data/results/figures/`: `grant_rate_by_reason.png`,
`grant_rate_by_app.png`, `heatmap_reason_app.png`, `trust_by_reason.png`,
`response_time_by_reason.png`. All are also rendered in the Streamlit
dashboard.

## Reproducing this run

```bash
pip install -r requirements.txt
python -m experiments.run_all         # writes data/results/results.json
streamlit run dashboard/app.py        # explore interactively
```
