# TrustLens
Modeling Perceived AI Authorship and Human Agreement with AI-Generated Judgments.

This is a compact, runnable research prototype inspired by the PLOS ONE paper:
https://doi.org/10.1371/journal.pone.0353391

Research questions:
1. Can human vs AI authorship be predicted from text?
2. Which linguistic features affect perceived AI authorship?
3. Does perceived authorship add information when predicting agreement?

The project is self-contained and generates a synthetic dataset so it runs without external data. Synthetic labels are NOT human-subject findings. Replace data/raw/trustlens.csv with appropriately licensed real data for research.

Setup:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Run:
python scripts/run_all.py

Outputs:
results/metrics/
results/figures/

Suggested research extension:
replace the synthetic data with the paper's released data, use participant/scenario-aware splits, and add mixed-effects statistical analysis.
