# POLCA-Lab Unified v3 Experiment Report

## Research goal
Study stochastic generative optimization across prompts, RAG policies, and tool-using agents while measuring sample efficiency and robustness rather than only final reward.

## v3 changes
The benchmark was upgraded from coarse binary trait rewards to weighted continuous metrics with independent per-metric stochastic noise. This avoids the immediate plateaus observed in the earlier version.

## Core metrics
- best held-out reward
- optimization AUC
- evaluations required to reach 0.80
- evaluations required to reach 0.90
- mean and standard deviation over seeds
- candidate memory size
- noise sensitivity

## Recommended experimental comparison
1. Full POLCA
2. POLCA without epsilon-net
3. POLCA without summarization
4. UCB/priority variant

Run `python scripts/run_research_suite.py --iterations 20 --seeds 3` and inspect `results/research_suite.json`.
