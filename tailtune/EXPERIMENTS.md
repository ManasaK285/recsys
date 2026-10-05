# Experiment plan

## Baselines

1. Most popular unseen items.
2. SASRec Transformer.

## Main method

Train BT-SR with:

- alpha = 0.05
- alpha = 0.10
- alpha = 0.20
- alpha = 0.30
- alpha = 0.50

## Metrics

Ranking:
- HR@5/10/20
- NDCG@5/10/20
- MRR@5/10/20

Long-tail:
- head exposure
- mid exposure
- tail exposure
- catalog coverage

Representation:
- singular-value spectrum
- effective rank

## Ablations

A. SASRec without BT
B. Random positive pairs
C. Same-target positive pairs
D. Alignment-only auxiliary loss
E. Full Barlow Twins

## Important scientific rule

Do not fabricate results. Run the experiments and report the actual values.
The project's hypothesis is about a controllable trade-off; the data determine
whether and how strongly that trade-off appears on each benchmark.
