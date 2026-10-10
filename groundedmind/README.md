# GroundedMind
 
Statistical Evaluation of Embodied vs. Statistical Concept Learning in Multimodal AI.

This repository is a runnable research prototype inspired by work comparing human and LLM sensory associations. It includes:
- concrete, abstract, and synthetic concepts
- text embedding baseline
- synthetic grounded experiences
- multimodal fusion
- contrastive learning
- human/model distribution comparison
- bootstrap confidence intervals
- permutation testing
- RSA
- ablation-ready experiment scripts
- visualization utilities

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

python experiments/run_association.py
python experiments/run_statistics.py
python experiments/run_grounding.py
```

The initial experiments use lightweight local models and synthetic data. Replace the synthetic human benchmark with real human responses for a research study.

## Folder structure

```text
groundedmind/
├── README.md
├── requirements.txt
├── configs/config.yaml
├── data/concepts/
├── src/data/
├── src/models/
├── src/experiments/
├── src/statistics/
├── src/evaluation/
├── experiments/
└── results/
```
