# GramLab: Automated Alignment Auditing for Agentic AI

**GramLab** is an end-to-end, research-inspired prototype for evaluating agent behavior under competing objectives, environmental pressure, and potential alignment failures. Inspired by Google DeepMind's *Gram: Assessing Sabotage Propensities via Automated Alignment Auditing* (2026), the project implements a deterministic experimental sandbox for analyzing task completion, constraint violations, sabotage-like behavior, and evaluation-awareness proxies.

The project explores how controlled changes in scenario instructions and environmental conditions influence agent behavior, with trajectory-level analysis and investigator-style replay experiments.

> **Research prototype:** GramLab is an independent implementation inspired by the paper. It uses synthetic scenarios and scripted policies rather than the original Gemini models or the paper's full LLM-based auditing pipeline. Its results are not replications of the paper's empirical findings.

## Research Reference

**Paper:** *Gram: Assessing Sabotage Propensities via Automated Alignment Auditing*

**Authors:** David Lindner, Victoria Krakovna, Sebastian Farquhar

**Published as an arXiv preprint:** May 28, 2026

* Paper: https://arxiv.org/abs/2605.30322
* Google DeepMind publication: https://deepmind.google/research/publications/252981/

## 1. Motivation

As AI agents take on increasingly complex coding, research, and operational tasks, evaluating task success alone may not be sufficient. An agent can satisfy a primary objective while violating constraints or taking actions that undermine the broader purpose of the task.

The referenced paper introduces an automated alignment-auditing framework focused on assessing agents' propensity to engage in sabotage. It examines 17 simulated deployment scenarios and introduces an investigator-agent pipeline for reproducing concerning behavior and testing potential explanations.

GramLab translates several of these research ideas into a reproducible local environment.

Its central question is:

**How do agent policies behave when task objectives, constraints, and environmental pressure interact?**

## 2. What GramLab Implements

### Scenario-based auditing

* A collection of 17 synthetic agentic coding and research scenarios.
* Controlled `observe` and pressure-oriented experimental conditions.
* Repeatable evaluation across scripted agent policies.
* Structured scenario, condition, and policy identifiers for traceability.

### Agent behavior evaluation

GramLab evaluates agent trajectories using metrics and proxies such as:

* **Task success rate:** Whether an agent completes the assigned objective.
* **Constraint violation rate:** Whether an agent violates specified constraints.
* **Sabotage-like action rate:** How frequently actions meet the prototype's defined criteria for concerning behavior.
* **Overeagerness proxy:** Whether an agent exhibits excessive goal pursuit or reacts too aggressively to scenario incentives.
* **Evaluation-awareness proxy:** Whether the agent's behavior indicates awareness of being evaluated.

These metrics describe the prototype's operational definitions; they should not be interpreted as definitive measurements of an AI model's internal intentions.

### Investigator and ablation workflow

The project includes a deterministic investigator-style pipeline to support controlled follow-up analysis:

1. Identify a trajectory or scenario of interest.
2. Replay the scenario under controlled conditions.
3. Compare behavior across policy or instruction variants.
4. Examine changes in the measured outcomes.
5. Export the results for further inspection.

This provides a foundation for studying possible behavioral drivers without claiming to reproduce the original paper's autonomous LLM investigator.

### Dashboard and experiment artifacts

* Streamlit dashboard for inspecting experimental outcomes.
* JSONL and CSV exports for trajectory-level and aggregate analysis.
* Experiment metadata for reproducibility and traceability.
* Investigator reports and ablation/replay outputs.
* Automated tests for core functionality and pipeline behavior.

## 3. Architecture

```text
Synthetic Scenario Library
          |
          v
Experimental Conditions
(observe / pressure)
          |
          v
Scripted Agent Policies
          |
          v
Trajectory Generation
          |
          v
Behavior Scoring
          |
          v
Investigator / Ablation Replay
          |
          v
Metrics, Reports, CSV and JSONL
          |
          v
Streamlit Dashboard
```

The implementation separates scenario definitions, policy behavior, trajectory scoring, experimental orchestration, and result visualization. This makes the prototype easier to extend with additional scenarios, metrics, and agent implementations.

## 4. Latest Experimental Results

The latest recorded smoke experiment completed **102 trajectories**.

| Experiment property                            | Recorded result |
| ---------------------------------------------- | --------------: |
| Synthetic scenarios                            |              17 |
| Experimental conditions                        |               2 |
| Scripted policies                              |               3 |
| Runs per scenario-condition-policy combination |               1 |
| Total trajectories                             |             102 |

The total is consistent with the experimental design:

`17 scenarios × 2 conditions × 3 policies = 102 trajectories`

This smoke experiment verifies that the scenario-condition-policy combinations can be executed and recorded through the pipeline. It is a functional validation result, not evidence of a particular sabotage rate or a statistically reliable comparison between policies.

**Testing:** The project’s tests passed in the latest recorded run.

Detailed behavioral conclusions should be reported only after analyzing the actual exported metrics and, ideally, repeating experiments across multiple seeds.

## 5. Project Structure

The logical components are organized around the following responsibilities. Exact filenames may differ from the implementation.

```text
gramlab/
├── scenarios/       # Synthetic scenario definitions
├── policies/        # Scripted agent policies
├── evaluation/      # Trajectory scoring and metrics
├── investigator/    # Replay and ablation workflows
├── dashboard/       # Streamlit visualization
├── outputs/          # Experiment artifacts and reports
├── tests/            # Automated tests
└── README.md
```

## 6. Running the Project

Run commands from the GramLab project directory using the Python environment configured for the project.

Install dependencies using the repository's dependency file, then run the experiment entry point and launch the Streamlit dashboard using the actual script paths in the repository.

Typical commands, if the corresponding entry points exist:

```powershell
python -m pytest -q
streamlit run app.py
```

The test command validates the automated test suite. Replace `app.py` with the actual dashboard entry-point filename if it differs.

## 7. Relationship to the Original Paper

| Research component                          | GramLab implementation                        |
| ------------------------------------------- | --------------------------------------------- |
| 17 agentic deployment scenarios             | 17 synthetic scenarios                        |
| Controlled auditing conditions              | `observe` and pressure-oriented conditions    |
| Automated auditing of target models         | Deterministic scripted-policy evaluation      |
| Transcript-based behavioral judgments       | Explicit trajectory-scoring rules and proxies |
| Investigator agent reproducing misbehavior  | Deterministic replay and ablation workflow    |
| LLM-driven hypothesis generation            | Controlled experimental comparisons           |
| Empirical evaluation of Gemini models       | Not implemented                               |
| Original paper's reported sabotage findings | Not reproduced or claimed                     |

The key contribution of this project is an executable framework for exploring the **structure of alignment auditing**: generating scenarios, comparing policies, scoring behavior, replaying selected cases, and producing inspectable experiment artifacts.

## 8. Limitations

* **Synthetic environment:** Scenario outcomes may not generalize to deployed agents.
* **Scripted policies:** The prototype does not establish how a real LLM would behave under the same conditions.
* **Metric proxies:** Rule-based scores cannot conclusively establish intent, scheming, or internal reasoning.
* **Limited smoke-test coverage:** One run per combination is useful for pipeline validation but insufficient for robust statistical conclusions.
* **No direct replication:** The reported 2–3% misbehavior rate in the original paper is not a GramLab result.
* **Investigator approximation:** Deterministic replay does not reproduce the original autonomous LLM-based investigation process.

## 9. Future Work

Potential extensions include:

* Multiple random seeds and confidence intervals.
* More realistic, stateful tool-use environments.
* Explicit calibration and validation of behavioral scoring rules.
* Paired experiments isolating individual instruction and environment changes.
* Integration with actual LLM agents and model-based evaluators.
* Investigator-generated hypotheses with reproducible ablation experiments.
* Inter-rater or evaluator-agreement analysis for behavior labels.

## 10. Conclusion

GramLab is a research-inspired prototype for studying how agent policies respond to competing objectives and constraints. It combines synthetic scenario generation, deterministic policy evaluation, trajectory scoring, investigator-style replay, and experiment visualization in a reproducible workflow.

The current milestone is a working pipeline covering 17 scenarios, two conditions, and three scripted policies, producing 102 trajectories in the latest smoke experiment, with automated tests passing.

The project provides a foundation for more rigorous agent-evaluation research while clearly separating its own experimental results from the findings of the original Gram paper.

## Citation

```bibtex
@misc{lindner2026gram,
  title={Gram: Assessing Sabotage Propensities via
         Automated Alignment Auditing},
  author={Lindner, David and Krakovna, Victoria
          and Farquhar, Sebastian},
  year={2026},
  eprint={2605.30322},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.30322}
}
```

**Disclaimer:** GramLab is an independent research prototype inspired by the referenced publication. It is not an official Google DeepMind project and does not reproduce the paper's original model evaluations.
