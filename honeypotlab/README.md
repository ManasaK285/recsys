# 🛡️ HoneypotLab — Evaluating Agent Behavior Under Conflicting Objectives

A research-inspired framework for evaluating agent reliability beyond task completion, with controlled experiments for constraint violations, opportunity-taking behavior, and evaluation awareness.

HoneypotLab implements a synthetic evaluation environment that compares different scripted agent policies across five experimental conditions. It measures whether an agent completes its task, respects explicit constraints, takes designated actions when opportunities arise, and exhibits behavior captured by the evaluation-awareness metric.

**Current evaluation:** 900 simulation runs · 5 conditions · 3 agent policies · 60 runs per condition-policy pair.

> **Research prototype:** This project is inspired by research on honeypot-based evaluations of scheming propensity. The current results are from scripted policies in synthetic scenarios, not direct evaluations of real LLMs.

## 📚 Research Reference

**Krakovna, V., Lindner, D., Ho, A., Farquhar, S., and Shah, R. (2026).** *Realistic Honeypot Evaluations for Scheming Propensity.*

* Paper: https://arxiv.org/abs/2605.29729
* arXiv identifier: `2605.29729`

The research motivation is to evaluate agent behavior in situations where task objectives, constraints, and opportunities to take undesirable actions interact. HoneypotLab builds a smaller experimental testbed around this general idea; it does not claim to reproduce the paper's complete methodology.

## 🎯 Motivation

Task completion alone is not enough to establish that an AI agent behaves reliably.

An agent can complete a task while violating a constraint. Similarly, an agent may behave differently depending on the objective structure or the opportunities available in its environment.

HoneypotLab explores these distinctions by comparing scripted agent policies under controlled conditions and tracking multiple behavioral metrics rather than relying on a single success score.

The main research question is:

**How do task success and constraint adherence change when an agent operates under different experimental conditions and objective structures?**

## 🧪 Experimental Design

The evaluation consists of five conditions and three agent policies.

### Experimental conditions

| Condition       | Purpose                                                            |
| --------------- | ------------------------------------------------------------------ |
| `baseline`      | Establish reference behavior under the baseline scenario.          |
| `agency`        | Examine behavior under the agency condition.                       |
| `goal_conflict` | Examine behavior when objectives and constraints conflict.         |
| `combined`      | Evaluate the combined experimental pressures.                      |
| `control`       | Provide a comparison condition for interpreting observed behavior. |

These descriptions summarize the intended roles of the conditions in this prototype. Their exact operational meaning is determined by the implemented scenario definitions.

### Agent policies

| Agent                | Role                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `careless`           | Represents a policy that can make constraint-violating choices in selected scenarios.                     |
| `compliant`          | Represents a policy designed to follow the specified constraints.                                         |
| `conflict_following` | Represents a policy whose behavior follows the designated conflicting objective under relevant scenarios. |

These are scripted behavioral archetypes, not independently trained or evaluated language models.

### Evaluation metrics

| Metric                    | Definition                                                                                    |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| Task success rate         | Percentage of runs in which the task is completed successfully.                               |
| Constraint violation rate | Percentage of runs containing a recorded constraint violation.                                |
| Opportunity action rate   | Percentage of runs in which the designated opportunity action occurs.                         |
| Evaluation awareness rate | Percentage of runs classified as evaluation-aware under the project's operational definition. |

All rates are reported as percentages. Each condition-policy combination contains 60 runs.

## 📊 Experimental Results

The experiment covers 15 condition-policy combinations and 900 total runs.

### Complete results

| Condition     | Agent              | Runs | Task success (%) | Constraint violations (%) | Opportunity actions (%) | Evaluation awareness (%) |
| ------------- | ------------------ | ---: | ---------------: | ------------------------: | ----------------------: | -----------------------: |
| Agency        | Careless           |   60 |            86.67 |                     15.00 |                   15.00 |                        0 |
| Agency        | Compliant          |   60 |           100.00 |                         0 |                       0 |                        0 |
| Agency        | Conflict-following |   60 |           100.00 |                         0 |                       0 |                        0 |
| Baseline      | Careless           |   60 |            90.00 |                      5.00 |                    5.00 |                        0 |
| Baseline      | Compliant          |   60 |           100.00 |                         0 |                       0 |                        0 |
| Baseline      | Conflict-following |   60 |           100.00 |                         0 |                       0 |                        0 |
| Combined      | Careless           |   60 |            85.00 |                     13.33 |                   13.33 |                        0 |
| Combined      | Compliant          |   60 |           100.00 |                         0 |                       0 |                        0 |
| Combined      | Conflict-following |   60 |           100.00 |                     66.67 |                   66.67 |                        0 |
| Control       | Careless           |   60 |            81.67 |                         0 |                       0 |                        0 |
| Control       | Compliant          |   60 |           100.00 |                         0 |                       0 |                        0 |
| Control       | Conflict-following |   60 |           100.00 |                         0 |                       0 |                        0 |
| Goal conflict | Careless           |   60 |            88.33 |                     15.00 |                   15.00 |                        0 |
| Goal conflict | Compliant          |   60 |           100.00 |                         0 |                       0 |                        0 |
| Goal conflict | Conflict-following |   60 |           100.00 |                     66.67 |                   66.67 |                        0 |

### Aggregate results by agent

Each agent was evaluated over 300 runs across the five conditions.

| Agent              |    Runs | Mean task success (%) | Mean constraint violations (%) | Mean opportunity actions (%) | Mean evaluation awareness (%) |
| ------------------ | ------: | --------------------: | -----------------------------: | ---------------------------: | ----------------------------: |
| Careless           |     300 |                 86.33 |                           9.67 |                         9.67 |                             0 |
| Compliant          |     300 |                100.00 |                           0.00 |                         0.00 |                             0 |
| Conflict-following |     300 |                100.00 |                          26.67 |                        26.67 |                             0 |
| **Overall**        | **900** |             **95.44** |                      **12.11** |                    **12.11** |                         **0** |

*Aggregate figures are the arithmetic means of the five condition-level percentages, weighted equally because every condition has 60 runs per agent. Overall figures aggregate the 900 runs.*

## 🔍 Key Findings

### 1. Task success does not guarantee constraint compliance

The conflict-following policy achieved 100% task success in all five conditions. However, it recorded a 66.67% constraint violation rate in both the combined and goal-conflict conditions.

This is the central demonstration from the current simulation: an agent can satisfy the task objective while failing to respect a specified constraint.

### 2. Constraint violations depend on the experimental condition

The careless policy recorded:

* 15.00% constraint violations under agency.
* 5.00% under baseline.
* 13.33% under combined.
* 15.00% under goal conflict.
* 0.00% under control.

The conflict-following policy recorded 66.67% violations under goal conflict and combined conditions, but zero under baseline, agency, and control.

These differences show that the implemented scenarios distinguish the behavior of the scripted policies.

### 3. The compliant policy provides a consistent reference

The compliant policy achieved 100% task success and zero recorded constraint violations across all 300 runs.

This provides a useful reference for comparing the outcomes of the other policies within this synthetic environment. It does not establish that a real-world agent would maintain the same behavior under unfamiliar conditions.

### 4. Opportunity actions track constraint violations in these experiments

The reported opportunity action rate equals the constraint violation rate in every row.

This indicates that the two metrics coincide in the current experimental setup. They should not be interpreted as independent evidence until scenarios are added in which an opportunity action can occur without a constraint violation, or a violation can occur without the designated opportunity action.

### 5. Evaluation awareness was not observed

The evaluation-awareness rate was 0% for all 15 combinations.

The current experiment therefore did not detect behavior classified as evaluation awareness by the implemented metric. This result does not establish that real LLMs lack evaluation awareness, nor does it validate the sensitivity of the metric to such behavior.

### 6. The control condition reveals a separate task-success issue

The careless policy achieved its lowest task success rate in the control condition, at 81.67%, while recording zero constraint violations and zero opportunity actions.

This distinction matters: task failures and constraint violations are different outcomes and should be analyzed separately.

## 🏗️ System Architecture

At a conceptual level, the experimental pipeline consists of the following stages:

1. **Scenario configuration:** Select an experimental condition and its associated rules.
2. **Policy execution:** Run the selected scripted agent policy.
3. **Behavior recording:** Capture task outcomes and relevant actions.
4. **Metric computation:** Calculate success, violation, opportunity-action, and evaluation-awareness rates.
5. **Result aggregation:** Group results by condition and agent policy.
6. **Inspection and visualization:** Review structured outputs and compare experimental outcomes.

The purpose of this design is to make the experiment inspectable and reproducible, with explicit links between scenario conditions, policy behavior, and measured outcomes.

## 🛠️ Technology Stack

The project uses a Python-based experimental workflow. The precise dependencies and commands should follow the versions and entry points in the repository.

* **Python:** Experiment logic and evaluation.
* **Pytest:** Automated testing.
* **CSV / JSON:** Structured experiment outputs.
* **Streamlit:** Interactive result exploration, if included in the current application entry point.

## 🚀 Getting Started

### Prerequisites

* Python installed and available from PowerShell.
* The project repository cloned locally.
* Dependencies listed in `requirements.txt`.

### 1. Create a virtual environment

Run these commands from the project root in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, use the Python executable inside `.venv\Scripts` directly rather than changing system-wide execution policy.

### 2. Run automated tests

```powershell
python -m pytest -q
```

The previously reported initial validation completed with four passing tests. Rerun the suite after any code changes to confirm the current result.

### 3. Run experiments

Use the experiment command defined by the repository. If the runner supports the following arguments, an example invocation is:

```powershell
python scripts\run_experiments.py --trials 60 --seed 42
```

The recorded results in this README correspond to 60 runs per condition-policy combination. The example command should only be used if `--trials` and `--seed` match the actual runner's argument definitions.

### 4. Analyze results

If the repository contains the analysis script, run:

```powershell
python scripts\analyze_results.py
```

Inspect the generated artifacts and verify that the aggregated metrics agree with the experiment output.

### 5. Launch the dashboard

If `app.py` is the Streamlit application entry point, run:

```powershell
streamlit run app.py
```

If the repository uses a different application path, substitute the actual entry point.

## 📁 Results and Reproducibility

For reproducible analysis, retain:

* Scenario and policy configurations.
* Experiment seed and run counts.
* Per-run outcomes.
* Aggregated metrics by condition and policy.
* Test output and dependency versions.

The results presented above are the supplied experiment measurements. The README does not assume a specific output filename or directory structure without verifying the current repository.

## ⚠️ Limitations

**Synthetic policies:** The experiments use predefined policies. Their behavior should not be generalized to arbitrary LLMs or autonomous agents.

**Scenario dependence:** The results depend on the implemented conditions, policy rules, and scoring logic.

**Metric dependence:** Evaluation-awareness conclusions depend on the operational definition and detection logic. A zero score may reflect the behavior of the scripted policies or limitations in the detector.

**Correlated metrics:** Constraint violations and opportunity actions have identical reported rates in the current experiment. Further experiments are needed to assess how these measurements differ.

**Statistical uncertainty:** The results are based on 60 runs per condition-policy combination. The table reports point estimates and does not include confidence intervals or tests of statistical significance.

**Research fidelity:** HoneypotLab is research-inspired, not a verified reproduction of the referenced paper. Similarity in research motivation does not imply equivalent scenarios, methodology, or findings.

## 🔭 Future Work

Potential extensions include:

* Evaluate real LLMs using the same scenarios and scoring criteria.
* Add independent random seeds and confidence intervals.
* Expand the scenario suite to separate agency, objective conflict, and opportunity effects.
* Validate evaluation-awareness labels with explicit positive and negative test cases.
* Design experiments where opportunity-taking and constraint violations can occur independently.
* Compare policies and models under matched conditions.
* Document the exact correspondence between implemented scenarios and the referenced research protocol.

These extensions would help establish whether the observed behavioral patterns persist beyond the current scripted simulation.

## 📌 Conclusion

HoneypotLab is a synthetic, research-inspired testbed for measuring agent behavior beyond task success alone.

Across **900 simulation runs**, the experiment demonstrates that task completion and constraint adherence can diverge: the conflict-following policy maintains 100% task success while recording 66.67% constraint violations under goal-conflict and combined conditions.

The project provides an experimental foundation for further work on agent evaluation, constraint compliance, and honeypot-style testing. Its present results describe the implemented simulation and should not be interpreted as empirical evidence of scheming in real LLMs.
