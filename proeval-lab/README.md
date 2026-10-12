# ProEval-Lab: Proactive Evaluation of AI Systems Under Limited Budgets

**A research-oriented implementation inspired by the ProEval paper**

ProEval-Lab is a research prototype for evaluating AI systems efficiently under limited evaluation budgets. It combines Gaussian Process (GP) surrogate modeling, uncertainty-aware sampling, risk-based failure discovery, and hybrid evaluation strategies to estimate overall system performance while identifying potentially problematic regions of the evaluation space.

The project is inspired by the central goals of the ProEval paper: reducing evaluation costs, estimating performance over a broader population of cases, and proactively identifying regions where an AI system may fail.

The current implementation provides an executable experimental pipeline, multiple evaluation strategies, result reporting, an interactive dashboard, an API layer, and automated tests.

> **Research status:** This is a working prototype inspired by ProEval, not an exact reproduction of the paper. Current experiments use synthetic evaluation data. The paper's full transfer-learning and Bayesian-quadrature methodology has not yet been independently reproduced.

---

## 1. Motivation

Evaluating an AI system across a large population of inputs can be expensive. Exhaustively evaluating every input may be impractical when each evaluation requires model inference, human review, external tools, or other costly resources.

A useful evaluation framework should address three questions:

1. **Performance estimation:** How accurately can we estimate overall model performance from a limited evaluation budget?
2. **Failure discovery:** Can we identify inputs or regions where the model is likely to fail?
3. **Evaluation efficiency:** Which evaluation cases should we prioritize to obtain the most useful information?

ProEval-Lab explores these questions through surrogate modeling and budget-aware evaluation strategies.

## 2. Research alignment with the ProEval paper

The project is organized around the following research concepts.

| Research concept                               | Implementation status                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------- |
| Budget-constrained evaluation                  | Implemented experimental pipeline with a configurable evaluation budget |
| Gaussian Process surrogate modeling            | Implemented                                                             |
| Uncertainty-aware sampling                     | Implemented                                                             |
| Risk-oriented failure discovery                | Implemented                                                             |
| Hybrid evaluation strategy                     | Implemented                                                             |
| Random-sampling baseline                       | Implemented                                                             |
| Performance estimation and error comparison    | Evaluated in prototype experiments                                      |
| Failure-region identification                  | Evaluated using synthetic labels                                        |
| Transfer-aware GP modeling                     | Not yet faithfully reproduced                                           |
| Bayesian quadrature for performance estimation | Not yet faithfully reproduced as the paper's complete method            |
| Real-world benchmark validation                | Future work                                                             |

The implementation is intended to establish an executable foundation for studying the paper's evaluation objectives and comparing alternative sampling policies.

## 3. System architecture

The current prototype includes the following components:

* **Experiment pipeline:** Generates or loads evaluation cases, applies the configured budget, runs the sampling strategy, and collects results.
* **GP surrogate model:** Learns an approximation of the underlying performance or risk landscape from observed cases.
* **Sampling strategies:** Select evaluation cases using random, uncertainty-based, risk-based, or hybrid policies.
* **Evaluation and reporting:** Compares estimated performance against the reference performance and evaluates failure-discovery behavior.
* **Interactive dashboard:** Provides a user-facing interface for exploring evaluation results.
* **API layer:** Exposes the pipeline through a FastAPI application.
* **Automated tests:** Checks core functionality and a smoke-test execution of the pipeline.
* **Containerization:** Includes Docker support for a reproducible deployment workflow.

### Evaluation workflow

1. Create a synthetic evaluation population.
2. Establish reference performance and failure labels.
3. Set an evaluation budget.
4. Select cases using the chosen sampling strategy.
5. Fit or update the surrogate model using observed evaluations.
6. Estimate population-level performance and prioritize potential failures.
7. Calculate evaluation metrics and generate reports.

## 4. Implemented evaluation strategies

### Random sampling

Selects cases without prioritizing uncertainty or predicted risk. This serves as a baseline for determining whether adaptive strategies improve evaluation efficiency.

### Uncertainty sampling

Prioritizes cases where the surrogate model is uncertain. The objective is to obtain informative observations that improve the model's approximation of the evaluation landscape.

### Risk-based sampling

Prioritizes cases predicted to belong to a failure-prone region. This strategy emphasizes identifying failures rather than minimizing overall performance-estimation error.

### Hybrid sampling

Combines uncertainty and risk signals to balance information gathering with targeted failure discovery.

These strategies represent the prototype's current policy implementations. Their exact correspondence to any named algorithm in the paper should be established through a method-by-method comparison.

## 5. Experimental setup

The initial experiment uses a synthetic evaluation environment.

| Parameter             |    Configuration |
| --------------------- | ---------------: |
| Evaluation population |        300 cases |
| Evaluation budget     |         60 cases |
| Budget fraction       |              20% |
| Random seeds          |                3 |
| Sampling strategies   |                4 |
| Surrogate model       | Gaussian Process |
| Test framework        |           pytest |

The evaluation population contains a reference performance landscape and synthetic failure labels. These allow the experiment to measure estimation accuracy and failure-discovery behavior without requiring an expensive external model evaluation.

**Important:** Synthetic data is useful for validating the mechanics of an evaluation algorithm, but it does not establish that the method generalizes to production AI systems or reproduces the paper's reported results.

## 6. Initial experimental results

The first completed experiment produced the following prototype results.

| Strategy    | Performance estimation error | Failure precision | Failure recall |
| ----------- | ---------------------------: | ----------------: | -------------: |
| Random      |                       0.0097 |                 — |              — |
| Uncertainty |                       0.0101 |                 — |              — |
| Risk        |                       0.0309 |             97.2% |          30.7% |
| Hybrid      |                   **0.0055** |                 — |              — |

*Dashes indicate metrics not included in the recorded results summarized here; they should not be interpreted as zero.*

### Key findings

**1. Hybrid sampling achieved the lowest recorded estimation error.**

The hybrid strategy obtained an estimation error of 0.0055, compared with 0.0097 for random sampling and 0.0101 for uncertainty sampling.

This is approximately a 43.3% reduction in estimation error relative to random sampling in this experiment.

**2. Risk-based sampling favored targeted failure discovery.**

The risk strategy achieved 97.2% precision and 30.7% recall on the synthetic failure labels. This suggests that its selected cases were highly concentrated among actual synthetic failures, while its coverage of all failures remained limited.

**3. Estimation and failure discovery are different objectives.**

The risk strategy had the highest recorded estimation error, while the hybrid strategy had the lowest. This illustrates why an evaluation framework should assess both population-level estimation and targeted failure discovery rather than relying on one metric.

**4. These are preliminary prototype findings.**

The results are specific to the synthetic dataset, selected seeds, budget, and implementation. More repetitions, uncertainty intervals, ablation studies, and external benchmarks are needed before making general claims about the relative effectiveness of these methods.

## 7. Metrics

The current experimental analysis focuses on two categories.

### Performance estimation

* **Estimation error:** Difference between estimated and reference population performance, using the project's configured error definition.
* **Budget efficiency:** Amount of evaluation performed relative to the total evaluation population.
* **Cross-seed stability:** Variation in results across random seeds.

### Failure discovery

* **Precision:** Fraction of identified cases that are actual failures.
* **Recall:** Fraction of actual failures that are identified.

For a stronger paper-aligned evaluation, the next iteration should report the exact error metric, mean and standard deviation across seeds, confidence intervals, and failure-discovery curves as the evaluation budget increases.

## 8. Installation

Requirements: Python and pip, plus the dependencies declared by the project.

From the repository root:

```powershell
cd proeval-lab

python -m pip install -r requirements.txt
```

If the project uses editable installation metadata, install it with:

```powershell
python -m pip install -e .
```

Run the tests:

```powershell
python -m pytest -q
```

### Test status

The current test suite completes successfully:

```text
6 passed, 9 warnings in 3.41s
```

The warnings originate from scikit-learn's Gaussian Process kernel optimization. They are non-fatal convergence warnings indicating that an estimated noise parameter reached its configured lower bound.

## 9. Running experiments

Run the experiment script from the repository root using the options supported by the current implementation. The initial experiment configuration is 300 cases, a budget of 60, and three random seeds.

For example, if the script exposes these command-line options:

```powershell
python scripts/run_experiments.py --n-cases 300 --budget 60 --seeds 3
```

Check the script's argument parser or `--help` output if your local version uses different option names.

Review the generated result files and compare all strategies using the same dataset, reference labels, budget, and seeds.

## 10. Dashboard and API

The project includes an interactive Streamlit dashboard and a FastAPI application.

These components provide interfaces for exploring experiments and integrating the evaluation pipeline into other applications. Their presence in the repository does not by itself establish production readiness; deployment, input validation, monitoring, and security should be assessed separately.

Start either interface using the entry-point files and commands documented in the repository. Verify the actual script names before running them.

## 11. Limitations

The current prototype has several important limitations:

* **Synthetic evaluation data:** Results do not yet demonstrate performance on real AI systems.
* **Methodological fidelity:** Transfer-aware modeling and the paper's complete Bayesian-quadrature approach remain to be reproduced and validated.
* **Limited experimental scale:** The initial study uses 300 cases, a budget of 60, and three seeds.
* **Limited uncertainty reporting:** Confidence intervals and comprehensive stability analyses remain to be added.
* **Baseline coverage:** More ablations and established evaluation baselines would strengthen the comparison.
* **Generalization:** Results from this experimental configuration should not be extrapolated to other datasets, model families, or production settings without validation.

## 12. Roadmap

### Phase 1 — Working prototype: completed

* [x] Implement the core evaluation pipeline.
* [x] Implement GP surrogate modeling.
* [x] Add random, uncertainty, risk, and hybrid strategies.
* [x] Add experiment reporting and evaluation metrics.
* [x] Include dashboard and API components.
* [x] Add automated tests.
* [x] Pass the current six-test suite.
* [x] Run the initial 300-case, 60-evaluation-budget experiment.

### Phase 2 — Experimental rigor

* [ ] Run repeated experiments with more random seeds.
* [ ] Report confidence intervals and estimation-error distributions.
* [ ] Evaluate performance across different failure rates and population sizes.
* [ ] Conduct ablation studies for the uncertainty and risk components.
* [ ] Plot estimation error and failure recall against evaluation budget.
* [ ] Document all metric definitions and experiment configurations.

### Phase 3 — Faithful paper implementation

* [ ] Map each implemented component to the corresponding paper section and equation.
* [ ] Implement the paper's transfer-aware Gaussian Process methodology.
* [ ] Implement and validate the specified Bayesian-quadrature estimator.
* [ ] Reproduce the paper's experimental protocol and baselines.
* [ ] Compare results against published findings where the experimental conditions are comparable.

### Phase 4 — External validation

* [ ] Evaluate on real model outputs and labeled benchmark datasets.
* [ ] Measure the cost-quality trade-off under realistic evaluation constraints.
* [ ] Validate failure discovery on held-out data.
* [ ] Improve experiment reproducibility and deployment documentation.

## 13. Reproducibility

For every reported experiment, record:

* Dataset or synthetic-data configuration.
* Model and surrogate hyperparameters.
* Sampling policy and acquisition settings.
* Evaluation budget and population size.
* Random seeds.
* Metric definitions.
* Software and dependency versions.
* Generated outputs and experiment configuration.

Keeping these details together makes results easier to reproduce and helps separate implementation improvements from differences in experimental setup.

## 14. Research significance

ProEval-Lab provides an initial experimental framework for studying how surrogate modeling and adaptive sampling can reduce evaluation effort while balancing performance estimation against failure discovery.

The initial findings suggest that a hybrid strategy can be promising for population-level estimation, while a risk-focused strategy can prioritize likely failures. These findings motivate a more rigorous investigation of the trade-offs between estimation accuracy, failure coverage, and evaluation cost.

The next major milestone is to move from a working, synthetic-data prototype to a methodologically faithful implementation of the ProEval paper and validate it on external benchmarks.

---

**Project status:** Working research prototype; initial experiments completed; automated tests passing; exact paper reproduction and real-world validation remain ongoing.

**Disclaimer:** All experimental metrics in this README describe the current project implementation. They are not results reported by the ProEval paper unless explicitly verified against the original publication.
