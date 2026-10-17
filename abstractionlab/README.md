# AbstractionLab

**A reproducible research-inspired sandbox for studying observer-defined representations and behavioral agreement in toy computational systems.**

AbstractionLab investigates how different discretization maps represent the same continuous trajectory and how different mathematical implementations can produce similar outputs despite differences in their internal states.

The project is inspired by Alexander Lerchner's essay *The Abstraction Fallacy: Why AI Can Simulate But Not Instantiate Consciousness* (March 10, 2026). It translates a narrow set of ideas about observer-defined symbolic representations and behavioral versus implementation-level differences into controlled computational experiments.
 
> **Scientific limitation:** AbstractionLab does not detect, prove, or disprove consciousness. It does not implement a theory of subjective experience, reproduce the essay's full philosophical argument, or evaluate real AI systems or physical hardware. All reported findings concern the project's mathematical toy simulations.

## Reference

Alexander Lerchner, *The Abstraction Fallacy: Why AI Can Simulate But Not Instantiate Consciousness* (March 10, 2026).

* Paper record: https://philpapers.org/rec/LERTAF

The essay discusses the distinction between behavioral simulation and physical instantiation, including the role of observers in mapping processes to symbolic descriptions. This project uses those ideas as motivation for limited computational questions; its experiments do not establish the essay's philosophical claims.

## Research Questions

AbstractionLab explores three questions:

1. How much can symbolic labels differ when different discretization maps observe the same underlying state?
2. Can different symbolic representations produce similar policy outputs despite substantial disagreement in their labels?
3. Can two mathematical implementations exhibit similar observable outputs while maintaining different internal state trajectories?

## Experiments

### Experiment A: Observer-Defined Representation Sensitivity

A bounded continuous state evolves toward a configurable target while receiving small disturbances. Coarse and balanced discretization maps assign symbolic labels to the same trajectory.

The experiment measures symbolic disagreement, policy-action agreement, and target reachability.

The updated environment includes an explicit target-attraction term. This corrects an earlier modeling issue in which a constant positive drift caused the state to settle around 0.36 instead of approaching the target of 0.75.

**This modification changes the toy environment's dynamics; it is not evidence for or against the paper's philosophical argument.**

### Experiment B: Quantized vs. Continuous Toy Dynamics

Two mathematical implementations process the same input sequence:

* **Quantized-state implementation:** Uses discretized numerical state updates.
* **Continuous-state implementation:** Preserves finer numerical precision.

The experiment compares their output agreement and mean absolute internal-state difference.

This comparison illustrates how output similarity and internal numerical similarity are distinct measurements. It does not establish general behavioral equivalence or make claims about biological or physical substrates.

## Experimental Results

The following results are from the recorded experiment with seed 42 and 500 steps.

### Experiment A — Representation Sensitivity

| Metric                                   |     Result |
| ---------------------------------------- | ---------: |
| Random seed                              |         42 |
| Steps                                    |        500 |
| Symbol disagreement, coarse vs. balanced |      97.4% |
| Action agreement, coarse vs. balanced    |      99.0% |
| Target                                   |     0.7500 |
| Final state                              |     0.7312 |
| Absolute final distance from target      |     0.0188 |
| Target tolerance                         |       0.05 |
| Minimum target distance                  | 0.00002622 |
| Task success                             |     `true` |

**Interpretation**

* **97.4% symbol disagreement:** The two maps frequently assign different symbolic labels to the same underlying trajectory. The result is sensitive to the maps' partition boundaries and label definitions.
* **99.0% action agreement:** Despite differing labels, the simple symbolic policies produce the same action for most evaluated states.
* **Successful target reachability:** The final state is approximately 0.0188 from the target, within the configured tolerance of 0.05. The experiment therefore satisfies its defined task-success criterion.

The high symbolic-disagreement rate should not be interpreted as a universal measure of representational difference. Different maps may use different numbers of bins, so their integer labels do not necessarily share a common semantic scale.

**Important methodological limitation:** The symbolic actions are calculated from the shared state trajectory but do not control the environment's state transitions. Action agreement is therefore a comparison of policy outputs, not a closed-loop control result.

### Experiment B — Quantized vs. Continuous Dynamics

| Metric                                  | Result |
| --------------------------------------- | -----: |
| Steps                                   |    500 |
| Output agreement                        |  86.4% |
| Mean absolute internal-state difference | 0.1625 |

**Interpretation**

The two toy implementations produce matching outputs for 86.4% of the evaluated steps, while their internal numerical states differ by a mean absolute difference of approximately 0.1625.

This demonstrates that, under the selected input sequence and output definition, output agreement does not require identical internal numerical states. It does not establish equivalence under other inputs, parameters, tasks, or implementations.

## What Changed in the Updated Version

* **Target-seeking dynamics:** Added an explicit target-attraction term so the continuous state approaches the configured target.
* **Configurable target:** The target is passed into the simulator rather than being implicitly fixed.
* **Input validation:** Checks invalid step counts, noise values, thresholds, and non-finite inputs.
* **Automated tests:** The recorded development run reported 8 passing tests.
* **Reproducible artifacts:** Exports aggregate metrics and experiment traces to JSON and CSV.
* **Interactive dashboard:** Supports exploring parameters through Streamlit.
* **Scientific caveats:** Documents the distinction between representation comparisons, toy implementation comparisons, and claims about consciousness.

## Technology Stack

* Python
* NumPy
* Pandas
* Streamlit
* pytest

## Project Structure

```text
abstractionlab/
├── abstractionlab/
│   ├── __init__.py
│   ├── core.py
│   └── experiments.py
├── tests/
│   └── test_core.py
├── app.py
├── run_experiment.py
├── requirements.txt
└── README.md
```

## Quick Start — Windows PowerShell

Run these commands from the project root directory.

### 1. Create a virtual environment

```powershell
py -m venv .venv
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run the test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The previously recorded development run passed all 8 tests. Run the command above to verify the current local checkout.

### 4. Run the experiments

```powershell
.\.venv\Scripts\python.exe run_experiment.py --steps 500 --seed 42 --target 0.75 --out outputs
```

### 5. Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Output Files

The experiment exports the following artifacts to the configured output directory:

| File                               | Description                                                              |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `outputs/summary.json`             | Configuration, aggregate metrics, and interpretation notes               |
| `outputs/trajectory.csv`           | Continuous trajectory, disturbances, symbolic labels, and policy actions |
| `outputs/substrate_comparison.csv` | Shared inputs, quantized and continuous states, and binary outputs       |

These artifacts make it possible to inspect individual observations in addition to the aggregate metrics.

## Limitations

* The environment and both implementation models are mathematical toy systems, not physical substrates.
* Symbolic disagreement depends on partition design, thresholds, and label conventions.
* Action agreement does not establish equivalent task performance because actions do not control the simulated trajectory.
* Output agreement is measured using a specific binary output definition and input sequence.
* The reported metrics describe a single seeded configuration and do not establish statistical significance or generalization.
* The project cannot determine whether a system has subjective experience.

## Future Work

Potential extensions include:

1. **Multi-seed evaluation:** Report averages, standard deviations, and confidence intervals across independent runs.
2. **Parameter sweeps:** Vary noise, target values, initial states, and discretization thresholds.
3. **Alternative symbol maps:** Compare partitions with different resolutions and boundary placements.
4. **Expanded output definitions:** Evaluate whether agreement persists under different observation and decision rules.
5. **Closed-loop control:** Let symbolic actions influence state transitions and compare actual task performance.
6. **Failure analysis:** Identify conditions under which representation differences lead to different actions or unsuccessful tasks.

## Summary

AbstractionLab provides a reproducible toy environment for investigating observer-defined representations and the relationship between symbolic labels, policy outputs, and numerical dynamics.

In the recorded 500-step experiment, the coarse and balanced maps showed **97.4% symbol disagreement** and **99.0% action agreement**, while the target-seeking task succeeded. In a separate comparison, quantized and continuous toy implementations achieved **86.4% output agreement** with a mean absolute internal-state difference of **0.1625**.

These results illustrate limited properties of the implemented simulations. They are not results reported by Lerchner's essay and do not establish or rule out consciousness.

**Project status:** Research-inspired computational prototype with two toy experiments and recorded experimental outputs.