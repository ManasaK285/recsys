# OptiSimplicity Lab

**A Reproducible Experimental Study of Simplicity, Combinatorial Optimization, Search Strategies, and Coincidence of Extrema**

OptiSimplicity Lab is a Python research project inspired by Dingle and Hutter, *Simplicity and Complexity in Combinatorial Optimization*, published in *Entropy* 28(2):226 (2026).

The project implements three computational experiments to investigate relationships between compressed description length, objective-function optimality, search performance, and overlap between optimal sets.

The experiments use exactly enumerable binary search spaces, deterministic random seeds, fixed evaluation budgets, and CSV-based result exports. The implementation is intended as a reproducible experimental laboratory rather than a reproduction of every theoretical or empirical result in the original paper.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Research Questions](#research-questions)
- [Project Features](#project-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
- [Experimental Design](#experimental-design)
- [Experiment A: Complexity of Optima](#experiment-a-complexity-of-optima)
- [Experiment B: Search Benchmark](#experiment-b-search-benchmark)
- [Experiment C: Coincidence of Extrema](#experiment-c-coincidence-of-extrema)
- [Observed Results](#observed-results)
- [Scientific Interpretation](#scientific-interpretation)
- [Testing and Validation](#testing-and-validation)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Reference](#reference)

---

## Research Motivation

Combinatorial optimization involves finding high-quality solutions within a finite or extremely large set of possible candidates.

A natural question is whether simple, compressible solutions are more likely to be optimal or easier to discover through search.

This project investigates that question from three complementary perspectives:

1. **Description length:** Are optimal candidates more compressible than typical candidates?
2. **Search behavior:** How do different search strategies perform under a common objective-evaluation budget?
3. **Coincidence of extrema:** How often do different objectives share optimal candidates, and how does observed overlap compare with a randomized baseline?

These questions are related but distinct. A compressible candidate is not necessarily optimal, and a search algorithm that finds simple candidates does not necessarily outperform other algorithms.

The project makes these distinctions explicit through separate experiments and exported artifacts.

## Research Questions

**RQ1 — Complexity of optima**

How does the compressed description length of globally optimal candidates compare with the description length of the full candidate population?

**RQ2 — Search performance**

How do uniform random sampling, simplicity-biased sampling, hill climbing, and simulated annealing compare in solution quality, optimality gap, optimum discovery rate, candidate coverage, and runtime?

**RQ3 — Coincidence of extrema**

Do different objective functions share optimal candidates, and how does their observed optimal-set overlap compare with a null model that independently shuffles score assignments?

---

## Project Features

- Exact enumeration of binary candidate spaces up to 16 bits.
- Four synthetic objective families.
- Zlib-based compressed description-length measurements.
- Four search algorithms evaluated under fixed budgets.
- Multiple seeded search runs.
- Optimality-gap and optimum-discovery tracking.
- Pairwise optimal-set overlap analysis.
- Score-permutation null model preserving objective score multisets.
- CSV outputs suitable for pandas, Excel, and further statistical analysis.
- JSON experiment configuration summary.
- Command-line interface and automated tests.

---

## Repository Structure

```text
optisimplicity-lab/
├── optisimplicity/
│   ├── __init__.py
│   ├── complexity.py
│   ├── encoding.py
│   ├── problems.py
│   ├── search.py
│   └── experiments/
│       ├── __init__.py
│       └── run_all.py
├── tests/
├── outputs/
│   ├── complexity_optima.csv
│   ├── search_runs.csv
│   ├── search_curves.csv
│   ├── search_aggregates.csv
│   ├── extrema_pairs.csv
│   ├── extrema_null_samples.csv
│   └── summary.json
├── README.md
└── requirements.txt
```

The experiment runner is implemented in `optisimplicity/experiments/run_all.py`. The extrema experiment is defined inside that module; it does not require a separate `extrema.py` file.

---

## Installation

Requirements:

- Python 3.10 or newer, subject to compatibility with the pinned dependencies.
- NumPy
- pandas
- pytest for testing

From PowerShell, navigate to the project directory:

```powershell
cd "C:\Users\Manasa\Desktop\MK\Personal\mk\Google2\G2\optisimplicity-lab"
```

Create and activate a virtual environment if necessary:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use the Python executable inside `.venv` directly or follow your local execution policy.

## Running the Experiments

First, inspect the available arguments:

```powershell
python -m optisimplicity.experiments.run_all --help
```

Run the complete experiment suite using the baseline configuration:

```powershell
python -m optisimplicity.experiments.run_all `
    --out outputs `
    --n-bits 10 `
    --seeds 5 `
    --budget 250 `
    --beta 0.25 `
    --extrema-pairs 30 `
    --null-samples 200 `
    --seed 42
```

The runner executes all three experiments and writes their artifacts to `outputs/`.

### Configuration

| Parameter | Value | Meaning |
|---|---:|---|
| `n_bits` | 10 | Number of bits in each candidate |
| Candidate space | 1,024 | \(2^{10}\) binary candidates |
| `seeds` | 5 | Search seeds per objective family |
| `budget` | 250 | Objective-evaluation budget per search run |
| `beta` | 0.25 | Simplicity-bias parameter |
| `extrema_pairs` | 30 | Number of generated pair records |
| `null_samples` | 200 | Null samples per pair record |
| `seed` | 42 | Main experiment seed |

With four objective families, four search algorithms, and five search seeds, the benchmark contains **80 search runs**.

The extrema experiment produces 30 observed pair records and 6,000 null-sample records.

---

## Experimental Design

### Candidate representation

A candidate is an \(n\)-bit binary string represented internally by an integer in the range

\[
0 \leq x < 2^n.
\]

For \(n=10\), the search space contains 1,024 candidates. Because the space is small enough to enumerate, globally optimal objective values and optimal sets can be calculated exactly.

### Objective families

| Objective | Construction | Purpose |
|---|---|---|
| `ones` | Number of one bits | Simple, regular objective |
| `target_similarity` | Number of matching bits against a seeded binary target | Tests target-dependent structure |
| `structured_blocks` | Repeating-pattern matches plus an adjacent-pair bonus | Tests structured objective design |
| `random_table` | Seeded random score assigned to each candidate | Provides an irregular score landscape |

All four objectives are maximized in the reported experiments.

### Description-length proxy

Each candidate is converted to a canonical fixed-width ASCII bitstring and compressed using zlib. The compressed size, converted to bits, is used as the description-length proxy.

This procedure is deterministic for a fixed implementation and encoding. It is not a direct calculation of Kolmogorov complexity.

---

## Experiment A: Complexity of Optima

### Objective

Compare the description length of globally optimal candidates with the median and mean description lengths of all candidates.

### Output

`outputs/complexity_optima.csv`

Important columns:

- `problem`: objective family.
- `n_bits`: candidate length.
- `n_candidates`: total candidate count.
- `best_objective`: globally optimal objective value.
- `number_of_optima`: number of globally optimal candidates.
- `mean_optimum_description_length_bits`: average compressed length of optimal candidates.
- `median_all_description_length_bits`: population median.
- `mean_all_description_length_bits`: population mean.
- `optimum_minus_population_median_bits`: difference between optimum mean length and population median.

A negative difference indicates that the optimal candidate is shorter under the chosen compression representation than the population median. A positive difference indicates a longer representation.

### Baseline results

For \(n=10\) and seed 42:

| Objective | Best score | Optima | Optimum length (bits) | Population median (bits) |
|---|---:|---:|---:|---:|
| `ones` | 10.000 | 1 | 88 | 128 |
| `target_similarity` | 10.000 | 1 | 120 | 128 |
| `structured_blocks` | 11.250 | 1 | 112 | 128 |
| `random_table` | 3.17885 | 1 | 144 | 128 |

All four objective instances have one global optimum.

The `ones` and `structured_blocks` optima have shorter compressed representations than the population median. The `target_similarity` optimum is also shorter, although the difference is smaller. The `random_table` optimum has a longer representation than the median.

These observations show that optimality and compressed description length do not coincide uniformly across these synthetic objectives.

They do not establish a general relationship between simplicity and optimization across arbitrary problem classes.

---

## Experiment B: Search Benchmark

### Objective

Compare four algorithms using a fixed objective-evaluation budget.

Algorithms:

1. **Uniform sampling:** Selects candidates uniformly from the binary search space.
2. **Simplicity-biased sampling:** Uses the project's simplicity-bias mechanism to influence candidate selection.
3. **Hill climbing:** Iteratively seeks improvements in the objective landscape.
4. **Simulated annealing:** Uses a temperature-controlled acceptance mechanism to explore the landscape.

The simplicity-biased sampler is an experimental heuristic, not a true implementation of universal algorithmic probability.

### Outputs

- `outputs/search_runs.csv`
- `outputs/search_curves.csv`
- `outputs/search_aggregates.csv`

### Evaluation metrics

| Metric | Interpretation |
|---|---|
| `mean_best_objective` | Mean best score achieved |
| `mean_optimality_gap` | Mean difference from the global optimum |
| `optimum_success_rate` | Fraction of runs reaching the exact optimum |
| `mean_unique_candidates` | Mean number of distinct candidates evaluated |
| `mean_runtime_seconds` | Mean measured execution time |

For a maximization problem, the optimality gap is calculated relative to the globally optimal score. Lower gaps are better; zero indicates that the optimum has been reached.

### Baseline results

The following results were generated with five seeds and a budget of 250 evaluations per run.

| Objective | Algorithm | Mean best score | Mean gap | Optimum success rate |
|---|---|---:|---:|---:|
| `ones` | Hill climbing | 10.000 | 0.000 | 100% |
| `ones` | Simplicity-biased | 10.000 | 0.000 | 100% |
| `ones` | Simulated annealing | 10.000 | 0.000 | 100% |
| `ones` | Uniform | 9.000 | 1.000 | 0% |
| `target_similarity` | Hill climbing | 10.000 | 0.000 | 100% |
| `target_similarity` | Simplicity-biased | 8.000 | 2.000 | 0% |
| `target_similarity` | Simulated annealing | 10.000 | 0.000 | 100% |
| `target_similarity` | Uniform | 9.200 | 0.800 | 20% |
| `structured_blocks` | Hill climbing | 11.250 | 0.000 | 100% |
| `structured_blocks` | Simplicity-biased | 8.700 | 2.550 | 0% |
| `structured_blocks` | Simulated annealing | 11.250 | 0.000 | 100% |
| `structured_blocks` | Uniform | 10.250 | 1.000 | 20% |
| `random_table` | Hill climbing | 1.92481 | 1.28622 | 0% |
| `random_table` | Simplicity-biased | 2.44413 | 0.76690 | 20% |
| `random_table` | Simulated annealing | 2.39940 | 0.81163 | 0% |
| `random_table` | Uniform | 2.70817 | 0.50286 | 0% |

### Interpretation

The results suggest that search performance depends on the objective landscape.

- Hill climbing and simulated annealing consistently find the optimum for `ones`, `target_similarity`, and `structured_blocks` in these five runs.
- Simplicity-biased sampling performs well on `ones` but fails to reach the optimum in the observed `target_similarity` and `structured_blocks` runs.
- Uniform sampling has the highest mean best score for `random_table`, although none of the algorithms reaches the exact optimum in these five runs.

The random-table results are particularly useful because they demonstrate that a search heuristic does not automatically benefit from preferring simple candidates.

These are descriptive results from a small experiment, not statistically conclusive algorithm rankings.

### Runtime observations

In this baseline run, the simplicity-biased sampler generally has a higher measured runtime than the other algorithms. The observed times are on the order of milliseconds.

Runtime differences at this scale can be affected by Python overhead, hardware, and timer resolution. A more rigorous performance comparison should use repeated timing measurements and larger workloads.

---

## Experiment C: Coincidence of Extrema

### Objective

Measure overlap between optimal sets for different objective functions and compare the observed overlap against a shuffled-score null model.

The implementation is contained in `run_extrema_experiment()` inside `optisimplicity/experiments/run_all.py`.

### Observed objective pairs

The three objective families used in this experiment are:

- `ones`
- `target_similarity`
- `structured_blocks`

The runner cycles through three ordered pairings:

1. `ones` versus `target_similarity`
2. `target_similarity` versus `structured_blocks`
3. `structured_blocks` versus `ones`

With 30 pair records, each pairing is repeated ten times with varying seeds where the objective definition uses its seed.

These are repeated seeded instances of three family-pair categories, not 30 distinct objective-family pairs.

### Null model

For each pair, the experiment independently permutes the score arrays of both objectives.

The permutation preserves each objective's score multiset and its number of optimal candidates while breaking the association between candidate identity and objective score.

This provides a transparent randomized baseline for optimal-set overlap.

It is not the only possible null model and should not be interpreted as a universal model of unrelated objective functions.

### Outputs

- `outputs/extrema_pairs.csv`
- `outputs/extrema_null_samples.csv`

The observed-pair output contains objective names, optimal-set sizes, intersection size, Jaccard overlap, and null-summary statistics.

The null-sample output contains individual randomized overlap measurements for each pair record.

### Metrics

For optimal sets \(A\) and \(B\), Jaccard overlap is

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

The experiment also reports the intersection size and intersection relative to the smaller optimal set.

### Baseline observations

For the tested \(n=10\) instances, each objective has one optimum, and the observed optimal sets do not overlap in the 30 generated records.

The null experiment generates 200 samples per pair, for 6,000 null records in total.

Across all null records, the observed mean intersection size is approximately 0.00033, with values ranging from zero to one.

Most null samples have zero overlap, which is expected for singleton optimal sets in a space of 1,024 candidates.

Under independently randomized assignments with unique maxima, the probability that two singleton optima coincide is

\[
\frac{1}{1024}\approx 0.000977.
\]

The expected number of intersections in 200 null samples is therefore approximately 0.195 for a pair with singleton optimal sets.

### Interpretation

The observed zero overlap is consistent with the shuffled-score null model. It does not establish that the optima are unusually dissimilar.

Because the three family-pair categories are repeated, their rows must not be treated as independent objective-family replications. Also, the `ones` and `structured_blocks` objectives are deterministic for a fixed bit width, whereas `target_similarity` changes with its seed.

The results should be interpreted as a preliminary comparison of three synthetic family-pair categories under a particular randomization baseline.

---

## Observed Results

The baseline experiment supports the following limited findings:

1. **Optimality and compressibility differ across objectives.** The random-table optimum is less compressible than the population median under the chosen representation, while the other three optima are more compressible.
2. **Search performance is objective-dependent.** Local search performs well on three structured objectives in the observed runs, while uniform sampling obtains the highest mean score on the random-table objective.
3. **Simplicity bias is not uniformly beneficial.** Its performance varies across objective families, and it incurs additional computational overhead in this implementation.
4. **Zero optimal-set overlap is not surprising under the chosen null model.** The singleton optimal sets and the small candidate space make the observed zero intersections consistent with the randomized baseline.
5. **More extensive experiments are needed.** Five search seeds and 200 null samples per pair provide preliminary evidence rather than broad statistical conclusions.

No hard-coded results are required by the experiment runner. Reported findings should always be tied to the configuration and CSV artifacts from the actual run.

---

## Testing and Validation

Run the test suite from the project root:

```powershell
python -m pytest -q
```

Check the generated files:

```powershell
Get-ChildItem .\outputs -File |
    Select-Object Name, Length
```

Inspect aggregate search results:

```powershell
Import-Csv .\outputs\search_aggregates.csv |
    Format-Table -AutoSize
```

Inspect the optimum-complexity results:

```powershell
Import-Csv .\outputs\complexity_optima.csv |
    Format-Table -AutoSize
```

Inspect the extrema results:

```powershell
Import-Csv .\outputs\extrema_pairs.csv |
    Select-Object -First 10 |
    Format-Table -AutoSize
```

The existence of output files confirms artifact generation, not the scientific validity of every conclusion. Tests should be used to validate code behavior, and experiment outputs should be examined independently.

---

## Reproducibility

The experiment suite exposes parameters for:

- Candidate bit width
- Search seeds
- Evaluation budget
- Simplicity-bias strength
- Number of extrema pair records
- Null samples per pair
- Main random seed

To reproduce the baseline configuration, use the command in [Running the Experiments](#running-the-experiments).

The generated `summary.json` records the experiment configuration, artifact names, and scientific caveats.

Reproducibility also depends on the Python environment, dependency versions, implementation, and random-number generation behavior. Preserve the configuration and source revision when comparing different runs.

---

## Limitations

### 1. Description length is not Kolmogorov complexity

Zlib compressed size depends on encoding and compression overhead. It is a practical proxy, not an implementation of the theoretical minimum description length of a string.

### 2. Small candidate spaces

At ten bits, there are only 1,024 candidates. Results may change at larger bit widths or under different objective constructions.

### 3. Synthetic objectives

The four objective families are designed for controlled experimentation. They do not represent the full diversity of real-world combinatorial optimization problems.

### 4. Limited search replications

Five seeds per algorithm and objective provide a preliminary benchmark. Larger numbers of independent instances and seeds are needed for more stable comparisons.

### 5. Simplicity-biased sampling is heuristic

The sampler is not true universal algorithmic probability. Results should not be generalized to all possible simplicity-based search strategies.

### 6. Null-model assumptions

Independently shuffled score assignments preserve score distributions and tie counts but destroy candidate-score structure. Other null models may yield different overlap distributions.

### 7. Repeated objective-pair categories

Thirty generated pair records do not constitute thirty independent objective-family comparisons. Future versions should distinguish family-level effects from instance-level variation.

### 8. Runtime measurement

Millisecond-scale measurements may be sensitive to implementation overhead and machine conditions. Timing conclusions should be validated with larger workloads and repeated benchmarks.

### 9. Generalization

The experiments do not establish a universal law that simple solutions are always optimal, that simplicity-biased search is superior, or that unrelated objective functions cannot share extrema.

---

## Future Work

Potential improvements include:

- Evaluate additional bit widths and objective families.
- Increase search seeds and null samples.
- Add confidence intervals and appropriately designed statistical comparisons.
- Separate objective-family comparisons from independently generated instances.
- Investigate alternative null models for optimal-set overlap.
- Compare multiple compression methods and representations.
- Benchmark larger workloads to obtain more reliable runtime estimates.
- Evaluate structured and random objectives with controlled variations in landscape properties.
- Add plots for search trajectories, optimality gaps, description lengths, and null overlap distributions.
- Compare simplicity-biased sampling against additional search heuristics under matched evaluation budgets.

These extensions would help determine which findings are robust to changes in objective structure, encoding, search budget, and null-model assumptions.

---

## Reference

Dingle, K., and Hutter, M. *Simplicity and Complexity in Combinatorial Optimization*. Entropy, 28(2), 226, 2026.

Publication: https://doi.org/10.3390/e28020226

---

## Summary

OptiSimplicity Lab provides a reproducible starting point for investigating the relationships between description length, optimization, search strategy, and optimal-set overlap.

Its main contribution is an experimental framework with exact enumeration, controlled search budgets, seeded runs, explicit null-model assumptions, and inspectable artifacts. The current results illustrate that compressibility and optimality are not interchangeable, search performance depends on the objective landscape, and observed extrema overlap must be interpreted relative to a carefully chosen baseline.

The findings are preliminary and apply to the synthetic objectives and configurations evaluated here. Broader claims require additional experiments and stronger statistical validation.