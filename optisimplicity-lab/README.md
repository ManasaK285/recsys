# OptiSimplicity Lab

**A research-inspired experimental sandbox for studying algorithmic simplicity, optimization, and the coincidence of extrema in finite search spaces.**

OptiSimplicity Lab investigates how different search strategies perform on small binary optimization problems, whether simpler candidate representations are associated with better objective values, and how often independently defined objectives share the same optimal candidates.

The project uses a finite binary search space, compressed representation length as a proxy for description complexity, multiple search algorithms, and a randomized null model for comparing optima.

> **Research prototype:** This is a synthetic computational study, not a proof of a general principle about algorithmic information theory or universal search.

## Research Questions

1. **Complexity of optima:** Do optimal solutions have shorter compressed representations than typical candidates?
2. **Search performance:** How do uniform random search, simplicity-biased sampling, hill climbing, and simulated annealing compare under a fixed evaluation budget?
3. **Coincidence of extrema:** Do different objective functions share optimal candidates more frequently than expected under an independently shuffled-score baseline?

## Experimental Setup

| Parameter                             | Configuration    |
| ------------------------------------- | ---------------- |
| Binary representation                 | 10 bits          |
| Candidate search space                | 1,024 candidates |
| Random seeds per search family        | 5                |
| Evaluation budget per search run      | 250              |
| Simplicity-bias parameter, β          | 0.25             |
| Extrema pairs                         | 30               |
| Null samples per extrema pair         | 200              |
| Random seed for complexity experiment | 42               |

The search benchmark evaluates four objective families:

* **Ones:** Rewards candidates with more set bits.
* **Target similarity:** Rewards candidates that match a seeded binary target.
* **Structured blocks:** Rewards agreement with a repeating binary pattern, with an additional adjacency bonus.
* **Random table:** Assigns seeded random objective scores to candidates.

## Methods

### 1. Complexity of Optima

The project estimates candidate description length using zlib-compressed representations. It compares the compressed length of each objective's optimum with the distribution of compressed lengths across the full candidate population.

The resulting measure is a practical compression-based proxy, not Kolmogorov complexity. Experiment A uses seed 42 for the objective instances.

### 2. Search Algorithm Benchmark

Four strategies are compared:

* **Uniform random search:** Samples candidates uniformly from the search space.
* **Simplicity-biased search:** Samples candidates with probabilities weighted toward shorter compressed representations.
* **Hill climbing:** Explores bit-flip neighbors and retains improvements.
* **Simulated annealing:** Uses a temperature-controlled local search that can accept some non-improving moves.

Each search run uses a budget of 250 evaluations. Search experiments use seeds 0–4 to construct objective instances, with separate seeds for the search procedures. Performance is summarized across runs.

The reported optimality gap is computed against the exact optimum of the specific objective instance used in that run. Therefore, the seed-42 optimum in the complexity experiment should not be treated as the reference optimum for every search run.

### 3. Coincidence of Extrema

The experiment compares optimal-candidate sets for pairs of objective families using set overlap and Jaccard similarity.

For the null baseline, each objective's score array is independently shuffled. This preserves its score distribution and tie counts while breaking the original association between candidate identity and score. The experiment then compares the observed overlap with overlap under these shuffled assignments.

This is one specific null model, not a universal test of whether objective optima coincide by chance.

## Results

### Complexity of Optima

Results from `outputs/complexity_optima.csv`:

| Objective family  | Best objective | Number of optima | Optimum compressed length (bits) | Median population length (bits) |
| ----------------- | -------------: | ---------------: | -------------------------------: | ------------------------------: |
| Ones              |          10.00 |                1 |                               88 |                             128 |
| Target similarity |          10.00 |                1 |                              120 |                             128 |
| Structured blocks |          11.25 |                1 |                              112 |                             128 |
| Random table      |          3.179 |                1 |                              144 |                             128 |

In this particular seeded experiment, the optimum for the ones objective has a shorter compressed representation than the population median. The target-similarity and structured-block optima are also shorter than the median, while the random-table optimum is longer.

**Interpretation:** These results illustrate that the relationship between compression length and objective quality depends on the objective. The four examples do not establish that simpler candidates are generally more likely to be optimal.

### Search Performance

Selected results from `outputs/search_aggregates.csv`:

| Objective         | Algorithm           | Mean best objective | Mean optimality gap | Optimum success rate |
| ----------------- | ------------------- | ------------------: | ------------------: | -------------------: |
| Ones              | Hill climbing       |               10.00 |                0.00 |                 100% |
| Ones              | Simplicity-biased   |               10.00 |                0.00 |                 100% |
| Ones              | Simulated annealing |               10.00 |                0.00 |                 100% |
| Ones              | Uniform random      |                9.00 |                1.00 |                   0% |
| Target similarity | Hill climbing       |               10.00 |                0.00 |                 100% |
| Target similarity | Simulated annealing |               10.00 |                0.00 |                 100% |
| Target similarity | Simplicity-biased   |                8.00 |                2.00 |                   0% |
| Target similarity | Uniform random      |                9.20 |                0.80 |                  20% |
| Structured blocks | Hill climbing       |               11.25 |                0.00 |                 100% |
| Structured blocks | Simulated annealing |               11.25 |                0.00 |                 100% |
| Structured blocks | Simplicity-biased   |                8.70 |                2.55 |                   0% |
| Structured blocks | Uniform random      |               10.25 |                1.00 |                  20% |
| Random table      | Hill climbing       |               1.925 |               1.286 |                   0% |
| Random table      | Simplicity-biased   |               2.444 |               0.767 |                  20% |
| Random table      | Simulated annealing |               2.399 |               0.812 |                   0% |
| Random table      | Uniform random      |               2.708 |               0.503 |                   0% |

*Rates are aggregated over five seeded runs. Mean best objective and mean gap are averages over those runs.*

**Interpretation:**

* Hill climbing and simulated annealing reliably found the optimum in the tested ones, target-similarity, and structured-block instances.
* Simplicity-biased search did not consistently find the optimum in the target-similarity or structured-block experiments.
* Uniform search explored more distinct candidates on average, but did not reliably find the optimum within the budget.
* The random-table results show that a strategy's performance depends on the structure of the objective; a local search advantage is not guaranteed on an unstructured landscape.
* Simplicity-biased sampling has additional computational overhead from calculating compressed representation lengths.

These findings describe this finite benchmark and should not be generalized to large-scale optimization without further experiments.

### Coincidence of Extrema

The project evaluates 30 comparisons among the ones, target-similarity, and structured-block objectives, using 200 shuffled null samples per comparison.

Across the recorded comparisons:

* Observed optimal-set overlap was zero for every pair.
* Each observed objective instance had a single optimum.
* Most shuffled null comparisons also had zero overlap.
* A small number of null comparisons produced nonzero overlap, illustrating that independently assigned optima can occasionally coincide by chance.

Because the optimal sets contain one candidate each, Jaccard overlap is either zero or one for each comparison. With only 200 null samples, rare coincidences may not appear in every comparison. The experiment therefore provides a small-scale illustration of the chosen null model, rather than strong evidence for a general theory of extrema.

## Project Outputs

The `outputs/` directory contains the generated experiment results, including:

* `summary.json` — configuration, experiment summary, and caveats.
* `complexity_optima.csv` — objective optima and compressed-length statistics.
* `search_aggregates.csv` — aggregated search performance by objective and algorithm.
* `extrema_pairs.csv` — observed extrema overlap and shuffled-null statistics.
* `search_runs.csv` — per-run search measurements, including the objective instance's known optimum and the observed optimality gap.

Exact filenames may depend on the current project version.

## Reproducibility

Run commands from the project root in PowerShell:

```powershell
python -m pytest -q
```

To regenerate experiments, use the experiment runner configured for this repository. For example, if the script accepts the parameters shown in the project configuration:

```powershell
python -m optisimplicity.experiments.run_all
```

Check the script's argument parser or README configuration if the runner requires explicit command-line arguments.

## Limitations

* **Small search space:** A 10-bit space contains only 1,024 candidates and is not representative of modern, high-dimensional optimization.
* **Compression proxy:** zlib length depends on representation and compressor settings; it is not a direct measurement of Kolmogorov complexity.
* **Heuristic simplicity prior:** The sampling distribution is a chosen compression-based heuristic, not universal algorithmic probability.
* **Limited repetitions:** Five search seeds and 200 null samples per pair provide limited statistical resolution.
* **Objective dependence:** Different objective families have different landscape structures, which can favor different search strategies.
* **Null-model dependence:** Independently shuffled scores preserve score values and tie counts, but are only one way to construct a comparison baseline.
* **No real-world validation:** The project does not establish performance on large language models, scientific discovery, or production optimization tasks.

## Conclusion

OptiSimplicity Lab provides a reproducible sandbox for investigating the relationships between compressed description length, search behavior, and optimal-candidate overlap. Its results demonstrate that the performance of simplicity-biased and local-search strategies varies across synthetic objective families, and that observed extrema overlap should be interpreted relative to an explicit null model.

The project is best presented as a **research-inspired experimental implementation** that makes its assumptions measurable and its limitations explicit, rather than as a reproduction or validation of a universal theory of simplicity and optimization.
