# CollectiveAlign Lab

**A reproducible research-inspired benchmark for collective human–AI alignment.**

CollectiveAlign Lab investigates how candidate presentation, agent policies, and collective decision protocols affect leader selection. It includes matched anonymous/visible scenarios, simulated agent baselines, voting and aggregation protocols, statistical metrics, experiment outputs, and an optional Streamlit dashboard.

> **Scientific integrity:** The default benchmark uses synthetic scenarios and explicitly coded toy agents. Results are not observations from the paper *To Mask or to Mirror: Human-AI Alignment in Collective Reasoning*, are not human-subject data, and do not establish how any real commercial LLM behaves.

## Research motivation

The project is inspired by:

Qian et al., **“To Mask or to Mirror: Human-AI Alignment in Collective Reasoning,” EMNLP 2025**.

- Publication: https://aclanthology.org/2025.emnlp-main.122/
- PDF: https://aclanthology.org/2025.emnlp-main.122.pdf

The paper motivates examining collective alignment rather than evaluating only individual model responses. This implementation is a research-inspired simulation scaffold, not an exact replication of the study.

## Research questions

1. Does leader selection change when identity presentation changes while candidate qualifications remain fixed?
2. How do competence-oriented, status-biased, identity-sensitive, and random policies differ?
3. How do independent voting, free-discussion approximation, and evidence-first aggregation affect selection disparity?
4. Does a protocol that emphasizes task evidence reduce the optimality gap or selection disparity in the simulator?
5. How can a controlled, reproducible benchmark be extended to evaluate actual LLMs and appropriately collected human-reference data?

## Included features

- Matched anonymous and identity-visible candidate scenarios.
- Four transparent baseline policies:
  - `competence`
  - `identity_sensitive`
  - `random`
  - `status_bias`
- Three decision protocols:
  - `independent_vote`
  - `free_discussion`
  - `evidence_first`
- Trial-level CSV and JSONL outputs.
- Summary metrics, identity-sensitivity estimates, selection-disparity tables, plots, and a Markdown report.
- Streamlit dashboard with model/protocol/condition filters and CSV export.
- Automated tests.
- No API key or GPU required for the local synthetic benchmark.

## Technology stack

- Python 3.10+ (Python 3.12 recommended)
- NumPy and pandas
- SciPy
- Matplotlib
- PyYAML and Pydantic
- Streamlit and Plotly for the optional dashboard
- pytest for tests

## Installation (Windows PowerShell)

Open PowerShell in the extracted `collectivealign-lab` project directory.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
```

If Python 3.12 is not installed, use another supported Python version:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
```

## Run the tests

```powershell
python -m pytest -q
```

## Run the full local experiment

```powershell
python run_all.py --config configs/local.yaml --out outputs
```

Or use the direct runner:

```powershell
python run_experiment.py --config configs/local.yaml --out outputs
```

The default configuration uses:

- 120 scenario seeds
- 2 presentation conditions (`anonymous`, `visible`)
- 4 simulated agent policies
- 3 decision protocols
- Group size 5
- 4 candidates per scenario

This produces **2,880 result rows** (120 × 2 × 4 × 3).

## Launch the dashboard

After running the experiment:

```powershell
streamlit run app.py
```

The dashboard reads `outputs/trial_results.csv`. If that file does not exist, run the benchmark first.

## Outputs

The benchmark writes the following files to `outputs/`:

| File | Description |
|---|---|
| `trial_results.csv` | Trial-level records for every condition, policy, and protocol |
| `trial_results.jsonl` | The same records in JSON Lines format |
| `run_metadata.json` | Configuration snapshot and synthetic-data warning |
| `summary_metrics.csv` | Mean/median optimality gap and selection rates |
| `identity_sensitivity.csv` | Fraction of paired trials where the winning candidate changes |
| `selection_disparity.csv` | Absolute difference between profile-group A and B selection rates |
| `optimality_gap.png` | Plot comparing task-performance gaps |
| `identity_sensitivity.png` | Plot of paired identity sensitivity |
| `REPORT.md` | Automatically generated results summary and interpretation cautions |

## Metrics

### 1. Optimality gap

For each trial:

`optimality_gap = best_candidate_qualification - elected_candidate_qualification`

Lower is better. A value of zero means the elected candidate had the highest qualification score in that scenario.

### 2. Identity sensitivity

For each model/protocol pair, calculate the fraction of matched scenarios where the winning candidate ID differs between the anonymous and visible conditions.

A changed winner does not automatically mean the decision was biased. Interpret this alongside qualification scores and selection disparities.

### 3. Selection disparity

The absolute difference between the proportion of trials won by profile group A and the proportion won by profile group B.

`selection_disparity = abs(selection_rate_A - selection_rate_B)`

This is a descriptive statistic. It is not, by itself, proof of unfairness or discrimination.

### 4. Mean winner qualification

The average qualification score of the selected candidate. Interpret together with the optimality gap and selection disparity.

## Interpretation of the current synthetic results

In the supplied run, each model/protocol/condition combination has 120 trials. The selection-disparity table shows several patterns:

- **Competence policy:** the visible and anonymous conditions have the same reported disparity for each protocol; the displayed disparity ranges from about 3.3 to 5.0 percentage points.
- **Identity-sensitive policy:** the reported disparity is larger in the visible condition for all three protocols. For independent voting, it rises from about 6.7 to 20.0 percentage points; for free-discussion aggregation, from about 8.3 to 23.3 points; and for evidence-first aggregation, from about 8.3 to 13.3 points.
- **Random policy:** results vary by protocol, but the reported disparities are the same across the two identity conditions in this run.
- **Status-bias policy:** disparity is substantial in both conditions. It is about 18.3 percentage points with evidence-first aggregation, 48.3 points with free-discussion aggregation, and 55.0 points with independent voting.

These are descriptive observations from the supplied synthetic output, not statistical conclusions about real AI models or people. The status-bias and identity-sensitive agents intentionally encode specific scoring rules, so their behavior is expected to reflect those rules.

## How the simulator works

1. A seed creates candidates with qualification scores and synthetic profile groups.
2. The anonymous and visible scenarios are intended to be matched on candidate IDs, qualifications, profile groups, and task evidence.
3. Simulated agents score each candidate using a transparent policy.
4. Candidate scores are aggregated under a selected protocol.
5. The winning candidate and task-performance gap are recorded.
6. Metrics, plots, and a report are generated from the trial records.

### Important implementation caveat

The current implementation is a baseline scaffold. The `free_discussion` protocol is a deterministic score-aggregation approximation, not an actual discussion between language models. `evidence_first` explicitly blends policy scores with qualification scores. These design choices can explain observed changes in disparity; they should not be presented as evidence that real-world deliberation will have the same effect.

Also, because the current anonymous/visible scenario generation should be checked for strict pairing in any confirmatory study, verify that every pair shares the exact candidate qualification and group assignments before interpreting paired effects. The supplied tests cover basic scenario matching, but do not replace a full experimental-design audit.

## Extending the project with real LLMs

A future LLM adapter should implement a stable interface such as:

`rank_candidates(scenario, prompt_config) -> structured ranking`

For each run, record:

- Provider and exact model/version
- Prompt template and configuration
- Decoding parameters
- Scenario seed and condition
- Parsed ranking, vote, and decision
- API errors, retries, and token usage when available

Use the same scenarios and output schema across models, validate structured responses, and randomize condition order where appropriate. Keep API keys in environment variables and never commit them to source control. The current project does **not** include working Gemini, GPT, Claude, or Gemma integrations.

## Research and ethics considerations

- Do not infer a person's demographic identity from their name.
- Use carefully controlled identity manipulations and document their limitations.
- Obtain informed consent and appropriate institutional review for new human-subject research where required.
- Pre-register confirmatory hypotheses and analysis choices.
- Use uncertainty estimates and an appropriate experimental unit; agents within the same group are not automatically independent observations.
- Separate task performance, behavioral alignment, and fairness-related metrics.
- Clearly label synthetic data and do not present it as a human cohort or published-paper result.

## Troubleshooting

**`ModuleNotFoundError: No module named 'collectivealign'`**

Activate the virtual environment and run:

```powershell
python -m pip install -e ".[dev,dashboard]"
```

**Dashboard says no results were found**

Run the experiment from the project root:

```powershell
python run_all.py --config configs/local.yaml --out outputs
```

**Streamlit or Plotly is missing**

Install the dashboard extras:

```powershell
python -m pip install -e ".[dashboard]"
```

**Tests fail**

Check that the virtual environment is active and that dependencies installed successfully, then rerun:

```powershell
python -m pytest -q
```

## License

This is a research/educational starter project. Add a formal license file before redistributing or publishing the repository.
