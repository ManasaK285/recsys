# SignalScope AI — Intelligent Telemetry Analytics Platform

**Turn service telemetry into evidence-backed engineering decisions.**

SignalScope AI is an end-to-end telemetry analytics research prototype that simulates service performance data, detects anomalous behavior, compares software releases, and generates evidence-grounded investigation summaries.

The platform combines **Python, FastAPI, Pydantic, DuckDB, Parquet, Pandas, NumPy, Streamlit, and Plotly** to demonstrate how an engineering team can move from raw telemetry to anomaly detection and release-regression investigation.

The project uses controlled synthetic telemetry with known anomaly labels, allowing detection results to be evaluated against ground truth instead of relying on convincing-looking dashboard output alone.

> **Important:** This is a research prototype using synthetic data. Detected anomalies indicate unusual behavior; they do not independently prove that a software release caused an incident.

## Key capabilities

- **Telemetry simulation:** Generate 30 days of hourly service telemetry across 50 services and 3 regions.
- **Controlled regression injection:** Introduce known anomalies affecting `checkout`, `search`, and `recommendations` in `us-east` during the final three days.
- **Data ingestion:** Validate and ingest telemetry through FastAPI and Pydantic.
- **Analytical storage:** Use DuckDB for local analytical queries and Parquet for columnar data export.
- **Anomaly detection:** Flag unusual latency, error rate, CPU, memory, and queue-depth measurements using robust statistical baselines.
- **Release comparison:** Compare metrics across releases and report absolute changes, relative changes, and sample counts.
- **Evidence-grounded analytics:** Answer supported natural-language questions using computed analytical results.
- **Interactive dashboard:** Explore telemetry, anomalies, service performance, and release comparisons through Streamlit and Plotly.
- **Evaluation and testing:** Measure anomaly-detection performance against synthetic ground-truth labels and run automated tests.
- **Operational visibility:** Expose health information and basic Prometheus-format metrics through the API.

## Architecture

```text
Synthetic Telemetry Generator
            |
            v
   FastAPI + Pydantic
   Ingestion and Validation
            |
            v
      DuckDB Storage
            |
            +--------------------+
            |                    |
            v                    v
     Parquet Export        Analytics Engine
                                 |
                    +------------+------------+
                    |            |            |
                    v            v            v
                 Anomaly      Release       Evidence-
                 Detection    Comparison    Grounded Q&A
                    |            |            |
                    +------------+------------+
                                 |
                     +-----------+-----------+
                     |                       |
                     v                       v
              FastAPI Endpoints      Streamlit Dashboard
                     |
                     v
              Evaluation + Tests
```

## Repository structure

```text
signalscope-ai/
├── src/
│   └── signalscope/
│       ├── __init__.py
│       ├── schemas.py
│       ├── simulator.py
│       ├── storage.py
│       ├── analytics.py
│       ├── insights.py
│       └── api.py
├── dashboard/
│   └── app.py
├── scripts/
│   ├── generate_demo.py
│   └── run_evaluation.py
├── tests/
│   ├── test_analytics.py
│   ├── test_insights.py
│   └── test_simulator.py
├── data/
├── outputs/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer; Python 3.11 is a recommended environment.
- Windows PowerShell, macOS, or Linux.
- Internet access for initial Python dependency installation.
- Docker Desktop is optional.

## Installation — Windows PowerShell

Open PowerShell and navigate to the project directory.

```powershell
cd C:\path\to\signalscope-ai

py -0p
py -3.11 -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If Python 3.11 is not installed, use an available compatible version listed by `py -0p`. If virtual-environment activation is blocked, use the environment's Python executable directly.

## Quick start

### 1. Generate the demo dataset

```powershell
python scripts\generate_demo.py
```

The simulator creates hourly telemetry covering:

- 30 days of simulated observations
- 50 services
- 3 regions
- Multiple software releases
- Injected regression behavior for three services in `us-east`

The generated data is stored in:

- `data/telemetry.parquet`
- `data/signalscope.duckdb`

Running the generation script resets and reseeds the demo dataset.

### 2. Start the API

```powershell
python -m uvicorn signalscope.api:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

Useful endpoints:

| Endpoint | Purpose |
|---|---|
| `/health` | API health and telemetry row count |
| `/summary` | Overall telemetry summary |
| `/metrics` | Prometheus-format metrics |
| `/anomalies?limit=100` | Ranked anomaly results |
| `/releases/compare` | Release comparison |
| `/evaluation` | Detection evaluation results |
| `POST /telemetry` | Ingest validated telemetry |
| `POST /ask` | Ask a supported analytics question |

The API initializes demo data if its database is empty.

### 3. Start the dashboard

Open a second PowerShell terminal, navigate to the project directory, and activate the virtual environment.

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard\app.py
```

Keep the API running in the first terminal. The dashboard is intended to provide an interactive view of the telemetry analytics workflow.

### 4. Ask an analytics question

In the Swagger UI at `http://127.0.0.1:8000/docs`, open `POST /ask` and submit:

```json
{
  "question": "Which services regressed after the latest release?"
}
```

Other example questions include:

```json
{
  "question": "Which services have the highest latency?"
}
```

```json
{
  "question": "Summarize the current telemetry anomalies."
}
```

The question router supports a limited set of analytical intents. Unsupported questions should receive an explicit response rather than fabricated measurements.

### 5. Run evaluation and tests

```powershell
python scripts\run_evaluation.py
python -m pytest
```

The evaluation script writes its results to `outputs/evaluation.json`.

Use the values produced by your own run when reporting benchmark results.

## Latest investigation: release v2.4.1

The flagged records examined during this project show a recurring pattern across three services in the `us-east` region.

| Service | Observed p95 latency in shared flagged rows | Observed error rate | Observed queue depth | Flagged severity |
|---|---:|---:|---:|---|
| `checkout` | 361–408 ms | 1.8–2.6% | 18–26 | High |
| `recommendations` | 251–306 ms | 2.0–2.8% | 18–25 | High |
| `search` | 270–319 ms | 2.1–3.0% | 18–25 | High |

These ranges summarize the flagged rows shared for inspection; they are **not** full-period averages, percentile estimates, or independent benchmark results.

### Interpretation

- **Multiple services are affected:** The shared records show elevated latency, error rates, and queue depth for all three services.
- **Checkout has the highest observed latency:** Its shared flagged rows reach approximately 408 ms p95 latency.
- **Search has the highest observed error rate:** Its shared flagged rows reach approximately 3.0%.
- **The metrics move together:** Elevated latency, error rate, and queue depth provide stronger investigation evidence than a single isolated metric.
- **The release relationship requires verification:** Because these records are associated with `v2.4.1` and overlap with the injected regression window, they support the intended demo scenario. They do not independently establish that a real deployment caused the behavior.

### Recommended engineering investigation

1. Compare each affected service's metrics before and after deployment.
2. Examine request volume, CPU utilization, memory utilization, and queue depth during the same windows.
3. Inspect traces, application logs, and error categories for the affected services.
4. Compare affected services with unaffected services and regions as control cohorts.
5. Report the baseline, current measurements, sample sizes, and time window together.
6. Separate detected anomalies from hypotheses about their root cause.

A useful next extension would be a dedicated **Release Regression Analysis** view that ranks service-region pairs by metric change, includes sample counts, and distinguishes observed regression from confirmed causality.

## Analytics methodology

### Anomaly detection

The analytics engine compares service-region observations against robust statistical baselines derived from the older release. It evaluates:

- p95 latency
- Error rate
- CPU utilization
- Memory utilization
- Queue depth

Thresholds and robust statistics are used to identify deviations and assign anomaly severity.

This approach is interpretable and useful for a controlled prototype. It is not a substitute for calibration against representative production telemetry, seasonality, traffic shifts, or service-specific operating limits.

### Release comparison

The release comparison workflow reports aggregate metric values and changes between releases, including absolute changes, relative changes, and sample counts.

A comparison across all services can hide a localized regression. The `v2.4.1` demo scenario affects only selected service-region cohorts, so service-level and region-level analysis is important.

Release comparisons should also account for unequal observation counts and time-window differences. A change after deployment is evidence of temporal association, not proof of causation.

### Evidence-grounded analyst

The default analyst is a deterministic intent router rather than a general-purpose LLM. It uses existing analytics functions to return supported findings and computed measurements.

This design prioritizes traceability: an explanation should be grounded in data returned by the analytics layer. An optional LLM could be added later, provided its claims are validated against structured evidence.

## Evaluation

The synthetic simulator supplies ground-truth labels for injected anomalies. The evaluation script calculates:

- True positives
- False positives
- True negatives
- False negatives
- Precision
- Recall
- F1 score
- Row and detection counts

Run the evaluation script to obtain the current metrics:

```powershell
python scripts\run_evaluation.py
```

Results are written to `outputs/evaluation.json`.

**Interpretation matters:** evaluation against controlled synthetic anomalies measures performance on the simulated distribution. It does not establish production detection accuracy, generalization to unseen incidents, or a specific operational throughput.

For a stronger evaluation, extend the experiment to include:

- Multiple random seeds
- Different anomaly magnitudes and durations
- Benign spikes and seasonal patterns
- Detection delay
- False-positive rates by service and metric
- Release-regression magnitude error
- Query latency and throughput at different dataset sizes

## Docker

With Docker Desktop installed, run:

```powershell
docker compose up --build
```

Then open:

http://127.0.0.1:8000/docs

The Docker Compose configuration mounts local data and output directories so generated artifacts can persist across container restarts.

## Limitations

- **Synthetic data:** No real production telemetry is collected.
- **Heuristic detection:** Thresholds and baselines require calibration for real workloads.
- **Causality:** Release comparisons reveal associations and do not prove that a release caused an incident.
- **Analyst scope:** The default question-answering component is deterministic and supports a limited set of intents; it is not an unrestricted LLM.
- **Local storage:** DuckDB and Parquet support a reproducible local workflow, not a claim of distributed-scale performance.
- **Security:** Authentication, authorization, rate limiting, and production hardening are not implemented as a complete production security layer.
- **Benchmark scope:** Results from the synthetic dataset should not be presented as production performance claims.

## Future work

1. Add service-region-specific release regression ranking.
2. Introduce time-window matching and confidence intervals for release comparisons.
3. Add seasonality-aware and multivariate anomaly detectors.
4. Extend evaluation with multiple seeds, benign anomalies, and detection-delay metrics.
5. Add optional LLM-generated explanations with structured evidence validation.
6. Benchmark ingestion, query latency, throughput, and larger datasets.
7. Add authentication, rate limits, structured logging, and deployment hardening.
8. Evaluate a PySpark processing path against the DuckDB baseline.

## Resume positioning

Use measured results from your own run and avoid claims that have not been benchmarked.

**SignalScope AI — Intelligent Telemetry Analytics Platform**

- Built a telemetry analytics platform using Python, FastAPI, DuckDB, and Parquet to analyze service performance and compare software releases.
- Implemented robust-baseline anomaly detection across latency, error rate, CPU, memory, and queue depth, evaluated against controlled synthetic ground truth.
- Developed evidence-grounded analytics responses and an interactive Streamlit dashboard for anomaly investigation and release comparison.
- Investigated a controlled release-regression scenario across three services, using service-level metric evidence to guide follow-up checks and avoid unsupported causal claims.

## License

Independent educational and research prototype. Add an explicit license file before public distribution.
