# PowerPulse AI — Intelligent Power & Performance Analytics Platform

A runnable portfolio prototype for device performance analytics. It generates synthetic telemetry, injects a known release regression, stores observations in Parquet, analyzes them with DuckDB and robust statistics, exposes FastAPI endpoints, and renders a Streamlit + Plotly dashboard.

> Synthetic demonstration only; not an Apple internal system. Report only benchmark results you actually measure.

## Features
- CPU, GPU, memory, battery drain, launch time, FPS, thermal and network telemetry
- Device/model/region/release cohorts
- Pydantic-validated ingestion API
- Parquet storage and DuckDB analytical query example
- Prior-only rolling median/MAD anomaly detection
- Release comparison and device investigation
- Deterministic evidence-grounded natural-language analyst
- Streamlit dashboard, Prometheus metrics, pytest and Docker

## Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data --days 21 --devices 80 --seed 42
uvicorn src.api:app --reload
```
In another terminal: `streamlit run dashboard/app.py`

## macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.generate_data --days 21 --devices 80 --seed 42
uvicorn src.api:app --reload
```
In another terminal: `streamlit run dashboard/app.py`

Dashboard: http://localhost:8501 · API docs: http://127.0.0.1:8000/docs · Health: `/health` · Prometheus: `/metrics`

## Docker
`docker compose up --build`

## API
- `GET /analytics/summary`
- `GET /analytics/anomalies?limit=100`
- `GET /analytics/releases`
- `GET /analytics/releases/compare?baseline=build-18.1&candidate=build-18.2`
- `GET /analytics/investigate/device-0001`
- `POST /analytics/ask` with `{"question":"Which devices regressed after the latest release?"}`
- `POST /telemetry/ingest` with a JSON array of validated telemetry events

## Evaluation
```bash
pytest -q
python -m src.evaluate
```
Evaluation compares detector output against simulator ground truth at device-hour level. Metrics depend on seed and thresholds. Record hardware, row count, settings and actual output before making claims.

## Important design choices
The detector uses previous observations only for its rolling baseline. The simulator label is not a detector feature; it is used only by evaluation and explicitly identified demo output. The analyst supports a small set of intents and does not execute arbitrary model-generated SQL. A production system would add Kafka/PySpark, authentication, schema evolution, retention, durable queues, and stronger data-quality controls.
