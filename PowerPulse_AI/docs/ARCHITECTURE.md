# Architecture

1. Seeded generator creates device-hour telemetry and controlled build regression.
2. FastAPI/Pydantic validates incoming batches; Parquet stores observations.
3. DuckDB provides SQL aggregation over Parquet; pandas/NumPy implement prior-only rolling median/MAD anomaly detection.
4. Release comparison computes cohort means and deltas; device investigation produces evidence and cautious hypotheses.
5. Analyst maps supported natural-language intents to deterministic analytics. No arbitrary LLM-generated SQL is executed.
6. Streamlit calls the API and renders Plotly charts.

## Production gaps
This is a portfolio prototype. A production service needs durable streaming, idempotency keys, schema evolution, auth/RBAC, audit logs, retention controls, backpressure, late-event handling, data-quality SLAs, distributed query execution, alert routing, and deployment/runbook automation.

## Causality
A before/after difference is an association, not proof of causation. Validate with matched cohorts, traces, controlled experiments, rollback tests, and confounder analysis.
