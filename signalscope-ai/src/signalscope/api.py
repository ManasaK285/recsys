from functools import lru_cache
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from .schemas import TelemetryBatch, AskRequest
from .simulator import generate_telemetry
from .storage import TelemetryStore
from .analytics import detect_anomalies, summarize, release_comparison, evaluation_metrics
from .insights import answer_question

app = FastAPI(title="SignalScope AI", version="1.0.0", description="Evidence-grounded telemetry analytics prototype")
COUNTERS = {"requests": 0, "errors": 0, "start": time.time()}

@lru_cache(maxsize=1)
def get_store():
    return TelemetryStore()

def ensure_data():
    store = get_store()
    if store.count() == 0:
        store.replace(generate_telemetry())
    return store

@app.on_event("startup")
def startup():
    ensure_data()

@app.get("/health")
def health():
    store = ensure_data()
    return {"status": "ok", "rows": store.count(), "service": "signalscope-api"}

@app.get("/summary")
def summary_endpoint():
    COUNTERS["requests"] += 1
    return summarize(ensure_data().read())

@app.get("/anomalies")
def anomalies_endpoint(limit: int = Query(100, ge=1, le=5000), service: str | None = None, region: str | None = None):
    COUNTERS["requests"] += 1
    df = ensure_data().read()
    if service: df = df[df.service == service]
    if region: df = df[df.region == region]
    out = detect_anomalies(df)
    out = out[out.is_detected_anomaly].sort_values("anomaly_score", ascending=False).head(limit).copy()
    out["timestamp"] = out.timestamp.astype(str)
    cols = ["timestamp", "service", "region", "release", "latency_p95_ms", "error_rate", "cpu_pct", "memory_pct", "queue_depth", "anomaly_score", "severity", "triggered_metrics", "is_injected_anomaly"]
    return {"count": len(out), "items": out[cols].to_dict("records")}

@app.get("/releases/compare")
def compare_endpoint(service: str | None = None, region: str | None = None):
    COUNTERS["requests"] += 1
    return release_comparison(ensure_data().read(), service, region)

@app.get("/evaluation")
def evaluation_endpoint():
    COUNTERS["requests"] += 1
    return evaluation_metrics(ensure_data().read())

@app.post("/telemetry")
def ingest_endpoint(batch: TelemetryBatch):
    COUNTERS["requests"] += 1
    try:
        df = pd.DataFrame([e.model_dump() for e in batch.events])
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
        total = ensure_data().append(df)
        return {"accepted": len(df), "total_rows": total}
    except Exception as exc:
        COUNTERS["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

@app.post("/ask")
def ask_endpoint(request: AskRequest):
    COUNTERS["requests"] += 1
    return answer_question(request.question, ensure_data().read())

@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint():
    store = ensure_data()
    return ("# TYPE signalscope_api_requests_total counter\n"
      f"signalscope_api_requests_total {COUNTERS['requests']}\n"
      "# TYPE signalscope_api_errors_total counter\n"
      f"signalscope_api_errors_total {COUNTERS['errors']}\n"
      "# TYPE signalscope_telemetry_rows gauge\n"
      f"signalscope_telemetry_rows {store.count()}\n"
      "# TYPE signalscope_uptime_seconds gauge\n"
      f"signalscope_uptime_seconds {max(0, time.time()-COUNTERS['start']):.3f}\n")
