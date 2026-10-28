from contextlib import asynccontextmanager
from time import perf_counter
import logging
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.schemas import QueryRequest, QueryResponse, EvaluateRequest
from app.telemetry import telemetry
from app.retrieval import FaissRetriever
from app.rag import RAGEngine
from app.sinks import sink

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
HTTP_REQUESTS = Counter("rag_http_requests_total", "Total HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("rag_http_request_duration_seconds", "HTTP request duration", ["method", "path"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = FaissRetriever(settings.docs_dir, settings.embedding_model)
    app.state.rag = RAGEngine(app.state.retriever)
    yield

app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def collect_request_telemetry(request: Request, call_next):
    start = perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        path = request.url.path
        telemetry.record_request(request.method, path, status, elapsed_ms)
        HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, path).observe(elapsed_ms / 1000)

@app.get("/")
def root():
    return {"service": settings.app_name, "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}

@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request):
    start = perf_counter()
    answer, retrieved, input_tokens, output_tokens, mode = request.app.state.rag.answer(
        payload.question, payload.top_k)
    elapsed = (perf_counter() - start) * 1000
    telemetry.record_query(input_tokens, output_tokens, elapsed,
                           [doc["document_id"] for doc in retrieved])
    return QueryResponse(answer=answer, retrieved=retrieved, latency_ms=round(elapsed, 3),
                         estimated_input_tokens=input_tokens,
                         estimated_output_tokens=output_tokens, generation_mode=mode)

@app.get("/metrics")
def metrics():
    return telemetry.snapshot()

@app.get("/metrics/prometheus")
def prometheus_metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.get("/alerts")
def alerts():
    current = telemetry.alerts(settings)
    return {"active_alert_count": len(current), "alerts": current}

@app.post("/evaluate")
def evaluate(payload: EvaluateRequest, request: Request):
    result = request.app.state.retriever.evaluate_recall(settings.eval_set_path, payload.k)
    telemetry.record_recall(result["recall_at_k"], payload.k)
    result["alerts"] = telemetry.alerts(settings)
    return result

@app.post("/export/bigquery")
def export_bigquery():
    return sink.export_events(telemetry.events_copy())

@app.get("/events")
def events(limit: int = 100):
    if limit < 1 or limit > 1000:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    items = telemetry.events_copy()
    return {"events": items[-limit:], "count": min(limit, len(items))}
