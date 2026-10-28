from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_metrics_shape():
    with TestClient(app) as client:
        body = client.get("/metrics").json()
        assert "latency_ms" in body
        assert "estimated_tokens_per_query" in body

def test_query_and_retrieval():
    with TestClient(app) as client:
        response = client.post("/query", json={
            "question": "How do I monitor p95 latency?", "top_k": 2})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert len(body["retrieved"]) > 0
        assert body["estimated_input_tokens"] > 0
