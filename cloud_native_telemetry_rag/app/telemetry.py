from collections import deque
from datetime import datetime, timezone
from threading import Lock
import math
import statistics

class TelemetryStore:
    """Thread-safe in-memory metrics store for a single-process demo."""
    def __init__(self, max_samples=5000):
        self.lock = Lock()
        self.requests = 0
        self.errors = 0
        self.latencies_ms = deque(maxlen=max_samples)
        self.query_tokens = deque(maxlen=max_samples)
        self.events = deque(maxlen=max_samples)
        self.last_recall = None
        self.recall_history = deque(maxlen=200)

    def record_request(self, method, path, status, latency_ms):
        with self.lock:
            self.requests += 1
            self.errors += int(status >= 500)
            self.latencies_ms.append(latency_ms)
            self.events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "http_request", "method": method, "path": path,
                "status_code": status, "latency_ms": round(latency_ms, 3)
            })

    def record_query(self, input_tokens, output_tokens, latency_ms, retrieved_ids):
        with self.lock:
            self.query_tokens.append(input_tokens + output_tokens)
            self.events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "rag_query", "latency_ms": round(latency_ms, 3),
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "retrieved_document_ids": retrieved_ids
            })

    def record_recall(self, recall, k):
        with self.lock:
            self.last_recall = recall
            self.recall_history.append(recall)
            self.events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "retrieval_evaluation", "k": k, "recall_at_k": recall
            })

    @staticmethod
    def percentile(values, p):
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
        return float(ordered[index])

    def snapshot(self):
        with self.lock:
            latencies = list(self.latencies_ms)
            tokens = list(self.query_tokens)
            requests, errors = self.requests, self.errors
            recall = self.last_recall
            return {
                "requests_total": requests,
                "errors_total": errors,
                "error_rate": errors / requests if requests else 0.0,
                "latency_ms": {
                    "p50": round(self.percentile(latencies, .50), 3),
                    "p95": round(self.percentile(latencies, .95), 3),
                    "max": round(max(latencies), 3) if latencies else 0.0,
                    "samples": len(latencies),
                },
                "estimated_tokens_per_query": {
                    "average": round(statistics.mean(tokens), 2) if tokens else 0.0,
                    "samples": len(tokens),
                },
                "retrieval": {"latest_recall_at_k": recall, "evaluation_count": len(self.recall_history)}
            }

    def events_copy(self):
        with self.lock:
            return list(self.events)

    def alerts(self, settings):
        snap = self.snapshot()
        alerts = []
        p95 = snap["latency_ms"]["p95"]
        if snap["latency_ms"]["samples"] and p95 > settings.p95_latency_threshold_ms:
            alerts.append({"severity": "warning", "rule": "p95_latency", "value": p95,
                           "threshold": settings.p95_latency_threshold_ms,
                           "message": "p95 request latency exceeded threshold"})
        if snap["requests_total"] >= 20 and snap["error_rate"] > settings.error_rate_threshold:
            alerts.append({"severity": "critical", "rule": "error_rate", "value": snap["error_rate"],
                           "threshold": settings.error_rate_threshold, "message": "HTTP error rate exceeded threshold"})
        avg_tokens = snap["estimated_tokens_per_query"]["average"]
        if snap["estimated_tokens_per_query"]["samples"] and avg_tokens > settings.avg_token_threshold:
            alerts.append({"severity": "warning", "rule": "token_utilization", "value": avg_tokens,
                           "threshold": settings.avg_token_threshold,
                           "message": "Average estimated token utilization exceeded threshold"})
        recall = snap["retrieval"]["latest_recall_at_k"]
        if recall is not None:
            if recall < settings.min_recall_at_k:
                alerts.append({"severity": "critical", "rule": "retrieval_recall", "value": recall,
                               "threshold": settings.min_recall_at_k, "message": "Recall@k is below minimum threshold"})
            drift = settings.recall_baseline - recall
            if drift > settings.recall_drift_tolerance:
                alerts.append({"severity": "warning", "rule": "retrieval_recall_drift", "value": drift,
                               "threshold": settings.recall_drift_tolerance,
                               "message": "Recall@k drifted below configured baseline"})
        return alerts

telemetry = TelemetryStore()
