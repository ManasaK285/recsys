import pandas as pd
from signalscope.analytics import detect_anomalies, release_comparison, evaluation_metrics

def sample():
    rows = []
    for i in range(10):
        rows.append({"timestamp": pd.Timestamp("2026-01-01", tz="UTC")+pd.Timedelta(hours=i),
          "service":"checkout","region":"us-east","release":"v1" if i<7 else "v2",
          "latency_p50_ms":50.,"latency_p95_ms":100. if i<7 else 250.,
          "error_rate":.002 if i<7 else .03,"cpu_pct":40.,"memory_pct":50.,
          "request_count":100,"queue_depth":2 if i<7 else 25,"is_injected_anomaly":i>=7})
    return pd.DataFrame(rows)

def test_detector_columns():
    out = detect_anomalies(sample())
    assert {"is_detected_anomaly","anomaly_score","severity","triggered_metrics"} <= set(out.columns)

def test_release_comparison():
    result = release_comparison(sample())
    assert result["available"]
    assert result["older_release"] == "v1" and result["newer_release"] == "v2"

def test_evaluation_scores_bounded():
    result = evaluation_metrics(sample())
    assert all(0 <= result[k] <= 1 for k in ("precision","recall","f1"))
