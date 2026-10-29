import pandas as pd
from signalscope.insights import answer_question

def frame():
    return pd.DataFrame([{"timestamp":pd.Timestamp("2026-01-01",tz="UTC"),"service":"checkout","region":"us-east","release":"v1","latency_p50_ms":50.,"latency_p95_ms":100.,"error_rate":.01,"cpu_pct":50.,"memory_pct":60.,"request_count":100,"queue_depth":1,"is_injected_anomaly":False}])

def test_unsupported_question_is_transparent():
    result = answer_question("Tell me a joke", frame())
    assert "could not map" in result["answer"]
    assert "supported_intents" in result

def test_summary_uses_computed_values():
    result = answer_question("Summarize telemetry", frame())
    assert result["evidence"][0]["row_count"] == 1
