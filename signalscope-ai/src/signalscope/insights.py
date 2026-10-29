import pandas as pd
from .analytics import detect_anomalies, release_comparison, summarize

NEXT_CHECKS = [
 "Inspect traces and error logs for the affected service and time window.",
 "Compare request volume and resource utilization before and after deployment.",
 "Compare affected services with unaffected controls before attributing causality to a release."
]

def answer_question(question, df):
    q = question.lower().strip()
    result = {"question": question, "evidence": [], "hypotheses": [], "next_checks": NEXT_CHECKS}
    if df.empty:
        result["answer"] = "No telemetry is available. Generate or ingest data first."
        return result
    result["evidence_window"] = {"start": pd.Timestamp(df.timestamp.min()).isoformat(), "end": pd.Timestamp(df.timestamp.max()).isoformat(), "rows_examined": len(df)}
    if any(k in q for k in ("regress", "release", "version", "deployment", "deploy")):
        comp = release_comparison(df)
        if not comp["available"]:
            result["answer"] = comp["message"]
            return result
        result["evidence"] = comp["metrics"]
        latency = next((m for m in comp["metrics"] if m["metric"] == "latency_p95_ms"), None)
        error = next((m for m in comp["metrics"] if m["metric"] == "error_rate"), None)
        anomalies = detect_anomalies(df)
        grouped = anomalies[anomalies.is_detected_anomaly].groupby(["service", "region"]).size().sort_values(ascending=False).head(10)
        result["affected_service_candidates"] = [{"service": s, "region": r, "flagged_rows": int(n)} for (s, r), n in grouped.items()]
        msg = f"Compared {comp['newer_release']} with {comp['older_release']}."
        if latency: msg += f" Mean p95 latency changed by {latency['percent_change']}% ({latency['older_mean']} to {latency['newer_mean']} ms)."
        if error: msg += f" Mean error rate changed by {error['percent_change']}% ({error['older_mean']} to {error['newer_mean']})."
        result["answer"] = msg + " This is an association, not proof of causation."
        result["hypotheses"] = ["The newer release may be associated with a performance change; validate with traces and unaffected controls."]
        return result
    if any(k in q for k in ("anomal", "unusual", "abnormal", "incident")):
        out = detect_anomalies(df)
        top = out[out.is_detected_anomaly].sort_values("anomaly_score", ascending=False).head(20)
        top = top[["timestamp", "service", "region", "release", "latency_p95_ms", "error_rate", "anomaly_score", "severity", "triggered_metrics"]].copy()
        top["timestamp"] = top.timestamp.astype(str)
        result["answer"] = f"Detected {int(out.is_detected_anomaly.sum())} anomalous telemetry rows using configured thresholds. Showing up to 20."
        result["evidence"] = top.to_dict("records")
        return result
    if any(k in q for k in ("latency", "slow", "highest")):
        top = df.groupby("service", as_index=False).agg(mean_p95_latency_ms=("latency_p95_ms", "mean"), max_p95_latency_ms=("latency_p95_ms", "max"), rows=("latency_p95_ms", "size")).sort_values("mean_p95_latency_ms", ascending=False).head(10)
        result["answer"] = "Services ranked by mean p95 latency across the current dataset."
        result["evidence"] = top.round(3).to_dict("records")
        return result
    if any(k in q for k in ("summar", "overview", "telemetry", "status")):
        result["answer"] = "Telemetry overview computed from the current dataset."
        result["evidence"] = [summarize(df)]
        return result
    result["answer"] = "I could not map that question to a supported analytics operation. Try release regressions, anomalies, highest latency services, or telemetry summary."
    result["supported_intents"] = ["release comparison", "anomaly investigation", "latency ranking", "telemetry summary"]
    return result
