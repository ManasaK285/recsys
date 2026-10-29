import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API = os.getenv("SIGNALSCOPE_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="SignalScope AI", page_icon="📡", layout="wide")
st.title("📡 SignalScope AI")
st.caption("Evidence-grounded telemetry analytics and release investigation")

@st.cache_data(ttl=10)
def get_json(path, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

try:
    summary, evaluation, comparison = get_json("/summary"), get_json("/evaluation"), get_json("/releases/compare")
except Exception as exc:
    st.error(f"Cannot reach {API}. Start the API with: python -m uvicorn signalscope.api:app --reload")
    st.exception(exc)
    st.stop()

a,b,c,d = st.columns(4)
a.metric("Telemetry rows", f"{summary['row_count']:,}")
b.metric("Services", summary["service_count"])
c.metric("Regions", summary["region_count"])
d.metric("Injected anomaly rows", f"{evaluation['ground_truth_anomalies']:,}")

left,right = st.columns(2)
left.metric("Mean p95 latency", f"{summary.get('mean_latency_p95_ms', 0):.1f} ms")
left.metric("Total requests", f"{summary.get('total_requests', 0):,}")
right.metric("Detection precision (synthetic)", f"{evaluation['precision']:.3f}")
right.metric("Detection recall (synthetic)", f"{evaluation['recall']:.3f}")
st.caption("Synthetic-data metrics are not production accuracy estimates.")

st.subheader("Release comparison")
if comparison.get("available"):
    cmp = pd.DataFrame(comparison["metrics"])
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    row = cmp[cmp.metric == "latency_p95_ms"]
    if not row.empty:
        x = row.iloc[0]
        st.info(f"Mean p95 latency: {x.older_mean:.2f} → {x.newer_mean:.2f} ms ({x.percent_change:.1f}%). Association does not prove causation.")
else:
    st.warning(comparison.get("message", "No comparison available."))

st.subheader("Anomaly explorer")
with st.form("filters"):
    x,y,z = st.columns(3)
    service = x.text_input("Service (optional)")
    region = y.text_input("Region (optional)")
    limit = z.slider("Rows", 10, 500, 100, 10)
    submitted = st.form_submit_button("Find anomalies")
if submitted:
    params = {"limit": limit}
    if service.strip(): params["service"] = service.strip()
    if region.strip(): params["region"] = region.strip()
    try:
        payload = get_json("/anomalies", params)
        anomalies = pd.DataFrame(payload["items"])
        st.write(f"Returned {payload['count']} rows")
        if not anomalies.empty:
            st.dataframe(anomalies, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(anomalies.head(20), x="service", y="anomaly_score", color="severity", title="Top anomaly scores"), use_container_width=True)
        else: st.info("No anomalies matched these filters.")
    except Exception as exc: st.error(str(exc))

st.subheader("Ask the telemetry analyst")
question = st.text_input("Question", "Which services regressed after the latest release?")
if st.button("Investigate"):
    try:
        r = requests.post(f"{API}/ask", json={"question": question}, timeout=60)
        r.raise_for_status()
        answer = r.json()
        st.markdown(answer.get("answer", "No answer"))
        if answer.get("hypotheses"):
            st.markdown("**Hypotheses to validate**")
            for item in answer["hypotheses"]: st.write(f"- {item}")
        if answer.get("evidence"): st.dataframe(pd.DataFrame(answer["evidence"]), use_container_width=True, hide_index=True)
        if answer.get("next_checks"):
            st.markdown("**Recommended next checks**")
            for item in answer["next_checks"]: st.write(f"- {item}")
        st.caption("Findings are computed from telemetry; correlation is not proof of causation.")
    except Exception as exc: st.error(str(exc))
st.divider()
st.caption("Local research prototype • Synthetic telemetry • No external LLM required")
