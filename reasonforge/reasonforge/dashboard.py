from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ReasonForge",
    page_icon="🧪",
    layout="wide",
)

st.title("ReasonForge")
st.caption("Reasoning-driven synthetic data generation and evaluation")

report_path = Path("artifacts/final_report.json")
data_path = Path("artifacts/synthetic_dataset.jsonl")

if not report_path.exists():
    st.warning("Run the pipeline first: `python -m reasonforge.pipeline --n 300`")
    st.stop()

report = json.loads(report_path.read_text(encoding="utf-8"))
rows = [
    json.loads(x)
    for x in data_path.read_text(encoding="utf-8").splitlines()
    if x.strip()
]
df = pd.DataFrame(rows)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Generated", report["generated"])
c2.metric("Accepted", report["accepted"])
c3.metric("Acceptance rate", report["acceptance_rate"])
c4.metric("Diversity", report["diversity_score"])

if "downstream_accuracy" in report:
    c5, c6 = st.columns(2)
    c5.metric("Downstream accuracy", report["downstream_accuracy"])
    c6.metric("Downstream macro-F1", report["downstream_macro_f1"])

st.subheader("Intent coverage")
st.bar_chart(pd.Series(report["intent_coverage"]))

st.subheader("Complexity coverage")
st.bar_chart(pd.Series(report["complexity_coverage"]))

st.subheader("Synthetic examples")
if not df.empty:
    cols = [
        "sample_id",
        "intent",
        "complexity",
        "instruction",
        "quality_score",
    ]
    st.dataframe(df[cols], use_container_width=True, height=500)

st.subheader("Quality report")
st.json(report)
