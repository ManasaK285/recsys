import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="POLCA-Lab", layout="wide")
st.title("POLCA-Lab — Stochastic Generative Optimization")
st.caption("Inspect candidate memory, stochastic search, diversity filtering, and convergence.")
files = sorted(Path("results").glob("*.json"))
if not files:
    st.info("Run scripts/run_experiment.py first.")
    st.stop()
selected = st.sidebar.selectbox("Experiment", files, format_func=lambda p: p.name)
data = json.loads(selected.read_text(encoding="utf-8"))
traj = pd.DataFrame(data["iterations"])

c1,c2,c3,c4 = st.columns(4)
c1.metric("Evaluations", data["total_evals"])
c2.metric("Memory", len(data["candidates"]))
c3.metric("Rejected", data["rejections"])
c4.metric("Final test", f"{traj.iloc[-1].best_test_score:.3f}")

st.subheader("Convergence")
st.line_chart(traj.set_index("evaluations")[["best_train_mean", "best_test_score"]])

st.subheader("Search efficiency")
st.line_chart(traj.set_index("iteration")[["accepted", "rejected", "memory_size"]])

st.subheader("Candidate memory")
candidates = pd.DataFrame(data["candidates"])
show = candidates[["cid", "mean", "variance", "count", "created_at", "program"]].sort_values("mean", ascending=False)
st.dataframe(show, use_container_width=True, hide_index=True)

st.subheader("Best candidate")
best = show.iloc[0]
st.code(best["program"], language="text")
