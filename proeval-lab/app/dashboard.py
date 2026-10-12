from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_experiment
from src.discovery.acquisition import STRATEGIES

st.set_page_config(page_title="ProEval-Lab", page_icon="🧪", layout="wide")
st.title("🧪 ProEval-Lab")
st.caption("Uncertainty-aware evaluation and proactive failure discovery — synthetic research prototype")

with st.sidebar:
    st.header("Experiment settings")
    n_cases = st.slider("Benchmark size", 100, 800, 300, 50)
    budget = st.slider("Evaluation budget", 20, min(250, n_cases - 1), min(60, n_cases - 1), 10)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)
    threshold = st.slider("Failure threshold", 0.30, 0.90, 0.65, 0.05)
    target_mae = st.slider("Target estimation MAE", 0.01, 0.10, 0.03, 0.01)
    strategy = st.selectbox("Acquisition strategy", STRATEGIES, index=3)
    run = st.button("Run evaluation", type="primary", use_container_width=True)

st.info("This demo uses synthetic labels. It demonstrates evaluation algorithms and experiment design, not real-world LLM performance.")
if run:
    with st.spinner("Fitting Gaussian Process and running active evaluation..."):
        result, curve, queried_df = run_experiment(
            n_cases=int(n_cases), budget=int(budget), seed=int(seed), strategy=strategy,
            checkpoints=12, threshold=float(threshold), target_mae=float(target_mae)
        )
    st.session_state["result"] = result
    st.session_state["curve"] = curve
    st.session_state["queried_df"] = queried_df
if "result" in st.session_state:
    r = st.session_state["result"]
    a,b,c,d = st.columns(4)
    a.metric("Ground-truth mean severity", f'{r["ground_truth_mean_severity"]:.3f}')
    b.metric("Estimated severity", f'{r["gp_estimate"]:.3f}')
    c.metric("Absolute estimation error", f'{r["gp_estimation_abs_error"]:.3f}')
    d.metric("Failures found", f'{r["failures_found"]} / {r["evaluations_used"]}')
    st.subheader("Evaluation budget")
    curve = st.session_state["curve"]
    if not curve.empty:
        fig = px.line(curve, x="budget_used", y=["gp_estimation_abs_error"], markers=True,
                      labels={"budget_used":"Evaluated cases", "value":"Absolute error", "variable":"Estimator"})
        fig.add_hline(y=target_mae, line_dash="dash", annotation_text="Target MAE")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Discovered cases")
    qdf = st.session_state["queried_df"].copy()
    qdf["failure_label"] = qdf["error_severity"].apply(lambda x: "Failure" if x >= threshold else "Non-failure")
    fig2 = px.scatter(qdf, x="difficulty", y="error_severity", color="category", symbol="failure_label",
                      hover_data=["case_id","text","ambiguity","distribution_shift"])
    fig2.add_hline(y=threshold, line_dash="dash", annotation_text="Failure threshold")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(qdf[["case_id","category","error_severity","is_failure","text"]].sort_values("error_severity", ascending=False),
                 use_container_width=True, hide_index=True)
    st.download_button("Download discovered cases CSV", qdf.to_csv(index=False), "discovered_cases.csv", "text/csv")
    st.json({k:v for k,v in r.items() if k != "queried_indices"})
else:
    st.markdown("### What this dashboard demonstrates")
    st.markdown("- Gaussian Process predictions and uncertainty\n- Budgeted selection of evaluation cases\n- Error-rate estimation against a known synthetic benchmark\n- Failure discovery by category\n- Exportable experiment artifacts")
