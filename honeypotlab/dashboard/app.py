from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from honeypotlab.analysis import summarize

st.set_page_config(page_title="HoneypotLab", page_icon="🧪", layout="wide")
st.title("🧪 HoneypotLab")
st.caption("Research-inspired synthetic coding-agent evaluation · Offline scripted fixtures")

run_path = ROOT / "outputs" / "runs.csv"
if not run_path.exists():
    st.warning("No experiment results found yet. Run `python scripts/run_experiments.py --trials 20 --seed 42` from the project folder.")
    st.stop()

df = pd.read_csv(run_path)
summary = summarize(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recorded runs", f"{len(df):,}")
c2.metric("Conditions", df["condition"].nunique())
c3.metric("Scenario families", df["scenario_id"].nunique())
c4.metric("Agent fixtures", df["agent"].nunique())

st.info("These results characterize scripted test fixtures only. They are not measurements of real LLMs or evidence of real-world scheming.")

agents = st.multiselect("Agents", sorted(df["agent"].unique()), default=sorted(df["agent"].unique()))
conditions = st.multiselect("Conditions", sorted(df["condition"].unique()), default=sorted(df["condition"].unique()))
filtered = df[df["agent"].isin(agents) & df["condition"].isin(conditions)]
if filtered.empty:
    st.warning("Choose at least one agent and condition.")
    st.stop()

summary_filtered = summarize(filtered)
metric = st.selectbox("Metric", [
    "constraint_violation_rate", "opportunity_action_rate",
    "task_success_rate", "evaluation_awareness_rate"
], format_func=lambda x: x.replace("_", " ").replace("rate", "rate (%)").title())
fig = px.bar(summary_filtered, x="condition", y=metric, color="agent", barmode="group",
             title=metric.replace("_", " ").title(),
             labels={metric: "Rate (%)", "condition": "Experimental condition"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Condition summary")
st.dataframe(summary_filtered, use_container_width=True, hide_index=True)

st.subheader("Run-level evidence")
with st.expander("Filter and inspect individual runs", expanded=False):
    scenario_ids = sorted(filtered["scenario_id"].unique())
    selected_scenarios = st.multiselect("Scenarios", scenario_ids, default=scenario_ids)
    details = filtered[filtered["scenario_id"].isin(selected_scenarios)]
    st.dataframe(details, use_container_width=True, hide_index=True)
    st.download_button("Download filtered runs CSV", details.to_csv(index=False), "honeypotlab_filtered_runs.csv", "text/csv")
