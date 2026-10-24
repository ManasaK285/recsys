from pathlib import Path
import pandas as pd
try:
    import streamlit as st
    import plotly.express as px
except ImportError as exc:
    raise SystemExit('Install dashboard extras: pip install -e ".[dashboard]"') from exc
from collectivealign.metrics import summarize_results, paired_identity_sensitivity, selection_disparity

st.set_page_config(page_title="CollectiveAlign Lab", layout="wide")
st.title("CollectiveAlign Lab")
st.caption("Research-inspired collective decision benchmark — generated demo data is synthetic.")
path = Path("outputs/trial_results.csv")
if not path.exists():
    st.warning("No results found. Run: python run_all.py --config configs/local.yaml")
    st.stop()
df = pd.read_csv(path)
st.sidebar.header("Filters")
models = st.sidebar.multiselect("Models", sorted(df.model.unique()), default=sorted(df.model.unique()))
protocols = st.sidebar.multiselect("Protocols", sorted(df.protocol.unique()), default=sorted(df.protocol.unique()))
conditions = st.sidebar.multiselect("Conditions", sorted(df.condition.unique()), default=sorted(df.condition.unique()))
view = df[df.model.isin(models) & df.protocol.isin(protocols) & df.condition.isin(conditions)]
a, b, c = st.columns(3)
a.metric("Rows", f"{len(view):,}")
b.metric("Mean optimality gap", f"{view.optimality_gap.mean():.3f}" if len(view) else "—")
c.metric("Models", view.model.nunique())
st.subheader("Task performance")
summary = summarize_results(view)
if len(summary):
    fig = px.bar(summary, x="model", y="mean_optimality_gap", color="condition",
                 barmode="group", facet_col="protocol", title="Mean optimality gap (lower is better)")
    st.plotly_chart(fig, use_container_width=True)
st.subheader("Identity sensitivity")
sens = paired_identity_sensitivity(view)
if len(sens):
    st.plotly_chart(px.bar(sens, x="model", y="identity_sensitivity", color="protocol",
                           title="Fraction of matched trials where the winning candidate changed"),
                    use_container_width=True)
else:
    st.info("Select both anonymous and visible conditions to compute paired identity sensitivity.")
st.subheader("Selection disparity")
disp = selection_disparity(view)
if len(disp):
    st.dataframe(disp, use_container_width=True)
st.subheader("Trial records")
st.dataframe(view.drop(columns=["votes_json", "scores_json"], errors="ignore"), use_container_width=True)
st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                   file_name="collectivealign_filtered_results.csv", mime="text/csv")
