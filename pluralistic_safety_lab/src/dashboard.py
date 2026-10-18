"""Optional Streamlit dashboard for generated analysis artifacts."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
st.set_page_config(page_title="Pluralistic Safety Lens", layout="wide")
st.title("Pluralistic Safety Lens")
st.caption("Exploratory analysis of ordinal safety ratings across rater groups.")
st.warning("Group differences are descriptive, not causal. Demo data is synthetic unless you supplied a real CSV.")

summary_path = REPORTS / "summary.json"
if not summary_path.exists():
    st.info("No reports found. Run `python -m src.run_pipeline` from the project root first.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ratings", f"{summary['n_ratings']:,}")
c2.metric("Items", f"{summary['n_items']:,}")
c3.metric("Raters", f"{summary['n_raters']:,}")
c4.metric("Groups", summary["n_groups"])
st.write(f"**Data source:** `{summary['data_source']}`")

group_path = REPORTS / "group_metrics.csv"
items_path = REPORTS / "item_disagreement.csv"
if group_path.exists():
    st.subheader("Group responsiveness summaries")
    groups = pd.read_csv(group_path)
    st.dataframe(groups, use_container_width=True)
    metric = st.selectbox("Metric to compare", [
        "scale_span", "distinct_rating_levels", "monotonicity",
        "severity_spearman", "rating_entropy"
    ])
    st.bar_chart(groups.set_index("group")[[metric]])
for title, filename in [
    ("Rating distributions", "fig_group_rating_distributions.png"),
    ("Items with most disagreement", "fig_item_disagreement.png"),
    ("Responsiveness overview", "fig_group_responsiveness.png"),
]:
    path = REPORTS / filename
    if path.exists():
        st.subheader(title)
        st.image(str(path), use_container_width=True)
if items_path.exists():
    st.subheader("Item-level disagreement")
    items = pd.read_csv(items_path)
    min_entropy = st.slider("Minimum normalized entropy", 0.0, 1.0, 0.5, 0.05)
    st.dataframe(items[items["normalized_entropy"] >= min_entropy], use_container_width=True)
st.subheader("Highest-disagreement items")
st.dataframe(pd.DataFrame(summary["high_disagreement_items"]), use_container_width=True)
