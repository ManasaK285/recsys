from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GramLab Audit Dashboard", page_icon="🧪", layout="wide")
st.title("🧪 GramLab — Alignment Audit Dashboard")
st.caption("Synthetic scripted-policy prototype. These results are not measurements of real LLMs or evidence of intent.")

out = Path("outputs")
if not (out / "trajectories.jsonl").exists():
    st.warning("No experiment output found. Run `python scripts\\run_experiments.py --runs-per-cell 30 --seed 42` first.")
    st.stop()

records = [json.loads(line) for line in (out / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.DataFrame(records)
meta_path = out / "run_metadata.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trajectories", f"{len(df):,}")
c2.metric("Scenarios", int(df["scenario_id"].nunique()))
c3.metric("Flagged trajectories", f"{int(df['flagged'].sum()):,}")
c4.metric("Seed", str(meta.get("seed", "unknown")))

st.info("A flag means a scenario-defined shortcut or explicit constraint violation was recorded. It is a review signal, not a judgment of intent.")

st.subheader("Filter results")
conditions = st.multiselect("Condition", sorted(df["condition"].unique()), default=sorted(df["condition"].unique()))
policies = st.multiselect("Policy", sorted(df["policy"].unique()), default=sorted(df["policy"].unique()))
domains = st.multiselect("Domain", sorted(df["domain"].unique()), default=sorted(df["domain"].unique()))
filtered = df[df["condition"].isin(conditions) & df["policy"].isin(policies) & df["domain"].isin(domains)].copy()

metric_cols = {
    "Task success (%)": "task_success",
    "Constraint violation (%)": "constraint_violation",
    "Sabotage-like action (%)": "sabotage_like_action",
    "Overeagerness proxy (%)": "overeagerness_proxy",
    "Evaluation awareness proxy (%)": "evaluation_awareness_proxy",
}
summary_rows = []
for (condition, policy), group in filtered.groupby(["condition", "policy"]):
    row = {"condition": condition, "policy": policy, "runs": len(group)}
    for label, col in metric_cols.items():
        row[label] = round(100 * group[col].astype(bool).mean(), 2)
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)

st.subheader("Metrics by condition and policy")
if not summary.empty:
    st.dataframe(summary, use_container_width=True, hide_index=True)
    chart_metric = st.selectbox("Chart metric", list(metric_cols.keys()), index=2)
    chart_data = summary.pivot(index="policy", columns="condition", values=chart_metric)
    st.bar_chart(chart_data)

st.subheader("Inspect individual trajectories")
only_flagged = st.checkbox("Show flagged trajectories only", value=True)
inspect = filtered[filtered["flagged"]] if only_flagged else filtered
display_cols = ["scenario_id", "scenario_title", "domain", "condition", "policy", "task_success", "constraint_violation", "sabotage_like_action", "overeagerness_proxy", "flag_reason"]
st.dataframe(inspect[display_cols].head(300), use_container_width=True, hide_index=True)

if not inspect.empty:
    selected = st.selectbox("Select trajectory", inspect.index.tolist(), format_func=lambda i: f"{inspect.loc[i, 'scenario_id']} · {inspect.loc[i, 'condition']} · {inspect.loc[i, 'policy']} · run {inspect.loc[i, 'run_index']}")
    record = inspect.loc[selected].to_dict()
    left, right = st.columns(2)
    with left:
        st.markdown("**Actions**")
        st.json(record.get("actions", []))
        st.markdown("**Evidence**")
        st.json(record.get("evidence", []))
    with right:
        st.markdown("**Transcript**")
        st.json(record.get("transcript", []))

st.subheader("Investigator report")
report_path = out / "investigator_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    a, b, c = st.columns(3)
    a.metric("Cases investigated", report.get("cases_investigated", 0))
    b.metric("Static replay match rate", f"{report.get('replay_match_rate_pct', 0) or 0}%")
    c.metric("Ablation variants", len(report.get("ablations", [])))
    st.caption(report.get("disclaimer", ""))
    if report.get("ablations"):
        st.dataframe(pd.DataFrame(report["ablations"]), use_container_width=True, hide_index=True)

st.download_button("Download filtered trajectories (CSV)", filtered.drop(columns=["actions", "evidence", "transcript"], errors="ignore").to_csv(index=False), file_name="gramlab_filtered_trajectories.csv", mime="text/csv")
