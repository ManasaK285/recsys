"""Streamlit dashboard for OptiSimplicity Lab."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from optisimplicity.experiments.run_all import run_all

st.set_page_config(page_title="OptiSimplicity Lab", page_icon="🧩", layout="wide")
st.title("🧩 OptiSimplicity Lab")
st.caption("Research-inspired experiments on description length, optimization search, and coincidence of extrema.")
st.warning("Compressed length is only a proxy for Kolmogorov complexity. The simplicity-biased sampler is not true universal algorithmic probability.")

with st.sidebar:
    st.header("Experiment settings")
    n_bits = st.slider("Binary string length", min_value=4, max_value=12, value=8)
    seeds = st.slider("Seeds per search family", min_value=1, max_value=10, value=3)
    budget = st.slider("Objective evaluations per run", min_value=25, max_value=1000, value=200, step=25)
    beta = st.slider("Simplicity bias β", min_value=0.0, max_value=2.0, value=0.25, step=0.05)
    pairs = st.slider("Objective pairs", min_value=3, max_value=60, value=12)
    null_samples = st.slider("Null samples per pair", min_value=20, max_value=500, value=100, step=20)
    seed = st.number_input("Master seed", min_value=0, max_value=1000000, value=42)
    run_button = st.button("Run all experiments", type="primary", use_container_width=True)

out = Path("outputs")
if run_button:
    with st.spinner("Running exact enumeration, search benchmarks, and null-model comparisons..."):
        summary = run_all(out, n_bits=n_bits, seeds=seeds, budget=budget, beta=beta,
                          extrema_pairs=pairs, null_samples=null_samples, seed=int(seed))
    st.success("Experiments completed. Outputs were written to the outputs/ directory.")
    st.session_state["last_summary"] = summary

summary_path = out / "summary.json"
if not summary_path.exists():
    st.info("Configure the experiment in the sidebar and select **Run all experiments** to generate results.")
    st.markdown("""
    **Included studies**
    - A: Complexity of exact optima vs. the candidate population
    - B: Uniform sampling, simplicity-biased sampling, hill climbing, and simulated annealing
    - C: Optimal-set coincidence compared with a shuffled-score null model
    """)
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
st.caption(f"Latest saved configuration: {summary['configuration']}")
tab_a, tab_b, tab_c, tab_d = st.tabs(["A · Complexity of optima", "B · Search benchmark", "C · Extrema coincidence", "Artifacts"])

with tab_a:
    st.subheader("Estimated description length of exact optima")
    data = pd.read_csv(out / "complexity_optima.csv")
    st.dataframe(data, use_container_width=True)
    fig = px.bar(data, x="problem", y="optimum_minus_population_median_bits",
                 title="Optimum mean description length minus population median",
                 labels={"problem": "Objective family", "optimum_minus_population_median_bits": "Difference (bits)"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Negative values mean the optima's average compressed length is below the population median for this encoding. This is not a universal complexity result.")

with tab_b:
    st.subheader("Search algorithm comparison")
    runs = pd.read_csv(out / "search_runs.csv")
    agg = pd.read_csv(out / "search_aggregates.csv")
    left, right = st.columns(2)
    with left:
        fig = px.bar(agg, x="algorithm", y="mean_optimality_gap", color="problem",
                     barmode="group", title="Mean optimality gap (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(agg, x="algorithm", y="optimum_success_rate", color="problem",
                     barmode="group", title="Exact-optimum success rate")
        st.plotly_chart(fig, use_container_width=True)
    curves = pd.read_csv(out / "search_curves.csv")
    selected_problem = st.selectbox("Problem family for convergence curves", sorted(curves["problem"].unique()))
    selected = curves[curves["problem"] == selected_problem]
    curve_agg = selected.groupby(["algorithm", "evaluation"], as_index=False)["best_objective"].mean()
    fig = px.line(curve_agg, x="evaluation", y="best_objective", color="algorithm",
                  title=f"Mean best-so-far objective — {selected_problem}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(agg, use_container_width=True)
    st.caption("Compare both evaluation budget and runtime. Simplicity-biased candidate weights incur preprocessing cost.")

with tab_c:
    st.subheader("Observed optimal-set overlap vs. null model")
    pairs_df = pd.read_csv(out / "extrema_pairs.csv")
    st.dataframe(pairs_df, use_container_width=True)
    fig = px.scatter(pairs_df, x="null_mean_jaccard", y="jaccard_overlap", color="objective_a",
                     hover_data=["objective_b", "optima_a", "optima_b"],
                     title="Observed Jaccard overlap vs. mean shuffled-null overlap",
                     labels={"null_mean_jaccard": "Null mean Jaccard overlap", "jaccard_overlap": "Observed Jaccard overlap"})
    low = min(pairs_df["null_mean_jaccard"].min(), pairs_df["jaccard_overlap"].min())
    high = max(pairs_df["null_mean_jaccard"].max(), pairs_df["jaccard_overlap"].max())
    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points above the diagonal have more observed overlap than the null mean. This is descriptive; use repeated pairs and uncertainty estimates before making inferential claims.")

with tab_d:
    st.subheader("Reproducible output artifacts")
    for filename in ["summary.json", "complexity_optima.csv", "search_runs.csv", "search_curves.csv",
                     "search_aggregates.csv", "extrema_pairs.csv", "extrema_null_samples.csv"]:
        path = out / filename
        if path.exists():
            st.write(f"**{filename}**")
            st.download_button(f"Download {filename}", path.read_bytes(), file_name=filename,
                               mime="application/json" if filename.endswith(".json") else "text/csv",
                               key=f"download_{filename}")
    st.json(summary)
