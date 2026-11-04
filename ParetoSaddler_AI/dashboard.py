import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from paretosaddler.optimizer import run_experiment

st.set_page_config(page_title="ParetoSaddler AI", page_icon="⚖️", layout="wide")
st.title("ParetoSaddler AI")
st.caption("Pareto-guided agent harness optimization · Synthetic research simulator")

with st.sidebar:
    st.header("Experiment controls")
    iterations = st.slider("Optimization iterations", 5, 60, 24)
    repeats = st.slider("Rollouts per task", 1, 8, 3)
    draws = st.select_slider("Bayesian bootstrap draws", options=[300, 600, 1200, 2500], value=1200)
    gamma = st.slider("Promotion threshold γ", 0.50, 0.95, 0.70, 0.05)
    mode = st.selectbox("Deployment preference", ["pareto", "min_cost", "weighted"])
    quality_floor = st.slider("Quality floor", 0.0, 1.0, 0.60, 0.05)
    cost_weight = st.slider("Cost weight (weighted mode)", 0.0, 1.0, 0.15, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
    run = st.button("Run optimization", type="primary", use_container_width=True)

if run:
    with st.spinner("Running stochastic rollouts and Bayesian screening..."):
        result = run_experiment(iterations=iterations, repeats=repeats,
            bootstrap_draws=draws, gamma=gamma, mode=mode,
            quality_floor=quality_floor, cost_weight=cost_weight, seed=int(seed))
    st.session_state["result"] = result

if "result" not in st.session_state:
    st.info("Configure the experiment in the sidebar and click **Run optimization**.")
    st.markdown("""
    **Pipeline:** Candidate mutation → repeated screening → Bayesian bootstrap
    → candidate promotion → development evaluation → Pareto archive → test evaluation.

    No API key is required. All benchmark data and model behavior are simulated.
    """)
else:
    result = st.session_state["result"]
    history = result["history"]
    records = result["records"]
    front = result["pareto_front"]
    test = result["test_summary"]
    promoted = int(history["promoted"].sum()) - 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated harnesses", len(records))
    c2.metric("Promoted candidates", promoted)
    c3.metric("Pareto alternatives", len(front))
    c4.metric("Test-selected", len(test))

    st.subheader("Quality vs. cost")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([x["cost"] for x in records], [x["quality"] for x in records],
               alpha=.55, label="Evaluated")
    ordered = sorted(front, key=lambda x: x["cost"])
    if ordered:
        ax.plot([x["cost"] for x in ordered], [x["quality"] for x in ordered],
                marker="o", linewidth=2, label="Pareto frontier")
    ax.set_xlabel("Mean cost per task (USD-equivalent)")
    ax.set_ylabel("Mean quality")
    ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Optimization trace")
    st.line_chart(history.set_index("iteration")[["promotion_support"]])
    st.dataframe(history, use_container_width=True, hide_index=True)

    st.subheader("Held-out test evaluation")
    st.dataframe(pd.DataFrame(test), use_container_width=True, hide_index=True)
    st.caption("Test domains are held out from candidate generation and development selection.")
