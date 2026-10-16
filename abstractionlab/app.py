import json
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from abstractionlab.experiments import run_all

st.set_page_config(page_title="AbstractionLab", layout="wide")
st.title("AbstractionLab")
st.subheader("Simulation, substrate, and observer-defined representations")
st.warning("This is a toy research sandbox. Its metrics do not measure consciousness and cannot prove or disprove sentience.")

with st.sidebar:
    st.header("Experiment settings")
    steps = st.slider("Time steps", 100, 2000, 500, 100)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)
    noise = st.slider("World noise", 0.0, 0.15, 0.025, 0.005)
    target = st.slider("Target state", -0.5, 1.0, 0.75, 0.05)
    run = st.button("Run experiments", type="primary")

if run or "abstractionlab_summary" not in st.session_state:
    with st.spinner("Running reproducible toy experiments..."):
        summary = run_all(int(steps), int(seed), float(noise), float(target), "outputs")
        st.session_state["abstractionlab_summary"] = summary

summary = st.session_state["abstractionlab_summary"]
out = Path("outputs")
trajectory = pd.read_csv(out / "trajectory.csv")
substrate = pd.read_csv(out / "substrate_comparison.csv")

st.markdown("### Experiment A — Mapmaker sensitivity")
a, b, c = st.columns(3)
a.metric("Symbol disagreement", f"{summary['map_experiment']['symbol_disagreement_coarse_vs_balanced']:.1%}")
b.metric("Action agreement", f"{summary['map_experiment']['action_agreement_coarse_vs_balanced']:.1%}")
c.metric("Toy task success", str(summary["map_experiment"]["task_success"]))

fig, ax = plt.subplots()
ax.plot(trajectory["step"], trajectory["state"], label="Continuous state")
ax.axhline(target, linestyle="--", label="Target")
ax.set_xlabel("Step")
ax.set_ylabel("State")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.caption("The same continuous trajectory is classified through different observer-selected symbol maps.")
st.dataframe(trajectory.head(30), use_container_width=True)

st.markdown("### Experiment B — Behavioral output vs. internal state")
x, y = st.columns(2)
x.metric("Output agreement", f"{summary['substrate_comparison']['output_agreement']:.1%}")
y.metric("Mean absolute state difference", f"{summary['substrate_comparison']['mean_absolute_state_difference']:.4f}")

fig2, ax2 = plt.subplots()
ax2.plot(substrate["step"], substrate["discrete_state"], label="Discrete toy state")
ax2.plot(substrate["step"], substrate["continuous_state"], label="Continuous toy state")
ax2.set_xlabel("Step")
ax2.set_ylabel("Internal state")
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

st.dataframe(substrate.head(30), use_container_width=True)
st.markdown("### Interpretation guardrails")
st.write("- A changed label demonstrates sensitivity to the selected discretization.")
st.write("- Similar outputs demonstrate behavioral agreement on this tested input sequence.")
st.write("- Different state traces demonstrate differences between these toy dynamics.")
st.write("- None of these observations establish whether a system has subjective experience.")

st.download_button("Download summary.json", json.dumps(summary, indent=2), file_name="summary.json", mime="application/json")
st.download_button("Download trajectory.csv", trajectory.to_csv(index=False), file_name="trajectory.csv", mime="text/csv")
st.download_button("Download substrate_comparison.csv", substrate.to_csv(index=False), file_name="substrate_comparison.csv", mime="text/csv")
