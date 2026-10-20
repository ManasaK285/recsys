from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MemoryShapes", page_icon="🧠", layout="wide")
st.title("MemoryShapes")
st.caption("Hybrid neural–cognitive models for reward learning — synthetic-data research demo")

metrics_path = Path("results/metrics.csv")
preds_path = Path("results/predictions.csv")
if not metrics_path.exists() or not preds_path.exists():
    st.warning("No results found yet. Run `python -m src.run_experiment` from the project folder.")
    st.code("python -m src.run_experiment --participants 120 --trials 150 --epochs 12")
    st.stop()

metrics = pd.read_csv(metrics_path)
preds = pd.read_csv(preds_path)
left, right = st.columns(2)
with left:
    st.subheader("Held-out predictive performance")
    st.caption("Lower log loss and Brier score are better; higher accuracy is better.")
    st.dataframe(metrics.sort_values("log_loss"), use_container_width=True, hide_index=True)
with right:
    st.subheader("Log loss by model")
    st.bar_chart(metrics.set_index("model")[["log_loss"]])

st.subheader("Explore trial-level predictions")
model = st.selectbox("Model", sorted(preds.model.unique()))
pid = st.selectbox("Test participant", sorted(preds.participant_id.unique()))
view = preds[(preds.model == model) & (preds.participant_id == pid)].sort_values("trial").copy()
view["predicted_choice"] = view[["p0", "p1", "p2", "p3"]].to_numpy().argmax(axis=1)
view["chosen_probability"] = view.apply(lambda r: r[f"p{int(r['choice'])}"], axis=1)
st.line_chart(view.set_index("trial")[["p0", "p1", "p2", "p3"]])
st.dataframe(view[["trial", "choice", "predicted_choice", "chosen_probability"]], use_container_width=True, hide_index=True)

st.subheader("Interpretation guardrails")
st.markdown("""
- The current results are from simulated participants, not the published human dataset.
- This implementation is inspired by the paper but is not the authors' exact model.
- Model comparisons should be repeated across random seeds and reported with uncertainty.
- Predicting choices is not the same as maximizing rewards.
""")
