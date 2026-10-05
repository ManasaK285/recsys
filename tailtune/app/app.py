import json
from pathlib import Path
import streamlit as st
import pandas as pd

from app.recommender import Recommender

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="TailTune", layout="wide")

st.title("TailTune")
st.caption("Controllable sequential recommendation with redundancy-reduced representations")

proc = ROOT / "data/processed"
if not (proc / "mappings.json").exists():
    st.error("Processed data not found. Run: python -m src.data.download && python -m src.data.preprocess")
    st.stop()

with open(proc / "mappings.json") as f:
    mappings = json.load(f)

with st.sidebar:
    st.header("Controls")
    model_type = st.selectbox("Model", ["bt", "sasrec"])
    alpha = st.slider("Redundancy strength α", 0.0, 0.5, 0.2, 0.05)
    user_id = st.number_input(
        "Mapped user ID", min_value=1, max_value=int(mappings["num_users"]),
        value=1, step=1
    )
    k = st.slider("Recommendations", 5, 20, 10)

if model_type == "bt":
    checkpoint = ROOT / f"checkpoints/bt_sr_alpha_{alpha}.pt"
else:
    checkpoint = ROOT / "checkpoints/sasrec.pt"

if not checkpoint.exists():
    st.warning(f"Checkpoint not found: {checkpoint}")
    st.info("Train the selected model first.")
    st.stop()

rec = Recommender(checkpoint, model_type=model_type, alpha=alpha)

left, right = st.columns(2)

with left:
    st.subheader("Recent history")
    for title in rec.history(user_id):
        st.write("•", title)

with right:
    st.subheader("Recommendations")
    rows = rec.recommend(user_id, k)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("How to read the model")
st.write(
    "Head items are highly popular, while tail items are relatively rare. "
    "Increasing α strengthens the redundancy-reduction objective. "
    "Use the experiment outputs to determine the actual accuracy/exposure trade-off "
    "for your dataset rather than assuming that larger α is always better."
)
