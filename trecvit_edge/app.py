import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import torch
from src.inference.video_inference import predict_video

st.set_page_config(page_title="TRecViT Edge", page_icon="🎞️", layout="wide")
st.title("TRecViT Edge")
st.caption("Research-inspired causal video action recognition — upload a video and run a trained checkpoint.")

checkpoint = st.text_input("Checkpoint path", "artifacts/best.pt")
uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if not Path(checkpoint).exists():
    st.info("Train a model first, then provide its checkpoint path. See README.md.")
elif uploaded:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name
    st.video(uploaded)
    if st.button("Analyze video", type="primary"):
        try:
            with st.spinner("Running inference..."):
                result = predict_video(video_path, checkpoint)
            a, b = st.columns(2)
            a.metric("Predicted action", result["label"])
            b.metric("Model confidence", f"{result['confidence']:.1%}")
            st.subheader("Class scores")
            scores = pd.DataFrame({"class": list(result["probabilities"].keys()),
                                   "score": list(result["probabilities"].values())})
            st.bar_chart(scores.set_index("class"))
            st.subheader("Frame-level prediction timeline")
            st.dataframe(pd.DataFrame({
                "sampled_frame": range(1, len(result["frame_predictions"]) + 1),
                "prediction": result["frame_predictions"]
            }), use_container_width=True, hide_index=True)
            st.warning("Softmax confidence is not guaranteed to be calibrated. Validate before any real-world use.")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
else:
    st.write("Add a trained checkpoint and upload a video to begin.")
