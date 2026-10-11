import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import streamlit as st
from provenancelab.detectors import DemoBaselineDetector
from provenancelab.transformations import transform
from provenancelab.benchmark import run_benchmark

st.set_page_config(page_title="ProvenanceLab", page_icon="🔎", layout="wide")
st.title("ProvenanceLab")
st.caption("AI-content provenance research • robustness experiments • evaluation monitoring")
st.warning("Demo mode: the included lexical baseline is NOT SynthID and cannot reliably determine whether text was AI-generated. Do not use its score for consequential decisions.")

tab_verify, tab_robust, tab_bench, tab_about = st.tabs(["Text analysis", "Robustness lab", "Evaluation monitor", "About"])
detector = DemoBaselineDetector()
with tab_verify:
    text = st.text_area("Text to analyze", "Furthermore, this example demonstrates a structured approach to evaluating content provenance. It is important to note that a result from a toy baseline is not proof of authorship.", height=180)
    if st.button("Analyze demo signal", type="primary"):
        result = detector.detect(text)
        c1, c2 = st.columns(2)
        c1.metric("Demo score (not probability)", result.score)
        c2.metric("Demo band", result.status.replace("demo_signal_", ""))
        st.write("**Method:**", result.method)
        st.info(result.caveat)
with tab_robust:
    source = st.text_area("Reference text", "It is important to evaluate the reliability of content provenance tools. A robust system should report uncertainty and avoid treating missing watermark evidence as proof of human authorship.", height=130)
    if st.button("Run transformations"):
        rows = []
        for kind in ["original", "truncate", "light_edit"]:
            transformed = transform(source, kind)
            result = detector.detect(transformed)
            rows.append({"transformation": kind, "word_count": len(transformed.split()), "demo_score": result.score, "demo_band": result.status, "text": transformed})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("This measures the toy baseline's response to transformations, not SynthID watermark survival.")
with tab_bench:
    n = st.slider("Samples per class", 5, 100, 30)
    seed = st.number_input("Random seed", min_value=0, value=42)
    if st.button("Run reproducible smoke benchmark"):
        result = run_benchmark(int(n), int(seed))
        m = result["metrics"]
        cols = st.columns(5)
        for col, key in zip(cols, ["precision", "recall", "f1", "false_positive_rate", "n"]):
            col.metric(key.replace("_", " ").title(), m[key])
        st.warning(result["warning"])
        st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True, hide_index=True)
        st.download_button("Download benchmark CSV", pd.DataFrame(result["rows"]).to_csv(index=False), "provenancelab_demo_benchmark.csv", "text/csv")
with tab_about:
    st.subheader("Architecture")
    st.markdown("""
    1. **Input layer:** text entry and validation.
    2. **Detector adapter:** currently a deliberately weak educational baseline.
    3. **Transformation suite:** original, truncation, and controlled light edits.
    4. **Evaluation monitor:** precision, recall, F1, false-positive rate, deterministic seed.
    5. **API:** FastAPI endpoints for health, verification, and benchmark execution.

    **Next research milestone:** integrate a supported Hugging Face SynthID Text watermarking configuration and a detector calibrated for that exact model/configuration. A generation watermark configuration alone does not provide a trained detector or universally valid threshold.
    """)
    st.markdown("References: [Google DeepMind SynthID Text](https://github.com/google-deepmind/synthid-text) · [Hugging Face SynthID documentation](https://huggingface.co/docs/transformers)")
