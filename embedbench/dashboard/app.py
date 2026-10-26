from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import streamlit as st
import pandas as pd
from embedbench.data import load_sample_data
from embedbench.models import EmbeddingModel, DEFAULT_MODEL
from embedbench.compression import truncate_embeddings
from embedbench.retrieval import SemanticSearch

st.set_page_config(page_title="EmbedBench", layout="wide")
st.title("EmbedBench")
st.caption("Research-inspired semantic retrieval quality, compression, and efficiency explorer")
st.warning("The bundled dataset is intentionally tiny for smoke testing. Use public benchmark datasets for research conclusions.")

@st.cache_resource
def get_runtime():
    corpus, _ = load_sample_data()
    model = EmbeddingModel(DEFAULT_MODEL, allow_fallback=True)
    vecs = model.encode([d["text"] for d in corpus], normalize_embeddings=True)
    return corpus, model, vecs

corpus, model, full_vectors = get_runtime()
st.sidebar.header("Search configuration")
dimension = st.sidebar.select_slider("Embedding dimensions", options=sorted(set([min(x, full_vectors.shape[1]) for x in [128,256,512,768]])), value=min(256, full_vectors.shape[1]))
top_k = st.sidebar.slider("Top-k results", 1, min(10, len(corpus)), 5)
st.sidebar.write(f"Embedding backend: `{model.backend}`")
query = st.text_input("Search your sample knowledge base", placeholder="How do I get a refund?")
if st.button("Search", type="primary") and query.strip():
    docs = truncate_embeddings(full_vectors, dimension)
    q = truncate_embeddings(model.encode([query], normalize_embeddings=True), dimension)
    docs /= (docs**2).sum(axis=1, keepdims=True).clip(1e-12)**0.5
    q /= (q**2).sum(axis=1, keepdims=True).clip(1e-12)**0.5
    results = SemanticSearch(corpus, docs).search_vectors(q, top_k)[0]
    for rank, item in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {item['doc_id']}** · similarity `{item['score']:.4f}`")
            st.write(item["text"])
            st.caption(f"Language: {next(d['language'] for d in corpus if d['doc_id'] == item['doc_id'])}")

results_path = ROOT / "outputs" / "results.csv"
if results_path.exists():
    st.subheader("Latest benchmark results")
    df = pd.read_csv(results_path)
    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.pivot_table(index="dimension", columns="precision", values="ndcg_at_k"))
    with c2:
        st.scatter_chart(df, x="vector_storage_bytes", y="ndcg_at_k", color="precision")
else:
    st.info("Run the benchmark to populate charts: `python experiments/run_benchmark.py`")
