from __future__ import annotations
import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_sample_data
from .models import EmbeddingModel, DEFAULT_MODEL
from .compression import truncate_embeddings, quantize_float16, quantize_int8_symmetric, storage_bytes
from .indexing import build_index
from .metrics import evaluate_rankings

def run_benchmark(output_dir="outputs", model_name=DEFAULT_MODEL, dimensions=(768,512,256,128),
                  device="auto", batch_size=16, top_k=5, use_faiss=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus, queries = load_sample_data()
    corpus_texts = [d["text"] for d in corpus]
    query_texts = [q["query"] for q in queries]
    model = EmbeddingModel(model_name=model_name, device=device, allow_fallback=True)
    start = time.perf_counter()
    corpus_vecs, query_vecs = model.encode_corpus_and_queries(
        corpus_texts, query_texts, batch_size=batch_size, normalize_embeddings=True)
    encode_seconds = time.perf_counter() - start

    records = []
    detail_rows = []
    max_dim = corpus_vecs.shape[1]
    dims = sorted(set(min(int(d), max_dim) for d in dimensions if int(d) > 0), reverse=True)
    if max_dim not in dims:
        dims.insert(0, max_dim)

    for dim in dims:
        c = truncate_embeddings(corpus_vecs, dim)
        q = truncate_embeddings(query_vecs, dim)
        for precision in ("float32", "float16", "int8"):
            if precision == "float32":
                cp, qp, bpe = c, q, 4
            elif precision == "float16":
                cp, qp, bpe = quantize_float16(c), quantize_float16(q), 2
            else:
                cp, _ = quantize_int8_symmetric(c)
                qp, _ = quantize_int8_symmetric(q)
                bpe = 1
            norms_c = np.linalg.norm(cp, axis=1, keepdims=True)
            norms_q = np.linalg.norm(qp, axis=1, keepdims=True)
            cp = cp / np.maximum(norms_c, 1e-12)
            qp = qp / np.maximum(norms_q, 1e-12)
            index = build_index(cp, prefer_faiss=use_faiss)
            t0 = time.perf_counter()
            scores, indices = index.search(qp, top_k=top_k)
            search_seconds = time.perf_counter() - t0
            rankings = [[corpus[int(i)]["doc_id"] for i in row] for row in indices]
            per_query = evaluate_rankings(rankings, queries, k=top_k)
            df = pd.DataFrame(per_query)
            records.append({
                "backend": model.backend, "model_name": model_name, "dimension": dim,
                "precision": precision, "recall_at_k": float(df["recall_at_k"].mean()),
                "mrr": float(df["mrr"].mean()), "ndcg_at_k": float(df["ndcg_at_k"].mean()),
                "embedding_seconds_total": encode_seconds,
                "query_search_ms_mean": (search_seconds / max(len(queries),1)) * 1000,
                "query_search_ms_p95": float(np.percentile(
                    np.array([search_seconds / max(len(queries),1) * 1000]), 95)),
                "vector_storage_bytes": storage_bytes(len(corpus), dim, bpe),
                "corpus_size": len(corpus), "query_count": len(queries),
            })
            for row, ranking in zip(per_query, rankings):
                detail_rows.append({"dimension": dim, "precision": precision, **row,
                                    "retrieved_doc_ids": "|".join(ranking)})
    results = pd.DataFrame(records)
    results.to_csv(output / "results.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output / "per_query_results.csv", index=False)
    summary = {
        "model_name": model_name, "backend": model.backend,
        "fallback_note": getattr(model, "_fallback_reason", None),
        "device_requested": device, "python": platform.python_version(),
        "encode_seconds_total": encode_seconds,
        "corpus_count": len(corpus), "query_count": len(queries),
        "note": "Tiny bundled demo dataset; use public benchmarks for meaningful conclusions.",
        "best_ndcg_configuration": results.sort_values("ndcg_at_k", ascending=False).iloc[0].to_dict(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _make_plots(results, output / "figures")
    return results, summary

def _make_plots(results, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for precision, group in results.groupby("precision"):
        group = group.sort_values("dimension")
        ax.plot(group["dimension"], group["ndcg_at_k"], marker="o", label=precision)
    ax.set_xlabel("Embedding dimensions")
    ax.set_ylabel("nDCG@k")
    ax.set_title("Retrieval quality by dimension and representation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_by_dimension.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    for precision, group in results.groupby("precision"):
        ax.scatter(group["vector_storage_bytes"], group["ndcg_at_k"], label=precision)
        for _, row in group.iterrows():
            ax.annotate(str(int(row["dimension"])), (row["vector_storage_bytes"], row["ndcg_at_k"]))
    ax.set_xlabel("Estimated raw vector storage (bytes)")
    ax.set_ylabel("nDCG@k")
    ax.set_title("Quality-storage trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_vs_storage.png", dpi=150)
    plt.close(fig)
