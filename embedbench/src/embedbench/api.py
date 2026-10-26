from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .data import load_sample_data
from .models import EmbeddingModel, DEFAULT_MODEL
from .retrieval import SemanticSearch
from .compression import truncate_embeddings

app = FastAPI(title="EmbedBench API", version="0.1.0",
              description="Semantic search and resource-aware embedding benchmark demo.")

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    dimensions: int = Field(default=256, ge=1, le=768)

@lru_cache(maxsize=1)
def _runtime():
    corpus, _ = load_sample_data()
    model = EmbeddingModel(DEFAULT_MODEL, allow_fallback=True)
    vectors = model.encode([d["text"] for d in corpus], normalize_embeddings=True)
    vectors = truncate_embeddings(vectors, min(256, vectors.shape[1]))
    # If a model emits fewer dimensions than requested, actual dimension is used.
    vectors = vectors / (vectors**2).sum(axis=1, keepdims=True).clip(1e-12) ** 0.5
    return corpus, model, vectors

@app.get("/health")
def health():
    return {"status": "ok", "service": "embedbench"}

@app.post("/search")
def search(request: SearchRequest):
    try:
        corpus, model, base_vectors = _runtime()
        dim = min(request.dimensions, base_vectors.shape[1])
        # Rebuild a compatible index at requested dimension from cached full embeddings.
        full = model.encode([d["text"] for d in corpus], normalize_embeddings=True)
        doc_vecs = truncate_embeddings(full, dim)
        query_vec = model.encode([request.query], normalize_embeddings=True)
        query_vec = truncate_embeddings(query_vec, dim)
        doc_vecs /= (doc_vecs**2).sum(axis=1, keepdims=True).clip(1e-12) ** 0.5
        query_vec /= (query_vec**2).sum(axis=1, keepdims=True).clip(1e-12) ** 0.5
        engine = SemanticSearch(corpus, doc_vecs)
        return {"backend": model.backend, "dimension": dim,
                "results": engine.search_vectors(query_vec, request.top_k)[0]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
