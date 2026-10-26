from __future__ import annotations
import numpy as np

class NumpyIndex:
    """Exact cosine/dot-product search fallback; assumes vectors are normalized."""
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
    def search(self, queries, top_k=5):
        q = np.asarray(queries, dtype=np.float32)
        scores = q @ self.vectors.T
        k = min(int(top_k), self.vectors.shape[0])
        idx = np.argsort(-scores, axis=1)[:, :k]
        vals = np.take_along_axis(scores, idx, axis=1)
        return vals, idx

def build_index(vectors, prefer_faiss=True):
    vectors = np.asarray(vectors, dtype=np.float32)
    if prefer_faiss:
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            return FaissIndex(index)
        except Exception:
            pass
    return NumpyIndex(vectors)

class FaissIndex:
    def __init__(self, index):
        self.index = index
    def search(self, queries, top_k=5):
        return self.index.search(np.asarray(queries, dtype=np.float32), min(top_k, self.index.ntotal))
