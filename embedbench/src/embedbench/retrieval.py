from __future__ import annotations
from .indexing import build_index
from .compression import truncate_embeddings
import numpy as np

class SemanticSearch:
    def __init__(self, documents, vectors):
        self.documents = list(documents)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.index = build_index(self.vectors)
    def search_vectors(self, query_vectors, top_k=5):
        scores, indices = self.index.search(query_vectors, top_k)
        results = []
        for score_row, index_row in zip(scores, indices):
            results.append([
                {"doc_id": self.documents[int(i)]["doc_id"],
                 "text": self.documents[int(i)]["text"],
                 "score": float(s)}
                for s, i in zip(score_row, index_row)
            ])
        return results
