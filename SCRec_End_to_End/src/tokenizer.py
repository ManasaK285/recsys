from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ResidualVectorQuantizer:
    """
    Lightweight residual quantizer.

    Each item receives a sequence of code indices. At each level:
        residual <- residual - selected_code
    and the selected code is the nearest centroid.

    This is an engineering approximation of the hierarchical semantic-ID
    tokenizer used by generative recommendation systems.
    """

    def __init__(self, codebook_size=64, code_length=3, seed=42):
        self.K = codebook_size
        self.L = code_length
        self.rng = np.random.default_rng(seed)
        self.codebooks = None

    def fit(self, X, epochs=8, lr=0.002):
        X = np.asarray(X, dtype=np.float32)
        residual = X.copy()
        self.codebooks = []
        for level in range(self.L):
            ids = self.rng.choice(len(X), size=min(self.K, len(X)), replace=False)
            C = X[ids].copy() if level == 0 else residual[ids].copy()
            if len(C) < self.K:
                pad = np.repeat(C[-1:], self.K - len(C), axis=0)
                C = np.concatenate([C, pad], axis=0)
            for _ in range(epochs):
                dist = ((residual[:, None, :] - C[None, :, :]) ** 2).sum(-1)
                assign = dist.argmin(1)
                for k in range(self.K):
                    mask = assign == k
                    if mask.any():
                        C[k] = (1 - lr) * C[k] + lr * residual[mask].mean(0)
            dist = ((residual[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            assign = dist.argmin(1)
            residual = residual - C[assign]
            self.codebooks.append(C)
        self.codebooks = np.stack(self.codebooks)

    def encode(self, X):
        X = np.asarray(X, dtype=np.float32)
        residual = X.copy()
        codes = []
        for C in self.codebooks:
            dist = ((residual[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            assign = dist.argmin(1)
            codes.append(assign)
            residual = residual - C[assign]
        return np.stack(codes, axis=1)

    def reconstruct(self, codes):
        out = np.zeros((len(codes), self.codebooks.shape[-1]), dtype=np.float32)
        for level in range(self.L):
            out += self.codebooks[level][codes[:, level]]
        return out


class CollaborativeSemanticTokenizer:
    def __init__(self, model_name, top_k=5, codebook_size=64, code_length=3):
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.rq = ResidualVectorQuantizer(codebook_size, code_length)
        self.gate = None

    @staticmethod
    def item_text(row):
        vals = []
        for c in ["title", "brand", "category", "description"]:
            if c in row and str(row[c]) != "nan":
                vals.append(f"{c}: {row[c]}")
        return " | ".join(vals)

    def encode_text(self, items):
        texts = [self.item_text(row) for _, row in items.iterrows()]
        return self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def collaborative_text(self, items, neighbors):
        titles = items["title"].fillna("").astype(str).tolist()
        out = []
        for i, ns in enumerate(neighbors):
            neighbor_titles = [titles[j] for j in ns if j < len(titles)]
            if neighbor_titles:
                out.append(
                    "Collaboratively similar items: " + "; ".join(neighbor_titles)
                )
            else:
                out.append("Collaboratively similar items: none")
        return out

    def build(self, items, neighbors):
        sem = self.encode_text(items)
        coll_text = self.collaborative_text(items, neighbors)
        coll = self.encoder.encode(
            coll_text, normalize_embeddings=True, show_progress_bar=True
        )

        # Learned-gate analogue: initialize gate from semantic/collaborative
        # agreement, then expose it as a trainable module in generation.
        agreement = (sem * coll).sum(1, keepdims=True)
        gate = 1 / (1 + np.exp(-4.0 * agreement))
        fused = gate * coll + (1 - gate) * sem
        fused = fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)

        self.rq.fit(fused)
        codes = self.rq.encode(fused)
        return sem.astype(np.float32), coll.astype(np.float32), fused.astype(np.float32), codes
