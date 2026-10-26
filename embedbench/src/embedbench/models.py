from __future__ import annotations
import numpy as np

DEFAULT_MODEL = "google/embeddinggemma-300m"

class EmbeddingModel:
    """Thin wrapper around SentenceTransformers with a deterministic offline fallback."""
    def __init__(self, model_name=DEFAULT_MODEL, device="auto", allow_fallback=True):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.backend = "sentence-transformers"
        try:
            from sentence_transformers import SentenceTransformer
            chosen_device = None if device == "auto" else device
            self.model = SentenceTransformer(model_name, device=chosen_device, trust_remote_code=True)
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Could not load embedding model {model_name}: {exc}") from exc
            self.backend = "tfidf-fallback"
            self._fallback_reason = str(exc)
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
            self._fallback_fitted = False

    def encode(self, texts, batch_size=16, normalize_embeddings=True, prompt_name=None):
        texts = list(texts)
        if self.model is not None:
            kwargs = {"batch_size": batch_size, "convert_to_numpy": True,
                      "normalize_embeddings": normalize_embeddings, "show_progress_bar": False}
            if prompt_name:
                kwargs["prompt_name"] = prompt_name
            try:
                return np.asarray(self.model.encode(texts, **kwargs), dtype=np.float32)
            except TypeError:
                kwargs.pop("prompt_name", None)
                return np.asarray(self.model.encode(texts, **kwargs), dtype=np.float32)
        if not self._fallback_fitted:
            matrix = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
            self._fallback_fitted = True
            self._fallback_texts = texts
            self._fallback_matrix = matrix
            return self._normalize(matrix) if normalize_embeddings else matrix
        # Fit a stable vocabulary from the current text collection if caller supplies a new set.
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return self._normalize(matrix) if normalize_embeddings else matrix

    @staticmethod
    def _normalize(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def encode_corpus_and_queries(self, corpus_texts, query_texts, **kwargs):
        if self.model is not None:
            return self.encode(corpus_texts, **kwargs), self.encode(query_texts, **kwargs)
        from sklearn.feature_extraction.text import TfidfVectorizer
        all_texts = list(corpus_texts) + list(query_texts)
        mat = self.vectorizer.fit_transform(all_texts).toarray().astype(np.float32)
        if kwargs.get("normalize_embeddings", True):
            mat = self._normalize(mat)
        return mat[:len(corpus_texts)], mat[len(corpus_texts):]
