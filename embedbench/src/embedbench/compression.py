from __future__ import annotations
import numpy as np

def truncate_embeddings(embeddings: np.ndarray, dimensions: int, renormalize=True) -> np.ndarray:
    """Keep the first dimensions; optionally renormalize after truncation."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if dimensions < 1 or dimensions > arr.shape[1]:
        raise ValueError(f"dimensions must be in [1, {arr.shape[1]}]")
    out = arr[:, :dimensions].copy()
    if renormalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out /= np.maximum(norms, 1e-12)
    return out

def quantize_float16(embeddings: np.ndarray) -> np.ndarray:
    """Round-trip through float16; returned as float32 for compatible downstream math."""
    return np.asarray(embeddings, dtype=np.float16).astype(np.float32)

def quantize_int8_symmetric(embeddings: np.ndarray):
    """Simple per-matrix symmetric int8 quantization/dequantization approximation."""
    arr = np.asarray(embeddings, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = max(max_abs / 127.0, 1e-12)
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale, scale

def storage_bytes(n_vectors: int, dimensions: int, bytes_per_value: int = 4) -> int:
    return int(n_vectors) * int(dimensions) * int(bytes_per_value)
