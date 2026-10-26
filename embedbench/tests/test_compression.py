import numpy as np
import pytest
from embedbench.compression import truncate_embeddings, quantize_float16, quantize_int8_symmetric, storage_bytes
def test_truncate_shape():
    x = np.ones((3, 8), dtype=np.float32)
    assert truncate_embeddings(x, 4).shape == (3, 4)
def test_invalid_dimension():
    with pytest.raises(ValueError):
        truncate_embeddings(np.ones((2, 3)), 4)
def test_quantization_shape():
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    assert quantize_float16(x).shape == x.shape
    q, scale = quantize_int8_symmetric(x)
    assert q.shape == x.shape and scale > 0
def test_storage():
    assert storage_bytes(100, 10, 4) == 4000
