import torch
from src.models.trecvit_inspired import TRecViTInspired, CausalRecurrentVideoBlock
from src.models.baselines import CNNGRUBaseline

def test_trecvit_output_shape():
    model = TRecViTInspired(num_classes=3, dim=128)
    x = torch.randn(2, 4, 3, 112, 112)
    y = model(x)
    assert y.shape == (2, 4, 3)

def test_gru_output_shape():
    model = CNNGRUBaseline(num_classes=5)
    x = torch.randn(2, 4, 3, 112, 112)
    assert model(x).shape == (2, 4, 5)

def test_recurrent_block_shape():
    block = CausalRecurrentVideoBlock(dim=32, heads=4)
    x = torch.randn(2, 3, 9, 32)
    assert block(x).shape == x.shape
