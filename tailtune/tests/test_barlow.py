import torch

from src.models.barlow_twins import barlow_twins_loss


def test_barlow_loss_finite():
    z1 = torch.randn(16, 8)
    z2 = torch.randn(16, 8)

    loss = barlow_twins_loss(z1, z2)

    assert torch.isfinite(loss)
    assert loss.item() >= 0