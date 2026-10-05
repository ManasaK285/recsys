import torch

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def barlow_twins_loss(z1, z2, lambda_offdiag=0.005):
    """
    Barlow Twins objective:
      diagonal -> 1
      off-diagonal -> 0
    """
    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)

    c = (z1.T @ z2) / z1.size(0)

    on = torch.diagonal(c)
    off = off_diagonal(c)

    loss_on = ((on - 1.0) ** 2).sum()
    loss_off = (off ** 2).sum()

    return loss_on + lambda_offdiag * loss_off
