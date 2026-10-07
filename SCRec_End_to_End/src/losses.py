import torch
import torch.nn.functional as F


def info_nce(a, b, temperature=0.2):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def project_ball(x, eps=1e-5):
    x = torch.tanh(x)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp((1 - eps) / norm, max=1.0)
    return x * scale


def poincare_distance(x, y, eps=1e-5):
    x = project_ball(x, eps)
    y = project_ball(y, eps)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    diff2 = ((x - y) ** 2).sum(-1)
    denom = ((1 - x2) * (1 - y2)).clamp_min(eps)
    z = 1 + 2 * diff2 / denom
    return torch.acosh(z.clamp_min(1 + eps))


def manifold_alignment(semantic, code, projector_s, projector_c):
    s = projector_s(semantic)
    c = projector_c(code)
    return poincare_distance(s, c).mean()
