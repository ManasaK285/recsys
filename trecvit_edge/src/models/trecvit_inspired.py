import torch
from torch import nn

class FrameEncoder(nn.Module):
    """Small CNN returning a spatial grid [B,T,N,D]."""
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        y = self.net(x.reshape(b * t, c, h, w))
        _, d, gh, gw = y.shape
        return y.flatten(2).transpose(1, 2).reshape(b, t, gh * gw, d)

class CausalRecurrentVideoBlock(nn.Module):
    """
    Educational TRecViT-inspired block:
    spatial attention -> causal recurrent temporal update -> channel MLP.
    Uses GRUCell for a compact baseline, not the paper's exact LRU.
    """
    def __init__(self, dim=128, heads=4, mlp_ratio=2):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_cell = nn.GRUCell(dim, dim)
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        # x: [B,T,N,D]. Each frame attends only to its own spatial patches.
        b, t, n, d = x.shape
        spatial = x.reshape(b * t, n, d)
        q = self.spatial_norm(spatial)
        spatial = spatial + self.spatial_attn(q, q, q, need_weights=False)[0]
        spatial = spatial.reshape(b, t, n, d)

        # Causal temporal recurrence, independently for each spatial location.
        state = torch.zeros(b, n, d, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(t):
            current = self.temporal_norm(spatial[:, step])
            state = self.temporal_cell(
                current.reshape(b * n, d), state.reshape(b * n, d)
            ).reshape(b, n, d)
            outputs.append(state)
        y = torch.stack(outputs, dim=1)
        y = y + self.channel_mlp(self.channel_norm(y))
        return y

class TRecViTInspired(nn.Module):
    def __init__(self, num_classes, dim=128, heads=4):
        super().__init__()
        self.encoder = FrameEncoder(dim)
        self.block = CausalRecurrentVideoBlock(dim, heads)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        features = self.encoder(x)
        features = self.block(features)
        # Predict per frame by pooling spatial tokens. Output [B,T,C].
        return self.head(features.mean(dim=2))

    def predict_clip(self, x):
        return self.forward(x).mean(dim=1)
