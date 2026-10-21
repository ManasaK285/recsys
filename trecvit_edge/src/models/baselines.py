import torch
from torch import nn
from .trecvit_inspired import FrameEncoder

class CNNGRUBaseline(nn.Module):
    def __init__(self, num_classes, dim=128):
        super().__init__()
        self.encoder = FrameEncoder(dim)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        z = self.encoder(x).mean(dim=2)
        z, _ = self.temporal(z)
        return self.head(z)

    def predict_clip(self, x):
        return self.forward(x).mean(dim=1)
