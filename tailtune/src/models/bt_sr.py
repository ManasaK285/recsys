import torch.nn as nn
from .sasrec import SASRec
from .barlow_twins import barlow_twins_loss

class BTSR(SASRec):
    def __init__(self, *args, alpha=0.2, lambda_offdiag=0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.lambda_offdiag = lambda_offdiag

    def bt_loss(self, x1, x2):
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        return barlow_twins_loss(z1, z2, self.lambda_offdiag)

    def total_loss(self, logits, target, bt_loss):
        import torch.nn.functional as F
        ce = F.cross_entropy(logits[:, 1:], target - 1)
        return ce + self.alpha * bt_loss, ce
