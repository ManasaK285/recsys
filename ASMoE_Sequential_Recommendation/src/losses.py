import torch.nn.functional as F

def next_item_loss(scores,target): return F.cross_entropy(scores,target)
