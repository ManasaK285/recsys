import torch

def effective_rank(z):
    """
    Effective rank = exp(entropy(normalized singular values)).
    """
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / (s.sum() + 1e-12)
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item(), s.cpu()

def collect_representations(model, histories, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            x = torch.tensor(histories[start:start+batch_size],
                             dtype=torch.long, device=device)
            out.append(model.encode(x).cpu())
    return torch.cat(out, dim=0)
