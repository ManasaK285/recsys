import torch
import torch.nn.functional as F


def contrastive_loss(concept_embeddings, experience_embeddings, temperature=0.07):
    concept_embeddings = F.normalize(concept_embeddings, dim=-1)
    experience_embeddings = F.normalize(experience_embeddings, dim=-1)

    logits = concept_embeddings @ experience_embeddings.T
    logits = logits / temperature

    labels = torch.arange(len(logits), device=logits.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)

    return (loss_a + loss_b) / 2
