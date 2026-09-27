import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.grounded_model import GroundedConceptModel
from src.models.contrastive import contrastive_loss


def train(model, concept_embeddings, experience_embeddings,
          epochs=100, lr=1e-3, temperature=0.07):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    concept_embeddings = torch.as_tensor(
        concept_embeddings, dtype=torch.float32
    )
    experience_embeddings = torch.as_tensor(
        experience_embeddings, dtype=torch.float32
    )

    for epoch in range(epochs):
        optimizer.zero_grad()

        predicted = model(experience_embeddings)

        # This assumes concept embeddings have the same output dimension
        # as the model output. Project them before using this function
        # when dimensions differ.
        if predicted.shape[-1] != concept_embeddings.shape[-1]:
            raise ValueError(
                "Grounded model output dimension must match concept embedding dimension."
            )

        loss = contrastive_loss(
            concept_embeddings,
            predicted,
            temperature=temperature,
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch+1:03d} loss={loss.item():.4f}")

    return model


if __name__ == "__main__":
    print(
        "Training entry point loaded. Supply aligned concept and sensory "
        "embeddings from your experiment before training."
    )
