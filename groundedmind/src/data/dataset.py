import torch
from torch.utils.data import Dataset


class ConceptExperienceDataset(Dataset):
    def __init__(self, concept_embeddings, experience_embeddings):
        if len(concept_embeddings) != len(experience_embeddings):
            raise ValueError("Both collections must have equal length.")
        self.concept_embeddings = concept_embeddings
        self.experience_embeddings = experience_embeddings

    def __len__(self):
        return len(self.concept_embeddings)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.concept_embeddings[idx], dtype=torch.float32),
            torch.as_tensor(self.experience_embeddings[idx], dtype=torch.float32),
        )
