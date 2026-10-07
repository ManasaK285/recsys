import numpy as np
import torch
from torch.utils.data import Dataset

TASKS = ["liked", "strong_preference", "repeat_interest", "high_engagement"]

class InteractionDataset(Dataset):
    def __init__(self, frame):
        self.users = torch.tensor(frame.user_id.to_numpy(), dtype=torch.long)
        self.items = torch.tensor(frame.item_id.to_numpy(), dtype=torch.long)
        self.y = torch.tensor(frame[TASKS].to_numpy(dtype=np.float32))
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.y[idx]
