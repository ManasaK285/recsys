import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    def __init__(self, text_dim, vision_dim, audio_dim, hidden_dim=512):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, text, vision, audio):
        text = self.text_projection(text)
        vision = self.vision_projection(vision)
        audio = self.audio_projection(audio)
        combined = torch.cat([text, vision, audio], dim=-1)
        return self.fusion(combined)
