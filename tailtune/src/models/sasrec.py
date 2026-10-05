import torch
import torch.nn as nn


class SASRec(nn.Module):
    """
    SASRec-style sequential recommender.

    Defaults are intentionally lighter so the model is practical on a laptop.
    The architecture can still be scaled up for the final experiment.
    """

    def __init__(
        self,
        num_items,
        max_seq_len=50,
        d_model=64,
        nhead=2,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.item_embedding = nn.Embedding(
            num_items + 1,
            d_model,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        # norm_first=False allows PyTorch's TransformerEncoder to use
        # its optimized nested-tensor path when possible.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, x):
        """
        Encode a batch of item sequences.

        x:
            [B, L]

        Returns:
            [B, d_model]
        """
        batch_size, seq_len = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(
            seq_len,
            device=x.device,
        ).unsqueeze(0)

        h = (
            self.item_embedding(x)
            + self.position_embedding(positions)
        )

        h = self.dropout(h)

        padding_mask = x.eq(0)

        # Prevent attention to future items.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        h = self.encoder(
            h,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        # Last non-padding position for each sequence.
        lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1

        batch_indices = torch.arange(
            batch_size,
            device=x.device,
        )

        z = h[batch_indices, lengths]

        return self.norm(z)

    def score_all(self, z):
        """
        Score every item.

        z:
            [B, d_model]

        Returns:
            [B, num_items + 1]
        """
        return z @ self.item_embedding.weight.T

    def forward(self, x):
        z = self.encode(x)
        return self.score_all(z)