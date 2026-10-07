import torch
import torch.nn as nn
import torch.nn.functional as F


class SCRec(nn.Module):
    """
    Lightweight semantic-ID autoregressive recommender.

    CT is represented by the tokenizer-produced semantic/collaborative fused
    item vectors. SG dynamically mixes code embeddings with the target item's
    semantic prior. MA aligns dense semantic and code representations in a
    shared hyperbolic space.
    """

    def __init__(
        self,
        n_items,
        codebook_size,
        code_length,
        semantic_dim,
        hidden_dim=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        max_history=20,
        manifold_dim=64,
    ):
        super().__init__()
        self.n_items = n_items
        self.K = codebook_size
        self.L = code_length
        self.hidden = hidden_dim
        self.max_history = max_history
        ablation = cfg.get("ablation", {})

        self.use_collaborative = ablation.get(
            "collaborative_tokenization", True
        )

        self.use_semantic_guidance = ablation.get(
            "semantic_guided_generation", True
        )

        self.use_manifold_alignment = ablation.get(
            "manifold_alignment", True
        )
        self.item_semantic = nn.Embedding(n_items, semantic_dim)
        self.item_code = nn.Parameter(
            torch.randn(n_items, code_length, hidden_dim) * 0.02
        )
        self.code_embeddings = nn.Embedding(codebook_size, hidden_dim)
        self.pos = nn.Embedding(max_history + code_length + 2, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.input_proj = nn.Linear(semantic_dim, hidden_dim)
        self.semantic_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.semantic_to_hidden = nn.Linear(semantic_dim, hidden_dim)

        self.code_head = nn.Linear(hidden_dim, codebook_size)

        self.manifold_s = nn.Sequential(
            nn.Linear(semantic_dim, manifold_dim),
            nn.Tanh(),
        )
        self.manifold_c = nn.Sequential(
            nn.Linear(hidden_dim, manifold_dim),
            nn.Tanh(),
        )

    def history_representation(self, histories, mask):
        x = self.item_code[histories].mean(dim=2)
        x = x + self.pos(
            torch.arange(x.size(1), device=x.device).unsqueeze(0)
        )
        key_padding = ~mask
        h = self.encoder(x, src_key_padding_mask=key_padding)
        lengths = mask.sum(1).clamp_min(1) - 1
        return h[torch.arange(h.size(0), device=h.device), lengths]

    def semantic_guided_logits(self, user_h, target_semantic):
        semantic_h = self.semantic_to_hidden(target_semantic)
        gate = torch.sigmoid(
            self.semantic_gate(torch.cat([user_h, semantic_h], dim=-1))
        )
        fused = gate * semantic_h + (1 - gate) * user_h
        return self.code_head(fused), fused, gate

    def forward(self, histories, mask, target_semantic, target_codes):
        user_h = self.history_representation(histories, mask)

        # Predict all code positions autoregressively using teacher forcing.
        B = histories.size(0)
        device = histories.device
        prev = torch.zeros(B, self.L, dtype=torch.long, device=device)
        prev[:, 0] = 0

        logits = []
        h = user_h
        for level in range(self.L):
            semantic_h = self.semantic_to_hidden(target_semantic)
            gate = torch.sigmoid(
                self.semantic_gate(torch.cat([h, semantic_h], dim=-1))
            )
            fused = gate * semantic_h + (1 - gate) * h
            logits.append(self.code_head(fused))

            # Inject ground-truth code for the next position.
            if level < self.L - 1:
                h = fused + self.code_embeddings(target_codes[:, level])

        logits = torch.stack(logits, dim=1)
        code_repr = self.code_embeddings(target_codes).mean(dim=1)
        return logits, user_h, semantic_h, code_repr

    @torch.no_grad()
    def recommend(self, histories, mask, semantic_matrix, codes, top_k=10):
        self.eval()
        user_h = self.history_representation(histories, mask)
        semantic_h = self.semantic_to_hidden(
            torch.as_tensor(semantic_matrix, device=histories.device)
        )
        gate = torch.sigmoid(
            self.semantic_gate(
                torch.cat(
                    [user_h[:, None, :].expand(-1, semantic_h.size(0), -1), semantic_h[None, :, :].expand(user_h.size(0), -1, -1)],
                    dim=-1,
                )
            )
        )
        fused = gate * semantic_h[None, :, :] + (1 - gate) * user_h[:, None, :]
        logits = self.code_head(fused)

        # Score candidates using code-token likelihood proxy plus semantic affinity.
        probs = logits.softmax(-1)
        code_t = torch.as_tensor(codes, device=histories.device)
        score = torch.zeros(histories.size(0), len(codes), device=histories.device)
        for level in range(self.L):
            score += probs.gather(2, code_t[:, level][None, :, None].expand(histories.size(0), -1, 1)).squeeze(-1)
        return score.topk(top_k, dim=-1).indices
