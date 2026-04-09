"""Policy networks: MLP for bandits, CausalTransformer for sequence tasks."""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """2-layer ReLU MLP: obs_dim -> hidden -> hidden -> num_actions."""

    def __init__(self, obs_dim: int, hidden: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalTransformer(nn.Module):
    """Decoder-only transformer for autoregressive sequence tasks.

    Input: token IDs [B, T] -> logits [B, T, vocab_size].
    Uses pre-norm transformer blocks with causal masking.
    """

    def __init__(self, vocab_size: int, d_model: int, nhead: int,
                 num_layers: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Causal mask — registered as buffer so it moves with .to(device)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        h = self.tok_emb(tokens) * math.sqrt(self.d_model) + self.pos_emb(pos)
        h = self.transformer(h, mask=self.causal_mask[:T, :T])
        h = self.ln_f(h)
        return self.head(h)
