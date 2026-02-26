# src/danet_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractLayer(nn.Module):
    """
    DANet-style abstraction layer:
    - Learns soft grouping of input features into K groups
    - Creates group embeddings, gated by group assignment
    - Projects to hidden dimension + residual
    """
    def __init__(self, in_dim: int, hidden_dim: int, groups: int = 16, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.groups = groups

        # soft assignment to groups
        self.assign = nn.Linear(in_dim, groups)

        # map input into group embeddings (groups * hidden_dim)
        self.group_embed = nn.Linear(in_dim, groups * hidden_dim)

        # project concatenated groups back to hidden
        self.out = nn.Linear(groups * hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.res_map = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softmax(self.assign(x), dim=-1)  # (B, groups)

        g = self.group_embed(x)  # (B, groups*hidden)
        g = g.view(x.size(0), self.groups, self.hidden_dim)  # (B, groups, hidden)

        g = g * a.unsqueeze(-1)  # gate by assignment
        g = g.reshape(x.size(0), self.groups * self.hidden_dim)

        y = self.out(g)
        y = self.drop(F.gelu(y))
        y = self.norm(y + self.res_map(x))
        return y


class DANetClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        groups: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = [AbstractLayer(in_dim, hidden_dim, groups=groups, dropout=dropout)]
        for _ in range(num_layers - 1):
            layers.append(AbstractLayer(hidden_dim, hidden_dim, groups=groups, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x).squeeze(-1)
        return logits