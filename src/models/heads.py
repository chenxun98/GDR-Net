from __future__ import annotations
from torch import nn

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
    def forward(self, x):
        return self.net(x)
