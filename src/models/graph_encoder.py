"""Two graph convolution layers followed by attention pooling."""
from __future__ import annotations
import torch
from torch import nn
from .layers import DenseGraphConv, AttentionPooling

class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__(); self.conv1 = DenseGraphConv(in_dim, hidden_dim); self.conv2 = DenseGraphConv(hidden_dim, out_dim); self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(out_dim); self.act = nn.GELU(); self.drop = nn.Dropout(dropout); self.pool = AttentionPooling(out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = self.drop(self.act(self.norm1(self.conv1(x, adj))))
        h = self.drop(self.act(self.norm2(self.conv2(h, adj))))
        pooled, attn = self.pool(h)
        return h, pooled, attn
