from __future__ import annotations
import torch
from torch import nn
class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__(); self.linear = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        b, n, _ = x.shape
        eye = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, -1, -1)
        a = adj + eye
        deg = a.abs().sum(dim=-1).clamp_min(1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        a_norm = deg_inv_sqrt.unsqueeze(-1) * a * deg_inv_sqrt.unsqueeze(-2)
        return torch.bmm(a_norm, self.linear(x))

    @staticmethod
    def normalize_adjacency(adj: torch.Tensor):
        b, n, _ = adj.shape
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(b, -1, -1)
        a = adj + eye
        deg_inv_sqrt = a.abs().sum(dim=-1).clamp_min(1e-6).pow(-0.5)
        return deg_inv_sqrt.unsqueeze(-1) * a * deg_inv_sqrt.unsqueeze(-2)

class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.gate = nn.Linear(dim, 1)
    def forward(self, x: torch.Tensor):
        score = self.gate(x).squeeze(-1)
        weight = torch.softmax(score, dim=-1)
        pooled = torch.sum(weight.unsqueeze(-1) * x, dim=1)
        return pooled, weight
