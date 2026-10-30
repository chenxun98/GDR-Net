from __future__ import annotations
import torch
from torch import nn
from src.data.channel_map import build_static_graph

class GraphBuilder(nn.Module):
    def __init__(self, channel_names, static_graph_weight: float = 0.35, graph_mode: str = 'hybrid', correlation_mode: str = 'absolute'):
        super().__init__()
        self.register_buffer('static_adj', torch.from_numpy(build_static_graph(channel_names)))
        self.static_graph_weight = static_graph_weight
        self.graph_mode = str(graph_mode).lower()
        self.correlation_mode = str(correlation_mode).lower()
        if self.correlation_mode not in {'absolute', 'positive', 'signed'}:
            raise ValueError('correlation_mode must be absolute, positive, or signed.')
    def dynamic_graph(self, x: torch.Tensor):
        x = x.detach()
        x = x - x.mean(dim=-1, keepdim=True)
        scale = torch.sqrt((x * x).mean(dim=-1, keepdim=True).clamp_min(1e-6))
        x = x / scale
        corr = torch.matmul(x, x.transpose(1, 2)) / x.shape[-1]
        if self.correlation_mode == 'absolute':
            corr = corr.abs()
        elif self.correlation_mode == 'positive':
            corr = corr.clamp_min(0.0)
        corr = corr.clamp(-1.0, 1.0)
        eye = torch.eye(corr.shape[-1], device=corr.device, dtype=corr.dtype).unsqueeze(0)
        return corr * (1.0 - eye)
    def forward(self, x: torch.Tensor):
        static = self.static_adj.unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.graph_mode == 'static':
            return static
        dyn = self.dynamic_graph(x)
        if self.graph_mode == 'dynamic':
            return dyn
        if self.graph_mode == 'identity':
            return torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
        mixed = self.static_graph_weight * static + (1.0 - self.static_graph_weight) * dyn
        return mixed.clamp(-1.0, 1.0) if self.correlation_mode == 'signed' else mixed.clamp(0.0, 1.0)
