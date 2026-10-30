"""Complementary graph encoding and label-compatible feature recombination."""
from __future__ import annotations
import torch
from torch import nn
from .graph_encoder import GraphEncoder
from .graph_builder import GraphBuilder
from .heads import MLPHead
from .layers import AttentionPooling
from .outputs import ModelOutput
from .temporal_encoder import ChannelTemporalEncoder


class GraphMaskSplitter(nn.Module):
    """Partition non-diagonal edges with a symmetric learned soft mask."""
    def __init__(self, feat_dim, dropout=0.2, temperature=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError('disentangle_temperature must be positive.')
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.edge_mlp = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(feat_dim, 1))
        self.temperature = float(temperature)

    def forward(self, node_feat, base_adj):
        query, key = self.query(node_feat), self.key(node_feat)
        attention = query @ key.transpose(1, 2) / query.shape[-1] ** 0.5
        left = node_feat.unsqueeze(2).expand(-1, -1, node_feat.shape[1], -1)
        right = node_feat.unsqueeze(1).expand(-1, node_feat.shape[1], -1, -1)
        edges = self.edge_mlp(torch.cat((left, right), dim=-1)).squeeze(-1)
        logits = 0.5 * (attention + edges)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        off_diagonal = 1 - torch.eye(base_adj.shape[-1], device=base_adj.device, dtype=base_adj.dtype)
        mask = torch.sigmoid(logits / self.temperature) * off_diagonal
        return base_adj * mask, base_adj * (off_diagonal - mask), mask


class GDRNet(nn.Module):
    """Dual graph encoders with training-time feature-statistic transfer."""
    def __init__(
        self, channel_names, num_domains=1, temporal_hidden=64, graph_hidden=128,
        graph_out=128, dropout=0.2, static_graph_weight=0.35,
        disentangle_temperature=1.0, num_classes=3, temporal_tokens=64,
        graph_mode='hybrid', use_frequency_branch=True, sampling_rate=256.0,
        temporal_normalization='group', correlation_mode='absolute',
        recombine_alpha=0.5, cls_fusion='task_style_temporal',
        recombination_pooling='shared', detach_style_donor=False,
        enable_disentanglement=True, enable_recombination=True,
    ):
        super().__init__()
        if cls_fusion not in {'task', 'task_style_temporal'}:
            raise ValueError('cls_fusion must be task or task_style_temporal.')
        if recombination_pooling not in {'shared', 'separate'}:
            raise ValueError('recombination_pooling must be shared or separate.')
        if not 0 <= recombine_alpha <= 1:
            raise ValueError('recombine_alpha must be between 0 and 1.')
        self.enable_disentanglement = bool(enable_disentanglement)
        self.enable_recombination = bool(enable_recombination)
        self.requires_pairs = self.enable_recombination
        self.recombine_alpha = float(recombine_alpha)
        self.detach_style_donor = bool(detach_style_donor)
        self.cls_fusion = cls_fusion
        self.temporal = ChannelTemporalEncoder(temporal_hidden, dropout, temporal_tokens, use_frequency_branch, sampling_rate, temporal_normalization)
        self.graph_builder = GraphBuilder(channel_names, static_graph_weight, graph_mode, correlation_mode)
        self.graph_splitter = GraphMaskSplitter(temporal_hidden, dropout, disentangle_temperature) if self.enable_disentanglement else None
        self.task_encoder = GraphEncoder(temporal_hidden, graph_hidden, graph_out, dropout)
        self.style_encoder = GraphEncoder(temporal_hidden, graph_hidden, graph_out, dropout)
        self.cf_pool = AttentionPooling(graph_out) if recombination_pooling == 'separate' else None
        self.det_head = MLPHead(graph_out, 1, dropout)
        cls_dim = 2 * graph_out + temporal_hidden if cls_fusion == 'task_style_temporal' else graph_out
        self.cls_head = MLPHead(cls_dim, num_classes, dropout)

    @staticmethod
    def _adain(content_node, style_node):
        content_mean = content_node.mean(dim=1, keepdim=True)
        content_std = content_node.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        style_mean = style_node.mean(dim=1, keepdim=True)
        style_std = style_node.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (content_node - content_mean) / content_std * style_std + style_mean

    def _classification_features(self, task, style, temporal):
        if self.cls_fusion == 'task_style_temporal':
            return torch.cat((task, style, temporal.mean(dim=1)), dim=-1)
        return task

    def forward(self, x, cf_pair_index=None):
        node_feat = self.temporal(x)
        base_adj = self.graph_builder(x)
        if self.graph_splitter is None:
            a_task, a_style, task_mask = base_adj, base_adj, torch.ones_like(base_adj)
        else:
            a_task, a_style, task_mask = self.graph_splitter(node_feat, base_adj)
        task_nodes, task, task_attn = self.task_encoder(node_feat, a_task)
        style_nodes, style, style_attn = self.style_encoder(node_feat, a_style)
        cf_det_logits = cf_cls_logits = cf_pooled = None
        if self.enable_recombination and cf_pair_index is not None:
            donor_nodes, donor_style = style_nodes[cf_pair_index], style[cf_pair_index]
            if self.detach_style_donor:
                donor_nodes, donor_style = donor_nodes.detach(), donor_style.detach()
            cf_nodes = self._adain(task_nodes, donor_nodes)
            cf_nodes = self.recombine_alpha * cf_nodes + (1 - self.recombine_alpha) * task_nodes
            cf_pooled, _ = (self.task_encoder.pool if self.cf_pool is None else self.cf_pool)(cf_nodes)
            cf_det_logits = self.det_head(cf_pooled).squeeze(-1)
            cf_cls_logits = self.cls_head(self._classification_features(cf_pooled, donor_style, node_feat))
        return ModelOutput(
            det_logits=self.det_head(task).squeeze(-1),
            cls_logits=self.cls_head(self._classification_features(task, style, node_feat)),
            cf_det_logits=cf_det_logits, cf_cls_logits=cf_cls_logits,
            base_adj=base_adj, a_inv=a_task, a_dom=a_style, inv_mask=task_mask,
            z_inv_node=task_nodes, z_dom_node=style_nodes,
            z_inv_pooled=task, z_dom_pooled=style,
            inv_attn=task_attn, dom_attn=style_attn, node_feat=node_feat,
            z_cf_pooled=cf_pooled,
        )
