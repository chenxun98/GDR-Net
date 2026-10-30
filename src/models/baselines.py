"""EEGNet and CNN-LSTM baselines for both seizure tasks."""
from __future__ import annotations
import torch
from torch import nn
from .heads import MLPHead
from .outputs import ModelOutput


class CNNBiLSTMBaseline(nn.Module):
    """CNN-LSTM: two strided convolutions, one bidirectional LSTM, attention pooling.

    The configured hidden_dim=128 gives 64 convolutional channels and 64
    recurrent units per direction.
    """

    def __init__(self, num_channels, num_domains=1, hidden_dim=128, dropout=0.2, num_classes=3):
        super().__init__()
        conv_dim = max(hidden_dim // 2, 32)
        lstm_hidden = max(hidden_dim // 2, 32)
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, conv_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(conv_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(conv_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_dim, hidden_size=lstm_hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        out_dim = lstm_hidden * 2
        self.attn = nn.Linear(out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        self.det_head = MLPHead(out_dim, 1, dropout)
        self.cls_head = MLPHead(out_dim, num_classes, dropout)

    def forward(self, x, cf_pair_index=None):
        tokens, _ = self.lstm(self.conv(x).transpose(1, 2))
        weights = torch.softmax(self.attn(tokens).squeeze(-1), dim=-1)
        pooled = self.norm(torch.sum(weights.unsqueeze(-1) * tokens, dim=1))
        return ModelOutput(
            det_logits=self.det_head(pooled).squeeze(-1),
            cls_logits=self.cls_head(pooled), z_inv_pooled=pooled,
        )


class EEGNetBaseline(nn.Module):
    """Temporal, spatial depthwise, and separable convolutions with task heads."""
    def __init__(
        self, num_channels, num_domains=1, dropout=0.25, num_classes=3,
        sampling_rate=256.0, f1=8, depth_multiplier=2, f2=None, pool_tokens=16,
    ):
        super().__init__()
        f1, depth_multiplier = int(f1), int(depth_multiplier)
        f2 = int(f2 or f1 * depth_multiplier)
        temporal_kernel = max(3, int(round(sampling_rate / 2.0)))
        temporal_kernel += int(temporal_kernel % 2 == 0)
        separable_kernel = max(3, int(round(sampling_rate / 8.0)))
        separable_kernel += int(separable_kernel % 2 == 0)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth_multiplier, (num_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * depth_multiplier, f1 * depth_multiplier, (1, separable_kernel),
                      padding=(0, separable_kernel // 2), groups=f1 * depth_multiplier, bias=False),
            nn.Conv2d(f1 * depth_multiplier, f2, 1, bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, int(pool_tokens)))
        out_dim = f2 * int(pool_tokens)
        self.norm = nn.LayerNorm(out_dim)
        self.det_head = MLPHead(out_dim, 1, dropout)
        self.cls_head = MLPHead(out_dim, num_classes, dropout)

    def forward(self, x, cf_pair_index=None):
        features = self.block2(self.block1(x.unsqueeze(1)))
        pooled = self.norm(self.pool(features).flatten(1))
        return ModelOutput(det_logits=self.det_head(pooled).squeeze(-1), cls_logits=self.cls_head(pooled))
