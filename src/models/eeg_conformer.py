"""Independent EEG Conformer architecture implementation for seizure tasks.

Reference: Song et al., DOI: 10.1109/TNSRE.2022.3230250.
Author implementation: https://github.com/eeyhsong/EEG-Conformer

The convolutional tokenizer, pre-normalized attention blocks and flattened
classifier follow the published architecture. The project's 18-channel,
8-second input and separate detection/subtype heads are adaptations. This
module does not reproduce the authors' training or evaluation protocol.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from .outputs import ModelOutput


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}.')
    return int(value)


def _dropout_probability(name: str, value: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0.0 <= value < 1.0):
        raise ValueError(f'{name} must be finite and in [0, 1), got {value!r}.')
    return float(value)


class _SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.attention_scale = embedding_dim ** -0.5
        self.dropout = dropout
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        # The author implementation divides by sqrt(embedding_dim), rather
        # than sqrt(head_dim); preserve that scaling explicitly.
        attended = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.attention_scale,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.projection(attended)


class _EncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = _SelfAttention(embedding_dim, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention_dropout(
            self.attention(self.attention_norm(tokens)),
        )
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class EEGConformerBaseline(nn.Module):
    """Convolutional EEG tokens followed by global temporal self-attention.

    ``num_times`` fixes the window size because the classifier flattens all
    tokens. No positional embedding is added, matching the author architecture.
    """

    def __init__(
        self, num_channels, num_times, num_classes=3, embedding_dim=40,
        depth=6, num_heads=10, dropout=0.5, head_dropout=0.3,
        temporal_kernel=25, pool_kernel=75, pool_stride=15,
    ):
        super().__init__()
        self.num_channels = _positive_int('num_channels', num_channels)
        self.num_times = _positive_int('num_times', num_times)
        num_classes = _positive_int('num_classes', num_classes)
        embedding_dim = _positive_int('embedding_dim', embedding_dim)
        depth = _positive_int('depth', depth)
        num_heads = _positive_int('num_heads', num_heads)
        temporal_kernel = _positive_int('temporal_kernel', temporal_kernel)
        pool_kernel = _positive_int('pool_kernel', pool_kernel)
        pool_stride = _positive_int('pool_stride', pool_stride)
        dropout = _dropout_probability('dropout', dropout)
        head_dropout = _dropout_probability('head_dropout', head_dropout)
        if embedding_dim % num_heads:
            raise ValueError(
                f'embedding_dim ({embedding_dim}) must be divisible by '
                f'num_heads ({num_heads}).',
            )
        minimum_times = temporal_kernel + pool_kernel - 1
        if self.num_times < minimum_times:
            raise ValueError(
                f'num_times must be at least {minimum_times} for '
                f'temporal_kernel={temporal_kernel} and pool_kernel={pool_kernel}; '
                f'got {self.num_times}.',
            )
        self.num_tokens = (self.num_times - minimum_times) // pool_stride + 1
        self.tokenizer = nn.Sequential(
            nn.Conv2d(1, embedding_dim, (1, temporal_kernel)),
            nn.Conv2d(embedding_dim, embedding_dim, (self.num_channels, 1)),
            nn.BatchNorm2d(embedding_dim),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(dropout),
            nn.Conv2d(embedding_dim, embedding_dim, 1),
        )
        # Construct each block separately so the initial weights are independent.
        self.encoder = nn.Sequential(*[
            _EncoderBlock(embedding_dim, num_heads, dropout) for _ in range(depth)
        ])
        self.shared_head = nn.Sequential(
            nn.Linear(embedding_dim * self.num_tokens, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(head_dropout),
        )
        self.det_head = nn.Linear(32, 1)
        self.cls_head = nn.Linear(32, num_classes)

    def forward(self, x, cf_pair_index=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            shape = getattr(x, 'shape', None)
            raise ValueError(f'Expected an EEG tensor shaped (B, C, T), got {shape}.')
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f'Expected {self.num_channels} EEG channels, got {x.shape[1]}.',
            )
        if x.shape[2] != self.num_times:
            raise ValueError(
                f'Expected {self.num_times} samples per EEG window, got {x.shape[2]}.',
            )
        tokens = self.tokenizer(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        features = self.shared_head(self.encoder(tokens).flatten(1))
        return ModelOutput(
            det_logits=self.det_head(features).squeeze(-1),
            cls_logits=self.cls_head(features),
            z_inv_pooled=features,
        )
