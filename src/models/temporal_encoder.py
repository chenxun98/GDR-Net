from __future__ import annotations
import torch
from torch import nn


def conv_norm(num_channels: int, normalization: str):
    mode = str(normalization).lower()
    if mode in {'group', 'groupnorm', 'gn'}:
        groups = min(8, int(num_channels))
        while groups > 1 and int(num_channels) % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, int(num_channels))
    if mode in {'batch', 'batchnorm', 'bn'}:
        return nn.BatchNorm1d(int(num_channels))
    raise ValueError(f'Unsupported temporal normalization {normalization!r}; use group or batch.')

class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor):
        weight = torch.softmax(self.score(tokens).squeeze(-1), dim=-1)
        return torch.sum(weight.unsqueeze(-1) * tokens, dim=1)

class ChannelTemporalEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.2, temporal_tokens: int = 64, use_frequency_branch: bool = True, sampling_rate: float = 256.0, normalization: str = 'group'):
        super().__init__()
        self.use_frequency_branch = bool(use_frequency_branch)
        self.sampling_rate = float(sampling_rate)
        self.band_edges = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]
        branch_dim = max(hidden_dim // 4, 8)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, branch_dim, kernel_size=k, padding=(k // 2) * d, dilation=d), conv_norm(branch_dim, normalization), nn.GELU(), nn.Dropout(dropout))
            for k, d in [(5, 1), (9, 1), (15, 2), (31, 2)]
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(branch_dim * 4, hidden_dim, kernel_size=1),
            conv_norm(hidden_dim, normalization),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            conv_norm(hidden_dim, normalization),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_pool = nn.AdaptiveAvgPool1d(int(temporal_tokens))
        position = torch.arange(int(temporal_tokens), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        positional_encoding = torch.zeros(int(temporal_tokens), hidden_dim, dtype=torch.float32)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim > 1:
            positional_encoding[:, 1::2] = torch.cos(position * div_term[: positional_encoding[:, 1::2].shape[1]])
        self.register_buffer('positional_encoding', positional_encoding, persistent=False)
        heads = 4 if hidden_dim % 4 == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.temporal_context = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attn_pool = TemporalAttentionPooling(hidden_dim)
        if self.use_frequency_branch:
            self.freq_mlp = nn.Sequential(
                nn.Linear(len(self.band_edges), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(hidden_dim)

    def bandpower_features(self, x: torch.Tensor):
        spectrum = torch.fft.rfft(x.squeeze(1), dim=-1)
        power = spectrum.abs().pow(2)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sampling_rate).to(x.device)
        total = power[:, (freqs >= 0.5) & (freqs <= 40.0)].sum(dim=-1).clamp_min(1e-6)
        bands = []
        for low, high in self.band_edges:
            mask = (freqs >= low) & (freqs < high)
            bands.append(power[:, mask].sum(dim=-1) / total)
        return torch.stack(bands, dim=-1)

    def forward(self, x: torch.Tensor):
        b, c, t = x.shape
        x = x.reshape(b * c, 1, t)
        feat = self.proj(torch.cat([branch(x) for branch in self.branches], dim=1))
        tokens = self.token_pool(feat).transpose(1, 2)
        tokens = tokens + self.positional_encoding[:tokens.shape[1]].to(tokens.dtype).unsqueeze(0)
        tokens = self.temporal_context(tokens)
        temporal_feat = self.attn_pool(tokens)
        if self.use_frequency_branch:
            freq_feat = self.freq_mlp(self.bandpower_features(x))
            feat = self.fusion(torch.cat([temporal_feat, freq_feat], dim=-1))
        else:
            feat = temporal_feat
        feat = self.norm(feat)
        return feat.reshape(b, c, -1)
