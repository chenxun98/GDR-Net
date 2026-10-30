"""Independent supervised DCRNN adaptation for seizure detection/classification.

Reference: Tang et al., https://arxiv.org/abs/2104.08336
Author implementation: https://github.com/tsy935/eeg-gnn-ssl

This implementation uses the paper's directed correlation graph and forward /
backward random-walk power bases. It does not copy the reference cell's
Chebyshev recurrence. Online source-training spectral normalization replaces
the author's offline per-node scaler. The 18 bipolar channels, 256-Hz input,
and separate task heads adapt DCRNN to this project; self-supervised pretraining
and the original paper's complete experimental protocol are not reproduced.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from .outputs import ModelOutput


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}.')
    return int(value)


def _positive_real(name: str, value: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0.0):
        raise ValueError(f'{name} must be finite and positive, got {value!r}.')
    return float(value)


class _DiffusionGraphConv(nn.Module):
    """Learned combination of X, Sf X, ..., Sf^K X, Sb X, ..., Sb^K X."""

    def __init__(self, input_dim: int, output_dim: int, max_diffusion_step: int):
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        self.linear = nn.Linear(input_dim * (1 + 2 * max_diffusion_step), output_dim)
        nn.init.xavier_normal_(self.linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor, supports: tuple[torch.Tensor, torch.Tensor]):
        terms = [features]
        for support in supports:
            # Restart from X for each direction: backward diffusion must never
            # start from the final forward-diffusion state.
            diffused = features
            for _ in range(self.max_diffusion_step):
                diffused = torch.bmm(support, diffused)
                terms.append(diffused)
        return self.linear(torch.cat(terms, dim=-1))


class _DCGRUCell(nn.Module):
    """GRU whose reset/update and candidate projections are graph diffusions."""

    def __init__(self, input_dim: int, hidden_dim: int, max_diffusion_step: int):
        super().__init__()
        self.gates = _DiffusionGraphConv(
            input_dim + hidden_dim, 2 * hidden_dim, max_diffusion_step,
        )
        self.candidate = _DiffusionGraphConv(
            input_dim + hidden_dim, hidden_dim, max_diffusion_step,
        )

    def forward(
        self, features: torch.Tensor, hidden: torch.Tensor,
        supports: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        reset, update = torch.sigmoid(
            self.gates(torch.cat((features, hidden), dim=-1), supports),
        ).chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(torch.cat((features, reset * hidden), dim=-1), supports),
        )
        return update * hidden + (1.0 - update) * candidate


class DCRNNBaseline(nn.Module):
    """Clip-wise correlation graphs and stacked diffusion-convolutional GRUs.

    Each one-second EEG segment supplies log-FFT node features. A single graph
    is constructed from the entire clip before spectral normalization, then
    reused for all recurrent steps. Classification applies linear task heads
    at each node before taking the maximum node logit.
    """

    # EMA must preserve the normalization statistics observed during training,
    # rather than smooth their existing moving averages a second time.
    ema_copy_buffers = ('spectral_norm.running_mean', 'spectral_norm.running_var')

    def __init__(
        self, num_channels, num_times, sampling_rate=256.0, num_classes=3,
        hidden_dim=64, num_layers=2, max_diffusion_step=2, top_k=3,
        step_seconds=1.0, dropout=0.0,
    ):
        super().__init__()
        self.num_channels = _integer('num_channels', num_channels)
        self.num_times = _integer('num_times', num_times)
        self.hidden_dim = _integer('hidden_dim', hidden_dim)
        self.num_layers = _integer('num_layers', num_layers)
        self.max_diffusion_step = _integer('max_diffusion_step', max_diffusion_step, 0)
        self.top_k = _integer('top_k', top_k, 0)
        num_classes = _integer('num_classes', num_classes)
        sampling_rate = _positive_real('sampling_rate', sampling_rate)
        step_seconds = _positive_real('step_seconds', step_seconds)
        if (isinstance(dropout, bool) or not isinstance(dropout, Real)
                or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0):
            raise ValueError(f'dropout must be finite and in [0, 1), got {dropout!r}.')
        if self.top_k >= self.num_channels:
            raise ValueError('top_k must be smaller than num_channels (self edges are separate).')
        samples_per_step = sampling_rate * step_seconds
        if (not math.isfinite(samples_per_step)
                or not math.isclose(samples_per_step, round(samples_per_step), rel_tol=0.0, abs_tol=1e-8)):
            raise ValueError('sampling_rate * step_seconds must be an integer number of samples.')
        self.step_samples = int(round(samples_per_step))
        if self.step_samples < 2 or self.num_times % self.step_samples != 0:
            raise ValueError('num_times must contain complete recurrent steps of at least two samples.')
        self.num_steps = self.num_times // self.step_samples
        self.frequency_bins = self.step_samples // 2
        if self.num_steps * self.frequency_bins < 2:
            raise ValueError('Each channel must supply at least two spectral values for normalization.')
        self.spectral_norm = nn.BatchNorm1d(self.num_channels, affine=False)
        self.cells = nn.ModuleList([
            _DCGRUCell(
                self.frequency_bins if layer == 0 else self.hidden_dim,
                self.hidden_dim, self.max_diffusion_step,
            )
            for layer in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(float(dropout))
        self.det_head = nn.Linear(self.hidden_dim, 1)
        self.cls_head = nn.Linear(self.hidden_dim, num_classes)

    def _log_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Return B, steps, nodes, floor(step_samples/2) log-amplitude features."""
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        segments = fft_input.reshape(
            x.shape[0], self.num_channels, self.num_steps, self.step_samples,
        )
        # Include DC, exclude Nyquist for even lengths, matching the reference
        # FFT's first floor(N/2) bins; retain no phase features.
        amplitude = torch.fft.rfft(segments, dim=-1).abs()[..., :self.frequency_bins]
        return amplitude.clamp_min(1e-8).log().permute(0, 2, 1, 3).contiguous()

    def _correlation_graph(self, log_spectrum: torch.Tensor) -> torch.Tensor:
        """Absolute normalized zero-lag cross-correlation, without demeaning."""
        per_node = log_spectrum.permute(0, 2, 1, 3).flatten(start_dim=2)
        per_node = F.normalize(per_node, p=2, dim=-1, eps=1e-12)
        similarity = torch.bmm(per_node, per_node.transpose(1, 2)).abs().clamp(max=1.0)
        diagonal = torch.eye(self.num_channels, device=similarity.device, dtype=torch.bool)
        adjacency = torch.zeros_like(similarity)
        if self.top_k:
            values, indices = similarity.masked_fill(diagonal.unsqueeze(0), -1.0).topk(
                self.top_k, dim=-1,
            )
            adjacency.scatter_(-1, indices, values)
        return adjacency.masked_fill(diagonal.unsqueeze(0), 1.0)

    @staticmethod
    def _random_walk_supports(adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Graph multiplication uses the transpose of each row-normalized walk.
        # Normalize A and A.T independently; incoming/outgoing degrees differ
        # because top-k neighbour selection need not produce a symmetric graph.
        forward = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        reverse = adjacency.transpose(1, 2)
        backward = reverse / reverse.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return forward.transpose(1, 2).contiguous(), backward.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, cf_pair_index=None) -> ModelOutput:
        if (x.ndim != 3 or x.shape[0] < 1
                or x.shape[1:] != (self.num_channels, self.num_times)):
            raise ValueError(
                f'Expected (batch, {self.num_channels}, {self.num_times}), got {tuple(x.shape)}.',
            )
        if not x.is_floating_point():
            raise ValueError('DCRNN inputs must be floating-point EEG samples.')
        spectrum = self._log_spectrum(x)
        supports = self._random_walk_supports(self._correlation_graph(spectrum))
        batch = x.shape[0]
        per_node = spectrum.permute(0, 2, 1, 3).reshape(batch, self.num_channels, -1)
        sequence = self.spectral_norm(per_node).reshape(
            batch, self.num_channels, self.num_steps, self.frequency_bins,
        ).permute(0, 2, 1, 3)
        for cell in self.cells:
            hidden = sequence.new_zeros(batch, self.num_channels, self.hidden_dim)
            outputs = []
            for step in range(self.num_steps):
                hidden = cell(sequence[:, step], hidden, supports)
                outputs.append(hidden)
            sequence = torch.stack(outputs, dim=1)
        node_features = F.relu(self.dropout(sequence[:, -1]))
        return ModelOutput(
            det_logits=self.det_head(node_features).squeeze(-1).amax(dim=1),
            cls_logits=self.cls_head(node_features).amax(dim=1),
        )
