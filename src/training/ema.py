"""Optional weight averaging; source validation selects and saves the same weights."""
from copy import deepcopy
import torch


class ModelEMA:
    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError('EMA decay must be between 0 and 1.')
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        # Some input scalers already maintain running moments. Copy those
        # moments directly instead of smoothing them for a second time.
        copy_buffers = getattr(model, 'ema_copy_buffers', ())
        for name, value in self.model.state_dict().items():
            if value.is_floating_point() and name not in copy_buffers:
                value.mul_(decay).add_(source[name], alpha=1.0 - decay)
            else:
                value.copy_(source[name])
