"""Supervised losses for original and recombined features."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    omega_det: float = 1.0
    alpha_cls: float = 0.0
    rho_cf: float = 0.2
    sigma_cf_cls: float = 0.0


@dataclass
class LossOutput:
    total: torch.Tensor
    items: dict


class Criterion:
    def __init__(self, weights: LossWeights, pos_weight=1.0, cls_class_weight=None):
        self.weights = weights
        self.pos_weight = pos_weight
        self.cls_class_weight = cls_class_weight
        self.recombination_scale = 1.0

    def set_auxiliary_scales(self, recombination=1.0):
        self.recombination_scale = float(min(max(recombination, 0.0), 1.0))

    def __call__(self, output, batch):
        det = batch['det_label'].float()
        cls = batch['cls_label'].long()
        zero = output.det_logits.sum() * 0.0
        weight = torch.as_tensor(self.pos_weight, device=det.device)
        det_loss = F.binary_cross_entropy_with_logits(output.det_logits, det, pos_weight=weight)
        positive = (det > .5) & (cls >= 0)
        cls_loss = zero
        if self.weights.alpha_cls > 0 and positive.any():
            cw = None if self.cls_class_weight is None else self.cls_class_weight.to(output.cls_logits)
            cls_loss = F.cross_entropy(output.cls_logits[positive], cls[positive], weight=cw)
        cf_loss = zero
        cf_cls_loss = zero
        cf_mask = batch.get('cf_mask', torch.zeros_like(det)).float()
        cf_cls_mask = batch.get('cf_cls_mask', torch.zeros_like(det)).float() * positive.float()
        # Training supplies these booleans before transferring the batch to
        # the GPU. Empty masks keep unused branch gradients as None.
        cf_has_pairs = batch.get('cf_has_pairs')
        cf_has_cls_pairs = batch.get('cf_has_cls_pairs')
        if (self.weights.rho_cf > 0 and output.cf_det_logits is not None
                and (bool(cf_mask.any()) if cf_has_pairs is None else cf_has_pairs)):
            raw = F.binary_cross_entropy_with_logits(output.cf_det_logits, det, pos_weight=weight, reduction='none')
            cf_loss = (raw * cf_mask).sum() / cf_mask.sum().clamp_min(1)
        if (self.weights.sigma_cf_cls > 0 and output.cf_cls_logits is not None
                and (bool(cf_cls_mask.any()) if cf_has_cls_pairs is None else cf_has_cls_pairs)):
            raw = F.cross_entropy(output.cf_cls_logits, cls.clamp_min(0), reduction='none')
            cf_cls_loss = (raw * cf_cls_mask).sum() / cf_cls_mask.sum().clamp_min(1)
        total = (self.weights.omega_det * det_loss + self.weights.alpha_cls * cls_loss
                 + self.recombination_scale * (self.weights.rho_cf * cf_loss + self.weights.sigma_cf_cls * cf_cls_loss))
        return LossOutput(total, {k: v.detach() for k, v in
            {'loss': total, 'det': det_loss, 'cls': cls_loss, 'cf': cf_loss, 'cf_cls': cf_cls_loss}.items()})
