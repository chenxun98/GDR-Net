"""Task predictions and optional graph representations."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelOutput:
    det_logits: torch.Tensor
    cls_logits: torch.Tensor
    boundary_pred: Optional[torch.Tensor] = None
    cf_det_logits: Optional[torch.Tensor] = None
    cf_cls_logits: Optional[torch.Tensor] = None
    base_adj: Optional[torch.Tensor] = None
    a_inv: Optional[torch.Tensor] = None
    a_dom: Optional[torch.Tensor] = None
    inv_mask: Optional[torch.Tensor] = None
    z_inv_node: Optional[torch.Tensor] = None
    z_dom_node: Optional[torch.Tensor] = None
    z_inv_pooled: Optional[torch.Tensor] = None
    z_dom_pooled: Optional[torch.Tensor] = None
    inv_attn: Optional[torch.Tensor] = None
    dom_attn: Optional[torch.Tensor] = None
    node_feat: Optional[torch.Tensor] = None
    z_cf_pooled: Optional[torch.Tensor] = None
