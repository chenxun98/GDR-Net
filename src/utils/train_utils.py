from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def batch_to_device(batch: dict, device: torch.device):
    tensor_keys = {'x', 'det_label', 'cls_label', 'cf_pair_index', 'cf_mask', 'cf_cls_mask'}
    return {key: value.to(device, non_blocking=True)
            if key in tensor_keys and torch.is_tensor(value) else value
            for key, value in batch.items()}


def _numpy_column(value, dtype):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)

@torch.no_grad()
def predict_epoch(model, loader, device: torch.device, metrics_computer, gt_events_df: pd.DataFrame,
                  *, compute_metrics: bool = True, window_only: bool = False):
    model.eval(); frames = []
    for batch in tqdm(loader, desc='Eval', leave=False):
        output = model(batch['x'].to(device, non_blocking=True))
        det_prob = torch.sigmoid(output.det_logits).cpu().numpy()
        cls_prob = torch.softmax(output.cls_logits, dim=-1).cpu().numpy()
        columns = {
            'record_id': batch['record_id'],
            'patient_id': batch['patient_id'],
            'record_duration_sec': _numpy_column(batch['record_duration_sec'], np.float64),
            'start_sec': _numpy_column(batch['start_sec'], np.float64),
            'end_sec': _numpy_column(batch['end_sec'], np.float64),
            'center_sec': _numpy_column(batch['center_sec'], np.float64),
            'det_true': _numpy_column(batch['det_label'], np.int64),
            'det_prob': det_prob.astype(np.float64, copy=False),
            'cls_true': _numpy_column(batch['cls_label'], np.int64),
            'cls_pred': cls_prob.argmax(axis=-1).astype(np.int64, copy=False),
        }
        boundary_pred = getattr(output, 'boundary_pred', None)
        if boundary_pred is not None:
            scale = batch['boundary_scale'].to(boundary_pred.device, non_blocking=True)
            boundary = (boundary_pred * scale.unsqueeze(-1)).cpu().numpy()
            columns['pred_delta_on'] = boundary[:, 0].astype(np.float64, copy=False)
            columns['pred_delta_off'] = boundary[:, 1].astype(np.float64, copy=False)
        for cls_idx in range(cls_prob.shape[1]):
            columns[f'cls_prob_{cls_idx}'] = cls_prob[:, cls_idx].astype(np.float64, copy=False)
        frames.append(pd.DataFrame(columns))
    pred_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    scores = (metrics_computer.compute(pred_df, gt_events_df, include_events=not window_only)
              if compute_metrics else {})
    return scores, pred_df
