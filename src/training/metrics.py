from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from .event_decoder import EventDecoder, event_level_f1

def _record_hours(pred_df: pd.DataFrame):
    if pred_df.empty:
        return 0.0
    if 'record_duration_sec' in pred_df.columns:
        durations = pred_df.groupby('record_id')['record_duration_sec'].first().astype(float)
    else:
        durations = pred_df.groupby('record_id').apply(lambda x: float(x['end_sec'].max() - x['start_sec'].min()), include_groups=False)
    return float(durations.sum() / 3600.0)

def _binary_f1(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum()); fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-8)

def _balanced_acc(y_true, y_pred):
    pos = (y_true == 1); neg = (y_true == 0)
    tpr = ((y_pred[pos] == 1).sum() / max(pos.sum(), 1)) if pos.any() else 0.0
    tnr = ((y_pred[neg] == 0).sum() / max(neg.sum(), 1)) if neg.any() else 0.0
    return 0.5 * (float(tpr) + float(tnr))

def _auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = rankdata(y_score, method='average')
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)

def _average_precision(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score))

def _macro_f1(y_true, y_pred, num_classes=4):
    vals = []
    for c in range(num_classes):
        vals.append(_binary_f1((y_true == c).astype(int), (y_pred == c).astype(int)))
    return float(np.mean(vals))

class MetricsComputer:
    def __init__(
        self,
        threshold: float = 0.5,
        min_event_sec: float = 3.0,
        merge_gap_sec: float = 2.0,
        num_classes: int = 4,
        class_names=None,
        max_event_sec: float | None = None,
        compute_classification: bool = True,
        label_groups: dict | None = None,
        off_threshold: float | None = None,
        smooth_sec: float = 0.0,
        use_boundary_offsets: bool = False,
        event_support_mode: str = 'stride_center',
        event_support_sec: float | None = None,
        event_iou_threshold: float = 0.3,
        event_gt_coverage_threshold: float | None = None,
    ):
        self.decoder = EventDecoder(threshold, min_event_sec, merge_gap_sec, class_names, max_event_sec, off_threshold, smooth_sec, use_boundary_offsets, event_support_mode, event_support_sec)
        self.num_classes = int(num_classes)
        self.compute_classification = bool(compute_classification)
        self.label_groups = {str(k): {str(x) for x in v} for k, v in (label_groups or {}).items()}
        self.event_iou_threshold = float(event_iou_threshold)
        self.event_gt_coverage_threshold = None if event_gt_coverage_threshold is None else float(event_gt_coverage_threshold)
    def compute(self, pred_df: pd.DataFrame, gt_events_df: pd.DataFrame, *, include_events: bool = True):
        y_true = pred_df['det_true'].to_numpy(); y_prob = pred_df['det_prob'].to_numpy(); y_pred = (y_prob >= self.decoder.threshold).astype(int)
        y_cls_true = pred_df['cls_true'].to_numpy(); y_cls_pred = pred_df['cls_pred'].to_numpy()
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        hours = _record_hours(pred_df)
        out = {
            'win_f1': _binary_f1(y_true, y_pred),
            'win_bal_acc': _balanced_acc(y_true, y_pred),
            'win_fpr': fp / max(fp + tn, 1),
            'win_fp_per_hour': fp / max(hours, 1e-8),
            'win_auroc': _auroc(y_true, y_prob),
            'win_ap': _average_precision(y_true, y_prob),
        }
        out['win_auprc'] = out['win_ap']  # Historical column name; both report AP.
        if self.compute_classification:
            cls_mask = (y_true == 1) & (y_cls_true >= 0)
            if cls_mask.any():
                out['cls_macro_f1'] = _macro_f1(y_cls_true[cls_mask], y_cls_pred[cls_mask], self.num_classes)
                for class_id in range(self.num_classes):
                    name = self.decoder.id_to_class.get(class_id, str(class_id))
                    out[f'cls_f1_{name}'] = _binary_f1((y_cls_true[cls_mask] == class_id).astype(int), (y_cls_pred[cls_mask] == class_id).astype(int))
                    out[f'cls_support_{name}'] = int((y_cls_true[cls_mask] == class_id).sum())
            else:
                out['cls_macro_f1'] = 0.0
                for class_id in range(self.num_classes):
                    name = self.decoder.id_to_class.get(class_id, str(class_id))
                    out[f'cls_f1_{name}'] = 0.0
                    out[f'cls_support_{name}'] = 0
            ap = float(out['win_ap'])
            macro_f1 = float(out['cls_macro_f1'])
            out['joint_ap_cls_hmean'] = 2.0 * ap * macro_f1 / max(ap + macro_f1, 1e-8)
        if not include_events:
            return out
        pred_events = self.decoder.decode(pred_df)
        gt_use = gt_events_df[['record_id', 'start_sec', 'end_sec', 'label']].copy() if not gt_events_df.empty else pd.DataFrame(columns=['record_id', 'start_sec', 'end_sec', 'label'])
        ev = event_level_f1(pred_events, gt_use, self.event_iou_threshold, gt_coverage_thr=self.event_gt_coverage_threshold)
        strict_ev = event_level_f1(pred_events, gt_use, self.event_iou_threshold)
        out['event_precision'] = ev['precision']; out['event_recall'] = ev['recall']; out['event_f1'] = ev['f1']; out['event_fp'] = ev['fp']; out['event_fp_per_hour'] = ev['fp'] / max(hours, 1e-8); out['event_fp_per_24h'] = 24.0 * ev['fp'] / max(hours, 1e-8)
        out['event_onset_latency_median_sec'] = ev['onset_latency_median_sec']
        out['event_onset_abs_error_median_sec'] = ev['onset_abs_error_median_sec']
        out['event_duration_abs_error_median_sec'] = ev['duration_abs_error_median_sec']
        out['event_strict_iou_precision'] = strict_ev['precision']
        out['event_strict_iou_recall'] = strict_ev['recall']
        out['event_strict_iou_f1'] = strict_ev['f1']
        if not gt_use.empty:
            durations = gt_use['end_sec'].astype(float) - gt_use['start_sec'].astype(float)
            duration_groups = {
                'lt8s': durations < 8.0,
                '8to30s': (durations >= 8.0) & (durations <= 30.0),
                'gt30s': durations > 30.0,
            }
            for group_name, mask in duration_groups.items():
                group_gt = gt_use[mask].copy()
                out[f'event_recall_duration_{group_name}'] = _event_recall(pred_events, group_gt, self.event_iou_threshold, self.event_gt_coverage_threshold)
                out[f'event_gt_duration_{group_name}'] = int(len(group_gt))
        for group_name, labels in self.label_groups.items():
            group_gt = gt_use[gt_use['label'].astype(str).isin(labels)].copy()
            out[f'event_recall_{group_name}'] = _event_recall(pred_events, group_gt, self.event_iou_threshold, self.event_gt_coverage_threshold)
            out[f'event_gt_{group_name}'] = int(len(group_gt))
        if self.compute_classification:
            ev_cls = event_level_f1(pred_events, gt_use, self.event_iou_threshold, match_label=True, gt_coverage_thr=self.event_gt_coverage_threshold)
            out['event_cls_precision'] = ev_cls['precision']
            out['event_cls_recall'] = ev_cls['recall']
            out['event_cls_f1'] = ev_cls['f1']
        return out

def _event_recall(pred_events: pd.DataFrame, gt_events: pd.DataFrame, iou_thr: float = 0.3, gt_coverage_thr: float | None = None):
    if gt_events.empty:
        return 0.0
    if pred_events.empty:
        return 0.0
    return float(event_level_f1(pred_events, gt_events, iou_thr=iou_thr, gt_coverage_thr=gt_coverage_thr)['recall'])
