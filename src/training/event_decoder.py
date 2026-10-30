from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

class EventDecoder:
    def __init__(self, threshold: float = 0.5, min_event_sec: float = 3.0, merge_gap_sec: float = 2.0, class_names=None, max_event_sec: float | None = None, off_threshold: float | None = None, smooth_sec: float = 0.0, use_boundary_offsets: bool = False, support_mode: str = 'stride_center', support_sec: float | None = None):
        self.threshold = threshold; self.min_event_sec = min_event_sec; self.merge_gap_sec = merge_gap_sec
        self.max_event_sec = max_event_sec
        self.off_threshold = float(off_threshold) if off_threshold is not None else float(threshold)
        self.smooth_sec = float(smooth_sec or 0.0)
        self.use_boundary_offsets = bool(use_boundary_offsets)
        self.support_mode = str(support_mode).lower()
        self.support_sec = None if support_sec is None else float(support_sec)
        if self.support_mode not in {'stride_center', 'window_coverage'}:
            raise ValueError('support_mode must be stride_center or window_coverage.')
        self.id_to_class = {i: name for i, name in enumerate(class_names or ['ABSZ', 'TNSZ', 'TCSZ'])}

    def _prepare_record(self, part: pd.DataFrame):
        part = part.sort_values('start_sec').reset_index(drop=True).copy()
        part['_event_prob'] = part['det_prob'].astype(float)
        if self.support_mode == 'window_coverage':
            part['_support_start'] = part['start_sec'].astype(float)
            part['_support_end'] = part['end_sec'].astype(float)
        else:
            support_sec = self.support_sec
            if support_sec is None:
                steps = part['center_sec'].astype(float).diff().dropna()
                support_sec = float(steps.median()) if not steps.empty else float((part['end_sec'] - part['start_sec']).median() / 2.0)
            half = max(float(support_sec), 1e-6) / 2.0
            part['_support_start'] = part['center_sec'].astype(float) - half
            part['_support_end'] = part['center_sec'].astype(float) + half
        if self.smooth_sec > 0 and len(part) > 1:
            step = float(part['start_sec'].diff().dropna().median())
            if np.isfinite(step) and step > 0:
                win = max(1, int(round(self.smooth_sec / step)))
                if win > 1:
                    part['_event_prob'] = part['_event_prob'].rolling(win, center=True, min_periods=1).mean()
        return part

    def _positive_groups(self, part: pd.DataFrame):
        probs = part['_event_prob'].to_numpy(dtype=float)
        starts = part['_support_start'].to_numpy(dtype=float)
        ends = part['_support_end'].to_numpy(dtype=float)
        groups: list[tuple[int, int]] = []
        current_start = None
        active = False
        for idx, prob in enumerate(probs):
            if not active and prob >= self.threshold:
                active = True
                current_start = idx
            elif active:
                if prob >= self.off_threshold:
                    continue
                else:
                    groups.append((int(current_start), idx - 1))
                    active = False
                    current_start = None
        if active and current_start is not None:
            groups.append((int(current_start), len(probs) - 1))
        if not groups:
            return []
        merged = [groups[0]]
        for group_start, group_end in groups[1:]:
            prev_start, prev_end_idx = merged[-1]
            prev_end = float(ends[prev_end_idx])
            cur_start = float(starts[group_start])
            if cur_start - prev_end <= self.merge_gap_sec:
                merged[-1] = (prev_start, group_end)
            else:
                merged.append((group_start, group_end))
        return merged

    def decode(self, window_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for record_id, part in window_df.groupby('record_id'):
            part = self._prepare_record(part)
            groups = self._positive_groups(part)
            if not groups: continue
            for start_idx, end_idx in groups:
                g = part.iloc[start_idx:end_idx + 1]
                duration = float(g['_support_end'].max() - g['_support_start'].min())
                if duration < self.min_event_sec: continue
                if self.max_event_sec is not None and duration > self.max_event_sec:
                    continue
                start = float(g['_support_start'].min())
                end = float(g['_support_end'].max())
                if self.use_boundary_offsets:
                    required = {'pred_delta_on', 'pred_delta_off'}
                    if not required.issubset(g.columns):
                        raise ValueError('Boundary-offset decoding was requested, but prediction columns are missing.')
                    onset = g['center_sec'] + g['pred_delta_on']
                    offset = g['center_sec'] + g['pred_delta_off']
                    start = float(np.clip(onset.min(), start, end))
                    end = float(np.clip(offset.max(), start, end + self.merge_gap_sec))
                    if end <= start:
                        start, end = float(g['_support_start'].min()), float(g['_support_end'].max())
                label_id = int(g['cls_pred'].mode().iloc[0])
                score_col = '_event_prob' if '_event_prob' in g else 'det_prob'
                rows.append({'record_id': str(record_id), 'start_sec': start, 'end_sec': end, 'label': self.id_to_class.get(label_id, 'bg'), 'score': float(g[score_col].mean())})
        return pd.DataFrame(rows, columns=['record_id', 'start_sec', 'end_sec', 'label', 'score'])

def _optimal_event_matches(pred_events: pd.DataFrame, gt_events: pd.DataFrame, iou_thr: float, match_label: bool, gt_coverage_thr: float | None = None):
    matches = []
    if pred_events.empty or gt_events.empty:
        return pd.DataFrame(matches)
    # Group once; repeatedly filtering all events becomes costly across records
    # and threshold sweeps. Group rows retain their original event order.
    pred_groups = dict(tuple(pred_events.groupby(pred_events['record_id'].astype(str), sort=False)))
    gt_groups = dict(tuple(gt_events.groupby(gt_events['record_id'].astype(str), sort=False)))
    for record_id in sorted(pred_groups.keys() & gt_groups.keys()):
        pred, gt = pred_groups[record_id], gt_groups[record_id]
        pred_start = pred['start_sec'].to_numpy(dtype=np.float64)[:, None]
        pred_end = pred['end_sec'].to_numpy(dtype=np.float64)[:, None]
        gt_start = gt['start_sec'].to_numpy(dtype=np.float64)[None, :]
        gt_end = gt['end_sec'].to_numpy(dtype=np.float64)[None, :]
        inter = np.maximum(0.0, np.minimum(pred_end, gt_end) - np.maximum(pred_start, gt_start))
        union = np.maximum(pred_end, gt_end) - np.minimum(pred_start, gt_start)
        ious = inter / np.maximum(union, 1e-6)
        gt_coverage = inter / np.maximum(gt_end - gt_start, 1e-6)
        eligible = ious >= iou_thr
        if gt_coverage_thr is not None:
            eligible |= gt_coverage >= gt_coverage_thr
        if match_label:
            same_label = pred['label'].astype(str).to_numpy()[:, None] == gt['label'].astype(str).to_numpy()[None, :]
            eligible &= same_label
            ious = np.where(same_label, ious, 0.0)
        score = np.where(eligible, 1.0 + np.maximum(ious, gt_coverage), 0.0)
        pred_idx, gt_idx = linear_sum_assignment(score, maximize=True)
        for i, j in zip(pred_idx.tolist(), gt_idx.tolist()):
            if score[i, j] <= 1.0:
                continue
            p = pred.iloc[i]
            g = gt.iloc[j]
            matches.append({
                'record_id': record_id,
                'pred_index': i,
                'gt_index': j,
                'iou': float(ious[i, j]),
                'onset_latency_sec': float(p['start_sec'] - g['start_sec']),
                'duration_error_sec': float((p['end_sec'] - p['start_sec']) - (g['end_sec'] - g['start_sec'])),
            })
    return pd.DataFrame(matches)


def event_level_f1(pred_events: pd.DataFrame, gt_events: pd.DataFrame, iou_thr: float = 0.3, match_label: bool = False, gt_coverage_thr: float | None = None):
    diagnostics = {
        'onset_latency_median_sec': np.nan,
        'onset_abs_error_median_sec': np.nan,
        'duration_abs_error_median_sec': np.nan,
    }
    if gt_events.empty and pred_events.empty: return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0, **diagnostics}
    if gt_events.empty: return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_events), 'fn': 0, **diagnostics}
    if pred_events.empty: return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_events), **diagnostics}
    matches = _optimal_event_matches(pred_events, gt_events, iou_thr, match_label, gt_coverage_thr)
    tp = int(len(matches))
    fp = len(pred_events) - tp; fn = len(gt_events) - tp
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    onset = matches['onset_latency_sec'].to_numpy(float) if not matches.empty else np.array([], dtype=float)
    duration = matches['duration_error_sec'].to_numpy(float) if not matches.empty else np.array([], dtype=float)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'onset_latency_median_sec': float(np.median(onset)) if len(onset) else np.nan,
        'onset_abs_error_median_sec': float(np.median(np.abs(onset))) if len(onset) else np.nan,
        'duration_abs_error_median_sec': float(np.median(np.abs(duration))) if len(duration) else np.nan,
    }
