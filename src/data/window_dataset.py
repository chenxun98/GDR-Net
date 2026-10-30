from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.constants import type_class_names_to_id

def read_manifest_csv(path: str | Path, **kwargs):
    path = Path(path)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='gb18030', **kwargs)

@dataclass
class WindowSampleMeta:
    record_id: str
    patient_id: str
    eeg_path: str
    start_sec: float
    end_sec: float
    center_sec: float
    det_label: int
    cls_label: int
    boundary: np.ndarray
    boundary_mask: float
    domain_label: int
    record_duration_sec: float

class EEGWindowDataset(Dataset):
    def __init__(
        self,
        root: str,
        window_sec: float,
        stride_sec: float,
        positive_overlap_ratio: float = 0.2,
        positive_event_coverage_ratio: float | None = None,
        split_record_ids=None,
        patient_to_domain=None,
        record_to_domain=None,
        events_filename: str = 'events.csv',
        type_class_names=None,
        max_negative_windows_per_record: int | None = None,
        negative_pos_ratio: float | None = None,
        negative_only_record_windows: int | None = None,
        label_map=None,
        include_labels=None,
        drop_records_with_excluded_labels: bool = False,
        allow_unknown_positive_labels: bool = False,
        fixed_domain_label: int | None = None,
        window_normalization: str = 'channel_zscore',
        require_events_file: bool = True,
    ):
        self.root = Path(root)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.positive_overlap_ratio = float(positive_overlap_ratio)
        self.positive_event_coverage_ratio = None if positive_event_coverage_ratio is None else float(positive_event_coverage_ratio)
        self.type_class_to_id = type_class_names_to_id(type_class_names)
        self.label_map = {str(k): str(v) for k, v in (label_map or {}).items()}
        self.include_labels = {str(x) for x in include_labels} if include_labels else None
        self.drop_records_with_excluded_labels = bool(drop_records_with_excluded_labels)
        self.allow_unknown_positive_labels = bool(allow_unknown_positive_labels)
        self.fixed_domain_label = fixed_domain_label
        self.window_normalization = str(window_normalization).lower()
        if self.window_normalization not in {'channel_zscore', 'robust_channel', 'none'}:
            raise ValueError('window_normalization must be channel_zscore, robust_channel, or none.')
        self.fallback_positive_class_id = 0
        self.max_negative_windows_per_record = max_negative_windows_per_record
        self.negative_pos_ratio = negative_pos_ratio
        self.negative_only_record_windows = negative_only_record_windows
        self.events_filename = str(events_filename)
        self.records = read_manifest_csv(self.root / 'records.csv')
        events_path = self.root / self.events_filename
        if require_events_file and not events_path.exists():
            raise FileNotFoundError(f'Required event manifest does not exist: {events_path}')
        self.events = read_manifest_csv(events_path) if events_path.exists() else pd.DataFrame(columns=['record_id', 'start_sec', 'end_sec', 'label'])
        self.records['record_id'] = self.records['record_id'].astype(str)
        self.records['patient_id'] = self.records['patient_id'].astype(str)
        self.events['record_id'] = self.events['record_id'].astype(str)
        original_events = self.events.copy()
        if split_record_ids is not None:
            keep = set(str(x) for x in split_record_ids)
            self.records = self.records[self.records['record_id'].isin(keep)].reset_index(drop=True)
        if self.include_labels is not None and self.drop_records_with_excluded_labels and 'label' in original_events.columns:
            mapped_labels = original_events['label'].astype(str).map(lambda x: self.label_map.get(x, x))
            event_labels = pd.DataFrame({'record_id': original_events['record_id'].astype(str), 'label': mapped_labels})
            labels_by_record = event_labels.groupby('record_id')['label'].apply(lambda s: set(s)).to_dict()
            keep_records = []
            for rid in self.records['record_id'].astype(str):
                labels = labels_by_record.get(rid, set())
                keep_records.append(not labels or labels.issubset(self.include_labels))
            self.records = self.records[keep_records].reset_index(drop=True)
        if self.include_labels is not None and 'label' in self.events.columns:
            mapped_labels = self.events['label'].astype(str).map(lambda x: self.label_map.get(x, x))
            self.events = self.events[mapped_labels.isin(self.include_labels)].copy()
        kept_record_ids = set(self.records['record_id'].astype(str))
        self.events = self.events[self.events['record_id'].astype(str).isin(kept_record_ids)].copy()
        self.record_id_to_sr = {str(r.record_id): float(r.sampling_rate) for r in self.records.itertuples(index=False)}
        self.events_by_record = {rid: df.sort_values('start_sec').reset_index(drop=True) for rid, df in self.events.groupby('record_id')}
        if patient_to_domain is None:
            pats = sorted(self.records['patient_id'].astype(str).unique().tolist())
            patient_to_domain = {p: i for i, p in enumerate(pats)}
        self.patient_to_domain = patient_to_domain
        self.record_to_domain = {str(k): int(v) for k, v in (record_to_domain or {}).items()}
        self.samples = self._build_index()
    def _build_index(self):
        samples = []
        for row in self.records.itertuples(index=False):
            record_samples = []
            rid = str(row.record_id); pid = str(row.patient_id); eeg_path = str(row.eeg_path); sr = float(row.sampling_rate); duration_sec = int(row.n_samples) / sr
            if eeg_path and not Path(eeg_path).exists():
                local_path = self.root / 'signals' / Path(eeg_path).name
                if local_path.exists():
                    eeg_path = str(local_path)
            ev_df = self.events_by_record.get(rid, pd.DataFrame(columns=['start_sec', 'end_sec', 'label']))
            n_windows = max(0, int(np.floor((duration_sec - self.window_sec) / self.stride_sec)) + 1)
            starts = np.arange(n_windows, dtype=np.float64) * self.stride_sec
            best_overlap = np.zeros(n_windows, dtype=np.float32)
            best_event_duration = np.zeros(n_windows, dtype=np.float32)
            best_event_idx = np.full(n_windows, -1, dtype=np.int64)
            event_rows = list(ev_df.itertuples(index=False))
            for ev_idx, ev in enumerate(event_rows):
                ev_start = float(ev.start_sec)
                ev_end = float(ev.end_sec)
                lo = max(0, int(np.floor((ev_start - self.window_sec) / self.stride_sec)) - 2)
                hi = min(n_windows - 1, int(np.ceil(ev_end / self.stride_sec)) + 2)
                if hi < lo:
                    continue
                idx = np.arange(lo, hi + 1)
                win_start = starts[idx]
                win_end = win_start + self.window_sec
                overlap = np.maximum(0.0, np.minimum(win_end, ev_end) - np.maximum(win_start, ev_start)).astype(np.float32)
                update = overlap > best_overlap[idx]
                best_overlap[idx[update]] = overlap[update]
                best_event_idx[idx[update]] = ev_idx
                best_event_duration[idx[update]] = max(ev_end - ev_start, 1e-6)
            positive_mask = best_overlap >= self.positive_overlap_ratio * self.window_sec
            if self.positive_event_coverage_ratio is not None:
                event_coverage = np.divide(best_overlap, best_event_duration, out=np.zeros_like(best_overlap), where=best_event_duration > 0)
                positive_mask |= (best_overlap > 0) & (event_coverage >= self.positive_event_coverage_ratio)
            for i, start in enumerate(starts):
                end = float(start + self.window_sec); center = 0.5 * (float(start) + end)
                positive = bool(positive_mask[i])
                ev_idx = int(best_event_idx[i])
                if positive and ev_idx >= 0:
                    best_ev = event_rows[ev_idx]
                    label = self.label_map.get(str(best_ev.label), str(best_ev.label))
                    if label not in self.type_class_to_id:
                        if not self.allow_unknown_positive_labels:
                            raise ValueError(f'Unknown label {label!r}. Configure data.type_class_names or label_map to include it.')
                        cls = self.fallback_positive_class_id
                    else:
                        cls = self.type_class_to_id[label]
                    boundary = np.asarray([float(best_ev.start_sec) - center, float(best_ev.end_sec) - center], dtype=np.float32) / self.window_sec
                    mask = 1.0
                else:
                    cls = -1; boundary = np.zeros(2, dtype=np.float32); mask = 0.0
                if self.fixed_domain_label is not None:
                    domain_label = int(self.fixed_domain_label)
                elif rid in self.record_to_domain:
                    domain_label = int(self.record_to_domain[rid])
                else:
                    domain_label = int(self.patient_to_domain.get(pid, 0))
                record_samples.append(WindowSampleMeta(rid, pid, eeg_path, float(start), float(end), float(center), int(positive), int(cls), boundary, float(mask), domain_label, float(duration_sec)))
            neg_limit = None
            if self.negative_pos_ratio is not None and self.negative_pos_ratio > 0:
                pos_count = sum(1 for s in record_samples if s.det_label == 1)
                if pos_count > 0:
                    neg_limit = int(np.ceil(pos_count * float(self.negative_pos_ratio)))
                elif self.negative_only_record_windows is not None:
                    neg_limit = max(0, int(self.negative_only_record_windows))
            if self.max_negative_windows_per_record is not None and self.max_negative_windows_per_record > 0:
                neg_limit = int(self.max_negative_windows_per_record) if neg_limit is None else min(neg_limit, int(self.max_negative_windows_per_record))
            if neg_limit is not None:
                pos_samples = [s for s in record_samples if s.det_label == 1]
                neg_samples = [s for s in record_samples if s.det_label == 0]
                if neg_limit <= 0:
                    neg_samples = []
                elif len(neg_samples) > neg_limit:
                    idx = np.linspace(0, len(neg_samples) - 1, int(neg_limit), dtype=int)
                    neg_samples = [neg_samples[i] for i in idx]
                record_samples = sorted([*pos_samples, *neg_samples], key=lambda s: s.start_sec)
            samples.extend(record_samples)
        return samples
    @staticmethod
    @lru_cache(maxsize=64)
    def _load_npy(path: str):
        return np.load(path, mmap_mode='r')
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index: int):
        s = self.samples[index]; sr = self.record_id_to_sr[s.record_id]; start_idx = int(round(s.start_sec * sr)); end_idx = int(round(s.end_sec * sr))
        window = np.asarray(self._load_npy(s.eeg_path)[:, start_idx:end_idx], dtype=np.float32)
        if self.window_normalization == 'channel_zscore':
            window = window - window.mean(axis=-1, keepdims=True)
            window = window / (window.std(axis=-1, keepdims=True) + 1e-6)
        elif self.window_normalization == 'robust_channel':
            median = np.median(window, axis=-1, keepdims=True)
            mad = np.median(np.abs(window - median), axis=-1, keepdims=True)
            window = (window - median) / (1.4826 * mad + 1e-6)
        return {'x': torch.from_numpy(window), 'det_label': torch.tensor(s.det_label, dtype=torch.float32), 'cls_label': torch.tensor(s.cls_label, dtype=torch.long), 'boundary': torch.from_numpy(s.boundary), 'boundary_mask': torch.tensor(s.boundary_mask, dtype=torch.float32), 'boundary_scale': torch.tensor(self.window_sec, dtype=torch.float32), 'domain_label': torch.tensor(s.domain_label, dtype=torch.long), 'record_id': s.record_id, 'patient_id': s.patient_id, 'record_duration_sec': torch.tensor(s.record_duration_sec, dtype=torch.float32), 'start_sec': torch.tensor(s.start_sec, dtype=torch.float32), 'end_sec': torch.tensor(s.end_sec, dtype=torch.float32), 'center_sec': torch.tensor(s.center_sec, dtype=torch.float32)}

def build_patient_split(
    records_csv: str,
    fold: int,
    num_folds: int,
    strategy: str = 'stratified',
    eligible_record_ids=None,
    events_csv: str | None = None,
):
    rec = read_manifest_csv(records_csv)
    rec['record_id'] = rec['record_id'].astype(str)
    rec['patient_id'] = rec['patient_id'].astype(str)
    if eligible_record_ids is not None:
        eligible_record_ids = {str(x) for x in eligible_record_ids}
        rec = rec[rec['record_id'].isin(eligible_record_ids)].copy()
    if rec.empty:
        raise ValueError('No eligible records remain before patient-level splitting.')
    events_path = Path(events_csv) if events_csv else Path(records_csv).with_name('events.csv')
    events = read_manifest_csv(events_path) if events_path.exists() else pd.DataFrame(columns=['record_id', 'start_sec', 'end_sec', 'label'])
    if not events.empty:
        events['record_id'] = events['record_id'].astype(str)
        events['label'] = events['label'].astype(str)
        events['duration_sec'] = (events['end_sec'].astype(float) - events['start_sec'].astype(float)).clip(lower=0.0)
        event_patients = rec[['record_id', 'patient_id']].merge(events, on='record_id', how='inner')
    else:
        event_patients = pd.DataFrame(columns=['record_id', 'patient_id', 'label', 'duration_sec'])
    patients = sorted(rec['patient_id'].unique().tolist())
    strategy = str(strategy).lower()
    if strategy == 'round_robin':
        patient_folds = {p: (i % num_folds) for i, p in enumerate(patients)}
    else:
        record_label_cols = sorted({label for value in rec.get('standard_label', pd.Series(['bckg'] * len(rec))).fillna('bckg').astype(str) for label in value.split(';')})
        event_label_cols = sorted(event_patients['label'].dropna().astype(str).unique().tolist())
        patient_rows = []
        for pid, part in rec.groupby('patient_id'):
            labels = part.get('standard_label', pd.Series(['bckg'] * len(part))).fillna('bckg').astype(str)
            label_sets = labels.map(lambda value: set(str(value).split(';')))
            record_label_counts = {label: int(label_sets.map(lambda values: label in values).sum()) for label in record_label_cols}
            ev_part = event_patients[event_patients['patient_id'].astype(str) == str(pid)]
            event_counts = {label: int((ev_part['label'] == label).sum()) for label in event_label_cols}
            event_durations = {label: float(ev_part.loc[ev_part['label'] == label, 'duration_sec'].sum()) for label in event_label_cols}
            hours = float(part.get('duration_sec', pd.Series([0.0] * len(part))).sum() / 3600.0)
            seizure_records = int((labels != 'bckg').sum())
            patient_rows.append({
                'patient_id': str(pid),
                'hours': hours,
                'records': int(len(part)),
                'seizure_records': seizure_records,
                'events': int(len(ev_part)),
                'event_duration': float(ev_part['duration_sec'].sum()) if not ev_part.empty else 0.0,
                'record_label_counts': record_label_counts,
                'event_counts': event_counts,
                'event_durations': event_durations,
            })
        fold_state = [
            {
                'hours': 0.0,
                'records': 0,
                'seizure_records': 0,
                'events': 0,
                'event_duration': 0.0,
                'record_labels': {label: 0 for label in record_label_cols},
                'event_labels': {label: 0 for label in event_label_cols},
                'event_label_durations': {label: 0.0 for label in event_label_cols},
            }
            for _ in range(num_folds)
        ]
        patient_folds = {}
        rows_by_patient = {item['patient_id']: item for item in patient_rows}

        def score(item, fold_idx):
            state = fold_state[fold_idx]
            record_label_pressure = sum(state['record_labels'][label] * item['record_label_counts'][label] for label in record_label_cols)
            event_label_pressure = sum(state['event_labels'][label] * item['event_counts'][label] for label in event_label_cols)
            event_duration_pressure = sum(state['event_label_durations'][label] * item['event_durations'][label] for label in event_label_cols)
            return (
                5.0 * event_label_pressure
                + 0.01 * event_duration_pressure
                + 1.0 * record_label_pressure
                + 0.5 * state['events']
                + 0.1 * state['seizure_records']
                + 0.01 * state['hours']
                + 0.1 * state['records']
            )

        def assign(item, target):
            patient_folds[item['patient_id']] = target
            fold_state[target]['hours'] += item['hours']
            fold_state[target]['records'] += item['records']
            fold_state[target]['seizure_records'] += item['seizure_records']
            fold_state[target]['events'] += item['events']
            fold_state[target]['event_duration'] += item['event_duration']
            for label in record_label_cols:
                fold_state[target]['record_labels'][label] += item['record_label_counts'][label]
            for label in event_label_cols:
                fold_state[target]['event_labels'][label] += item['event_counts'][label]
                fold_state[target]['event_label_durations'][label] += item['event_durations'][label]

        label_patients = {
            label: [item['patient_id'] for item in patient_rows if item['event_counts'][label] > 0]
            for label in event_label_cols
        }
        for label in sorted(event_label_cols, key=lambda value: (len(label_patients[value]), value)):
            if len(label_patients[label]) < num_folds:
                raise ValueError(
                    f'Cannot build {num_folds} class-viable patient folds: label {label!r} '
                    f'has only {len(label_patients[label])} independent patients.'
                )
            covered_folds = {
                patient_folds[pid] for pid in label_patients[label] if pid in patient_folds
            }
            missing_folds = [fold_idx for fold_idx in range(num_folds) if fold_idx not in covered_folds]
            candidates = [rows_by_patient[pid] for pid in label_patients[label] if pid not in patient_folds]
            candidates.sort(key=lambda item: (item['event_counts'][label], item['event_duration'], item['hours']), reverse=True)
            if len(candidates) < len(missing_folds):
                raise ValueError(f'Greedy constrained assignment could not cover every fold for label {label!r}.')
            for target in sorted(missing_folds, key=lambda fold_idx: (fold_state[fold_idx]['hours'], fold_idx)):
                assign(candidates.pop(0), target)

        remaining = [item for item in patient_rows if item['patient_id'] not in patient_folds]
        for item in sorted(remaining, key=lambda x: (x['events'], x['event_duration'], x['seizure_records'], x['hours']), reverse=True):
            target = min(range(num_folds), key=lambda fold_idx: score(item, fold_idx))
            assign(item, target)

        for label in event_label_cols:
            if any(fold_state[fold_idx]['event_labels'][label] == 0 for fold_idx in range(num_folds)):
                raise RuntimeError(f'Internal split error: label {label!r} is absent from at least one fold.')
    val_patients = {p for p, f in patient_folds.items() if f == fold}
    train_records = rec[~rec['patient_id'].isin(val_patients)]['record_id'].astype(str).tolist()
    val_records = rec[rec['patient_id'].isin(val_patients)]['record_id'].astype(str).tolist()
    patient_to_domain = {p: i for i, p in enumerate(patients)}
    return train_records, val_records, patient_to_domain
