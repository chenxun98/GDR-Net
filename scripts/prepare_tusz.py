#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.io_utils import standardize_record
from src.utils.constants import TUSZ_TO_INTERNAL

DEFAULT_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]

MIN_VALID_AGE_YEARS = 0.5
MAX_VALID_AGE_YEARS = 120.0

# TUSZ contains one erroneous female header among 35 longitudinal records
# from a patient whose remaining records consistently report male.
PATIENT_SEX_OVERRIDES = {'aaaaampk': 'M'}

def normalize_sex(value: str):
    text = str(value or '').strip().lower()
    if text in {'m', 'male', '1', '男'}:
        return 'M'
    if text in {'f', 'female', '2', '女'}:
        return 'F'
    return ''

def read_edf_demographics(edf_path: Path):
    try:
        header = edf_path.read_bytes()[:256].decode('latin-1', errors='ignore')
    except OSError:
        return {'age_years': np.nan, 'sex': '', 'age_text': ''}
    patient_field = header[8:88].strip()
    sex = ''
    sex_match = re.search(r'(?<![A-Za-z])([MF])(?=\s)', patient_field, flags=re.IGNORECASE)
    if sex_match:
        sex = normalize_sex(sex_match.group(1))
    age_text = ''
    age_years = np.nan
    age_match = re.search(r'\bAge\s*:\s*([0-9]+(?:\.[0-9]+)?)', patient_field, flags=re.IGNORECASE)
    if age_match:
        age_text = age_match.group(1)
        age_years = float(age_text)
        if age_years < MIN_VALID_AGE_YEARS or age_years > MAX_VALID_AGE_YEARS:
            age_years = np.nan
    return {'age_years': age_years, 'sex': sex, 'age_text': age_text}

def harmonize_patient_sex(records: list[dict]):
    records_by_patient = {}
    for record in records:
        records_by_patient.setdefault(str(record['patient_id']), []).append(record)
    changed = 0
    for patient_id, patient_records in records_by_patient.items():
        observed = [str(record.get('sex', '')).strip() for record in patient_records]
        observed = [value for value in observed if value in {'M', 'F'}]
        if patient_id in PATIENT_SEX_OVERRIDES:
            consensus = PATIENT_SEX_OVERRIDES[patient_id]
        elif observed:
            counts = {value: observed.count(value) for value in {'M', 'F'}}
            if counts['M'] == counts['F'] and counts['M'] > 0:
                raise ValueError(f'Ambiguous patient-level sex for {patient_id}: {counts}')
            consensus = max(counts, key=counts.get)
        else:
            continue
        for record in patient_records:
            if record.get('sex') != consensus:
                record['sex'] = consensus
                changed += 1
    return changed

def parse_args():
    p = argparse.ArgumentParser(description='Prepare TUSZ into standardized manifests.')
    p.add_argument('--tusz_root', default='data/raw/tusz')
    p.add_argument('--output_root', default='./data/tusz_processed')
    p.add_argument('--target_sr', type=float, default=256.0)
    p.add_argument('--channels', nargs='+', default=DEFAULT_CHANNELS)
    p.add_argument('--low_hz', type=float, default=0.5)
    p.add_argument('--high_hz', type=float, default=40.0)
    p.add_argument('--notch_hz', type=float, default=60.0)
    p.add_argument('--skip_existing', action='store_true', help='Reuse existing .npy outputs when present.')
    return p.parse_args()

def _parse_csv_like(path: Path):
    rows = []
    known_labels = {str(key).lower() for key in TUSZ_TO_INTERNAL} | {'bckg', 'back', 'background', 'seiz', 'seizure'}
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            values = [str(x).strip() for x in row]
            lowered = [x.lower() for x in values]
            if lowered[0].startswith('#') or lowered[0].startswith('version'):
                continue
            if any(h in lowered for h in ['start', 'start_time', 'stop', 'stop_time', 'label', 'event', 'channel']):
                continue
            numeric = []
            for idx, value in enumerate(values):
                try:
                    numeric.append((idx, float(value)))
                except ValueError:
                    continue
            if len(numeric) < 2:
                continue
            start = numeric[0][1]
            end = numeric[1][1]
            label = ''
            for value in lowered:
                if value in known_labels:
                    label = value
                    break
            if not label:
                text_values = [value for value in lowered if value and not value.replace('.', '', 1).isdigit()]
                label = text_values[-1] if text_values else ''
            if end <= start:
                continue
            rows.append({'start_sec': start, 'end_sec': end, 'label': label, 'channel': values[0] if values else ''})
    return rows

def safe_record_id(root: Path, edf_path: Path):
    rel = edf_path.relative_to(root).with_suffix('')
    parts = [re.sub(r'[^A-Za-z0-9]+', '_', part).strip('_') for part in rel.parts]
    return '__'.join(part for part in parts if part)

def _parse_tse_like(path: Path):
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('version'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0]); end = float(parts[1])
            except Exception:
                continue
            rows.append({'start_sec': start, 'end_sec': end, 'label': str(parts[2]).strip().lower()})
    return rows

def _deduplicate_events(events: list[dict]):
    out = []
    seen = set()
    for ev in events:
        item = (
            round(float(ev['start_sec']), 6),
            round(float(ev['end_sec']), 6),
            str(ev['label']),
        )
        if item in seen:
            continue
        seen.add(item)
        out.append(ev)
    return out

def _events_from_csv_bi(edf_path: Path):
    binary_path = edf_path.with_suffix('.csv_bi')
    if not binary_path.exists():
        return None
    typed_path = edf_path.with_suffix('.csv')
    detection_events = []
    for ev in _parse_csv_like(binary_path):
        binary_tag = str(ev['label']).lower()
        if binary_tag in {'bckg', 'back', 'background'}:
            continue
        detection_events.append({
            'start_sec': float(ev['start_sec']),
            'end_sec': float(ev['end_sec']),
            'label': 'seizure',
            'source_label': binary_tag,
        })
    typed_events = []
    if typed_path.exists():
        for ev in _parse_csv_like(typed_path):
            source_tag = str(ev['label']).lower()
            if source_tag in TUSZ_TO_INTERNAL:
                typed_events.append({
                    'start_sec': float(ev['start_sec']),
                    'end_sec': float(ev['end_sec']),
                    'label': TUSZ_TO_INTERNAL[source_tag],
                    'source_label': source_tag,
                })
    return _deduplicate_events(detection_events), _deduplicate_events(typed_events)

def read_tusz_event_sets(edf_path: Path):
    csv_bi_events = _events_from_csv_bi(edf_path)
    if csv_bi_events is not None:
        return csv_bi_events

    for ann in [edf_path.with_suffix('.tse'), edf_path.with_suffix('.tse_bi'), edf_path.with_suffix('.rec'), edf_path.with_suffix('.csv')]:
        if ann.exists():
            events = _parse_csv_like(ann) if ann.suffix in {'.csv', '.rec'} else _parse_tse_like(ann)
            typed_events = []
            for ev in events:
                tag = str(ev['label']).lower()
                if tag in TUSZ_TO_INTERNAL:
                    typed_events.append({
                        'start_sec': float(ev['start_sec']),
                        'end_sec': float(ev['end_sec']),
                        'label': TUSZ_TO_INTERNAL[tag],
                        'source_label': tag,
                    })
            typed_events = _deduplicate_events(typed_events)
            detection_events = _deduplicate_events([
                {**ev, 'label': 'seizure'} for ev in typed_events
            ])
            return detection_events, typed_events
    return [], []

def apply_record_event_labels(records: list[dict], detection_events: list[dict], typed_events: list[dict]):
    labels_by_record = {}
    source_labels_by_record = {}
    detection_records = {str(ev['record_id']) for ev in detection_events}
    for ev in typed_events:
        rid = str(ev['record_id'])
        labels_by_record.setdefault(rid, set()).add(str(ev['label']))
        source_labels_by_record.setdefault(rid, set()).add(str(ev['source_label']))
    for rec in records:
        rid = str(rec['record_id'])
        labels = sorted(labels_by_record.get(rid, set()))
        source_labels = sorted(source_labels_by_record.get(rid, set()))
        rec['standard_label'] = ';'.join(labels) if labels else ('seizure' if rid in detection_records else 'bckg')
        rec['source_label'] = ';'.join(source_labels) if source_labels else ('seizure' if rid in detection_records else 'bckg')

def infer_patient_id_from_path(path: Path):
    for token in reversed(path.parts):
        if token.isdigit() and len(token) >= 6:
            return token
    return path.stem.split('_')[0]

def main():
    args = parse_args()
    root = Path(args.tusz_root)
    out = Path(args.output_root)
    sig_dir = out / 'signals'
    sig_dir.mkdir(parents=True, exist_ok=True)
    records, detection_events, typed_events, failures = [], [], [], []
    edf_paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.edf')
    for edf_path in tqdm(edf_paths, desc='Preparing TUSZ'):
        rid = safe_record_id(root, edf_path); pid = infer_patient_id_from_path(edf_path)
        npy_path = sig_dir / f'{rid}.npy'
        try:
            demographics = read_edf_demographics(edf_path)
            det_evs, type_evs = read_tusz_event_sets(edf_path)
            if args.skip_existing and npy_path.exists():
                signal = np.load(npy_path, mmap_mode='r')
                sr = float(args.target_sr)
                n_channels, n_samples = int(signal.shape[0]), int(signal.shape[1])
            else:
                signal, sr, ch_names = standardize_record(str(edf_path), list(args.channels), args.target_sr, args.low_hz, args.high_hz, args.notch_hz)
                np.save(npy_path, signal.astype(np.float32))
                n_channels, n_samples = int(signal.shape[0]), int(signal.shape[1])
            records.append({
                'record_id': rid,
                'patient_id': pid,
                'eeg_path': str(npy_path.resolve()),
                'source_path': str(edf_path.resolve()),
                'dataset': 'tusz',
                **demographics,
                'standard_label': '',
                'source_label': '',
                'sampling_rate': sr,
                'n_channels': n_channels,
                'n_samples': n_samples,
                'duration_sec': float(n_samples / sr),
            })
            for ev in det_evs:
                detection_events.append({
                    'record_id': rid,
                    'start_sec': ev['start_sec'],
                    'end_sec': ev['end_sec'],
                    'label': ev['label'],
                    'source_label': ev['source_label'],
                })
            for ev in type_evs:
                typed_events.append({
                    'record_id': rid,
                    'start_sec': ev['start_sec'],
                    'end_sec': ev['end_sec'],
                    'label': ev['label'],
                    'source_label': ev['source_label'],
                })
        except Exception as exc:
            failures.append({
                'record_id': rid,
                'source_path': str(edf_path.resolve()),
                'error_type': type(exc).__name__,
                'error': str(exc),
            })
    harmonized_sex_records = harmonize_patient_sex(records)
    apply_record_event_labels(records, detection_events, typed_events)
    pd.DataFrame(
        records,
        columns=[
            'record_id',
            'patient_id',
            'eeg_path',
            'source_path',
            'dataset',
            'age_years',
            'sex',
            'age_text',
            'standard_label',
            'source_label',
            'sampling_rate',
            'n_channels',
            'n_samples',
            'duration_sec',
        ],
    ).to_csv(out / 'records.csv', index=False)
    event_columns = ['record_id', 'start_sec', 'end_sec', 'label', 'source_label']
    pd.DataFrame(detection_events, columns=event_columns).to_csv(out / 'events.csv', index=False)
    pd.DataFrame(detection_events, columns=event_columns).to_csv(out / 'detection_events.csv', index=False)
    pd.DataFrame(typed_events, columns=event_columns).to_csv(out / 'typed_events.csv', index=False)
    pd.DataFrame(failures, columns=['record_id', 'source_path', 'error_type', 'error']).to_csv(out / 'failures.csv', index=False)
    print(
        f'Saved {len(records)} TUSZ records with {len(detection_events)} detection events '
        f'and {len(typed_events)} typed events to {out}. failed={len(failures)} '
        f'patient_sex_harmonized_records={harmonized_sex_records}'
    )

if __name__ == '__main__':
    main()
