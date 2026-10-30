#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.io_utils import standardize_record

EVENT_LABEL = 'seizure'
DEFAULT_CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']

# CHB21 is a longitudinal follow-up recording of CHB01, not a new patient.
PATIENT_ALIASES = {'chb21': 'chb01'}

def normalize_sex(value: str):
    text = str(value or '').strip().lower()
    if text in {'m', 'male', '男'}:
        return 'M'
    if text in {'f', 'female', '女'}:
        return 'F'
    return ''

def load_subject_info(root: Path):
    path = root / 'SUBJECT-INFO'
    if not path.exists():
        return {}
    rows = {}
    for raw_line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith('case'):
            continue
        parts = re.split(r'\s+', line)
        if len(parts) < 3:
            continue
        patient_id = parts[0].lower()
        sex = normalize_sex(parts[1])
        age_text = ' '.join(parts[2:]).strip()
        try:
            age_years = float(age_text)
        except ValueError:
            age_years = np.nan
        rows[patient_id] = {'age_years': age_years, 'sex': sex, 'age_text': age_text}
    return rows

def parse_args():
    p = argparse.ArgumentParser(description='Prepare CHB-MIT EDF files into the project standardized format.')
    p.add_argument('--chbmit_root', default='data/raw/chbmit')
    p.add_argument('--output_root', default='./data/chbmit_processed')
    p.add_argument('--target_sr', type=float, default=256.0)
    p.add_argument('--channels', nargs='+', default=DEFAULT_CHANNELS)
    p.add_argument('--low_hz', type=float, default=0.5)
    p.add_argument('--high_hz', type=float, default=40.0)
    p.add_argument('--notch_hz', type=float, default=60.0)
    p.add_argument('--only_records_with_seizures', action='store_true')
    p.add_argument('--max_records', type=int, default=0, help='Optional global cap for quick smoke tests.')
    p.add_argument('--max_records_per_patient', type=int, default=0, help='Optional per-patient cap for quick smoke tests.')
    p.add_argument('--skip_failed', action='store_true', default=True, help='Skip EDFs that cannot be read or do not contain required channels.')
    return p.parse_args()

def read_chbmit_events(summary_path: Path):
    events_by_file = {}
    if not summary_path.exists():
        return events_by_file
    current_file = None
    pending_start = None
    for raw_line in summary_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw_line.strip()
        m = re.match(r'File Name:\s*(\S+)', line, flags=re.IGNORECASE)
        if m:
            current_file = m.group(1)
            events_by_file.setdefault(current_file, [])
            pending_start = None
            continue
        if current_file is None:
            continue
        m = re.match(r'Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)\s*seconds', line, flags=re.IGNORECASE)
        if m:
            pending_start = float(m.group(1))
            continue
        m = re.match(r'Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)\s*seconds', line, flags=re.IGNORECASE)
        if m and pending_start is not None:
            end = float(m.group(1))
            if end > pending_start:
                events_by_file[current_file].append({'start_sec': pending_start, 'end_sec': end, 'label': EVENT_LABEL})
            pending_start = None
    return events_by_file

def iter_edf_paths(root: Path):
    records_file = root / 'RECORDS'
    if records_file.exists():
        for rel in records_file.read_text(encoding='utf-8', errors='ignore').splitlines():
            rel = rel.strip()
            if rel and rel.lower().endswith('.edf'):
                path = root / rel
                if path.exists():
                    yield path
        return
    yield from sorted(root.rglob('*.edf'))

def summarize_raw_edfs(root: Path):
    records_file = root / 'RECORDS'
    if records_file.exists():
        listed = [rel.strip() for rel in records_file.read_text(encoding='utf-8', errors='ignore').splitlines() if rel.strip().lower().endswith('.edf')]
        existing = [root / rel for rel in listed if (root / rel).exists()]
        return {'records_listed': len(listed), 'edf_existing': len(existing), 'edf_missing': len(listed) - len(existing)}
    existing = list(root.rglob('*.edf'))
    return {'records_listed': len(existing), 'edf_existing': len(existing), 'edf_missing': 0}

def records_with_seizures(root: Path):
    path = root / 'RECORDS-WITH-SEIZURES'
    if not path.exists():
        return set()
    return {
        str((root / rel.strip()).resolve())
        for rel in path.read_text(encoding='utf-8', errors='ignore').splitlines()
        if rel.strip()
    }

def patient_id_from_path(edf_path: Path):
    for part in reversed(edf_path.parts):
        if re.fullmatch(r'chb\d+', part.lower()):
            return part.lower()
    return edf_path.stem.split('_')[0].lower()

def canonical_patient_id(source_patient_id: str):
    patient_id = str(source_patient_id).lower()
    return PATIENT_ALIASES.get(patient_id, patient_id)

def apply_record_event_labels(records: list[dict], events: list[dict]):
    labels_by_record = {}
    for ev in events:
        labels_by_record.setdefault(str(ev['record_id']), set()).add(str(ev['label']))
    for rec in records:
        labels = sorted(labels_by_record.get(str(rec['record_id']), set()))
        label_text = ';'.join(labels) if labels else 'bckg'
        rec['standard_label'] = label_text
        rec['source_label'] = label_text

def main():
    args = parse_args()
    root = Path(args.chbmit_root)
    out = Path(args.output_root)
    sig_dir = out / 'signals'
    sig_dir.mkdir(parents=True, exist_ok=True)

    events_cache = {}
    subject_info = load_subject_info(root)
    records, events = [], []
    per_patient_counts = {}
    raw_summary = summarize_raw_edfs(root)
    print(
        f"CHB-MIT raw summary: records_listed={raw_summary['records_listed']} "
        f"edf_existing={raw_summary['edf_existing']} edf_missing={raw_summary['edf_missing']}"
    )
    if raw_summary['edf_missing'] > 0:
        print('WARNING: RECORDS lists EDF files that are not present on disk. The processed dataset will only include existing EDF files.')
    edf_paths = list(iter_edf_paths(root))
    wanted_with_seizures = records_with_seizures(root)
    if args.only_records_with_seizures:
        wanted = wanted_with_seizures
        edf_paths = [p for p in edf_paths if str(p.resolve()) in wanted]

    listed_seizure_records_without_events = []
    failed_records = []
    for edf_path in tqdm(edf_paths, desc='Preparing CHB-MIT'):
        source_pid = patient_id_from_path(edf_path)
        pid = canonical_patient_id(source_pid)
        if args.max_records_per_patient > 0 and per_patient_counts.get(pid, 0) >= args.max_records_per_patient:
            continue
        if args.max_records > 0 and len(records) >= args.max_records:
            break

        summary_path = edf_path.parent / f'{source_pid}-summary.txt'
        if summary_path not in events_cache:
            events_cache[summary_path] = read_chbmit_events(summary_path)
        evs = events_cache[summary_path].get(edf_path.name, [])
        if str(edf_path.resolve()) in wanted_with_seizures and not evs:
            listed_seizure_records_without_events.append(str(edf_path.relative_to(root)))

        try:
            signal, sr, ch_names = standardize_record(str(edf_path), list(args.channels), args.target_sr, args.low_hz, args.high_hz, args.notch_hz)
        except Exception as exc:
            if args.skip_failed:
                failed_records.append((str(edf_path.relative_to(root)), type(exc).__name__, str(exc)))
                continue
            raise

        rid = edf_path.stem
        npy_path = sig_dir / f'{rid}.npy'
        np.save(npy_path, signal.astype(np.float32))
        records.append({
            'record_id': rid,
            'patient_id': pid,
            'eeg_path': str(npy_path.resolve()),
            'dataset': 'chbmit',
            # Keep visit-specific demographics while sharing the canonical
            # patient ID used for patient-disjoint splitting.
            **subject_info.get(source_pid, {'age_years': np.nan, 'sex': '', 'age_text': ''}),
            'standard_label': '',
            'source_label': '',
            'sampling_rate': sr,
            'n_channels': signal.shape[0],
            'n_samples': signal.shape[1],
            'duration_sec': float(signal.shape[1] / sr),
        })
        for ev in evs:
            events.append({'record_id': rid, **ev})
        per_patient_counts[pid] = per_patient_counts.get(pid, 0) + 1

    apply_record_event_labels(records, events)
    pd.DataFrame(
        records,
        columns=[
            'record_id',
            'patient_id',
            'eeg_path',
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
    pd.DataFrame(events, columns=['record_id', 'start_sec', 'end_sec', 'label']).to_csv(out / 'events.csv', index=False)
    patients = len({r['patient_id'] for r in records})
    seizure_records = len({e['record_id'] for e in events})
    print(f'Saved {len(records)} CHB-MIT records from {patients} patients with {len(events)} seizure events in {seizure_records} records to {out}')
    if listed_seizure_records_without_events:
        preview = ', '.join(listed_seizure_records_without_events[:10])
        print(f'WARNING: {len(listed_seizure_records_without_events)} RECORDS-WITH-SEIZURES files had no parsed summary events: {preview}')
    if failed_records:
        preview = '; '.join(f'{path} [{err}]' for path, err, _ in failed_records[:10])
        print(f'WARNING: skipped {len(failed_records)} EDF files because required channels/read failed: {preview}')
    if len(records) < 100:
        print('WARNING: processed dataset is small. If this was not intentional, rerun without --only_records_with_seizures, --max_records, or --max_records_per_patient.')

if __name__ == '__main__':
    main()
