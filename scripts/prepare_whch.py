#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.channel_map import build_montage_signal, canonicalize_channel_name
from src.data.io_utils import bandpass_and_notch, resample_signal

mne.set_log_level("ERROR")

WHCH_ANNOTATION_PARSER_VERSION = "hospital-marker-latest-compatible-start"

DEFAULT_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

SOURCE_DATASETS = {
    "ABSZ": {"root_name": "Absence", "source_label": "GP"},
    "TNSZ": {"root_name": "TP", "source_label": "TP"},
    "TCSZ": {"root_name": "TP_GP", "source_label": "TP-GP"},
}

GP_MARKERS = {"GP", "GPA", "GPJ"}
TP_MARKERS = {"TP"}
END_MARKERS = {"END"}
EVENT_MARKERS = GP_MARKERS | TP_MARKERS | END_MARKERS


def parse_args():
    p = argparse.ArgumentParser(description="Prepare WHCH into per-record .npy files plus records.csv/events.csv.")
    p.add_argument("--whch_root", default="data/raw/whch", help="Root folder containing Absence, TP and TP_GP.")
    p.add_argument("--output_root", default="./data/whch_processed")
    p.add_argument("--target_sr", type=float, default=256.0)
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    p.add_argument("--low_hz", type=float, default=0.5)
    p.add_argument("--high_hz", type=float, default=40.0)
    p.add_argument("--notch_hz", type=float, default=50.0)
    p.add_argument("--max_records", type=int, default=0)
    p.add_argument("--max_records_per_class", type=int, default=0)
    p.add_argument("--patient_map", default="")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def norm_desc(desc) -> str:
    return str(desc).strip().upper()


def patient_id_from_record(record_code: str) -> str:
    match = re.match(r"^([A-Za-z]+\d{4})", record_code)
    if match:
        return match.group(1).upper()
    return record_code[:6].upper()


def clean_patient_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip("\x00\r\n\t "))


def normalize_sex(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "男"}:
        return "M"
    if text in {"f", "female", "女"}:
        return "F"
    return ""


def parse_age_years(age_text: str) -> float:
    text = str(age_text or "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"(\d+)\s*[Yy]\s*(\d+)?\s*[Mm]?", text)
    if match:
        years = int(match.group(1))
        months = int(match.group(2) or 0)
        return years + months / 12.0
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[Mm]", text)
    if match:
        return float(match.group(1)) / 12.0
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2:
        return int(numbers[0]) + int(numbers[1]) / 12.0
    if len(numbers) == 1:
        return float(numbers[0])
    return np.nan


def read_pnt_metadata(eeg_path: Path) -> dict[str, str]:
    pnt_path = eeg_path.with_suffix(".PNT")
    if not pnt_path.exists():
        pnt_path = eeg_path.with_suffix(".pnt")
    if not pnt_path.exists():
        return {}
    text = pnt_path.read_bytes().decode("gb18030", errors="ignore")
    fields = {}
    parts = [clean_patient_value(part) for part in text.split("\x00")]
    parts = [part for part in parts if part]
    field_names = [
        "ID", "EEG No.", "Name", "Sex", "Date of Birth", "Age", "Refer Dept",
        "Physician", "In/Outpatient", "Height", "Weight", "Patient Condition",
    ]
    for key, out_key in [
        ("Name", "patient_name"),
        ("ID", "hospital_patient_id"),
        ("EEG No.", "eeg_number"),
        ("Sex", "sex_text"),
        ("Age", "age_text"),
    ]:
        if key in parts:
            idx = parts.index(key)
            if idx + 1 < len(parts) and parts[idx + 1] not in field_names:
                fields[out_key] = parts[idx + 1]
    fields["sex"] = normalize_sex(fields.get("sex_text", ""))
    fields["age_years"] = parse_age_years(fields.get("age_text", ""))
    return fields


def stable_patient_hash(patient_key: str) -> str:
    digest = hashlib.sha1(patient_key.encode("utf-8")).hexdigest()[:12]
    return f"P{digest}"


def load_patient_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise ValueError("--patient_map must contain a patient_id column.")
    lookup = {}
    for row in df.itertuples(index=False):
        values = row._asdict()
        patient_id = str(values["patient_id"])
        for key in ["record_id", "record_code", "source_path"]:
            value = values.get(key)
            if value is not None and not pd.isna(value):
                lookup[f"{key}:{str(value).strip().upper()}"] = patient_id
    return lookup


def resolve_patient_id(patient_lookup: dict[str, str], record_id: str, record_code: str, eeg_path: Path, pnt_meta: dict[str, str]):
    candidates = [
        f"record_id:{record_id.upper()}",
        f"record_code:{record_code.upper()}",
        f"source_path:{str(eeg_path.resolve()).upper()}",
    ]
    for candidate in candidates:
        if candidate in patient_lookup:
            return patient_lookup[candidate], "patient_map"
    if pnt_meta.get("hospital_patient_id"):
        return stable_patient_hash(pnt_meta["hospital_patient_id"]), "hospital_patient_id"
    if pnt_meta.get("patient_name"):
        return stable_patient_hash(pnt_meta["patient_name"]), "patient_name"
    return stable_patient_hash(f"UNKNOWN_{patient_id_from_record(record_code)}"), "record_code_fallback"


def read_raw_header(eeg_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mne.io.read_raw_nihon(str(eeg_path), preload=False, verbose=False)


def load_raw_data(eeg_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mne.io.read_raw_nihon(str(eeg_path), preload=True, verbose=False)


def marker_events(raw) -> list[tuple[float, str]]:
    events = []
    for ann in raw.annotations:
        desc = norm_desc(ann["description"])
        if desc in EVENT_MARKERS:
            events.append((float(ann["onset"]), desc))
    return sorted(events, key=lambda item: item[0])


def pair_start_to_next_end(events: list[tuple[float, str]], start_markers: set[str], qc_rows: list[dict] | None = None):
    intervals = []
    active_start = None
    active_marker = None
    for onset, marker in events:
        if marker in start_markers:
            if active_start is None:
                active_start = onset
                active_marker = marker
            else:
                if qc_rows is not None:
                    qc_rows.append({
                        "issue": "RepeatedStartBeforeEnd",
                        "onset_sec": onset,
                        "marker": marker,
                        "active_start_sec": active_start,
                        "active_marker": active_marker,
                    })
                # A repeated start invalidates the unfinished candidate. Pair END
                # with the most recent compatible marker instead of spanning the
                # intervening background recording.
                active_start = onset
                active_marker = marker
        elif marker in END_MARKERS and active_start is not None and active_start < onset:
            intervals.append((active_start, onset))
            active_start = None
            active_marker = None
        elif marker in END_MARKERS and qc_rows is not None:
            qc_rows.append({"issue": "OrphanEnd", "onset_sec": onset, "marker": marker})
    if active_start is not None and qc_rows is not None:
        qc_rows.append({
            "issue": "UnclosedStart",
            "onset_sec": active_start,
            "marker": active_marker,
        })
    return intervals


def parse_tcsz_with_qc(events: list[tuple[float, str]]):
    """Parse hospital marker patterns, including optional pre-tonic GP data."""
    intervals, phases, qc_rows = [], [], []
    state = "waiting"
    event_start = None
    pre_tonic_start = None
    tonic_start = None
    post_tonic_start = None
    for onset, marker in events:
        if state == "waiting":
            if marker in TP_MARKERS:
                event_start = onset
                pre_tonic_start = None
                tonic_start = onset
                state = "tonic"
            elif marker in GP_MARKERS:
                event_start = onset
                pre_tonic_start = onset
                state = "pre_tonic_gp"
            elif marker in END_MARKERS:
                qc_rows.append({"issue": "OrphanEnd", "onset_sec": onset, "marker": marker})
            continue

        if state == "pre_tonic_gp":
            if marker in TP_MARKERS:
                tonic_start = onset
                state = "tonic"
            elif marker in GP_MARKERS:
                qc_rows.append({
                    "issue": "RepeatedPreTonicGP", "onset_sec": onset, "marker": marker,
                    "active_start_sec": pre_tonic_start, "active_marker": "GP",
                })
                event_start = onset
                pre_tonic_start = onset
            elif marker in END_MARKERS:
                if event_start is not None and onset > event_start:
                    intervals.append({"start_sec": event_start, "end_sec": onset, "label": "ABSZ", "phase_label": "ABSZ"})
                event_start = None
                pre_tonic_start = None
                state = "waiting"
            continue

        if state == "tonic":
            if marker in TP_MARKERS:
                qc_rows.append({
                    "issue": "RepeatedTonicStart", "onset_sec": onset, "marker": marker,
                    "active_start_sec": tonic_start, "active_marker": "TP",
                })
                event_start = onset
                pre_tonic_start = None
                tonic_start = onset
            elif marker in GP_MARKERS:
                post_tonic_start = onset
                state = "post_tonic_gp"
            elif marker in END_MARKERS:
                if tonic_start is not None and onset > tonic_start:
                    intervals.append({"start_sec": tonic_start, "end_sec": onset, "label": "TNSZ", "phase_label": "TNSZ"})
                event_start = None
                pre_tonic_start = None
                tonic_start = None
                state = "waiting"
            continue

        if state == "post_tonic_gp":
            if marker in GP_MARKERS:
                qc_rows.append({
                    "issue": "RepeatedPhaseTransition", "onset_sec": onset, "marker": marker,
                    "active_start_sec": post_tonic_start, "active_marker": "GP",
                })
                post_tonic_start = onset
            elif marker in TP_MARKERS:
                qc_rows.append({
                    "issue": "TonicRestartBeforeEnd", "onset_sec": onset, "marker": marker,
                    "active_start_sec": tonic_start, "active_marker": "TP",
                })
                event_start = onset
                pre_tonic_start = None
                tonic_start = onset
                post_tonic_start = None
                state = "tonic"
            elif marker in END_MARKERS:
                if event_start is not None and tonic_start is not None and post_tonic_start is not None and event_start <= tonic_start < post_tonic_start < onset:
                    intervals.append({"start_sec": event_start, "end_sec": onset, "label": "TCSZ", "phase_label": "TCSZ"})
                    if pre_tonic_start is not None and pre_tonic_start < tonic_start:
                        phases.append({"start_sec": pre_tonic_start, "end_sec": tonic_start, "label": "TCSZ", "phase_label": "GP_SEGMENT"})
                    phases.extend([
                        {"start_sec": tonic_start, "end_sec": post_tonic_start, "label": "TCSZ", "phase_label": "TP_SEGMENT"},
                        {"start_sec": post_tonic_start, "end_sec": onset, "label": "TCSZ", "phase_label": "GP_SEGMENT"},
                    ])
                event_start = None
                pre_tonic_start = None
                tonic_start = None
                post_tonic_start = None
                state = "waiting"

    if state != "waiting":
        qc_rows.append({
            "issue": "UnclosedMarkerSequence", "onset_sec": event_start,
            "marker": "TP" if tonic_start is not None else "GP",
            "active_start_sec": event_start,
            "active_marker": "TP" if tonic_start is not None else "GP",
        })
    return intervals, qc_rows, phases


def parse_gtcs_intervals(events: list[tuple[float, str]]):
    return parse_tcsz_with_qc(events)[0]


def parse_gtcs_phases(events: list[tuple[float, str]]):
    return parse_tcsz_with_qc(events)[2]


def parse_intervals(standard_label: str, raw):
    intervals, _, _ = parse_intervals_with_qc(standard_label, raw)
    return intervals


def parse_intervals_with_qc(standard_label: str, raw):
    events = marker_events(raw)
    qc_rows = []
    if standard_label == "ABSZ":
        intervals = [{"start_sec": s, "end_sec": e, "label": "ABSZ", "phase_label": "ABSZ"} for s, e in pair_start_to_next_end(events, GP_MARKERS, qc_rows)]
        return intervals, qc_rows, []
    if standard_label == "TNSZ":
        intervals = [{"start_sec": s, "end_sec": e, "label": "TNSZ", "phase_label": "TNSZ"} for s, e in pair_start_to_next_end(events, TP_MARKERS, qc_rows)]
        return intervals, qc_rows, []
    return parse_tcsz_with_qc(events)


def standardize_raw(raw, target_channels, target_sampling_rate, low_hz, high_hz, notch_hz):
    signal = raw.get_data().astype(np.float32)
    signal_channels = [canonicalize_channel_name(ch) for ch in raw.ch_names]
    target_channels = [canonicalize_channel_name(ch) for ch in target_channels]
    signal = build_montage_signal(signal, signal_channels, target_channels)
    original_sr = float(raw.info["sfreq"])
    anti_alias_high_hz = min(float(high_hz), 0.45 * float(target_sampling_rate))
    signal = bandpass_and_notch(signal, original_sr, low_hz, anti_alias_high_hz, notch_hz)
    signal = resample_signal(signal, original_sr, target_sampling_rate)
    return signal.astype(np.float32), target_channels


def iter_raw_records(root: Path):
    for standard_label, spec in SOURCE_DATASETS.items():
        class_root = root / spec["root_name"]
        for eeg_path in sorted(p for p in class_root.rglob("*") if p.is_file() and p.suffix.lower() == ".eeg"):
            yield standard_label, eeg_path

def apply_record_event_labels(records: list[dict], events: list[dict]):
    labels_by_record = {}
    for ev in events:
        labels_by_record.setdefault(str(ev["record_id"]), set()).add(str(ev["label"]))
    for rec in records:
        labels = sorted(labels_by_record.get(str(rec["record_id"]), set()))
        rec["standard_label"] = ";".join(labels) if labels else "bckg"


def build_preprocessing_summary(records, events, phase_events, failures, marker_qc_rows, raw_input_count):
    record_df = pd.DataFrame(records)
    event_df = pd.DataFrame(events)
    failure_df = pd.DataFrame(failures)
    marker_df = pd.DataFrame(marker_qc_rows)
    record_count = int(len(record_df))
    duration_summary = {}
    if not event_df.empty and {"label", "duration_sec"}.issubset(event_df.columns):
        for label, group in event_df.groupby("label"):
            durations = pd.to_numeric(group["duration_sec"], errors="coerce").dropna()
            duration_summary[str(label)] = {
                "min": float(durations.min()) if not durations.empty else None,
                "median": float(durations.median()) if not durations.empty else None,
                "max": float(durations.max()) if not durations.empty else None,
                "under_2_sec": int((durations < 2.0).sum()),
                "over_300_sec": int((durations > 300.0).sum()),
            }
    summary = {
        "annotation_parser_version": WHCH_ANNOTATION_PARSER_VERSION,
        "raw_files": int(raw_input_count),
        "processed_records": record_count,
        "patients": int(record_df["patient_id"].nunique()) if not record_df.empty else 0,
        "recording_hours": float(record_df["duration_sec"].sum() / 3600.0) if not record_df.empty else 0.0,
        "events": int(len(event_df)),
        "events_by_label": {str(k): int(v) for k, v in event_df.get("label", pd.Series(dtype=str)).value_counts().items()},
        "event_duration_seconds_by_label": duration_summary,
        "phase_segments": int(len(phase_events)),
        "failures": int(len(failure_df)),
        "failures_by_type": {str(k): int(v) for k, v in failure_df.get("error_type", pd.Series(dtype=str)).value_counts().items()},
        "marker_qc_occurrences": int(len(marker_df)),
        "marker_qc_records": int(marker_df["record_id"].nunique()) if "record_id" in marker_df else 0,
        "marker_qc_by_issue": {str(k): int(v) for k, v in marker_df.get("issue", pd.Series(dtype=str)).value_counts().items()},
    }
    warnings_out = []
    if raw_input_count and len(failure_df) / raw_input_count > 0.05:
        warnings_out.append("More than 5% of discovered raw files failed preprocessing.")
    if record_count and summary["marker_qc_records"] / record_count > 0.05:
        warnings_out.append("More than 5% of processed records contain malformed marker sequences.")
    if any(summary["events_by_label"].get(label, 0) == 0 for label in ("ABSZ", "TNSZ", "TCSZ")):
        warnings_out.append("At least one required seizure label has zero parsed events.")
    if any(item["over_300_sec"] > 0 for item in duration_summary.values()):
        warnings_out.append("At least one parsed event exceeds 300 seconds; inspect marker QC before training.")
    summary["warnings"] = warnings_out
    return summary


def main():
    args = parse_args()
    root = Path(args.whch_root)
    out = Path(args.output_root)
    sig_dir = out / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)

    channels = [canonicalize_channel_name(ch) for ch in args.channels]
    patient_lookup = load_patient_map(args.patient_map)
    records, events, phase_events, failures, identity_rows, marker_qc_rows = [], [], [], [], [], []

    raw_inputs = list(iter_raw_records(root))
    if not raw_inputs:
        expected = ", ".join(str(root / spec["root_name"]) for spec in SOURCE_DATASETS.values())
        raise FileNotFoundError(
            f"No WHCH .eeg/.EEG files found under expected class folders: {expected}. "
            "Check --whch_root and the Absence/TP/TP_GP directory names."
        )
    if args.max_records_per_class > 0:
        counts, selected = {label: 0 for label in SOURCE_DATASETS}, []
        for label, path in raw_inputs:
            if counts[label] < args.max_records_per_class:
                selected.append((label, path))
                counts[label] += 1
        raw_inputs = selected
    if args.max_records > 0:
        raw_inputs = raw_inputs[: args.max_records]

    for standard_label, eeg_path in tqdm(raw_inputs, desc="Preparing WHCH"):
        record_code = eeg_path.stem.upper()
        record_id = f"{standard_label}_{record_code}"
        npy_path = sig_dir / f"{record_id}.npy"
        try:
            pnt_meta = read_pnt_metadata(eeg_path)
            patient_id, identity_source = resolve_patient_id(patient_lookup, record_id, record_code, eeg_path, pnt_meta)
            raw_header = read_raw_header(eeg_path)
            intervals, marker_qc, phases = parse_intervals_with_qc(standard_label, raw_header)
            for row in marker_qc:
                marker_qc_rows.append({"record_id": record_id, "standard_label": standard_label, **row})
            has_incomplete_markers = bool(marker_events(raw_header)) and not intervals
            if has_incomplete_markers:
                failures.append(
                    {
                        "record_id": record_id,
                        "standard_label": standard_label,
                        "source_path": str(eeg_path.resolve()),
                        "error_type": "IncompleteMarkers",
                        "error": "Seizure markers were present but no valid start/end interval could be parsed.",
                    }
                )
                continue

            if args.dry_run:
                n_samples = int(round(float(raw_header.n_times / raw_header.info["sfreq"]) * args.target_sr))
                n_channels = len(channels)
            elif args.skip_existing and npy_path.exists():
                signal = np.load(npy_path, mmap_mode="r")
                n_channels, n_samples = int(signal.shape[0]), int(signal.shape[1])
            else:
                raw = load_raw_data(eeg_path)
                signal, _ = standardize_raw(raw, channels, args.target_sr, args.low_hz, args.high_hz, args.notch_hz)
                np.save(npy_path, signal)
                n_channels, n_samples = int(signal.shape[0]), int(signal.shape[1])

            records.append(
                {
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "eeg_path": "" if args.dry_run else str(npy_path.resolve()),
                    "dataset": "whch",
                    "age_years": pnt_meta.get("age_years", np.nan),
                    "age_months": round(float(pnt_meta["age_years"]) * 12.0, 2) if np.isfinite(pnt_meta.get("age_years", np.nan)) else np.nan,
                    "sex": pnt_meta.get("sex", ""),
                    "age_text": pnt_meta.get("age_text", ""),
                    "standard_label": "",
                    "source_label": SOURCE_DATASETS[standard_label]["source_label"],
                    "sampling_rate": float(args.target_sr),
                    "n_channels": n_channels,
                    "n_samples": n_samples,
                    "duration_sec": float(n_samples / args.target_sr),
                }
            )
            identity_rows.append({
                "record_id": record_id,
                "patient_id": patient_id,
                "identity_source": identity_source,
                "hospital_id_hash": stable_patient_hash(pnt_meta["hospital_patient_id"]) if pnt_meta.get("hospital_patient_id") else "",
                "name_hash": stable_patient_hash(pnt_meta["patient_name"]) if pnt_meta.get("patient_name") else "",
            })
            for ev in intervals:
                duration_sec = float(ev["end_sec"] - ev["start_sec"])
                events.append({
                    "record_id": record_id,
                    **ev,
                    "duration_sec": duration_sec,
                })
            for phase in phases:
                phase_events.append({"record_id": record_id, **phase, "duration_sec": float(phase["end_sec"] - phase["start_sec"])})
        except Exception as exc:
            failures.append(
                {
                    "record_id": record_id,
                    "standard_label": standard_label,
                    "source_path": str(eeg_path.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    record_cols = ["record_id", "patient_id", "eeg_path", "dataset", "age_years", "age_months", "sex", "age_text", "standard_label", "source_label", "sampling_rate", "n_channels", "n_samples", "duration_sec"]
    event_cols = ["record_id", "start_sec", "end_sec", "label", "phase_label", "duration_sec"]
    failure_cols = ["record_id", "standard_label", "source_path", "error_type", "error"]
    apply_record_event_labels(records, events)
    pd.DataFrame(records, columns=record_cols).to_csv(out / "records.csv", index=False)
    pd.DataFrame(events, columns=event_cols).to_csv(out / "events.csv", index=False)
    pd.DataFrame(phase_events, columns=["record_id", "start_sec", "end_sec", "label", "phase_label", "duration_sec"]).to_csv(out / "phase_events.csv", index=False)
    pd.DataFrame(marker_qc_rows).to_csv(out / "marker_qc.csv", index=False)
    pd.DataFrame(failures, columns=failure_cols).to_csv(out / "failures.csv", index=False)
    identity_df = pd.DataFrame(identity_rows, columns=["record_id", "patient_id", "identity_source", "hospital_id_hash", "name_hash"])
    identity_df.to_csv(out / "identity_qc.csv", index=False)
    hospital_collisions = identity_df[identity_df["hospital_id_hash"] != ""].groupby("hospital_id_hash")["patient_id"].nunique()
    name_collisions = identity_df[identity_df["name_hash"] != ""].groupby("name_hash")["patient_id"].nunique()
    pd.Series({
        "records": len(identity_df),
        "patients": identity_df["patient_id"].nunique(),
        "hospital_ids_mapping_to_multiple_patients": int((hospital_collisions > 1).sum()),
        "names_mapping_to_multiple_patients": int((name_collisions > 1).sum()),
        **{f"source_{key}": int(value) for key, value in identity_df["identity_source"].value_counts().items()},
    }).to_json(out / "identity_qc_summary.json", indent=2)
    preprocessing_summary = build_preprocessing_summary(
        records, events, phase_events, failures, marker_qc_rows, len(raw_inputs),
    )
    preprocessing_summary["hospital_ids_mapping_to_multiple_patients"] = int((hospital_collisions > 1).sum())
    preprocessing_summary["names_mapping_to_multiple_patients"] = int((name_collisions > 1).sum())
    (out / "preprocessing_summary.json").write_text(
        json.dumps(preprocessing_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved {len(records)} WHCH records with {len(events)} events to {out}. failed={len(failures)}")


if __name__ == "__main__":
    main()
