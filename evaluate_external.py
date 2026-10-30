#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from train import (
    build_loader,
    build_metrics,
    build_model_from_config,
    maybe_build_external_datasets,
    predict_checkpoint_ensemble,
    type_confusion_matrix,
)
from src.utils.config import load_config, validate_config
from src.training.calibration import calibrate_prediction_frame, calibration_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='One-time evaluation of frozen source checkpoints on external datasets.')
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--external_config', required=True, help='YAML defining held-out evaluation datasets; source postprocessing stays frozen.')
    parser.add_argument('--checkpoint', choices=['best', 'ensemble'], default='ensemble')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_operating_point(run_dir: Path, prefix: str, fallback_threshold: float, fallback_min_event_sec: float):
    path = run_dir / f'{prefix}_val_metrics_tuned_threshold.csv'
    if not path.exists() and prefix == 'ensemble':
        path = run_dir / 'best_val_metrics_tuned_threshold.csv'
    if not path.exists():
        return float(fallback_threshold), float(fallback_min_event_sec)
    row = pd.read_csv(path).iloc[0]
    return float(row['selected_det_threshold']), float(row.get('selected_min_event_sec', fallback_min_event_sec))


def checkpoint_paths(run_dir: Path, prefix: str):
    if prefix == 'best':
        return [run_dir / 'best.pt']
    manifest = run_dir / 'ensemble_checkpoints.json'
    if not manifest.exists():
        raise FileNotFoundError(f'Missing frozen ensemble manifest: {manifest}')
    return [Path(item['path']) for item in json.loads(manifest.read_text(encoding='utf-8'))]


def read_temperature(run_dir: Path, prefix: str):
    path = run_dir / f'{prefix}_temperature.json'
    if not path.exists() and prefix == 'ensemble':
        path = run_dir / 'best_temperature.json'
    if not path.exists():
        return 1.0
    return float(json.loads(path.read_text(encoding='utf-8'))['temperature'])


def check_frozen_postprocessing(source_cfg, external_cfg):
    """External datasets may not supply newly tuned decoding parameters."""
    source = source_cfg.get('postprocess', {})
    for key, value in external_cfg.get('postprocess', {}).items():
        if key not in source or source[key] != value:
            raise ValueError(f'External postprocess.{key} differs from the frozen source configuration.')


def check_heldout_patients(run_dir: Path, datasets):
    """Reject train/selection patients in both within-domain and external tests."""
    source_patients = set()
    for filename in ('train_patients.csv', 'val_patients.csv'):
        path = run_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Patient-disjoint evaluation requires the source patient list: {path}')
        table = pd.read_csv(path, dtype=str)
        if 'patient_id' not in table:
            raise ValueError(f'Missing patient_id column in {path}')
        source_patients.update(table['patient_id'].dropna().astype(str))
    for item in datasets:
        patients = set(item['dataset'].records['patient_id'].dropna().astype(str))
        overlap = source_patients & patients
        if overlap:
            raise ValueError(f'Evaluation set {item["name"]!r} contains {len(overlap)} source training/validation patients.')


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    source_cfg = load_config(str(run_dir / 'used_config.yaml'))
    external_cfg = load_config(args.external_config)
    source_cfg['data']['external_sets'] = external_cfg.get('data', {}).get('external_sets', [])
    if not source_cfg['data']['external_sets']:
        raise ValueError('external_config must define at least one data.external_sets entry.')
    check_frozen_postprocessing(source_cfg, external_cfg)
    validate_config(source_cfg, mode='external')

    paths = checkpoint_paths(run_dir, args.checkpoint)
    payload = torch.load(paths[0], map_location='cpu', weights_only=False)
    domain_to_id = payload['domain_to_id']
    class_names = source_cfg['data'].get('type_class_names', ['ABSZ', 'TNSZ', 'TCSZ'])
    model = build_model_from_config(source_cfg, len(domain_to_id), len(class_names))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    threshold, min_event_sec = read_operating_point(
        run_dir, args.checkpoint, source_cfg['postprocess']['det_threshold'], source_cfg['postprocess']['min_event_sec'],
    )
    source_cfg['postprocess']['min_event_sec'] = min_event_sec
    temperature = read_temperature(run_dir, args.checkpoint)
    datasets = maybe_build_external_datasets(source_cfg, domain_to_id, class_names)
    if not datasets:
        raise ValueError('No external datasets were constructed from the frozen external configuration.')
    check_heldout_patients(run_dir, datasets)
    output_dir = run_dir / 'external_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    result_manifest = {
        'layout_version': 2,
        'source_run': str(run_dir.resolve()),
        'checkpoint': args.checkpoint,
        'external_config': str(Path(args.external_config).resolve()),
        'threshold_from_source_validation': threshold,
        'min_event_sec_from_source_validation': min_event_sec,
        'temperature_from_source_validation': temperature,
        'datasets': {},
    }
    for item in datasets:
        prefix = f'{args.checkpoint}_external_{item["name"]}'
        metric_path = output_dir / f'{prefix}_metrics.csv'
        if metric_path.exists() and not args.overwrite:
            raise FileExistsError(f'External result already exists: {metric_path}. Use --overwrite only for an intentional rerun.')
        data_cfg = source_cfg['data']
        loader = build_loader(item['dataset'], int(data_cfg.get('eval_batch_size', data_cfg['batch_size'])),
            int(data_cfg.get('num_workers', 0)), False,
            bool(data_cfg.get('pin_memory', True)) and device.type == 'cuda',
            seed=int(source_cfg.get('seed', 0)) + 1,
            persistent_workers=bool(data_cfg.get('persistent_workers', True)),
            prefetch_factor=int(data_cfg.get('prefetch_factor', 4)))
        metrics = build_metrics(
            source_cfg['postprocess'], len(class_names), class_names,
            compute_classification=(item['task'] != 'detection'),
            label_groups=item.get('label_groups'), threshold=threshold,
        )
        _, predictions = predict_checkpoint_ensemble(model, paths, loader, device, metrics, item['dataset'].events, compute_metrics=False)
        predictions = calibrate_prediction_frame(predictions, temperature)
        values = metrics.compute(predictions, item['dataset'].events)
        calibration_values, reliability = calibration_metrics(predictions['det_true'], predictions['det_prob'], int(source_cfg['postprocess'].get('calibration_bins', 15)))
        values.update({f'calibration_{key}': value for key, value in calibration_values.items()})
        predictions.to_csv(output_dir / f'{prefix}_window_predictions.csv', index=False)
        reliability.to_csv(output_dir / f'{prefix}_reliability.csv', index=False)
        pd.DataFrame([values]).to_csv(metric_path, index=False)
        metrics.decoder.decode(predictions).to_csv(output_dir / f'{prefix}_events.csv', index=False)
        if metrics.compute_classification:
            type_confusion_matrix(predictions, class_names).to_csv(output_dir / f'{prefix}_type_confusion_matrix.csv')
        result_manifest['datasets'][item['name']] = {
            'metrics': f'{prefix}_metrics.csv',
            'window_predictions': f'{prefix}_window_predictions.csv',
            'events': f'{prefix}_events.csv',
            'reliability': f'{prefix}_reliability.csv',
            'type_confusion_matrix': f'{prefix}_type_confusion_matrix.csv' if metrics.compute_classification else None,
        }
    (output_dir / f'{args.checkpoint}_result_manifest.json').write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2), encoding='utf-8',
    )


if __name__ == '__main__':
    main()
