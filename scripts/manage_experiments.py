#!/usr/bin/env python3
"""Prepare training jobs, prepare external evaluation, and summarize results."""
import argparse
import json
import math
from pathlib import Path
import sys
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.protocol import prepare_paper, save_manifests, write_json
from src.utils.config import load_config
from scripts.run_experiments import load_manifest, source_complete, external_complete


def source_results(payload):
    rows, missing = [], []
    for item in payload['experiments']:
        run = Path(item['output_dir'])
        if not source_complete(run):
            missing.append(item['id'])
            continue
        if load_config(run / 'used_config.yaml') != load_config(item['config']):
            raise ValueError(f'Configuration differs for {item["id"]}.')
        metric = 'win_ap' if item['task'] == 'detection' else 'cls_macro_f1'
        score = float(json.loads((run / 'ensemble_val_metrics.json').read_text())[metric])
        if not math.isfinite(score):
            raise ValueError(f'Non-finite primary metric for {item["id"]}.')
        rows.append({**{key: item[key] for key in ('id', 'task', 'model', 'seed', 'fold')},
                     'metric': metric, 'score': score})
    return pd.DataFrame(rows), missing


def summarize_source(payload, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, missing = source_results(payload)
    write_json(output / 'missing.json', missing)
    if rows.empty:
        return
    rows.to_csv(output / 'source_by_fold_seed.csv', index=False)
    groups = ['task', 'model', 'metric']
    seeds = rows.groupby([*groups, 'seed'], as_index=False).score.mean()
    seeds.to_csv(output / 'source_by_seed.csv', index=False)
    summary = seeds.groupby(groups).score.agg(['count', 'mean', 'std']).reset_index()
    summary.to_csv(output / 'source_summary.csv', index=False)
    print(summary.round(5).to_string(index=False))
    print('Incomplete:', len(missing))


def prepare_evaluation(manifest, output):
    payload = load_manifest(manifest)
    if payload['stage'] != 'source':
        raise ValueError('Provide the source-training manifest.')
    _, missing = source_results(payload)
    if missing:
        raise ValueError(f'{len(missing)} training runs are incomplete.')
    path = Path(output) / (Path(manifest).stem + '_external.json')
    save_manifests(path, payload['experiments'], stage='external', protocol=payload.get('protocol'))
    print('External-evaluation jobs:', len(payload['experiments']), '; manifest:', path)


def summarize_external(payload, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for item in payload['experiments']:
        run = Path(item['output_dir'])
        if not external_complete(run):
            missing.append(item['id'])
            continue
        result = json.loads((run / 'external_evaluation/ensemble_result_manifest.json').read_text())
        for target, artifacts in result['datasets'].items():
            values = pd.read_csv(run / 'external_evaluation' / artifacts['metrics']).iloc[0].to_dict()
            rows.append({**{key: item[key] for key in ('id', 'task', 'model', 'seed', 'fold')},
                         'target': target, **values})
    write_json(output / 'missing_external.json', missing)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'external_by_fold_seed.csv', index=False)
    metrics = [key for key in ('win_ap', 'win_auroc', 'event_f1', 'event_recall', 'event_precision',
               'event_fp_per_hour', 'event_strict_iou_f1', 'calibration_brier', 'calibration_nll',
               'calibration_ece', 'cls_macro_f1', 'cls_f1_ABSZ', 'cls_f1_TNSZ', 'cls_f1_TCSZ') if key in frame]
    groups = ['task', 'model', 'target']
    # Average source folds within each seed before calculating the seed SD.
    seeds = frame.groupby([*groups, 'seed'], as_index=False)[metrics].mean()
    seeds.to_csv(output / 'external_by_seed.csv', index=False)
    summary = seeds.groupby(groups).agg({metric: ['mean', 'std', 'count'] for metric in metrics})
    summary.to_csv(output / 'external_summary.csv')
    print('External runs complete:', len(payload['experiments']) - len(missing), '/', len(payload['experiments']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--study', default='configs/study.yaml')
    p.add_argument('--data-root', default='data')
    p.add_argument('--plan-root', default='runs/plan')
    p.add_argument('--run-root', default='runs/study')
    for name in ('evaluate', 'summarize'):
        p = sub.add_parser(name)
        p.add_argument('--manifest', required=True)
        p.add_argument('--output', default='runs/evaluation' if name == 'evaluate' else 'runs/summary')
    args = parser.parse_args()
    if args.command == 'prepare':
        study = yaml.safe_load(Path(args.study).read_text(encoding='utf-8'))
        items = prepare_paper(study, args.plan_root, args.run_root, args.data_root)
        print(json.dumps({'training_runs': len(items), 'manifest': str(Path(args.plan_root) / 'paper.json')}, indent=2))
    elif args.command == 'evaluate':
        prepare_evaluation(args.manifest, args.output)
    else:
        payload = load_manifest(args.manifest)
        if payload['stage'] == 'source':
            summarize_source(payload, args.output)
        else:
            summarize_external(payload, args.output)


if __name__ == '__main__':
    main()
