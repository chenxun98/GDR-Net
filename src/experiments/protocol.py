"""Configurations and experiment manifests for the five-model comparison."""
from copy import deepcopy
import json
from pathlib import Path
import yaml
from src.utils.config import load_config, validate_config

PROJECT = Path(__file__).resolve().parents[2]
MODELS = ('gdr_net', 'eegnet', 'cnn_lstm', 'eeg_conformer', 'dcrnn')
LABELS = dict(zip(MODELS, ('GDR-Net', 'EEGNet', 'CNN-LSTM', 'EEG Conformer', 'DCRNN')))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text(encoding='utf-8')) != config:
        raise ValueError(f'Existing configuration differs: {path}. Use a new --plan-root.')
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')


def model_config(task, model, data_root='data'):
    if task not in ('detection', 'subtype') or model not in MODELS:
        raise ValueError(f'Unknown task/model: {task}/{model}')
    cfg = load_config(PROJECT / 'configs' / f'{task}.yaml')
    for spec in cfg['data']['train_sets']:
        spec['root'] = (Path(data_root) / Path(spec['root']).name).as_posix()
    cfg['experiment'] = {'group': task, 'name': model}
    if model == 'gdr_net':
        return validate_config(cfg)
    if model == 'dcrnn':
        defaults = dict(hidden_dim=64, num_layers=2, max_diffusion_step=2,
                        top_k=3, step_seconds=1.0, dropout=0.0 if task == 'detection' else 0.5)
        cfg['model'] = {'architecture': model, **{
            'dcrnn_' + key: cfg['model'].get('dcrnn_' + key, value)
            for key, value in defaults.items()}}
    elif model == 'eeg_conformer':
        defaults = dict(embedding_dim=40, depth=6, num_heads=10, dropout=0.5,
                        head_dropout=0.3, temporal_kernel=25, pool_kernel=75, pool_stride=15)
        cfg['model'] = {'architecture': model, **{
            'conformer_' + key: cfg['model'].get('conformer_' + key, value)
            for key, value in defaults.items()}}
    else:
        keys = {'dropout', 'hidden_dim'} if model == 'cnn_lstm' else {
            'dropout', 'eegnet_f1', 'eegnet_depth_multiplier', 'eegnet_f2', 'eegnet_pool_tokens'}
        cfg['model'] = {key: value for key, value in cfg['model'].items() if key in keys}
        cfg['model']['architecture'] = model
    cfg['loss'].update(rho_cf=0.0, sigma_cf_cls=0.0)
    return validate_config(cfg)


def external_spec(task, data_root):
    spec = {'name': 'whch', 'root': (Path(data_root) / 'whch_processed').as_posix(),
            'task': 'detection' if task == 'detection' else 'multiclass', 'events_file': 'events.csv'}
    if task == 'subtype':
        spec.update(include_labels=['ABSZ', 'TNSZ', 'TCSZ'], drop_records_with_excluded_labels=True)
    return spec


def make_item(config, plan_root, run_root, task, model, fold, seed, data_root):
    cfg = deepcopy(config)
    cfg['seed'] = seed
    cfg['train']['patient_fold'] = fold
    cfg['experiment'] = {'group': 'core', 'name': model}
    identity = f'core/{task}/{model}/fold{fold}/seed{seed}'
    path = Path(plan_root) / 'resolved' / (identity + '.yaml')
    target_path = Path(plan_root) / 'targets' / (identity + '.yaml')
    out = Path(run_root) / identity
    validate_config(cfg)
    write_config(path, cfg)
    write_config(target_path, {'data': {'external_sets': [external_spec(task, data_root)]}})
    return dict(id=identity, task=task, model=model, display_name=LABELS[model], fold=fold,
                seed=seed, config=path.as_posix(), output_dir=out.as_posix(),
                train_argv=['train.py', '--config', path.as_posix(), '--output_dir', out.as_posix()],
                external_argv=['evaluate_external.py', '--run_dir', out.as_posix(),
                               '--external_config', target_path.as_posix(), '--checkpoint', 'ensemble'])


def save_manifests(path, items, stage='source', **metadata):
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('Duplicate experiment identities.')
    write_json(path, dict(stage=stage, experiments=items, **metadata))


def prepare_paper(study, plan_root, run_root, data_root):
    items = []
    for task in ('detection', 'subtype'):
        for model in study['models']:
            cfg = model_config(task, model, data_root)
            for fold in study[task + '_folds']:
                for seed in study['seeds']:
                    items.append(make_item(cfg, plan_root, run_root, task, model, fold, seed, data_root))
    save_manifests(Path(plan_root) / 'paper.json', items, study=study,
                   protocol='patient_disjoint_source_development')
    return items
