"""YAML inheritance and validation for training and external evaluation."""
from pathlib import Path
import math
import yaml


SUPPORTED_ARCHITECTURES = {'gdr_net', 'eegnet', 'cnn_lstm', 'eeg_conformer', 'dcrnn'}
SUPPORTED_SELECTION_METRICS = {
    'val_win_ap', 'val_win_auprc', 'val_win_auroc', 'val_win_f1',
    'val_win_bal_acc', 'val_cls_macro_f1', 'val_event_f1', 'val_joint_ap_cls_hmean',
}


def load_config(path: str):
    path = Path(path)
    with path.open('r', encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream) or {}
    parent_path = cfg.pop('extends', None)
    if not parent_path:
        return cfg
    parent_path = Path(parent_path)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    return deep_merge(load_config(str(parent_path)), cfg)


def deep_merge(base: dict, override: dict):
    merged = dict(base)
    for key, value in override.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def validate_config(cfg: dict, mode: str = 'train'):
    for section in ('data', 'model', 'loss'):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f'Missing required configuration section: {section}')
    model, loss = cfg['model'], cfg['loss']
    data = cfg['data']
    for key in ('batch_size', 'eval_batch_size', 'prefetch_factor'):
        if key in data and int(data[key]) < 1:
            raise ValueError(f'data.{key} must be positive.')
    if data.get('num_workers') is not None and int(data['num_workers']) < 0:
        raise ValueError('data.num_workers must be nonnegative.')
    architecture = str(model.get('architecture', '')).lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f'Unsupported model.architecture={architecture!r}.')
    for key in ('omega_det', 'alpha_cls', 'rho_cf', 'sigma_cf_cls'):
        value = float(loss.get(key, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'loss.{key} must be finite and nonnegative.')
    if bool(cfg.get('postprocess', {}).get('use_boundary_offsets', False)):
        raise ValueError('Window-based event decoding requires use_boundary_offsets=false.')
    cf_weight = float(loss.get('rho_cf', 0)) + float(loss.get('sigma_cf_cls', 0))
    if architecture == 'gdr_net':
        recombination = bool(model.get('enable_recombination', True))
        if recombination != (cf_weight > 0):
            raise ValueError('GDR enable_recombination must match the presence of positive recombination loss weights.')
        if model.get('cls_fusion', 'task_style_temporal') not in {'task', 'task_style_temporal'}:
            raise ValueError('GDR cls_fusion must be task or task_style_temporal.')
        if model.get('recombination_pooling', 'shared') not in {'shared', 'separate'}:
            raise ValueError('GDR recombination_pooling must be shared or separate.')
        if not 0 <= float(model.get('recombine_alpha', 0.5)) <= 1:
            raise ValueError('recombine_alpha must be between 0 and 1.')
    elif cf_weight != 0:
        raise ValueError('Recombination losses are available only for GDR-Net.')
    if str(mode).lower() != 'train':
        return cfg
    if 'train' not in cfg or 'postprocess' not in cfg:
        raise ValueError('Training requires train and postprocess sections.')
    train = cfg['train']
    if cfg['data'].get('external_sets') or cfg['data'].get('external_root'):
        raise ValueError('Source training must not contain external datasets; use evaluate_external.py after selection.')
    metric = str(train.get('selection_metric', '')).lower()
    metric = metric if metric.startswith('val_') else f'val_{metric}'
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f'Unsupported train.selection_metric={metric!r}; no fallback is allowed.')
    if metric in {'val_cls_macro_f1', 'val_joint_ap_cls_hmean'} and not cfg['data'].get('compute_classification', True):
        raise ValueError('Subtype selection metrics require compute_classification=true.')
    for key in ('recombination_start_epoch',):
        if int(train.get(key, 1)) < 1:
            raise ValueError(f'train.{key} must be at least 1.')
    for key in ('recombination_warmup_epochs',):
        if int(train.get(key, 0)) < 0:
            raise ValueError(f'train.{key} must be nonnegative.')
    weighted = bool(train.get('weighted_sampling', False))
    if weighted and float(loss.get('det_pos_weight', 1)) != 1:
        raise ValueError('Do not combine balanced sampling with det_pos_weight != 1.')
    strategy = str(train.get('sampling_strategy', 'detection')).lower()
    class_strategies = {'subtype', 'class', 'classification', 'subtype_balanced', 'patient_class', 'patient_aware', 'patient_class_balanced', 'class_patient'}
    if weighted and strategy in class_strategies and loss.get('cls_class_weight') not in (None, '', 'none', False):
        raise ValueError('Do not combine subtype-balanced sampling with class-weighted cross-entropy.')
    if strategy == 'class_patient':
        if not weighted or not 0 < float(train.get('seizure_sampling_fraction', 0.5)) < 1:
            raise ValueError('class_patient requires weighted_sampling=true and a seizure fraction in (0, 1).')
        if train.get('balance_domains', False) or float(train.get('domain_balance_alpha', 0)) != 0:
            raise ValueError('class_patient balances classes and patients; domain reweighting is not implemented.')
    if not 0 <= float(train.get('ema_decay', 0)) < 1:
        raise ValueError('ema_decay must be in [0, 1).')
    if not 1 <= int(train.get('checkpoint_start_epoch', 1)) <= int(train['epochs']):
        raise ValueError('checkpoint_start_epoch must lie within the training epochs.')
    return cfg
