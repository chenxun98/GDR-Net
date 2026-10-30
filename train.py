#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from time import perf_counter
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from src.data.pairing import build_counterfactual_pairs
from src.data.balanced_sampling import class_patient_sampler
from src.data.window_dataset import EEGWindowDataset, build_patient_split, read_manifest_csv
from src.models.baselines import CNNBiLSTMBaseline, EEGNetBaseline
from src.models.eeg_conformer import EEGConformerBaseline
from src.models.dcrnn import DCRNNBaseline
from src.models.gdr_net import GDRNet
from src.training.losses import Criterion, LossWeights
from src.training.ema import ModelEMA
from src.training.calibration import calibrate_prediction_frame, calibration_metrics, fit_binary_temperature
from src.training.metrics import MetricsComputer
from src.training.event_decoder import event_level_f1
from src.utils.config import load_config, validate_config
from src.utils.train_utils import batch_to_device, predict_epoch, seed_worker, set_seed

class CombinedEEGWindowDataset:
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.samples = []
        self.events = pd.DataFrame(columns=['record_id', 'start_sec', 'end_sec', 'label'])
        self._index = []
        for ds_idx, ds in enumerate(self.datasets):
            self.samples.extend(ds.samples)
            if hasattr(ds, 'events') and not ds.events.empty:
                self.events = ds.events.copy() if self.events.empty else pd.concat([self.events, ds.events], ignore_index=True)
            for local_idx in range(len(ds)):
                self._index.append((ds_idx, local_idx))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        ds_idx, local_idx = self._index[index]
        return self.datasets[ds_idx][local_idx]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/detection.yaml')
    p.add_argument('--output_dir', default=None, help='Defaults to runs/<group>/<name>/fold<fold>/seed<seed>.')
    p.add_argument('--fold', type=int, default=None, help='Override train.patient_fold without editing the config.')
    p.add_argument('--set', action='append', default=[], help='Override config values, e.g. --set model.graph_mode=static')
    p.add_argument('--overwrite', action='store_true', help='Archive an existing non-empty run directory before starting a clean rerun.')
    return p.parse_args()

def apply_overrides(cfg, overrides):
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Invalid override {item!r}. Expected key=value.')
        key, value = item.split('=', 1)
        target = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)

def build_loader(dataset, batch_size, num_workers, shuffle, pin_memory, sampler=None, seed: int = 0,
                 persistent_workers: bool = True, prefetch_factor: int = 4):
    generator = torch.Generator().manual_seed(int(seed))
    worker_options = {'persistent_workers': bool(persistent_workers), 'prefetch_factor': int(prefetch_factor)} if num_workers > 0 else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        worker_init_fn=seed_worker, generator=generator, **worker_options)

def build_detection_sampler(dataset, num_samples=None, balance_domains: bool = False):
    labels = torch.tensor([s.det_label for s in dataset.samples], dtype=torch.long)
    if balance_domains:
        domains = torch.tensor([s.domain_label for s in dataset.samples], dtype=torch.long)
        pair_to_count = {}
        for domain, label in zip(domains.tolist(), labels.tolist()):
            key = (int(domain), int(label))
            pair_to_count[key] = pair_to_count.get(key, 0) + 1
        weights = torch.tensor([1.0 / pair_to_count[(int(domain), int(label))] for domain, label in zip(domains.tolist(), labels.tolist())], dtype=torch.float32)
    else:
        counts = torch.bincount(labels, minlength=2).float().clamp_min(1.0)
        weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(weights=weights, num_samples=int(num_samples or len(weights)), replacement=True)

def build_subtype_sampler(dataset, num_samples=None, balance_domains: bool = False):
    keys = []
    for sample in dataset.samples:
        cls_label = int(sample.cls_label)
        det_label = int(sample.det_label)
        subtype_key = cls_label if det_label == 1 and cls_label >= 0 else -1
        key = (int(sample.domain_label), subtype_key) if balance_domains else subtype_key
        keys.append(key)
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = torch.tensor([1.0 / counts[key] for key in keys], dtype=torch.float32)
    return WeightedRandomSampler(weights=weights, num_samples=int(num_samples or len(weights)), replacement=True)

def build_patient_class_sampler(dataset, compute_classification: bool, num_samples=None, domain_balance_alpha: float = 0.0):
    keys = []
    domains = []
    for sample in dataset.samples:
        if compute_classification and int(sample.det_label) == 1 and int(sample.cls_label) >= 0:
            class_key = f'cls_{int(sample.cls_label)}'
        else:
            class_key = f'det_{int(sample.det_label)}'
        domain = int(sample.domain_label)
        keys.append((domain, str(sample.patient_id), class_key))
        domains.append(domain)
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    domain_group_counts = {}
    for key in set(keys):
        domain_group_counts[key[0]] = domain_group_counts.get(key[0], 0) + 1
    alpha = float(domain_balance_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('train.domain_balance_alpha must be in [0, 1].')
    weights = torch.tensor([
        (1.0 / counts[key]) * (1.0 / domain_group_counts[domain]) ** alpha
        for key, domain in zip(keys, domains)
    ], dtype=torch.float32)
    return WeightedRandomSampler(weights=weights, num_samples=int(num_samples or len(weights)), replacement=True)

def build_train_sampler(dataset, cfg, compute_classification: bool):
    train_cfg = cfg.get('train', {})
    if not bool(train_cfg.get('weighted_sampling', True)):
        return None
    strategy = str(train_cfg.get('sampling_strategy', 'detection')).lower()
    if strategy == 'class_patient':
        return class_patient_sampler(dataset, compute_classification,
            train_cfg.get('samples_per_epoch'), float(train_cfg.get('seizure_sampling_fraction', 0.5)))
    balance_domains = bool(train_cfg.get('balance_domains', False))
    if compute_classification and strategy in {'subtype', 'class', 'classification', 'subtype_balanced'}:
        return build_subtype_sampler(dataset, train_cfg.get('samples_per_epoch'), balance_domains)
    if strategy in {'patient_class', 'patient_aware', 'patient_class_balanced'}:
        return build_patient_class_sampler(
            dataset, compute_classification, train_cfg.get('samples_per_epoch'),
            float(train_cfg.get('domain_balance_alpha', 0.0)),
        )
    return build_detection_sampler(dataset, train_cfg.get('samples_per_epoch'), balance_domains)

def build_loss_weights(loss_cfg):
    return LossWeights(**{field: loss_cfg.get(field, getattr(LossWeights(), field)) for field in LossWeights.__dataclass_fields__})

def load_torch_checkpoint(path: str | Path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def build_optimizer(model, cfg):
    train_cfg = cfg.get('train', {})
    lr = float(train_cfg['lr'])
    weight_decay = float(train_cfg['weight_decay'])
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def auxiliary_warmup_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    """Linearly increase the recombination-loss weight from its start epoch."""
    start_epoch = int(start_epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min((epoch - start_epoch + 1) / warmup_epochs, 1.0)

def resolve_cls_class_weight(cfg, dataset, num_classes, compute_classification):
    if not compute_classification:
        return None
    mode = cfg.get('loss', {}).get('cls_class_weight')
    if mode is None or str(mode).lower() in {'', 'none', 'false'}:
        return None
    if isinstance(mode, (list, tuple)):
        if len(mode) != num_classes:
            raise ValueError(f'loss.cls_class_weight has {len(mode)} values, expected {num_classes}.')
        return torch.tensor([float(x) for x in mode], dtype=torch.float32)
    if str(mode).lower() != 'balanced':
        raise ValueError("loss.cls_class_weight must be omitted, 'balanced', or a list of per-class weights.")
    labels = [int(s.cls_label) for s in dataset.samples if int(s.det_label) == 1 and int(s.cls_label) >= 0]
    if not labels:
        return None
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float().clamp_min(1.0)
    weights = counts.sum() / (float(num_classes) * counts)
    return weights / weights.mean().clamp_min(1e-8)

def resolve_output_dir(args, cfg):
    if args.output_dir:
        return Path(args.output_dir)
    exp_cfg = cfg.get('experiment', {})
    name = str(exp_cfg.get('name') or Path(args.config).stem)
    group = exp_cfg.get('group')
    fold = int(cfg.get('train', {}).get('patient_fold', 0))
    seed = int(cfg.get('seed', 0))
    if group:
        return Path('runs') / str(group) / name / f'fold{fold}' / f'seed{seed}'
    return Path('runs') / name / f'fold{fold}' / f'seed{seed}'

def prepare_run_output_dir(out: Path, overwrite: bool = False):
    out = Path(out)
    backup = None
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f'Run directory is not empty: {out}. Refusing to overwrite a completed or partial experiment. '
                'Choose a new fold/seed/output directory or pass --overwrite to archive it first.'
            )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup = out.with_name(f'{out.name}.backup_{timestamp}')
        out.rename(backup)
    out.mkdir(parents=True, exist_ok=True)
    return backup

def build_model_from_config(cfg, num_domains, num_type_classes):
    model_cfg = cfg.get('model', {})
    sampling_rate = float(cfg.get('data', {}).get('sampling_rate', cfg.get('data', {}).get('target_sampling_rate', 256.0)))
    architecture = str(model_cfg.get('architecture', 'gdr_net')).lower()
    common = dict(
        channel_names=cfg['data']['channels'], num_domains=num_domains,
        temporal_hidden=int(model_cfg.get('temporal_hidden', 64)),
        graph_hidden=int(model_cfg.get('graph_hidden', 128)),
        graph_out=int(model_cfg.get('graph_out', 128)),
        dropout=float(model_cfg.get('dropout', 0.2)),
        static_graph_weight=float(model_cfg.get('static_graph_weight', 0.35)),
        num_classes=num_type_classes, temporal_tokens=int(model_cfg.get('temporal_tokens', 64)),
        graph_mode=str(model_cfg.get('graph_mode', 'hybrid')),
        use_frequency_branch=bool(model_cfg.get('use_frequency_branch', True)),
        sampling_rate=sampling_rate,
        temporal_normalization=str(model_cfg.get('temporal_normalization', 'group')),
        correlation_mode=str(model_cfg.get('correlation_mode', 'absolute')),
    )
    if architecture == 'gdr_net':
        return GDRNet(
            **common,
            disentangle_temperature=float(model_cfg.get('disentangle_temperature', 1.0)),
            recombine_alpha=float(model_cfg.get('recombine_alpha', 0.5)),
            cls_fusion=str(model_cfg.get('cls_fusion', 'task_style_temporal' if cfg['data'].get('compute_classification', True) else 'task')),
            recombination_pooling=str(model_cfg.get('recombination_pooling', 'shared')),
            detach_style_donor=bool(model_cfg.get('detach_style_donor', False)),
            enable_disentanglement=bool(model_cfg.get('enable_disentanglement', True)),
            enable_recombination=bool(model_cfg.get('enable_recombination', True)),
        )
    if architecture == 'dcrnn':
        return DCRNNBaseline(
            num_channels=len(cfg['data']['channels']),
            num_times=int(round(sampling_rate * float(cfg['data']['window_sec']))),
            sampling_rate=sampling_rate, num_classes=num_type_classes,
            **{key.removeprefix('dcrnn_'): value for key, value in model_cfg.items()
               if key.startswith('dcrnn_')},
        )
    if architecture == 'eeg_conformer':
        return EEGConformerBaseline(
            num_channels=len(cfg['data']['channels']),
            num_times=int(round(sampling_rate * float(cfg['data']['window_sec']))),
            num_classes=num_type_classes,
            **{key.removeprefix('conformer_'): value for key, value in model_cfg.items()
               if key.startswith('conformer_')},
        )
    dropout = float(model_cfg.get('dropout', 0.2))
    if architecture == 'cnn_lstm':
        return CNNBiLSTMBaseline(
            num_channels=len(cfg['data']['channels']), num_domains=num_domains,
            hidden_dim=int(model_cfg.get('hidden_dim', 128)),
            dropout=dropout, num_classes=num_type_classes,
        )
    if architecture == 'eegnet':
        return EEGNetBaseline(
            len(cfg['data']['channels']),
            num_domains,
            dropout,
            num_type_classes,
            sampling_rate,
            int(model_cfg.get('eegnet_f1', 8)),
            int(model_cfg.get('eegnet_depth_multiplier', 2)),
            model_cfg.get('eegnet_f2'),
            int(model_cfg.get('eegnet_pool_tokens', 16)),
        )
    raise ValueError(f'Unsupported model.architecture={architecture!r}. Use gdr_net, eegnet, cnn_lstm, eeg_conformer or dcrnn.')

def resolve_det_pos_weight(cfg, train_sampler, pos_count, neg_count):
    configured = cfg.get('loss', {}).get('det_pos_weight')
    if configured is not None:
        return float(configured)
    return 1.0 if train_sampler is not None else neg_count / max(pos_count, 1)

def build_external_specs(cfg):
    specs = cfg['data'].get('external_sets')
    if specs:
        return specs
    external_root = cfg['data'].get('external_root')
    if not external_root:
        return []
    return [{'name': 'external', 'root': external_root, 'task': 'multiclass'}]

def build_train_specs(cfg):
    specs = cfg['data'].get('train_sets')
    if specs:
        return specs
    return [{'name': 'train', 'root': cfg['data']['train_root']}]

def age_group_from_value(value):
    try:
        age = float(value)
    except (TypeError, ValueError):
        return 'unknown'
    if pd.isna(age):
        return 'unknown'
    if age < 13:
        return 'child'
    if age < 18:
        return 'adolescent'
    if age < 60:
        return 'adult'
    return 'older_adult'

def add_metadata_filter_columns(records: pd.DataFrame):
    records = records.copy()
    if 'age_group' not in records.columns:
        records['age_group'] = records['age_years'].map(age_group_from_value) if 'age_years' in records.columns else 'unknown'
    if 'sex' in records.columns:
        records['sex'] = records['sex'].fillna('unknown').astype(str).replace({'': 'unknown'})
    else:
        records['sex'] = 'unknown'
    if 'dataset' in records.columns:
        records['dataset'] = records['dataset'].fillna('unknown').astype(str).replace({'': 'unknown'})
    else:
        records['dataset'] = 'unknown'
    return records

def values_from_filter(value):
    if isinstance(value, (list, tuple, set)):
        return {str(x) for x in value}
    return {str(value)}

def apply_metadata_filter(records: pd.DataFrame, metadata_filter):
    if not metadata_filter:
        return records
    records = add_metadata_filter_columns(records)
    keep = pd.Series(True, index=records.index)
    for column, allowed in dict(metadata_filter).items():
        if column not in records.columns:
            raise ValueError(f'Metadata filter column {column!r} is not present in records.csv.')
        allowed_values = values_from_filter(allowed)
        keep &= records[column].fillna('unknown').astype(str).isin(allowed_values)
    return records[keep].copy()

def record_ids_from_csv(path):
    table = read_manifest_csv(path)
    if 'record_id' in table.columns:
        return set(table['record_id'].astype(str))
    if table.shape[1] == 0:
        return set()
    return set(table.iloc[:, 0].astype(str))

def resolve_spec_csv(root: Path, value):
    path = Path(value)
    return path if path.is_absolute() else root / path

def filtered_record_ids_for_spec(spec):
    root = Path(spec['root'])
    records = read_manifest_csv(root / 'records.csv')
    records['record_id'] = records['record_id'].astype(str)
    records = apply_metadata_filter(records, spec.get('metadata_filter') or spec.get('filter'))
    records = records.drop_duplicates('record_id')
    ids = set(records['record_id'].astype(str))
    record_ids_csv = spec.get('record_ids_csv')
    if record_ids_csv:
        ids &= record_ids_from_csv(resolve_spec_csv(root, record_ids_csv))
    exclude_filter = spec.get('exclude_metadata_filter') or spec.get('exclude_filter')
    if exclude_filter:
        excluded = apply_metadata_filter(read_manifest_csv(root / 'records.csv'), exclude_filter)
        ids -= set(excluded['record_id'].astype(str))
    include_labels = spec.get('include_labels')
    if include_labels and bool(spec.get('drop_records_with_excluded_labels', False)):
        events_path = root / str(spec.get('events_file', 'events.csv'))
        if events_path.exists():
            events = read_manifest_csv(events_path)
            events['record_id'] = events['record_id'].astype(str)
            label_map = {str(k): str(v) for k, v in (spec.get('label_map') or {}).items()}
            events['_mapped_label'] = events['label'].astype(str).map(lambda value: label_map.get(value, value))
            allowed_labels = {str(value) for value in include_labels}
            labels_by_record = events.groupby('record_id')['_mapped_label'].apply(set).to_dict()
            ids = {rid for rid in ids if not labels_by_record.get(rid) or labels_by_record[rid].issubset(allowed_labels)}
    return ids

def build_metrics(post_cfg, num_type_classes, type_class_names, compute_classification=True, label_groups=None, threshold=None, min_event_sec=None):
    return MetricsComputer(
        threshold=post_cfg['det_threshold'] if threshold is None else threshold,
        min_event_sec=post_cfg['min_event_sec'] if min_event_sec is None else float(min_event_sec),
        merge_gap_sec=post_cfg['merge_gap_sec'],
        num_classes=num_type_classes,
        class_names=type_class_names,
        max_event_sec=post_cfg.get('max_event_sec'),
        compute_classification=compute_classification,
        label_groups=label_groups,
        off_threshold=post_cfg.get('off_threshold'),
        smooth_sec=float(post_cfg.get('smooth_sec', 0.0) or 0.0),
        use_boundary_offsets=bool(post_cfg.get('use_boundary_offsets', False)),
        event_support_mode=str(post_cfg.get('event_support_mode', 'stride_center')),
        event_support_sec=post_cfg.get('event_support_sec'),
        event_iou_threshold=float(post_cfg.get('event_iou_threshold', 0.3)),
        event_gt_coverage_threshold=post_cfg.get('event_gt_coverage_threshold'),
    )

def threshold_sweep(pred_df, gt_events_df, cfg, num_type_classes, type_class_names, compute_classification=True, label_groups=None):
    thresholds = cfg['postprocess'].get('sweep_thresholds') or []
    min_event_values = cfg['postprocess'].get('sweep_min_event_sec') or [cfg['postprocess']['min_event_sec']]
    rows = []
    for threshold in thresholds:
        for min_event_sec in min_event_values:
            computer = build_metrics(cfg['postprocess'], num_type_classes, type_class_names, compute_classification, label_groups, float(threshold), float(min_event_sec))
            predicted_events = computer.decoder.decode(pred_df)
            event_scores = event_level_f1(predicted_events, gt_events_df, computer.event_iou_threshold,
                                         gt_coverage_thr=computer.event_gt_coverage_threshold)
            rows.append({'det_threshold': float(threshold), 'min_event_sec': float(min_event_sec),
                         **{f'event_{key}': event_scores[key] for key in ('f1', 'recall', 'precision')}})
    return pd.DataFrame(rows)

def type_confusion_matrix(pred_df, type_class_names):
    n = len(type_class_names)
    cm = torch.zeros((n, n), dtype=torch.long)
    if pred_df.empty or n == 0:
        return pd.DataFrame(cm.numpy(), index=type_class_names, columns=type_class_names)
    part = pred_df[(pred_df['det_true'].astype(int) == 1) & (pred_df['cls_true'].astype(int) >= 0)]
    for true_id, pred_id in zip(part['cls_true'].astype(int), part['cls_pred'].astype(int)):
        if 0 <= true_id < n and 0 <= pred_id < n:
            cm[true_id, pred_id] += 1
    return pd.DataFrame(cm.numpy(), index=type_class_names, columns=type_class_names)

def maybe_update_top_checkpoints(model, cfg, domain_to_id, out, records, epoch, score, metric, top_k):
    if top_k <= 0:
        return records
    score = float(score)
    if len(records) >= top_k and score <= min(item['score'] for item in records):
        return records
    ckpt_dir = out / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'epoch_{int(epoch):03d}.pt'
    torch.save({
        'model_state': model.state_dict(),
        'config': cfg,
        'domain_to_id': domain_to_id,
        'epoch': int(epoch),
        'score': score,
        'selection_metric': metric,
    }, path)
    records = [*records, {'epoch': int(epoch), 'score': score, 'metric': metric, 'path': str(path)}]
    records = sorted(records, key=lambda item: item['score'], reverse=True)
    for item in records[top_k:]:
        stale = Path(item['path'])
        if stale.exists():
            stale.unlink()
    records = records[:top_k]
    (out / 'top_checkpoints.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records

def average_prediction_frames(pred_frames):
    if not pred_frames:
        return pd.DataFrame()
    out = pred_frames[0].copy()
    for col in ['det_prob', 'pred_delta_on', 'pred_delta_off']:
        if col in out.columns:
            out[col] = sum(frame[col].astype(float).to_numpy() for frame in pred_frames) / len(pred_frames)
    cls_prob_cols = sorted([c for c in out.columns if c.startswith('cls_prob_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
    if cls_prob_cols:
        probs = sum(frame[cls_prob_cols].astype(float).to_numpy() for frame in pred_frames) / len(pred_frames)
        out.loc[:, cls_prob_cols] = probs
        out['cls_pred'] = probs.argmax(axis=1).astype(int)
    return out

def predict_checkpoint_ensemble(model, checkpoint_paths, loader, device, metrics_computer, gt_events_df, *, compute_metrics=True):
    pred_frames = []
    for path in checkpoint_paths:
        payload = load_torch_checkpoint(path, map_location=device)
        model.load_state_dict(payload['model_state'])
        _, pred_df = predict_epoch(model, loader, device, metrics_computer, gt_events_df, compute_metrics=False)
        pred_frames.append(pred_df)
    pred_df = average_prediction_frames(pred_frames)
    values = metrics_computer.compute(pred_df, gt_events_df) if compute_metrics else {}
    return values, pred_df

def maybe_build_external_datasets(cfg, domain_to_id, type_class_names):
    datasets = []
    domain_level = str(cfg['data'].get('domain_level', 'patient')).lower()
    for spec in build_external_specs(cfg):
        root = Path(spec['root'])
        if not (root / 'records.csv').exists():
            raise FileNotFoundError(f'External records manifest does not exist: {root / "records.csv"}')
        task = str(spec.get('task', 'multiclass')).lower()
        fixed_domain_label = 0 if domain_level == 'dataset' else None
        split_record_ids = sorted(filtered_record_ids_for_spec(spec))
        external_records = read_manifest_csv(root / 'records.csv')
        external_records['record_id'] = external_records['record_id'].astype(str)
        external_records['patient_id'] = external_records['patient_id'].astype(str)
        external_records = add_metadata_filter_columns(external_records)
        external_records = external_records[external_records['record_id'].isin(split_record_ids)].copy()
        record_to_domain = {}
        patient_to_domain = {}
        if domain_level == 'patient':
            patient_to_domain = {pid: int(domain_to_id.get(pid, 0)) for pid in external_records['patient_id'].unique()}
        elif domain_level != 'dataset':
            if domain_level not in external_records.columns:
                raise ValueError(f'External records do not contain domain field {domain_level!r}.')
            for row in external_records[['record_id', domain_level]].drop_duplicates('record_id').itertuples(index=False):
                value = str(getattr(row, domain_level) or 'unknown')
                record_to_domain[str(row.record_id)] = int(domain_to_id.get(value, domain_to_id.get('unknown', 0)))
        ds = EEGWindowDataset(
            str(root),
            cfg['data']['window_sec'],
            cfg['data']['stride_sec'],
            cfg['data']['positive_overlap_ratio'],
            positive_event_coverage_ratio=cfg['data'].get('positive_event_coverage_ratio'),
            split_record_ids=split_record_ids,
            patient_to_domain=patient_to_domain,
            record_to_domain=record_to_domain,
            type_class_names=type_class_names,
            label_map=spec.get('label_map'),
            include_labels=spec.get('include_labels'),
            drop_records_with_excluded_labels=bool(spec.get('drop_records_with_excluded_labels', False)),
            allow_unknown_positive_labels=(task == 'detection'),
            fixed_domain_label=fixed_domain_label,
            events_filename=str(spec.get('events_file', 'events.csv')),
            window_normalization=str(cfg['data'].get('window_normalization', 'channel_zscore')),
        )
        datasets.append({'name': str(spec.get('name', root.name)), 'task': task, 'dataset': ds, 'label_groups': spec.get('label_groups')})
    return datasets

def dataset_kwargs_from_spec(spec, task):
    spec_task = str(spec.get('task', task)).lower()
    return {
        'label_map': spec.get('label_map'),
        'include_labels': spec.get('include_labels'),
        'drop_records_with_excluded_labels': bool(spec.get('drop_records_with_excluded_labels', False)),
        'allow_unknown_positive_labels': bool(spec.get('allow_unknown_positive_labels', spec_task == 'detection')),
        'events_filename': str(spec.get('events_file', 'events.csv')),
    }

def read_filtered_records_for_output(root, record_ids, name):
    df = read_manifest_csv(Path(root) / 'records.csv')
    df['record_id'] = df['record_id'].astype(str)
    df['patient_id'] = df['patient_id'].astype(str)
    out = df[df['record_id'].isin(record_ids)].copy()
    out.insert(0, 'train_set', name)
    return out

def source_record_split(spec, train_cfg, records_df, allowed_ids):
    """Use a frozen patient-disjoint split when supplied by the study planner."""
    train_csv, val_csv = spec.get('train_record_ids_csv'), spec.get('val_record_ids_csv')
    if bool(train_csv) != bool(val_csv):
        raise ValueError('train_record_ids_csv and val_record_ids_csv must be specified together.')
    root = Path(spec['root'])
    if train_csv:
        train_ids = record_ids_from_csv(resolve_spec_csv(root, train_csv))
        val_ids = record_ids_from_csv(resolve_spec_csv(root, val_csv))
        if not train_ids or not val_ids:
            raise ValueError('Frozen source train and validation splits must both contain records.')
        excluded = (train_ids | val_ids) - set(allowed_ids)
        if excluded:
            raise ValueError(f'Frozen split contains {len(excluded)} records outside the eligible source cohort.')
        if train_ids & val_ids:
            raise ValueError('Frozen source train and validation record IDs overlap.')
    else:
        train_ids, val_ids, _ = build_patient_split(
            str(root / 'records.csv'), int(train_cfg['patient_fold']),
            int(train_cfg['num_folds']), str(train_cfg.get('split_strategy', 'stratified')),
            eligible_record_ids=allowed_ids, events_csv=str(root / str(spec.get('events_file', 'events.csv'))),
        )
    train_patients = set(records_df.loc[records_df['record_id'].isin(train_ids), 'patient_id'])
    val_patients = set(records_df.loc[records_df['record_id'].isin(val_ids), 'patient_id'])
    if train_patients & val_patients:
        raise ValueError('Source training and validation contain records from the same patient.')
    return sorted(train_ids), sorted(val_ids)

def build_multi_train_val(cfg, type_class_names, compute_classification):
    train_specs = build_train_specs(cfg)
    domain_level = str(cfg['data'].get('domain_level', 'patient')).lower()
    all_patient_ids = []
    domain_records = []
    split_items = []
    for spec in train_specs:
        root = str(spec['root'])
        allowed_ids = filtered_record_ids_for_spec(spec)
        records_df = read_manifest_csv(Path(root) / 'records.csv')
        records_df['record_id'] = records_df['record_id'].astype(str)
        records_df['patient_id'] = records_df['patient_id'].astype(str)
        train_ids, val_ids = source_record_split(spec, cfg['train'], records_df, allowed_ids)
        filtered_records = records_df[records_df['record_id'].isin(set(train_ids) | set(val_ids))]
        filtered_records = add_metadata_filter_columns(filtered_records)
        filtered_records.insert(0, '_train_set_name', str(spec.get('name', Path(root).name)))
        domain_records.append(filtered_records)
        all_patient_ids.extend(filtered_records['patient_id'].astype(str).unique().tolist())
        split_items.append({'spec': spec, 'train_ids': train_ids, 'val_ids': val_ids, 'allowed_ids': sorted(allowed_ids)})
    if domain_level == 'dataset':
        domain_to_id = {str(spec.get('name', Path(str(spec['root'])).name)): i for i, spec in enumerate(train_specs)}
        patient_to_domain = {}
    elif domain_level == 'patient':
        patients = sorted({str(p) for p in all_patient_ids})
        patient_to_domain = {p: i for i, p in enumerate(patients)}
        domain_to_id = patient_to_domain
    else:
        domain_df = pd.concat(domain_records, ignore_index=True) if domain_records else pd.DataFrame()
        domain_df = add_metadata_filter_columns(domain_df)
        if domain_level not in domain_df.columns:
            raise ValueError(f"data.domain_level={domain_level!r} is not available in records.csv. Use dataset, patient, sex, age_group, or another records.csv column.")
        domain_df[domain_level] = domain_df[domain_level].fillna('unknown').astype(str).replace({'': 'unknown'})
        domain_values = sorted(domain_df[domain_level].unique().tolist())
        domain_to_id = {value: i for i, value in enumerate(domain_values)}
        patient_to_domain = {}

    record_to_domain = {}
    if domain_level not in {'dataset', 'patient'}:
        domain_df = pd.concat(domain_records, ignore_index=True) if domain_records else pd.DataFrame()
        domain_df = add_metadata_filter_columns(domain_df)
        for row in domain_df[['record_id', domain_level]].drop_duplicates('record_id').itertuples(index=False):
            domain_value = str(getattr(row, domain_level) or 'unknown')
            record_to_domain[str(row.record_id)] = int(domain_to_id.get(domain_value, domain_to_id.get('unknown', 0)))

    train_datasets, val_datasets = [], []
    train_records_out, val_records_out = [], []
    max_neg = cfg['data'].get('max_negative_windows_per_record')
    negative_pos_ratio = cfg['data'].get('negative_pos_ratio')
    negative_only_record_windows = cfg['data'].get('negative_only_record_windows')
    val_max_neg = cfg['data'].get('val_max_negative_windows_per_record')
    val_negative_pos_ratio = cfg['data'].get('val_negative_pos_ratio')
    val_negative_only_record_windows = cfg['data'].get('val_negative_only_record_windows')
    task = 'multiclass' if compute_classification else 'detection'
    for item in split_items:
        spec = item['spec']
        root = str(spec['root'])
        kwargs = dataset_kwargs_from_spec(spec, task)
        name = str(spec.get('name', Path(root).name))
        fixed_domain_label = domain_to_id[name] if domain_level == 'dataset' else None
        train_dataset = EEGWindowDataset(
            root,
            cfg['data']['window_sec'],
            cfg['data']['stride_sec'],
            cfg['data']['positive_overlap_ratio'],
            positive_event_coverage_ratio=cfg['data'].get('positive_event_coverage_ratio'),
            split_record_ids=item['train_ids'],
            patient_to_domain=patient_to_domain,
            record_to_domain=record_to_domain,
            type_class_names=type_class_names,
            max_negative_windows_per_record=max_neg,
            negative_pos_ratio=negative_pos_ratio,
            negative_only_record_windows=negative_only_record_windows,
            fixed_domain_label=fixed_domain_label,
            window_normalization=str(cfg['data'].get('window_normalization', 'channel_zscore')),
            **kwargs,
        )
        val_dataset = EEGWindowDataset(
            root,
            cfg['data']['window_sec'],
            cfg['data']['stride_sec'],
            cfg['data']['positive_overlap_ratio'],
            positive_event_coverage_ratio=cfg['data'].get('positive_event_coverage_ratio'),
            split_record_ids=item['val_ids'],
            patient_to_domain=patient_to_domain,
            record_to_domain=record_to_domain,
            type_class_names=type_class_names,
            max_negative_windows_per_record=val_max_neg,
            negative_pos_ratio=val_negative_pos_ratio,
            negative_only_record_windows=val_negative_only_record_windows,
            fixed_domain_label=fixed_domain_label,
            window_normalization=str(cfg['data'].get('window_normalization', 'channel_zscore')),
            **kwargs,
        )
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        actual_train_ids = {str(sample.record_id) for sample in train_dataset.samples}
        actual_val_ids = {str(sample.record_id) for sample in val_dataset.samples}
        train_records_out.append(read_filtered_records_for_output(root, actual_train_ids, name))
        val_records_out.append(read_filtered_records_for_output(root, actual_val_ids, name))
    train_ds = train_datasets[0] if len(train_datasets) == 1 else CombinedEEGWindowDataset(train_datasets)
    val_ds = val_datasets[0] if len(val_datasets) == 1 else CombinedEEGWindowDataset(val_datasets)
    train_records_df = pd.concat(train_records_out, ignore_index=True) if train_records_out else pd.DataFrame()
    val_records_df = pd.concat(val_records_out, ignore_index=True) if val_records_out else pd.DataFrame()
    if set(train_records_df['patient_id']) & set(val_records_df['patient_id']):
        raise ValueError('Source training/validation patient overlap across dataset specifications.')
    return train_ds, val_ds, domain_to_id, train_records_df, val_records_df

def is_detection_only(cfg):
    task = str(cfg['data'].get('task', '')).lower()
    if task == 'detection':
        return True
    return not bool(cfg['data'].get('compute_classification', True))

def resolve_selection_metric(cfg):
    metric = str(cfg.get('train', {}).get('selection_metric', 'val_event_f1'))
    return metric if metric.startswith('val_') else f'val_{metric}'

def fit_and_save_calibration(predictions, out, prefix, bins=15):
    temperature = fit_binary_temperature(predictions['det_true'], predictions['det_prob'])
    calibrated = calibrate_prediction_frame(predictions, temperature)
    raw_metrics, _ = calibration_metrics(predictions['det_true'], predictions['det_prob'], bins)
    calibrated_metrics, reliability = calibration_metrics(calibrated['det_true'], calibrated['det_prob'], bins)
    payload = {'temperature': temperature, 'fit_split': 'source_validation', 'raw': raw_metrics, 'calibrated': calibrated_metrics}
    (out / f'{prefix}_temperature.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    calibrated.to_csv(out / f'{prefix}_val_window_predictions_calibrated.csv', index=False)
    reliability.to_csv(out / f'{prefix}_val_reliability_calibrated.csv', index=False)
    return calibrated

def split_viability_table(train_ds, val_ds, type_class_names, compute_classification):
    rows = []
    for split_name, dataset in [('train', train_ds), ('val', val_ds)]:
        positive = [sample for sample in dataset.samples if int(sample.det_label) == 1]
        rows.append({
            'split': split_name,
            'class_name': 'ANY_SEIZURE',
            'positive_windows': len(positive),
            'patients': len({str(sample.patient_id) for sample in positive}),
            'records': len({str(sample.record_id) for sample in positive}),
        })
        if compute_classification:
            for class_id, class_name in enumerate(type_class_names):
                selected = [sample for sample in positive if int(sample.cls_label) == class_id]
                rows.append({
                    'split': split_name,
                    'class_name': str(class_name),
                    'positive_windows': len(selected),
                    'patients': len({str(sample.patient_id) for sample in selected}),
                    'records': len({str(sample.record_id) for sample in selected}),
                })
    return pd.DataFrame(rows)

def save_validation_results(predictions, events, cfg, out, prefix, metrics, class_names, compute_classification):
    """Finalize source-only operating points once, after checkpoint selection."""
    predictions.to_csv(out / f'{prefix}_val_window_predictions.csv', index=False)
    values = metrics.compute(predictions, events)
    pd.DataFrame([values]).to_csv(out / f'{prefix}_val_metrics.csv', index=False)
    (out / f'{prefix}_val_metrics.json').write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics.decoder.decode(predictions).to_csv(out / f'{prefix}_val_events.csv', index=False)
    if compute_classification:
        type_confusion_matrix(predictions, class_names).to_csv(out / f'{prefix}_val_type_confusion_matrix.csv')
    operating = fit_and_save_calibration(predictions, out, prefix, int(cfg['postprocess'].get('calibration_bins', 15))) if bool(cfg['postprocess'].get('temperature_scaling', True)) else predictions
    sweep = threshold_sweep(operating, events, cfg, len(class_names), class_names, compute_classification)
    if sweep.empty:
        return
    sweep.to_csv(out / f'{prefix}_val_threshold_sweep.csv', index=False)
    selected = sweep.sort_values(['event_f1', 'event_recall', 'event_precision'], ascending=False).iloc[0]
    threshold, duration = float(selected['det_threshold']), float(selected['min_event_sec'])
    tuned = build_metrics(cfg['postprocess'], len(class_names), class_names, compute_classification,
                          threshold=threshold, min_event_sec=duration)
    values = tuned.compute(operating, events)
    values.update(selected_det_threshold=threshold, selected_min_event_sec=duration)
    pd.DataFrame([values]).to_csv(out / f'{prefix}_val_metrics_tuned_threshold.csv', index=False)
    (out / f'{prefix}_val_metrics_tuned_threshold.json').write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    tuned.decoder.decode(operating).to_csv(out / f'{prefix}_val_events_tuned_threshold.csv', index=False)

def enforce_split_viability(table, cfg, compute_classification):
    train_cfg = cfg.get('train', {})
    min_train = int(train_cfg.get('min_train_patients_per_class', 1))
    min_val = int(train_cfg.get('min_val_patients_per_class', 1))
    classes = set(table['class_name']) if compute_classification else {'ANY_SEIZURE'}
    failures = []
    for class_name in sorted(classes):
        for split_name, minimum in [('train', min_train), ('val', min_val)]:
            row = table[(table['split'] == split_name) & (table['class_name'] == class_name)]
            patients = int(row.iloc[0]['patients']) if not row.empty else 0
            windows = int(row.iloc[0]['positive_windows']) if not row.empty else 0
            if patients < minimum or windows == 0:
                failures.append(f'{split_name}/{class_name}: patients={patients}, windows={windows}, required_patients={minimum}')
    if failures:
        raise ValueError('Patient split is not class viable: ' + '; '.join(failures))

def main():
    args = parse_args(); cfg = load_config(args.config); apply_overrides(cfg, args.set); validate_config(cfg, mode='train'); set_seed(int(cfg['seed']))
    if args.fold is not None:
        cfg['train']['patient_fold'] = int(args.fold)
    out = resolve_output_dir(args, cfg)
    backup = prepare_run_output_dir(out, bool(args.overwrite))
    if backup is not None:
        print(json.dumps({'archived_previous_run': str(backup)}, ensure_ascii=False))
    (out / 'used_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    (out / 'experiment_summary.json').write_text(json.dumps({
        'config': args.config,
        'output_dir': str(out),
        'experiment': cfg.get('experiment', {}),
        'model_architecture': cfg.get('model', {}).get('architecture', 'gdr_net'),
        'task': cfg.get('data', {}).get('task'),
        'train_sets': [spec.get('name', spec.get('root')) for spec in build_train_specs(cfg)],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    type_class_names = cfg['data'].get('type_class_names', ['ABSZ', 'TNSZ', 'TCSZ'])
    compute_classification = not is_detection_only(cfg)
    train_ds, val_ds, domain_to_id, train_records_df, val_records_df = build_multi_train_val(cfg, type_class_names, compute_classification)
    train_records_df.to_csv(out / 'train_records.csv', index=False)
    val_records_df.to_csv(out / 'val_records.csv', index=False)
    pd.DataFrame({'patient_id': sorted(train_records_df['patient_id'].astype(str).unique())}).to_csv(out / 'train_patients.csv', index=False)
    pd.DataFrame({'patient_id': sorted(val_records_df['patient_id'].astype(str).unique())}).to_csv(out / 'val_patients.csv', index=False)
    viability = split_viability_table(train_ds, val_ds, type_class_names, compute_classification)
    enforce_split_viability(viability, cfg, compute_classification)
    max_neg = cfg['data'].get('max_negative_windows_per_record')
    negative_pos_ratio = cfg['data'].get('negative_pos_ratio')
    negative_only_record_windows = cfg['data'].get('negative_only_record_windows')
    num_workers = cfg['data'].get('num_workers')
    if num_workers is None:
        num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = bool(cfg['data'].get('pin_memory', device.type == 'cuda')) and device.type == 'cuda'
    balance_domains = bool(cfg['train'].get('balance_domains', False))
    train_sampler = build_train_sampler(train_ds, cfg, compute_classification)
    loader_options = dict(persistent_workers=bool(cfg['data'].get('persistent_workers', True)),
                          prefetch_factor=int(cfg['data'].get('prefetch_factor', 4)))
    eval_batch_size = int(cfg['data'].get('eval_batch_size', cfg['data']['batch_size']))
    train_loader = build_loader(train_ds, cfg['data']['batch_size'], num_workers, True, pin_memory, train_sampler, int(cfg['seed']), **loader_options)
    val_loader = build_loader(val_ds, eval_batch_size, num_workers, False, pin_memory, seed=int(cfg['seed']) + 1, **loader_options)
    print(json.dumps({'device': str(device), 'train_records': int(len(train_records_df)), 'val_records': int(len(val_records_df)), 'train_patients': len(set(s.patient_id for s in train_ds.samples)), 'val_patients': len(set(s.patient_id for s in val_ds.samples)), 'train_windows': len(train_ds), 'val_windows': len(val_ds), 'domain_level': cfg['data'].get('domain_level', 'patient'), 'num_domains': len(domain_to_id), 'patient_independent': True}, ensure_ascii=False))
    pos_count = sum(s.det_label for s in train_ds.samples); neg_count = max(len(train_ds.samples) - pos_count, 1)
    pos_weight = resolve_det_pos_weight(cfg, train_sampler, pos_count, neg_count)
    if len(train_records_df) + len(val_records_df) < int(cfg['data'].get('min_expected_records', 100)):
        print(json.dumps({'warning': 'Processed training dataset has very few records. Double-check preprocessing and split settings before reporting results.', 'processed_records': int(len(train_records_df) + len(val_records_df))}, ensure_ascii=False))
    print(json.dumps({
        'train_positive_windows': int(pos_count),
        'train_negative_windows': int(neg_count),
        'train_positive_rate': float(pos_count / max(pos_count + neg_count, 1)),
        'negative_pos_ratio': negative_pos_ratio,
        'negative_only_record_windows': negative_only_record_windows,
        'max_negative_windows_per_record': max_neg,
        'weighted_sampling': bool(train_sampler is not None),
        'sampling_strategy': str(cfg.get('train', {}).get('sampling_strategy', 'detection')),
        'balance_domains': balance_domains,
        'det_pos_weight': float(pos_weight),
        'configured_det_pos_weight': cfg.get('loss', {}).get('det_pos_weight'),
        'train_batch_size': cfg['data']['batch_size'], 'eval_batch_size': eval_batch_size,
        'num_workers_per_loader': num_workers, **loader_options,
    }, ensure_ascii=False))
    model = build_model_from_config(cfg, len(domain_to_id), len(type_class_names)).to(device)
    cls_class_weight = resolve_cls_class_weight(cfg, train_ds, len(type_class_names), compute_classification)
    criterion = Criterion(build_loss_weights(cfg['loss']), pos_weight, cls_class_weight)
    if cls_class_weight is not None:
        print(json.dumps({'cls_class_weight': [float(x) for x in cls_class_weight.tolist()]}, ensure_ascii=False))
    optimizer = build_optimizer(model, cfg)
    ema_decay = float(cfg['train'].get('ema_decay', 0.0))
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    metrics = build_metrics(cfg['postprocess'], len(type_class_names), type_class_names, compute_classification=compute_classification)
    best_val = -1.0; history = []; no_improve_epochs = 0
    early_stop_patience = int(cfg['train'].get('early_stop_patience', 0) or 0)
    early_stop_min_epochs = int(cfg['train'].get('early_stop_min_epochs', 1) or 1)
    selection_metric = resolve_selection_metric(cfg)
    checkpoint_start_epoch = int(cfg['train'].get('checkpoint_start_epoch', 1))
    use_counterfactual = bool(getattr(model, 'requires_pairs', False)) or criterion.weights.rho_cf > 0 or criterion.weights.sigma_cf_cls > 0
    pairing_task = 'subtype' if compute_classification else 'detection'
    pairing_generator = torch.Generator().manual_seed(int(cfg['seed']) + 104729)
    checkpoint_top_k = int(cfg['train'].get('checkpoint_top_k', 0) or 0)
    ensemble_top_k = int(cfg['train'].get('ensemble_top_k', 0) or 0)
    top_checkpoints = []
    top_predictions = {}  # Only probability columns for the retained ensemble members.
    best_predictions = None
    validate_before_checkpoint = bool(cfg['train'].get('validate_before_checkpoint', False))
    for epoch in range(1, int(cfg['train']['epochs']) + 1):
        epoch_start = perf_counter()
        print(json.dumps({'epoch': epoch, 'phase': 'training'}), flush=True)
        recombination_scale = auxiliary_warmup_scale(
            epoch,
            int(cfg['train'].get('recombination_start_epoch', 1)),
            int(cfg['train'].get('recombination_warmup_epochs', 0)),
        )
        criterion.set_auxiliary_scales(recombination=recombination_scale)
        model.train(); loss_sums = {}; pair_stats = {
            'anchors': 0, 'valid': 0, 'cls_valid': 0,
            'cross_domain': 0, 'same_domain': 0,
        }
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            if use_counterfactual:
                pairs = build_counterfactual_pairs(batch, pairing_task, pairing_generator)
                batch['cf_pair_index'] = pairs.pair_index
                batch['cf_mask'] = pairs.mask
                batch['cf_cls_mask'] = pairs.cls_mask
                batch['cf_has_pairs'] = bool(pairs.mask.any())
                batch['cf_has_cls_pairs'] = bool((pairs.cls_mask.bool() & (batch['det_label'] > .5) & (batch['cls_label'] >= 0)).any())
                pair_stats['anchors'] += int(len(pairs.mask))
                pair_stats['valid'] += int(pairs.mask.sum().item())
                pair_stats['cls_valid'] += int(pairs.cls_mask.sum().item())
                valid = pairs.mask.bool()
                if valid.any():
                    anchor_domain = batch['domain_label'].long()
                    donor_domain = anchor_domain[pairs.pair_index]
                    cross = donor_domain != anchor_domain
                    pair_stats['cross_domain'] += int(cross[valid].sum().item())
                    pair_stats['same_domain'] += int((~cross[valid]).sum().item())
                    for domain in donor_domain[valid].unique().tolist():
                        key = f'donor_domain_{int(domain)}'
                        pair_stats[key] = pair_stats.get(key, 0) + int((donor_domain[valid] == int(domain)).sum().item())
                class_keys = batch['cls_label'].long() if compute_classification else batch['det_label'].long()
                for class_id in class_keys.unique().tolist():
                    anchors = class_keys == int(class_id)
                    pair_stats[f'class_{int(class_id)}_anchors'] = pair_stats.get(f'class_{int(class_id)}_anchors', 0) + int(anchors.sum().item())
                    pair_stats[f'class_{int(class_id)}_valid'] = pair_stats.get(f'class_{int(class_id)}_valid', 0) + int(pairs.mask[anchors].sum().item())
            batch = batch_to_device(batch, device); optimizer.zero_grad(set_to_none=True)
            output = model(batch['x'], batch.get('cf_pair_index')); loss_out = criterion(output, batch)
            if cfg['train'].get('fail_on_nonfinite', False) and not torch.isfinite(loss_out.total):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}; run is incomplete.')
            loss_out.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['train']['grad_clip']),
                error_if_nonfinite=bool(cfg['train'].get('fail_on_nonfinite', False)))
            if cfg['train'].get('fail_on_nonfinite', False):
                loss_out.items['grad_norm'] = grad_norm.detach()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            for k, v in loss_out.items.items(): loss_sums[k] = loss_sums.get(k, 0.0) + v
        train_values = {f'train_{k}': float(v / max(len(train_loader), 1)) for k, v in loss_sums.items()}
        train_seconds = perf_counter() - epoch_start
        evaluation_model = ema.model if ema is not None else model
        checkpoint_eligible = epoch >= checkpoint_start_epoch
        run_validation = checkpoint_eligible or validate_before_checkpoint
        validation_start = perf_counter()
        val_metrics, val_pred = ({}, None)
        if run_validation:
            print(json.dumps({'epoch': epoch, 'phase': 'validation', 'train_seconds': train_seconds}), flush=True)
            val_metrics, val_pred = predict_epoch(evaluation_model, val_loader, device, metrics, val_ds.events,
                window_only=(selection_metric != 'val_event_f1'))
        validation_seconds = perf_counter() - validation_start if run_validation else 0.0
        pair_metrics = {}
        if use_counterfactual:
            pair_metrics = {
                'cf_valid_rate': pair_stats['valid'] / max(pair_stats['anchors'], 1),
                'cf_cls_valid_rate': pair_stats['cls_valid'] / max(pair_stats['anchors'], 1),
                'cf_cross_domain_rate': pair_stats['cross_domain'] / max(pair_stats['valid'], 1),
                'cf_same_domain_rate': pair_stats['same_domain'] / max(pair_stats['valid'], 1),
            }
            for key, value in pair_stats.items():
                if key.startswith('donor_domain_'):
                    domain = key.removeprefix('donor_domain_')
                    pair_metrics[f'cf_donor_domain_{domain}_rate'] = value / max(pair_stats['valid'], 1)
                elif key.startswith('class_') and key.endswith('_anchors'):
                    class_id = key.removeprefix('class_').removesuffix('_anchors')
                    valid_count = pair_stats.get(f'class_{class_id}_valid', 0)
                    pair_metrics[f'cf_valid_class_{class_id}_rate'] = valid_count / max(value, 1)
        rec = {
            'epoch': epoch, 'recombination_scale': recombination_scale, 'ema_updates': ema.updates if ema else 0,
            'validation_ran': run_validation, 'train_seconds': train_seconds,
            'validation_seconds': validation_seconds,
            **train_values,
            **{f'train_{k}': v for k, v in pair_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        if not run_validation:
            history.append(rec)
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            continue
        if selection_metric not in rec:
            raise KeyError(f'Selection metric {selection_metric!r} was not produced by validation.')
        current_score = float(rec[selection_metric])
        if not math.isfinite(current_score):
            raise FloatingPointError(f'Nonfinite validation score at epoch {epoch}; no checkpoint was selected.')
        if checkpoint_eligible:
            top_checkpoints = maybe_update_top_checkpoints(evaluation_model, cfg, domain_to_id, out, top_checkpoints, epoch, current_score, selection_metric, checkpoint_top_k)
        if checkpoint_eligible and ensemble_top_k > 1:
            retained = {item['epoch'] for item in top_checkpoints[:ensemble_top_k]}
            probability_columns = ['det_prob', *[f'cls_prob_{i}' for i in range(len(type_class_names))]]
            if epoch in retained:
                top_predictions[epoch] = val_pred[probability_columns].copy()
            top_predictions = {key: value for key, value in top_predictions.items() if key in retained}
        improved = checkpoint_eligible and current_score > best_val
        if improved:
            best_val = current_score
            no_improve_epochs = 0
            best_predictions = val_pred
            torch.save({'model_state': evaluation_model.state_dict(), 'config': cfg, 'domain_to_id': domain_to_id}, out / 'best.pt')
        elif checkpoint_eligible:
            no_improve_epochs += 1
        history.append(rec)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if early_stop_patience > 0 and epoch >= early_stop_min_epochs and no_improve_epochs >= early_stop_patience:
            print(json.dumps({'early_stop': True, 'epoch': epoch, 'selection_metric': selection_metric, 'best_val_score': best_val, 'patience': early_stop_patience}, ensure_ascii=False), flush=True)
            break
    if best_predictions is None:
        raise RuntimeError('No finite eligible checkpoint was selected.')
    postprocess_start = perf_counter()
    print('Finalizing source validation: best checkpoint and ensemble.', flush=True)
    save_validation_results(best_predictions, val_ds.events, cfg, out, 'best', metrics, type_class_names, compute_classification)
    if ensemble_top_k > 1 and len(top_checkpoints) >= 2:
        selected_checkpoints = top_checkpoints[:ensemble_top_k]
        (out / 'ensemble_checkpoints.json').write_text(json.dumps(selected_checkpoints, ensure_ascii=False, indent=2), encoding='utf-8')
        # Validation is deterministic and unshuffled. Reuse the predictions that
        # selected these exact saved EMA/online checkpoints, rather than re-run them.
        averaged = average_prediction_frames([top_predictions[item['epoch']] for item in selected_checkpoints])
        ensemble_val_pred = best_predictions.copy()
        for column in averaged:
            ensemble_val_pred[column] = averaged[column].to_numpy()
        save_validation_results(ensemble_val_pred, val_ds.events, cfg, out, 'ensemble', metrics, type_class_names, compute_classification)
    postprocess_seconds = perf_counter() - postprocess_start
    print(json.dumps({'source_postprocess_seconds': postprocess_seconds}), flush=True)
    (out / 'training_complete.json').write_text(json.dumps({
        'epochs_completed': len(history), 'selection_metric': selection_metric,
        'best_val_score': best_val, 'ema_decay': ema_decay,
        'checkpoint_weights': 'ema' if ema else 'online',
        'source_postprocess_seconds': postprocess_seconds,
    }, indent=2), encoding='utf-8')
    print(f'Training complete. Best {selection_metric}={best_val:.4f}')
if __name__ == '__main__': main()
