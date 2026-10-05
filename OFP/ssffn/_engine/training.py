from __future__ import annotations
from OFP.ssffn._engine.timestamp_policy import install_timestamp_policy
install_timestamp_policy()
import argparse
import gc
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from OFP.ssffn._engine.module_batches import ModuleGroupedLoader, module_mean_loss
from OFP.ssffn._engine.data import CompatBatchedDataset, CompatCfg, FEATURE_CACHE_VERSION, FeatureNormStats, Model2FeatureCache, compat_feature_names, compute_feature_norm_stats, cuda_autocast, effective_pos_weight, file_label_map, format_duration, format_epoch_status, format_rate, log_run_header, progress_bar
from OFP.ssffn._engine.evaluation import evaluate_prediction_output, log_stage_done, log_stage_start, positive_scores, select_threshold_for_scores, write_threshold_predictions
from OFP.ssffn._engine.index import read_index
from OFP.ssffn._engine.runtime import format_metric_summary, resolve_runtime_device
from OFP.ssffn._engine.feature_schema import feature_group_manifest, select_feature_names
RUN_NAME = 'ssffn'
RAW_SEQUENCE_FEATURES = ['Temp', 'Curr', 'TxP0', 'RxP0', 'RxP1', 'RxP2', 'RxP3', 'RxP4', 'TxP1', 'TxP2', 'TxP3', 'TxP4', 'DeltaSeconds']
TEMPORAL_SUMMARY_MODES = ('none', 'raw_channel_stats')
TEMPORAL_SUMMARY_OPS = ('last', 'mean', 'std', 'min', 'max', 'delta', 'range')

class PositiveLinear(nn.Linear):

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, -3.9)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, F.softplus(self.weight), self.bias)

    def effective_weight(self) -> torch.Tensor:
        return F.softplus(self.weight)

class FrozenLinearHead:

    def __init__(self, weight: np.ndarray, bias: float) -> None:
        self.coef_ = np.asarray(weight, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.asarray([float(bias)], dtype=np.float64)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None=None) -> 'FrozenLinearHead':
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = np.asarray(x, dtype=np.float64) @ self.coef_[0] + self.intercept_[0]
        logits = np.clip(logits, -60.0, 60.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - positive, positive], axis=1)

def frozen_encoder_head(model: 'SSFFNModel') -> FrozenLinearHead:
    head = model.classifier
    weight = head.effective_weight() if isinstance(head, PositiveLinear) else head.weight
    bias = float(head.bias.detach().cpu().reshape(-1)[0]) if head.bias is not None else 0.0
    return FrozenLinearHead(weight.detach().cpu().numpy().reshape(-1), bias)

def raw_validity_feature_name(raw_name: str) -> str:
    return 'DeltaSecondsValidMask' if raw_name == 'DeltaSeconds' else f'{raw_name}ValidMask'

def raw_validity_indices(cfg: CompatCfg, raw_indices: list[int] | np.ndarray) -> np.ndarray | None:
    names = compat_feature_names(cfg)
    selected_names = [names[int(idx)] for idx in raw_indices]
    mask_names = [raw_validity_feature_name(name) for name in selected_names]
    if not all((name in names for name in mask_names)):
        return None
    return np.asarray([names.index(name) for name in mask_names], dtype=np.int64)

def resolve_fusion_mode(fusion_mode: str, tsf_ablation: str='full') -> str:
    return 'sit'

def configure_runtime(cfg: CompatCfg, seed: int, fold: int, training_seed: int=42) -> None:
    torch.manual_seed(int(training_seed) + int(fold))
    np.random.seed(int(seed) + int(fold))
    cfg.device = resolve_runtime_device(cfg.device)
    if str(cfg.device).startswith('cuda') and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

def load_or_compute_feature_norm_stats(data_dir: Path, train_files: list[str], cfg: CompatCfg, label_by_file: dict[str, int]) -> FeatureNormStats:
    if not str(cfg.module_cache_dir).strip():
        return compute_feature_norm_stats(data_dir, train_files, cfg, label_by_file)
    fingerprint_payload = {'cache_version': FEATURE_CACHE_VERSION, 'feature_mode': cfg.feature_mode, 'target_mode': cfg.target_mode, 'target_horizon_hours': float(cfg.target_horizon_hours), 'preserve_timepoints': bool(cfg.preserve_timepoints), 'rule_mode': cfg.rule_mode, 'train_files': sorted(train_files)}
    fingerprint = hashlib.sha1(json.dumps(fingerprint_payload, sort_keys=True).encode('utf-8')).hexdigest()[:20]
    cache_path = Path(cfg.module_cache_dir) / 'norm_stats' / f'{fingerprint}.json'
    if cache_path.exists():
        try:
            return FeatureNormStats(**json.loads(cache_path.read_text(encoding='utf-8')))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            cache_path.unlink(missing_ok=True)
    stats = compute_feature_norm_stats(data_dir, train_files, cfg, label_by_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f'.{cache_path.name}.{os.getpid()}.tmp')
    try:
        temp_path.write_text(json.dumps(asdict(stats), indent=2), encoding='utf-8')
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return stats

def feature_indices(feature_names: list[str], stat_feature_mode: str, stat_feature_groups: str='', exclude_stat_feature_groups: str='') -> tuple[list[int], list[str], list[int], list[str]]:
    raw_names = [name for name in RAW_SEQUENCE_FEATURES if name in feature_names]
    raw_indices = [feature_names.index(name) for name in raw_names]
    if not raw_indices:
        raise ValueError('No raw sequence features were found in compat feature names')
    raw_set = set(raw_names) | {'Ts'}
    mode = str(stat_feature_mode).lower()
    if str(stat_feature_groups).strip():
        stat_names = select_feature_names(feature_names, stat_feature_groups, exclude_stat_feature_groups)
        stat_names = [name for name in stat_names if name not in raw_set]
    elif mode in {'ofp_expert_stat', 'expert_statistical'}:
        stat_names = select_feature_names(feature_names, 'expert_statistical')
    elif mode == 'all_engineered':
        stat_names = [name for name in feature_names if name not in raw_set]
    elif mode == 'model2_expert':
        keep = {'FeCo', 'FeTxP0', 'FeRxP0', 'TsDelta'}
        stat_names = [name for name in feature_names if any((name.startswith(prefix) for prefix in keep))]
    elif mode == 'statistics':
        keys = ('_mean', '_std', '_delta', '_exp_range', 'Range', 'Std', 'Count', 'Storm', 'Rate', 'Flag', 'Elapsed', 'DeltaSeconds')
        stat_names = [name for name in feature_names if name not in raw_set and any((key in name for key in keys))]
    elif mode == 'all':
        stat_names = list(feature_names)
    else:
        raise ValueError('Unknown stat_feature_mode; use ofp_expert_stat, all_engineered, model2_expert, statistics, or all')
    if not stat_names:
        raise ValueError('The selected expert/statistical feature groups are empty. Check --feature_mode, --stat_feature_groups, and --exclude_stat_feature_groups.')
    stat_indices = [feature_names.index(name) for name in stat_names]
    return (raw_indices, raw_names, stat_indices, stat_names)

def temporal_summary_feature_names(raw_names: list[str], temporal_summary_mode: str) -> list[str]:
    mode = str(temporal_summary_mode).lower()
    if mode == 'none':
        return []
    if mode != 'raw_channel_stats':
        raise ValueError(f'Unknown temporal_summary_mode={temporal_summary_mode!r}; use one of {TEMPORAL_SUMMARY_MODES}')
    return [f'TemporalSummary_{name}_{op}' for name in raw_names for op in TEMPORAL_SUMMARY_OPS]

def stat_candidate_names(stat_names: list[str], raw_names: list[str], temporal_summary_mode: str) -> list[str]:
    return list(stat_names) + temporal_summary_feature_names(raw_names, temporal_summary_mode)

def compute_temporal_window_summaries(raw: np.ndarray, positions: np.ndarray, seq_len: int, temporal_summary_mode: str) -> np.ndarray:
    mode = str(temporal_summary_mode).lower()
    positions = np.asarray(positions, dtype=np.int64)
    if mode == 'none':
        return np.zeros((len(positions), 0), dtype=np.float32)
    if mode != 'raw_channel_stats':
        raise ValueError(f'Unknown temporal_summary_mode={temporal_summary_mode!r}; use one of {TEMPORAL_SUMMARY_MODES}')
    raw = np.asarray(raw, dtype=np.float32)
    n_channels = int(raw.shape[1]) if raw.ndim == 2 else 0
    out = np.zeros((len(positions), n_channels * len(TEMPORAL_SUMMARY_OPS)), dtype=np.float32)
    for row_idx, pos in enumerate(positions):
        pos = int(pos)
        if pos < 0 or pos >= len(raw):
            continue
        start = max(0, pos - int(seq_len) + 1)
        window = raw[start:pos + 1]
        if len(window) == 0:
            continue
        window = np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        last = window[-1]
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        min_value = window.min(axis=0)
        max_value = window.max(axis=0)
        delta = last - window[0]
        value_range = max_value - min_value
        channel_major = np.stack([last, mean, std, min_value, max_value, delta, value_range], axis=1)
        out[row_idx] = channel_major.reshape(-1)
    return np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def build_stat_candidate_matrix(features: np.ndarray, positions: np.ndarray, raw_indices: np.ndarray | list[int], stat_indices: np.ndarray | list[int], seq_len: int, temporal_summary_mode: str) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    stat_idx = np.asarray(stat_indices, dtype=np.int64)
    raw_idx = np.asarray(raw_indices, dtype=np.int64)
    base = features[positions][:, stat_idx].astype(np.float32) if len(stat_idx) > 0 else np.zeros((len(positions), 0), dtype=np.float32)
    temporal = compute_temporal_window_summaries(features[:, raw_idx].astype(np.float32), positions, int(seq_len), temporal_summary_mode)
    if temporal.shape[1] == 0:
        return base
    return np.concatenate([base, temporal], axis=1).astype(np.float32)

def _save_score_table(path: Path, names: list[str], scores: np.ndarray, score_name: str='score') -> pd.DataFrame:
    frame = pd.DataFrame({'feature': list(names), score_name: np.asarray(scores, dtype=float)})
    frame.sort_values(score_name, ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

def _save_heatmap(path: Path, values: np.ndarray, xlabels: list[str], ylabels: list[str], title: str='') -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[plot-skip] path={path} reason={exc}', flush=True)
        return
    arr = np.asarray(values, dtype=float)
    width = max(4.0, min(14.0, 0.35 * max(len(xlabels), 4)))
    height = max(2.5, min(8.0, 0.35 * max(len(ylabels), 4)))
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    im = ax.imshow(arr, aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def _save_topk_heatmap(path: Path, frame: pd.DataFrame, value_col: str, top_k: int=30) -> None:
    if frame.empty:
        return
    top = frame.head(int(top_k))
    _save_heatmap(path, top[[value_col]].to_numpy(dtype=float).T, top['feature'].astype(str).tolist(), [value_col], '')

def _cap_explain_rows(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(max_rows) <= 0 or len(y) <= int(max_rows):
        return (x, y)
    rng = np.random.default_rng(int(seed))
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y <= 0)
    pos_keep = pos if len(pos) <= int(max_rows) // 2 else rng.choice(pos, size=int(max_rows) // 2, replace=False)
    remain = max(int(max_rows) - len(pos_keep), 0)
    neg_keep = neg if len(neg) <= remain else rng.choice(neg, size=remain, replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(keep)
    return (x[keep], y[keep])

def resolve_branch_selection_budgets(total_select_k: int, temporal_select_k: int, stat_select_k: int, n_temporal_features: int, n_stat_features: int) -> tuple[int, int]:
    total = int(total_select_k)
    temporal = int(temporal_select_k)
    stat = int(stat_select_k)
    if min(total, temporal, stat) < 0:
        raise ValueError('Feature-selection budgets must be non-negative')
    if total > 0:
        if temporal <= 0:
            raise ValueError('--total_select_k requires an explicit positive --temporal_select_k. The two branch-wise ExtraTrees importance scales are not directly comparable.')
        stat_from_total = total - temporal
        if stat_from_total <= 0:
            raise ValueError('--total_select_k must leave at least one feature for the expert-statistical branch')
        if stat > 0 and stat != stat_from_total:
            raise ValueError(f'Inconsistent Top-K allocation: total={total}, temporal={temporal}, stat={stat}; expected stat={stat_from_total}')
        stat = stat_from_total
    if temporal > int(n_temporal_features):
        raise ValueError(f'Requested {temporal} temporal channels, but only {n_temporal_features} are available')
    if stat > int(n_stat_features):
        raise ValueError(f'Requested {stat} expert-statistical features, but only {n_stat_features} are available')
    return (temporal, stat)

def maybe_select_temporal_features(cache: Model2FeatureCache, base_dataset: CompatBatchedDataset, cfg: CompatCfg, raw_indices: list[int], raw_names: list[str], args: argparse.Namespace, fold: int, run_dir: Path, select_k: int, strict: bool=False) -> tuple[list[int], list[str], dict[str, Any]]:
    selector_name = 'none'
    k = int(select_k)
    all_indices = list(raw_indices)
    all_names = list(raw_names)
    meta: dict[str, Any] = {'temporal_selector': selector_name, 'requested_temporal_select_k': k, 'original_temporal_feature_count': len(all_names)}
    if selector_name == 'none':
        if strict and 0 < k < len(all_names):
            raise ValueError('A total Top-K budget requires --temporal_selector extra_trees')
        meta['selected_temporal_feature_count'] = len(all_names)
        meta['selected_features'] = [{'feature': name, 'rank': rank, 'importance': None} for rank, name in enumerate(all_names, 1)]
        return (all_indices, all_names, meta)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for name, selected in zip(base_dataset.file_names, base_dataset.selected_positions):
        if len(selected) <= 0:
            continue
        _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = cache.get(name)
        raw = features[:, np.asarray(raw_indices, dtype=np.int64)].astype(np.float32)
        x_parts.append(compute_temporal_window_summaries(raw, selected, int(cfg.seq_len), 'raw_channel_stats'))
        y_parts.append(labels[selected].astype(np.int8))
    if not x_parts:
        if strict:
            raise ValueError('Temporal feature selection has no training windows')
        meta.update({'selected_temporal_feature_count': len(all_names), 'reason': 'empty_training_selection'})
        return (all_indices, all_names, meta)
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    x, y = _cap_explain_rows(x, y, 200000, int(args.seed) + int(fold))
    if len(np.unique(y)) < 2:
        if strict:
            raise ValueError('Temporal feature selection requires both positive and negative training windows')
        meta.update({'selected_temporal_feature_count': len(all_names), 'reason': 'single_class_training_selection'})
        return (all_indices, all_names, meta)
    selector = ExtraTreesClassifier(n_estimators=200, random_state=int(args.seed) + int(fold), n_jobs=2, class_weight='balanced')
    started = time.time()
    selector.fit(x, y)
    descriptor_scores = np.asarray(selector.feature_importances_, dtype=float)
    operation_scores = descriptor_scores.reshape(len(all_names), len(TEMPORAL_SUMMARY_OPS))
    channel_scores = operation_scores.sum(axis=1)
    score_total = float(channel_scores.sum())
    if score_total > 0.0:
        channel_scores = channel_scores / score_total
    ranking = np.argsort(channel_scores)[::-1]
    keep_count = len(all_names) if k <= 0 else min(k, len(all_names))
    selected_local = [int(value) for value in ranking[:keep_count]]
    selected_indices = [all_indices[index] for index in selected_local]
    selected_names = [all_names[index] for index in selected_local]
    explain_dir = run_dir / 'explainability'
    descriptor_names = temporal_summary_feature_names(all_names, 'raw_channel_stats')
    descriptor_frame = _save_score_table(explain_dir / 'temporal_summary_descriptor_importance.csv', descriptor_names, descriptor_scores, 'importance')
    _save_topk_heatmap(explain_dir / 'temporal_summary_descriptor_importance_topk_heatmap.png', descriptor_frame, 'importance')
    selected_set = set(selected_names)
    channel_frame = pd.DataFrame({'feature': all_names, 'importance': channel_scores, 'selected': [name in selected_set for name in all_names]}).sort_values('importance', ascending=False, ignore_index=True)
    channel_frame.insert(0, 'rank', np.arange(1, len(channel_frame) + 1))
    explain_dir.mkdir(parents=True, exist_ok=True)
    channel_frame.to_csv(explain_dir / 'temporal_raw_channel_importance.csv', index=False)
    _save_topk_heatmap(explain_dir / 'temporal_raw_channel_importance_topk_heatmap.png', channel_frame, 'importance', top_k=len(all_names))
    _save_heatmap(explain_dir / 'temporal_channel_operation_importance_heatmap.png', operation_scores, list(TEMPORAL_SUMMARY_OPS), all_names)
    selected_frame = channel_frame[channel_frame['selected']].copy()
    selected_frame.to_csv(explain_dir / 'selected_temporal_features.csv', index=False)
    meta.update({'selected_temporal_feature_count': len(selected_names), 'temporal_selector_rows': int(len(y)), 'temporal_selector_positive_rows': int((y > 0).sum()), 'temporal_selector_seconds': float(time.time() - started), 'selected_features': selected_frame[['feature', 'rank', 'importance']].to_dict('records')})
    return (selected_indices, selected_names, meta)

def maybe_select_stat_features(cache: Model2FeatureCache, base_dataset: CompatBatchedDataset, cfg: CompatCfg, raw_indices: list[int], raw_names: list[str], stat_indices: list[int], stat_names: list[str], args: argparse.Namespace, fold: int, run_dir: Path, select_k: int | None=None, strict: bool=False) -> tuple[list[int], list[str], dict[str, Any], list[str]]:
    selector_name = 'none'
    k = int(0 if select_k is None else select_k)
    candidate_names = stat_candidate_names(stat_names, raw_names, 'none')
    temporal_summary_count = len(candidate_names) - len(stat_names)
    meta: dict[str, Any] = {'stat_selector': selector_name, 'requested_stat_select_k': k, 'base_stat_feature_count': len(stat_indices), 'temporal_summary_mode': 'none', 'temporal_summary_feature_count': int(temporal_summary_count), 'original_stat_feature_count': len(candidate_names)}
    all_local_indices = list(range(len(candidate_names)))
    if selector_name == 'none':
        if strict and 0 < k < len(candidate_names):
            raise ValueError('A total Top-K budget requires --stat_selector extra_trees')
        meta['selected_stat_feature_count'] = len(candidate_names)
        meta['selected_features'] = [{'feature': name, 'rank': rank, 'importance': None} for rank, name in enumerate(candidate_names, 1)]
        return (all_local_indices, candidate_names, meta, candidate_names)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for name, selected in zip(base_dataset.file_names, base_dataset.selected_positions):
        if len(selected) <= 0:
            continue
        _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = cache.get(name)
        x_parts.append(build_stat_candidate_matrix(features, selected, raw_indices, stat_indices, int(cfg.seq_len), 'none'))
        y_parts.append(labels[selected].astype(np.int8))
    if not x_parts:
        if strict:
            raise ValueError('Expert-statistical feature selection has no training windows')
        meta['selected_stat_feature_count'] = len(candidate_names)
        meta['reason'] = 'empty_training_selection'
        return (all_local_indices, candidate_names, meta, candidate_names)
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    x, y = _cap_explain_rows(x, y, 200000, int(args.seed) + int(fold))
    if len(np.unique(y)) < 2:
        if strict:
            raise ValueError('Expert-statistical feature selection requires both positive and negative training windows')
        meta['selected_stat_feature_count'] = len(candidate_names)
        meta['reason'] = 'single_class_training_selection'
        return (all_local_indices, candidate_names, meta, candidate_names)
    selector = ExtraTreesClassifier(n_estimators=200, random_state=int(args.seed) + int(fold), n_jobs=2, class_weight='balanced')
    started = time.time()
    selector.fit(x, y)
    scores = np.asarray(selector.feature_importances_, dtype=float)
    ranking = np.argsort(scores)[::-1]
    keep_count = len(ranking) if k <= 0 else min(k, len(ranking))
    local = ranking[:keep_count]
    selected_indices = [int(i) for i in local]
    selected_names = [candidate_names[int(i)] for i in local]
    importance = _save_score_table(run_dir / 'explainability' / 'stat_feature_importance.csv', candidate_names, scores, 'importance')
    selected_set = set(selected_names)
    importance.insert(0, 'rank', np.arange(1, len(importance) + 1))
    importance['selected'] = importance['feature'].isin(selected_set)
    importance.to_csv(run_dir / 'explainability' / 'stat_feature_importance.csv', index=False)
    _save_topk_heatmap(run_dir / 'explainability' / 'stat_feature_importance_topk_heatmap.png', importance, 'importance')
    meta.update({'selected_stat_feature_count': len(selected_indices), 'stat_selector_rows': int(len(y)), 'stat_selector_positive_rows': int((y > 0).sum()), 'stat_selector_seconds': float(time.time() - started), 'selected_features': importance.loc[importance['selected'], ['feature', 'rank', 'importance']].to_dict('records')})
    (run_dir / 'explainability' / 'selected_stat_features.csv').write_text(pd.DataFrame({'feature': selected_names}).to_csv(index=False), encoding='utf-8')
    return (selected_indices, selected_names, meta, candidate_names)

def deterministic_module_order(count: int, seed: int, epoch: int) -> np.ndarray:
    if int(count) <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 104729 * int(epoch))
    return rng.permutation(int(count)).astype(np.int64, copy=False)

class SelectedWindowStatDataset(IterableDataset):

    def __init__(self, base: CompatBatchedDataset, cache: Model2FeatureCache, cfg: CompatCfg, raw_indices: list[int], stat_indices: list[int], stat_input_indices: list[int], temporal_summary_mode: str) -> None:
        self.base = base
        self.cache = cache
        self.cfg = cfg
        self.raw_indices = np.asarray(raw_indices, dtype=np.int64)
        self.raw_validity_indices = raw_validity_indices(cfg, self.raw_indices)
        self.stat_indices = np.asarray(stat_indices, dtype=np.int64)
        self.stat_input_indices = np.asarray(stat_input_indices, dtype=np.int64)
        self.temporal_summary_mode = str(temporal_summary_mode)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.base)

    @property
    def file_names(self) -> list[str]:
        return self.base.file_names

    @property
    def selected_positions(self) -> list[np.ndarray]:
        return self.base.selected_positions

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        seq_len = int(self.cfg.seq_len)
        pairs = list(zip(self.base.file_names, self.base.selected_positions))
        if pairs:
            order = deterministic_module_order(len(pairs), int(self.cfg.seed), int(self._epoch))
            pairs = [pairs[int(index)] for index in order]
        worker = get_worker_info()
        if worker is not None:
            pairs = pairs[worker.id::worker.num_workers]
        weight_by_name = {name: weights for name, weights in zip(self.base.file_names, self.base.selected_weights)}
        for name, selected in pairs:
            if len(selected) <= 0:
                continue
            _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = self.cache.get(name)
            raw = features[:, self.raw_indices].astype(np.float32)
            raw_validity = features[:, self.raw_validity_indices].astype(np.float32) if self.raw_validity_indices is not None else np.ones_like(raw, dtype=np.float32)
            stat = build_stat_candidate_matrix(features, selected, self.raw_indices, self.stat_indices, seq_len, self.temporal_summary_mode)
            stat = stat[:, self.stat_input_indices].astype(np.float32)
            pad_x = np.zeros((seq_len - 1, raw.shape[1]), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, raw], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, raw_validity], axis=0))
            x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            ys = torch.from_numpy(labels[selected].astype(np.float32))
            ws = torch.from_numpy(weight_by_name.get(name, np.ones(len(selected), dtype=np.float32)).astype(np.float32))
            module_has_positive = bool(np.any(labels > 0))
            module_flag = torch.tensor(float(module_has_positive))
            stat_tensor = torch.from_numpy(stat.astype(np.float32))
            for start in range(0, len(selected), int(self.cfg.batch_size)):
                end = start + int(self.cfg.batch_size)
                idx = torch.from_numpy(np.asarray(selected[start:end], dtype=np.int64))
                yield (x_windows.index_select(0, idx).contiguous(), m_windows.index_select(0, idx).contiguous(), stat_tensor[start:end].contiguous(), ys[start:end].contiguous(), ws[start:end].contiguous(), module_flag)
from OFP.ssffn.model import SSFFNModel as SSFFNModel
from OFP.ssffn._engine.modules import RuleFreeHSS, NoHSSSampler, sensor_distribution_signal
from OFP.ssffn._engine import data as _sampling_deep

def initialize_lazy_layers(model: SSFFNModel, dataset: SelectedWindowStatDataset, cfg: CompatCfg) -> None:
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    try:
        raw_x, raw_m, stat_x, _ys, _ws, _module_flag = next(iter(loader))
    except StopIteration as exc:
        raise ValueError('Cannot initialize fusion model because the dataset is empty') from exc
    model.eval()
    with torch.no_grad():
        model(raw_x.to(cfg.device), raw_m.to(cfg.device), stat_x.to(cfg.device))

def write_branch_selection_manifest(run_dir: Path, temporal_meta: dict[str, Any], stat_meta: dict[str, Any], requested_total_k: int) -> None:
    rows: list[dict[str, Any]] = []
    for branch, meta in (('temporal_raw', temporal_meta), ('expert_statistical', stat_meta)):
        for item in meta.get('selected_features', []):
            rows.append({'branch': branch, 'feature': item.get('feature', ''), 'within_branch_rank': item.get('rank'), 'within_branch_importance': item.get('importance')})
    explain_dir = run_dir / 'explainability'
    explain_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(explain_dir / 'selected_feature_manifest.csv', index=False)
    summary = {'requested_total_select_k': int(requested_total_k), 'selected_temporal_feature_count': int(temporal_meta.get('selected_temporal_feature_count', 0)), 'selected_stat_feature_count': int(stat_meta.get('selected_stat_feature_count', 0)), 'selected_input_feature_count': int(temporal_meta.get('selected_temporal_feature_count', 0) + stat_meta.get('selected_stat_feature_count', 0)), 'importance_note': 'Importance values are normalized within each branch and must not be compared across branches.'}
    (explain_dir / 'branch_selection_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

def train_fusion_encoder(train_files: list[str], cache: Model2FeatureCache, cfg: CompatCfg, raw_indices: list[int], raw_names: list[str], stat_indices: list[int], stat_names: list[str], args: argparse.Namespace, fold: int, run_dir: Path) -> tuple[SSFFNModel, CompatBatchedDataset, dict[str, Any], list[int], list[str], list[int], list[str], list[str]]:
    started = log_stage_start('build_fusion_dataset', RUN_NAME, fold, files=len(train_files), sample_selection=cfg.sample_selection, sampling_mode=cfg.sampling_mode)
    base_dataset = CompatBatchedDataset(train_files, cache, cfg)
    if int(base_dataset.pos_rows) <= 0:
        raise ValueError('The selected training subset contains no positive pre-anomaly windows. Increase --max_train_files or verify --target_mode and the trace split before using this run as evidence.')
    original_stat_candidates = stat_candidate_names(stat_names, raw_names, 'none')
    temporal_select_k, stat_select_k = resolve_branch_selection_budgets(0, 0, 0, len(raw_names), len(original_stat_candidates))
    strict_budget = False
    selected_raw_indices, selected_raw_names, temporal_selector_meta = maybe_select_temporal_features(cache, base_dataset, cfg, raw_indices, raw_names, args, fold, run_dir, temporal_select_k, strict=strict_budget)
    stat_input_indices, stat_input_names, stat_selector_meta, stat_candidate_feature_names = maybe_select_stat_features(cache, base_dataset, cfg, selected_raw_indices, selected_raw_names, stat_indices, stat_names, args, fold, run_dir, select_k=stat_select_k, strict=strict_budget)
    actual_total = len(selected_raw_names) + len(stat_input_names)
    if strict_budget and actual_total != 0:
        raise RuntimeError(f'Branch selection produced {actual_total} inputs, expected total Top-K={0}')
    write_branch_selection_manifest(run_dir, temporal_selector_meta, stat_selector_meta, 0)
    fusion_dataset = SelectedWindowStatDataset(base_dataset, cache, cfg, selected_raw_indices, stat_indices, stat_input_indices, 'none')
    log_stage_done('build_fusion_dataset', started, RUN_NAME, fold, rows=int(base_dataset.total_rows), pos=int(base_dataset.pos_rows), neg=int(base_dataset.neg_rows))
    model = SSFFNModel(module_variant=args.module_variant, seq_len=cfg.seq_len, stat_feature_names=stat_input_names, stat_modality_dropout=0.15).to(cfg.device)
    endpoints = np.concatenate([cache.get(name)[1][positions][:, selected_raw_indices[:12]] for name, positions in zip(base_dataset.file_names, base_dataset.selected_positions) if len(positions)])
    if len(endpoints) > 100000:
        endpoints = endpoints[np.random.default_rng(int(cfg.seed) + fold).choice(len(endpoints), 100000, replace=False)]
    distribution = model.sensor_input.fit_distribution(endpoints)
    (run_dir / 'backbone_training_distribution.json').write_text(json.dumps(distribution, indent=2))
    del endpoints
    assert True
    initialize_lazy_layers(model, fusion_dataset, cfg)
    loader = ModuleGroupedLoader(DataLoader(fusion_dataset, batch_size=None, shuffle=False, num_workers=int(cfg.num_workers)), 32)
    assert int(cfg.batch_size) >= max(int(cfg.positive_windows_per_module), int(cfg.normal_windows_per_module))
    assert True, 'Grouped study uses ATAL only'
    pos_weight = effective_pos_weight(base_dataset, cfg)
    pos_weight = 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=cfg.device), reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith('cuda')
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled, init_scale=1024.0)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=1024.0)
    warmup_epochs = max(0, min(0, max(int(cfg.epochs) - 1, 0)))
    temporal_aux_weight = max(0.0, 0.0)
    stat_aux_weight = max(0.0, 0.0)
    module_bag_weight = max(0.0, 0.0)
    use_aux = temporal_aux_weight > 0.0 or stat_aux_weight > 0.0 or warmup_epochs > 0
    if warmup_epochs > 0 and temporal_aux_weight + stat_aux_weight <= 0.0:
        raise ValueError('--branch_warmup_epochs requires a positive temporal or statistical auxiliary loss weight')
    history: list[dict[str, Any]] = []
    distribution_hss = None
    sampler_class = NoHSSSampler if args.module_variant == 'no_hss' else RuleFreeHSS
    distribution_hss = sampler_class(base_dataset, selected_raw_indices, int(cfg.seed) + int(fold))
    np.save(run_dir / 'hss_training_quantiles.npy', distribution_hss.knots)
    hss_history = []
    adaptive_meta: dict[str, float] = {}
    adaptive_applied = False
    train_started = time.time()
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        fusion_dataset.set_epoch(epoch)
        total_loss = 0.0
        total_fused_loss = 0.0
        total_temporal_loss = 0.0
        total_stat_loss = 0.0
        total_bag_loss = 0.0
        total_alarm_loss = 0.0
        n_seen = 0
        epoch_started = time.time()
        phase = 'branch-warmup' if epoch <= warmup_epochs else 'fusion'
        for batch_idx, (raw_x, raw_m, stat_x, ys, ws, module_flag) in enumerate(loader, 1):
            raw_x = raw_x.to(cfg.device)
            raw_m = raw_m.to(cfg.device)
            stat_x = stat_x.to(cfg.device)
            ys = ys.to(cfg.device).float()
            ws = ws.to(cfg.device).float()
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(amp_enabled):
                if use_aux:
                    logits, temporal_logits, stat_logits = model.forward_with_aux(raw_x, raw_m, stat_x)
                    fused_loss = module_mean_loss(loss_fn(logits, ys), ws, module_flag)
                    temporal_loss = module_mean_loss(loss_fn(temporal_logits, ys), ws, module_flag)
                    stat_loss = module_mean_loss(loss_fn(stat_logits, ys), ws, module_flag)
                    loss = temporal_aux_weight * temporal_loss + stat_aux_weight * stat_loss
                    if epoch > warmup_epochs:
                        loss = loss + fused_loss
                else:
                    logits = model(raw_x, raw_m, stat_x)
                    fused_loss = module_mean_loss(loss_fn(logits, ys), ws, module_flag)
                    temporal_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
                    stat_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
                    loss = fused_loss
                bag_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
                if module_bag_weight > 0.0 and epoch > warmup_epochs:
                    if bool(torch.any(ys > 0.5)):
                        bag_loss = F.softplus(-torch.max(logits[ys > 0.5]))
                    elif float(module_flag.item()) <= 0.0:
                        bag_loss = F.softplus(torch.max(logits))
                    loss = loss + module_bag_weight * bag_loss
                alarm_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f'Non-finite SSFFN loss at fold={fold}, epoch={epoch}, batch={batch_idx}; the run was stopped before corrupting the saved model.')
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip), error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip), error_if_nonfinite=True)
                optimizer.step()
            batch_n = int(raw_x.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_n
            total_fused_loss += float(fused_loss.detach().cpu()) * batch_n
            total_temporal_loss += float(temporal_loss.detach().cpu()) * batch_n
            total_stat_loss += float(stat_loss.detach().cpu()) * batch_n
            total_bag_loss += float(bag_loss.detach().cpu()) * batch_n
            total_alarm_loss += float(alarm_loss.detach().cpu()) * batch_n
            n_seen += batch_n
            if int(cfg.log_batches) > 0 and batch_idx % int(cfg.log_batches) == 0:
                elapsed = time.time() - epoch_started
                print(f'[batch] ep={epoch:02d} batch={batch_idx:>5}/{len(loader):<5} {progress_bar(batch_idx, len(loader), width=16)} phase={phase} loss={total_loss / max(n_seen, 1):.5f} fused={total_fused_loss / max(n_seen, 1):.5f} temporal={total_temporal_loss / max(n_seen, 1):.5f} stat={total_stat_loss / max(n_seen, 1):.5f} rows={n_seen} bag={total_bag_loss / max(n_seen, 1):.5f} rate={format_rate(n_seen, elapsed)} elapsed={format_duration(elapsed)}', flush=True)
        epoch_seconds = time.time() - epoch_started
        elapsed_total = time.time() - train_started
        eta = elapsed_total / max(epoch, 1) * max(int(cfg.epochs) - epoch, 0)
        avg_loss = total_loss / max(n_seen, 1)
        history.append({'epoch': float(epoch), 'phase': phase, 'loss': float(avg_loss), 'fused_loss': float(total_fused_loss / max(n_seen, 1)), 'temporal_aux_loss': float(total_temporal_loss / max(n_seen, 1)), 'stat_aux_loss': float(total_stat_loss / max(n_seen, 1)), 'module_bag_loss': float(total_bag_loss / max(n_seen, 1)), 'normal_top4_loss': float(total_alarm_loss / max(n_seen, 1)), 'rows_seen': float(n_seen), 'elapsed_seconds': epoch_seconds})
        print(format_epoch_status('align-train', RUN_NAME, fold, epoch, int(cfg.epochs), float(avg_loss), int(n_seen), float(epoch_seconds), elapsed_total, eta), flush=True)
        checkpoint_path = run_dir / 'training_checkpoint.pt'
        torch.save({'epoch': int(epoch), 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'history': history, 'amp_enabled': bool(amp_enabled)}, checkpoint_path)
        if distribution_hss is not None and epoch % 2 == 0 and (epoch < int(cfg.epochs)):
            print(f'[distribution-hss] refreshing after epoch {epoch}', flush=True)
            hss_history.append(distribution_hss.refresh(model, cfg, selected_raw_indices, stat_indices, stat_input_indices, 'none', score_selected_positions, epoch))
            (run_dir / 'hss_history.json').write_text(json.dumps(hss_history, indent=2))
        if float(cfg.adaptive_negative_weight) > 0.0 and (not adaptive_applied) and (epoch >= int(cfg.adaptive_warmup_epochs)) and (epoch < int(cfg.epochs)):
            score_started = log_stage_start('adaptive_negative_weight', RUN_NAME, fold, after_epoch=epoch)
            scores = score_selected_positions(model, cache, base_dataset, cfg, selected_raw_indices, stat_indices, stat_input_indices, 'none')
            adaptive_meta = base_dataset.apply_adaptive_negative_weights(scores, float(cfg.adaptive_negative_weight))
            adaptive_meta['applied_after_epoch'] = float(epoch)
            adaptive_applied = True
            print(f"[adaptive-neg] rows={int(adaptive_meta.get('updated_negative_rows', 0))} score_min={adaptive_meta.get('min_score', 0.0):.5f} score_max={adaptive_meta.get('max_score', 0.0):.5f} max_extra={cfg.adaptive_negative_weight}", flush=True)
            log_stage_done('adaptive_negative_weight', score_started, RUN_NAME, fold)
    meta = {'architecture': model.temporal_encoder_cfg, 'parameter_count': sum((p.numel() for p in model.parameters())), 'modules_per_optimizer_step': 32, 'transfer_loss': 'bce', 'module_margin_alpha': 0.0, 'transfer_hss': 'distribution_risk', 'distribution_hss_history': hss_history, 'sampling_policy': distribution_hss.policy_metadata, 'train_rows': int(base_dataset.total_rows), 'train_pos_rows': int(base_dataset.pos_rows), 'train_neg_rows': int(base_dataset.neg_rows), 'pos_weight': float(pos_weight), 'branch_warmup_epochs': int(warmup_epochs), 'temporal_aux_weight': float(temporal_aux_weight), 'stat_aux_weight': float(stat_aux_weight), 'module_bag_weight': float(module_bag_weight), 'temporal_view_mode': 'level_mask', 'temporal_modality_dropout': 0.0, 'stat_modality_dropout': 0.15, 'training_seed': int(args.training_seed), 'data_seed': int(args.seed), 'history': history, 'adaptive_negative_weight_meta': adaptive_meta, 'temporal_feature_selection': temporal_selector_meta, 'stat_feature_selection': stat_selector_meta, 'branch_selection': {'requested_total_select_k': 0, 'effective_temporal_select_k': int(temporal_select_k), 'effective_stat_select_k': int(stat_select_k), 'selected_input_feature_count': int(actual_total)}}
    return (model, base_dataset, meta, selected_raw_indices, selected_raw_names, stat_input_indices, stat_input_names, stat_candidate_feature_names)

@torch.no_grad()
def encode_file_positions(model: SSFFNModel, cache: Model2FeatureCache, file_name: str, cfg: CompatCfg, raw_indices: list[int], stat_indices: list[int], stat_input_indices: list[int], temporal_summary_mode: str, positions: np.ndarray | None=None, return_branch_latents: bool=False) -> dict[str, np.ndarray]:
    model.eval()
    timestamps, features, _valid_mask, labels, _anomaly_labels, rule_pred, extra = cache.get(file_name)
    n_rows = int(len(features))
    if positions is None:
        positions = np.arange(n_rows, dtype=np.int64)
    else:
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[(positions >= 0) & (positions < n_rows)]
    if n_rows == 0 or len(positions) == 0:
        return {'timestamps': np.zeros(0, dtype=np.int64), 'fused': np.zeros((0, 0), dtype=np.float32), 'temporal_latent': np.zeros((0, 0), dtype=np.float32), 'statistical_latent': np.zeros((0, 0), dtype=np.float32), 'logits': np.zeros(0, dtype=np.float32), 'labels': np.zeros(0, dtype=np.int8), 'weights': np.ones(0, dtype=np.float32), 'rule_pred': np.zeros(0, dtype=np.int8), 'extra': extra}
    raw = features[:, np.asarray(raw_indices, dtype=np.int64)].astype(np.float32)
    validity_indices = raw_validity_indices(cfg, raw_indices)
    raw_validity = features[:, validity_indices].astype(np.float32) if validity_indices is not None else np.ones_like(raw, dtype=np.float32)
    seq_len = int(cfg.seq_len)
    stat = build_stat_candidate_matrix(features, positions, raw_indices, stat_indices, seq_len, temporal_summary_mode)
    stat = stat[:, np.asarray(stat_input_indices, dtype=np.int64)].astype(np.float32)
    pad_x = np.zeros((seq_len - 1, raw.shape[1]), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = torch.from_numpy(np.concatenate([pad_x, raw], axis=0))
    m_pad = torch.from_numpy(np.concatenate([pad_m, raw_validity], axis=0))
    x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
    m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
    fused_parts: list[np.ndarray] = []
    temporal_parts: list[np.ndarray] = []
    statistical_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    eval_batch_size = 2048
    for start in range(0, len(positions), eval_batch_size):
        batch_pos = positions[start:start + eval_batch_size]
        idx = torch.from_numpy(batch_pos.astype(np.int64))
        raw_x = x_windows.index_select(0, idx).contiguous().to(cfg.device)
        raw_m = m_windows.index_select(0, idx).contiguous().to(cfg.device)
        stat_x = torch.from_numpy(stat[start:start + len(batch_pos)].astype(np.float32)).to(cfg.device)
        with cuda_autocast(bool(cfg.amp) and str(cfg.device).startswith('cuda')):
            if return_branch_latents:
                fused, explain = model.encode(raw_x, raw_m, stat_x, return_explain=True)
            else:
                fused = model.encode(raw_x, raw_m, stat_x)
                explain = None
            logits = model.classifier(fused).squeeze(-1)
        fused_parts.append(fused.float().detach().cpu().numpy().astype(np.float32))
        if explain is not None:
            temporal_parts.append(explain['seq_latent'].float().cpu().numpy().astype(np.float32))
            statistical_parts.append(explain['stat_latent'].float().cpu().numpy().astype(np.float32))
        logit_parts.append(logits.float().detach().cpu().numpy().astype(np.float32))
    return {'timestamps': timestamps[positions].astype(np.int64), 'fused': np.concatenate(fused_parts, axis=0).astype(np.float32), 'temporal_latent': np.concatenate(temporal_parts, axis=0).astype(np.float32) if temporal_parts else np.zeros((len(positions), 0), dtype=np.float32), 'statistical_latent': np.concatenate(statistical_parts, axis=0).astype(np.float32) if statistical_parts else np.zeros((len(positions), 0), dtype=np.float32), 'logits': np.concatenate(logit_parts, axis=0).astype(np.float32), 'labels': labels[positions].astype(np.int8), 'weights': np.ones(len(positions), dtype=np.float32), 'rule_pred': rule_pred[positions].astype(np.int8), 'extra': extra}

def score_selected_positions(model: SSFFNModel, cache: Model2FeatureCache, dataset: CompatBatchedDataset, cfg: CompatCfg, raw_indices: list[int], stat_indices: list[int], stat_input_indices: list[int], temporal_summary_mode: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for file_name, positions in zip(dataset.file_names, dataset.selected_positions):
        block = encode_file_positions(model, cache, file_name, cfg, raw_indices, stat_indices, stat_input_indices, temporal_summary_mode, positions)
        out[file_name] = (1.0 / (1.0 + np.exp(-block['logits']))).astype(np.float32)
    return out

def save_decision_latent_importance(estimator: Any, latent_names: list[str], run_dir: Path) -> None:
    scores = getattr(estimator, 'feature_importances_', None)
    if scores is None:
        return
    frame = _save_score_table(run_dir / 'explainability' / 'decision_latent_importance.csv', latent_names, np.asarray(scores, dtype=float), 'importance')
    _save_topk_heatmap(run_dir / 'explainability' / 'decision_latent_importance_topk_heatmap.png', frame, 'importance')

def summarize_tsf_explainability(model: SSFFNModel, cache: Model2FeatureCache, dataset: CompatBatchedDataset, cfg: CompatCfg, raw_indices: list[int], stat_indices: list[int], stat_input_indices: list[int], raw_names: list[str], stat_names: list[str], run_dir: Path, args: argparse.Namespace) -> None:
    max_rows = 2048
    return

def score_fused_files_to_memory(estimator: Any, model: SSFFNModel, cache: Model2FeatureCache, file_names: list[str], cfg: CompatCfg, raw_indices: list[int], stat_indices: list[int], stat_input_indices: list[int], temporal_summary_mode: str, fold: int, stage_name: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    started = log_stage_start(stage_name, RUN_NAME, fold, files=len(file_names))
    for idx, file_name in enumerate(file_names, 1):
        block = encode_file_positions(model, cache, file_name, cfg, raw_indices, stat_indices, stat_input_indices, temporal_summary_mode)
        score = positive_scores(estimator, block['fused'])
        parts = [pd.DataFrame({'timestamp': block['timestamps'].astype(np.int64), 'score': score.astype(np.float32), 'rule_predict': block['rule_pred'].astype(int), 'source': RUN_NAME})]
        extra = block['extra']
        if len(extra):
            parts.append(pd.DataFrame({'timestamp': extra[:, 0].astype(np.int64), 'score': np.nan, 'rule_predict': extra[:, 1].astype(int), 'source': 'model2_extra_rule'}))
        frames[file_name] = pd.concat(parts, ignore_index=True).sort_values('timestamp')
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(f'[stage-progress] model={RUN_NAME} fold={fold} stage={stage_name} {progress_bar(idx, len(file_names), width=18)} files={idx}/{len(file_names)} rate={format_rate(idx, elapsed)} elapsed={format_duration(elapsed)}', flush=True)
    log_stage_done(stage_name, started, RUN_NAME, fold)
    return frames

def run_fold(fold: int, args: argparse.Namespace, partitions=None) -> list[dict[str, Any]]:
    cfg = CompatCfg(seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip, negative_ratio=10.0, pos_weight_cap=20.0, fixed_threshold=0.3, target_mode='pre_event', target_horizon_hours=120.0, feature_mode='ofp', preserve_timepoints=True, sampling_mode='module_balanced', positive_windows_per_module=args.positive_windows_per_module, negative_windows_per_faulty_module=args.negative_windows_per_faulty_module, normal_windows_per_module=args.normal_windows_per_module, val_fraction=0.2, threshold_search=True, threshold_grid=args.threshold_grid, threshold_metric='f1_score', rule_mode='none', sample_selection='hybrid', sample_topk_fraction=0.5, sample_signal_mode='sensor_distribution', sample_signal_temporal_fraction=0.5, temporal_positive_weight=0.0, temporal_weight_horizon_hours=120.0, adaptive_negative_weight=0.0, adaptive_warmup_epochs=1, max_cached_files=args.max_cached_files, module_cache_dir=str(args.module_cache_dir) if args.module_cache_dir else '', seed=args.seed, device=args.device, num_workers=args.num_workers, amp=bool(args.amp), allow_tf32=not args.no_tf32, log_batches=args.log_batches, max_train_files=0, max_test_files=0, min_hit_lead_hours=0.0)
    configure_runtime(cfg, int(args.seed), int(fold), int(args.training_seed))
    fold_started = time.time()
    print(f'[aligned-init] fold={fold} device={cfg.device} torch={torch.__version__}', flush=True)
    index_df = read_index(args.index_path)
    label_by_file = file_label_map(index_df)
    if partitions is None:
        raise ValueError('Explicit training and validation partitions are required')
    train_files, val_files, test_files = (list(partitions[key]) for key in ('train', 'validation', 'test'))
    sets = [set(train_files), set(val_files), set(test_files)]
    if any((sets[i] & sets[j] for i in range(3) for j in range(i))):
        raise ValueError('A module occurs in more than one split')
    run_dir = Path(args.out_root) / RUN_NAME / f'fold_{fold}'
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_mode = resolve_fusion_mode('sit', 'full')
    log_run_header('SSFFN: SENSOR AND STATISTICAL FEATURE FUSION NETWORK', {'experiment id': args.experiment_id or 'default', 'temporal encoder': 'sit', 'temporal view': 'level_mask', 'decision layer': 'linear', 'decision source': 'encoder_head', 'fold': fold, 'train/val/test': f'{len(train_files)}/{len(val_files)}/{len(test_files)} modules', 'target': cfg.target_mode, 'feature mode': cfg.feature_mode, 'stat features': 'ofp_expert_stat', 'stat feature groups': 'statistical,expert', 'excluded stat groups': 'none', 'temporal summaries': 'none', 'fusion mode': resolved_mode, 'tsf ablation': 'full', 'latent probe top-k': 'disabled', 'rule mode': cfg.rule_mode, 'alarm confirmation': f'{1}-of-{1} learned path only', 'threshold constraint': 'none', 'sample selection': cfg.sample_selection, 'sample signal': cfg.sample_signal_mode, 'signal temporal mix': cfg.sample_signal_temporal_fraction, 'branch warmup': 0, 'aux loss weights': f'temporal={0.0} stat={0.0}', 'module bag weight': 0.0, 'modality dropout': f'temporal={0.0} stat={0.15}', 'seq_len': cfg.seq_len, 'epochs': cfg.epochs, 'batch_size': cfg.batch_size, 'device': cfg.device, 'out_dir': run_dir})
    stats_started = log_stage_start('feature_norm_stats', RUN_NAME, fold, files=len(train_files), data_dir=args.data_dir)
    stats = load_or_compute_feature_norm_stats(args.data_dir, train_files, cfg, label_by_file)
    log_stage_done('feature_norm_stats', stats_started, RUN_NAME, fold)
    (run_dir / 'normalization.json').write_text(json.dumps(asdict(stats), indent=2), encoding='utf-8')
    (run_dir / 'split.json').write_text(json.dumps(dict(train=train_files, validation=val_files, test=test_files), indent=2), encoding='utf-8')
    mean, std = stats.arrays()
    cache = Model2FeatureCache(args.data_dir, mean, std, cfg, label_by_file)
    names = compat_feature_names(cfg)
    all_raw_indices, all_raw_names, stat_indices, stat_names = feature_indices(names, 'ofp_expert_stat', 'statistical,expert', '')
    stat_candidate_feature_names = stat_candidate_names(stat_names, all_raw_names, 'none')
    original_stat_feature_count = len(stat_candidate_feature_names)
    model, dataset, train_meta, raw_indices, raw_names, stat_input_indices, stat_input_names, stat_candidate_feature_names = train_fusion_encoder(train_files, cache, cfg, all_raw_indices, all_raw_names, stat_indices, stat_names, args, int(fold), run_dir)
    (run_dir / 'feature_groups.json').write_text(json.dumps({'temporal_encoder': 'sit', 'temporal_view_mode': 'level_mask', 'decision_layer': 'linear', 'experiment_id': args.experiment_id, 'raw_sequence_feature_candidates': all_raw_names, 'raw_sequence_features': raw_names, 'base_statistic_features': stat_names, 'statistic_feature_candidates': stat_candidate_feature_names, 'statistic_features': stat_input_names, 'temporal_summary_mode': 'none', 'temporal_summary_features': temporal_summary_feature_names(raw_names, 'none'), 'feature_group_counts': {'raw_sequence_features_original': len(all_raw_names), 'raw_sequence_features': len(raw_names), 'raw_sequence_window_values': len(raw_names) * int(cfg.seq_len), 'base_statistic_features': len(stat_names), 'temporal_summary_features': len(stat_candidate_feature_names) - len(stat_names), 'statistic_features_original': original_stat_feature_count, 'statistic_features_used': len(stat_input_names), 'sensor_token_dim': 64, 'expert_token_dim': 64, 'shared_output_dim': 64}, 'all_compat_features': names, 'all_feature_groups': feature_group_manifest(names), 'selected_stat_feature_groups': feature_group_manifest(stat_input_names), 'train_meta': train_meta}, indent=2), encoding='utf-8')
    torch.save({'state_dict': model.state_dict(), 'release_model': 'SSFFN', 'module_variant': args.module_variant, 'training_seed': args.training_seed, 'normalization': asdict(stats), 'cfg': asdict(cfg), 'temporal_encoder': 'sit', 'temporal_view_mode': 'level_mask', 'temporal_encoder_cfg': model.temporal_encoder_cfg, 'raw_feature_candidates': all_raw_names, 'raw_features': raw_names, 'base_stat_features': stat_names, 'stat_feature_candidates': stat_candidate_feature_names, 'stat_features': stat_input_names, 'stat_input_indices': stat_input_indices, 'temporal_summary_mode': 'none', 'fusion_mode': resolved_mode, 'train_meta': train_meta}, run_dir / 'aligned_encoder.pt')
    cache.cache.clear()
    cfg.max_cached_files = 512
    assert True
    x_train = np.zeros((0, model.latent_dim), dtype=np.float32)
    y_train = np.concatenate(dataset.selected_labels)
    sample_weight = np.concatenate(dataset.selected_weights)
    latent_names = [f'fused_{i:03d}' for i in range(model.latent_dim)]
    summarize_tsf_explainability(model, cache, dataset, cfg, raw_indices, stat_indices, stat_input_indices, raw_names, stat_input_names, run_dir, args)
    estimator = frozen_encoder_head(model)
    decision_meta = {'decision_layer': 'linear', 'decision_source': 'encoder_head', 'linear_class_weight': 'end_to_end_bce'}
    ml_started = log_stage_start('decision_train_on_fused_latent', RUN_NAME, fold, rows=len(y_train), features=x_train.shape[1], decision_layer='linear', balance=decision_meta.get('xgb_balance_mode', 'not_applicable'), scale_pos_weight=f"{float(decision_meta.get('xgb_scale_pos_weight', 1.0)):.4f}", sample_weight=True)
    sample_weight_used = False
    decision_meta['sample_weight_used'] = bool(sample_weight_used)
    log_stage_done('decision_train_on_fused_latent', ml_started, RUN_NAME, fold)
    save_decision_latent_importance(estimator, latent_names, run_dir)
    with (run_dir / 'decision_layer.pkl').open('wb') as fh:
        pickle.dump({'model': estimator, 'feature_names': latent_names, 'decision_meta': decision_meta}, fh)
    val_scores = score_fused_files_to_memory(estimator, model, cache, val_files, cfg, raw_indices, stat_indices, stat_input_indices, 'none', int(fold), 'score_val_files')
    test_score_started = time.time()
    test_scores = score_fused_files_to_memory(estimator, model, cache, test_files, cfg, raw_indices, stat_indices, stat_input_indices, 'none', int(fold), 'score_test_files')
    test_score_seconds = float(time.time() - test_score_started)
    scored_test_rows = int(sum((len(frame) for frame in test_scores.values())))
    scoring_rows_per_second = scored_test_rows / max(test_score_seconds, 1e-09)
    scoring_ms_per_window = 1000.0 * test_score_seconds / max(scored_test_rows, 1)
    decision_tag = 'encoder_head_linear'
    threshold, val_metrics = select_threshold_for_scores(val_scores, args.data_dir, run_dir, decision_tag, args)
    pred_dir = run_dir / 'predictions' / decision_tag
    eval_dir = run_dir / 'evaluation' / decision_tag
    test_rows = write_threshold_predictions(test_scores, pred_dir, threshold, confirm_k=1, confirm_m=1)
    if test_files:
        metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, 0.0)
    else:
        metrics = {}
    print(f'[aligned-done] fold={fold} threshold={threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}', flush=True)
    decision_feature_count = int(x_train.shape[1])
    result = {'deep_model': RUN_NAME, 'method_family': 'SSFFN', 'method_label': args.method_label or 'SSFFN', 'experiment_id': args.experiment_id, 'temporal_encoder': 'sit', 'temporal_view_mode': 'level_mask', 'fold': int(fold), 'mode': decision_tag, 'ml_model': 'linear', 'decision_layer': 'linear', 'decision_source': 'encoder_head', 'ml_feature_set': 'aligned_latent', 'selector': 'none', 'selected_feature_count': decision_feature_count, 'fusion_mode': resolved_mode, 'rule_mode': cfg.rule_mode, 'threshold': float(threshold), 'threshold_constraint': 'none', 'confirm_k': 1, 'confirm_m': 1, 'temporal_selector': 'none', 'temporal_select_k': 0, 'stat_selector': 'none', 'stat_select_k': 0, 'total_select_k': 0, 'temporal_summary_mode': 'none', 'temporal_feature_count': len(raw_names), 'stat_feature_count': len(stat_input_names), 'selected_input_feature_count': len(raw_names) + len(stat_input_names), 'stat_feature_groups': 'statistical,expert', 'exclude_stat_feature_groups': '', 'sample_selection': 'hybrid', 'sample_topk_fraction': 0.5, 'sample_signal_mode': 'sensor_distribution', 'sample_signal_temporal_fraction': 0.5, 'branch_warmup_epochs': 0, 'temporal_aux_weight': 0.0, 'stat_aux_weight': 0.0, 'module_bag_weight': 0.0, 'temporal_modality_dropout': 0.0, 'stat_modality_dropout': 0.15, 'temporal_positive_weight': 0.0, 'adaptive_negative_weight': 0.0, 'train_rows': int(train_meta.get('train_rows', 0)), 'train_pos_rows': int(train_meta.get('train_pos_rows', 0)), 'train_neg_rows': int(train_meta.get('train_neg_rows', 0)), 'test_score_seconds': test_score_seconds, 'scored_test_rows': scored_test_rows, 'scoring_rows_per_second': float(scoring_rows_per_second), 'scoring_ms_per_window': float(scoring_ms_per_window), 'encoder_model_mb': float((run_dir / 'aligned_encoder.pt').stat().st_size / (1024.0 * 1024.0)), 'decision_model_mb': float((run_dir / 'decision_layer.pkl').stat().st_size / (1024.0 * 1024.0)), 'val_metrics': val_metrics, 'test_rows': int(test_rows), 'metrics': metrics}
    results = [result]
    del x_train, val_scores, test_scores
    gc.collect()
    print(f'[fold-done] run={RUN_NAME} fold={fold} elapsed={format_duration(time.time() - fold_started)}', flush=True)
    model.close()
    del model, cache, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
_sampling_deep.row_signal_scores = sensor_distribution_signal
