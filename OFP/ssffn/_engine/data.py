from __future__ import annotations
import math
import os
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from OFP.ssffn._engine.allocation import allocate_stratified_negative_counts
from OFP.ssffn._engine.metrics import first_positive_timestamp
from OFP.ssffn._engine.feature_base import MODEL2_FEATURES, NA_DEFAULT, OFP_BASE_COLUMNS, OFP_NAME_MAP, TEMP_OUTLIER, first_anomaly_ahead_label, first_event_valid_mask, make_model2_features, rolling_corr
from OFP.ssffn._engine.feature_schema import OFP_ENGINEERED_FEATURES, RULE_FEATURES, add_ofp_expert_stat_features, deduplicate
FEATURE_CACHE_VERSION = 'ssffn_features'

@dataclass
class CompatCfg:
    seq_len: int = 64
    epochs: int = 8
    batch_size: int = 256
    lr: float = 0.0003
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    negative_ratio: float = 10.0
    pos_weight_cap: float = 20.0
    fixed_threshold: float = 0.5
    target_mode: str = 'module_fault'
    target_horizon_hours: float = 120.0
    feature_mode: str = 'model2_plus'
    preserve_timepoints: bool = False
    sampling_mode: str = 'module_balanced'
    positive_windows_per_module: int = 96
    negative_windows_per_faulty_module: int = 24
    normal_windows_per_module: int = 24
    val_fraction: float = 0.2
    threshold_search: bool = True
    threshold_grid: str = '0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50'
    threshold_metric: str = 'f1_score'
    rule_mode: str = 'model2_simple'
    sample_selection: str = 'random'
    sample_topk_fraction: float = 0.5
    sample_signal_mode: str = 'expert'
    sample_signal_temporal_fraction: float = 0.5
    temporal_positive_weight: float = 0.0
    temporal_weight_horizon_hours: float = 120.0
    adaptive_negative_weight: float = 0.0
    adaptive_warmup_epochs: int = 1
    max_cached_files: int = 512
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 0
    amp: bool = False
    allow_tf32: bool = True
    log_batches: int = 1000
    module_cache_dir: str = ''
    max_train_files: int = 0
    max_test_files: int = 0
    min_hit_lead_hours: float = 0.0

@dataclass
class FeatureNormStats:
    mean: list[float]
    std: list[float]
    rows_seen: int
    files_seen: int
    feature_names: list[str]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.mean, dtype=np.float32), np.maximum(np.asarray(self.std, dtype=np.float32), 1e-06))
ROLLING_BASE_COLS = ['Temp', 'Curr', 'TxP0', 'RxP0', 'RxP1', 'RxP2', 'RxP3', 'RxP4', 'TxP1', 'TxP2', 'TxP3', 'TxP4', 'FeTxP0-Max', 'FeTxP0-Min', 'FeRxP0-Max', 'FeRxP0-Min']
ROLL_WINDOWS = (3, 6, 12, 24, 64)
LANE_PLUS_FEATURES = ['TxLaneRange', 'RxLaneRange', 'TxLaneStd', 'RxLaneStd', 'TxP0MinusLaneMean', 'RxP0MinusLaneMean']
ROLLING_PLUS_FEATURES = []
DRAM_PLUS_FEATURES = ['DeltaSeconds', 'ElapsedHours', 'TempInvalidFlag', 'CurrLowFlag', 'TxNegativeCount', 'RxNegativeCount', 'PowerNegativeCount', 'PowerHighCount', 'RuleLikeAbnormalCount', 'AnyRuleLikeAbnormal']
COMPAT_PLUS_FEATURES = list(MODEL2_FEATURES) + LANE_PLUS_FEATURES + ROLLING_PLUS_FEATURES + DRAM_PLUS_FEATURES
COMPAT_OFP_FEATURES = deduplicate([*MODEL2_FEATURES, *OFP_ENGINEERED_FEATURES])
COMPAT_OFP_PLUS_FEATURES = deduplicate([*COMPAT_OFP_FEATURES, *LANE_PLUS_FEATURES, *ROLLING_PLUS_FEATURES, *DRAM_PLUS_FEATURES])
RAW_SENSOR_FEATURES = [name for name in OFP_BASE_COLUMNS if name != 'Ts']
RAW_VALID_MASK_FEATURES = [f'{name}ValidMask' for name in RAW_SENSOR_FEATURES]
SSFFN_TIME_FEATURES = ['DeltaSeconds', 'DeltaSecondsValidMask', 'TsValidMask']
SSFFN_EXPERT_QUALITY_FEATURES = ['TempInvalidFlag', 'AnySensorMissingFlag', 'SensorMissingCount', 'SamplingGapFlag']
SSFFN_PRESERVED_FEATURES = [*SSFFN_TIME_FEATURES, *RAW_VALID_MASK_FEATURES, *SSFFN_EXPERT_QUALITY_FEATURES]

def compat_feature_names(cfg: CompatCfg) -> list[str]:
    mode = str(cfg.feature_mode).lower()
    if mode in {'model2', 'base'}:
        names = list(MODEL2_FEATURES)
    elif mode in {'model2_plus', 'plus'}:
        names = list(COMPAT_PLUS_FEATURES)
    elif mode in {'ofp', 'ofp_expert_stat'}:
        names = list(COMPAT_OFP_FEATURES)
    elif mode in {'ofp_plus', 'ofp_expert_stat_plus'}:
        names = list(COMPAT_OFP_PLUS_FEATURES)
    else:
        raise ValueError(f'Unknown feature_mode={cfg.feature_mode!r}')
    if bool(cfg.preserve_timepoints):
        names = deduplicate([*names, *SSFFN_PRESERVED_FEATURES])
    return names

def add_model2_plus_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    extras: dict[str, object] = {}
    tx_cols = ['TxP1', 'TxP2', 'TxP3', 'TxP4']
    rx_cols = ['RxP1', 'RxP2', 'RxP3', 'RxP4']
    tx = out[tx_cols].apply(pd.to_numeric, errors='coerce') if set(tx_cols) <= set(out.columns) else pd.DataFrame(index=out.index)
    rx = out[rx_cols].apply(pd.to_numeric, errors='coerce') if set(rx_cols) <= set(out.columns) else pd.DataFrame(index=out.index)
    if len(tx.columns):
        extras['TxLaneRange'] = tx.max(axis=1) - tx.min(axis=1)
        extras['TxLaneStd'] = tx.std(axis=1).fillna(0.0)
        extras['TxP0MinusLaneMean'] = pd.to_numeric(out.get('TxP0', 0.0), errors='coerce') - tx.mean(axis=1)
    else:
        extras['TxLaneRange'] = 0.0
        extras['TxLaneStd'] = 0.0
        extras['TxP0MinusLaneMean'] = 0.0
    if len(rx.columns):
        extras['RxLaneRange'] = rx.max(axis=1) - rx.min(axis=1)
        extras['RxLaneStd'] = rx.std(axis=1).fillna(0.0)
        extras['RxP0MinusLaneMean'] = pd.to_numeric(out.get('RxP0', 0.0), errors='coerce') - rx.mean(axis=1)
    else:
        extras['RxLaneRange'] = 0.0
        extras['RxLaneStd'] = 0.0
        extras['RxP0MinusLaneMean'] = 0.0
    ts = pd.to_numeric(out.get('Ts', pd.Series(np.arange(len(out)), index=out.index)), errors='coerce').ffill().fillna(0.0)
    extras['DeltaSeconds'] = ts.diff().fillna(0.0).clip(lower=0.0)
    extras['ElapsedHours'] = ((ts - ts.iloc[0]) / 3600.0).fillna(0.0) if len(ts) else 0.0
    temp = pd.to_numeric(out.get('Temp', 0.0), errors='coerce').fillna(NA_DEFAULT)
    curr = pd.to_numeric(out.get('Curr', 0.0), errors='coerce').fillna(NA_DEFAULT)
    extras['TempInvalidFlag'] = temp.le(-254.0).astype(float)
    extras['CurrLowFlag'] = curr.lt(5000.0).astype(float)
    tx_all_cols = ['TxP0', 'TxP1', 'TxP2', 'TxP3', 'TxP4']
    rx_all_cols = ['RxP0', 'RxP1', 'RxP2', 'RxP3', 'RxP4']
    tx_all = out[[c for c in tx_all_cols if c in out.columns]].apply(pd.to_numeric, errors='coerce')
    rx_all = out[[c for c in rx_all_cols if c in out.columns]].apply(pd.to_numeric, errors='coerce')
    tx_neg = tx_all.lt(0.0).sum(axis=1) if len(tx_all.columns) else pd.Series(0.0, index=out.index)
    rx_neg = rx_all.lt(0.0).sum(axis=1) if len(rx_all.columns) else pd.Series(0.0, index=out.index)
    tx_high = tx_all.gt(1000.0).sum(axis=1) if len(tx_all.columns) else pd.Series(0.0, index=out.index)
    rx_high = rx_all.gt(1000.0).sum(axis=1) if len(rx_all.columns) else pd.Series(0.0, index=out.index)
    power_neg = tx_neg + rx_neg
    power_high = tx_high + rx_high
    rule_like = extras['TempInvalidFlag'] + extras['CurrLowFlag'] + power_neg + power_high
    extras['TxNegativeCount'] = tx_neg.astype(float)
    extras['RxNegativeCount'] = rx_neg.astype(float)
    extras['PowerNegativeCount'] = power_neg.astype(float)
    extras['PowerHighCount'] = power_high.astype(float)
    extras['RuleLikeAbnormalCount'] = rule_like.astype(float)
    extras['AnyRuleLikeAbnormal'] = pd.Series(rule_like, index=out.index).gt(0).astype(float)
    for win in ROLL_WINDOWS:
        abnormal_roll = pd.Series(rule_like, index=out.index).rolling(window=win, min_periods=1)
        power_neg_roll = pd.Series(power_neg, index=out.index).rolling(window=win, min_periods=1)
        power_high_roll = pd.Series(power_high, index=out.index).rolling(window=win, min_periods=1)
        extras[f'RuleLikeStorm_r{win}'] = abnormal_roll.sum().fillna(0.0)
        extras[f'RuleLikeRate_r{win}'] = abnormal_roll.mean().fillna(0.0)
        extras[f'PowerNegStorm_r{win}'] = power_neg_roll.sum().fillna(0.0)
        extras[f'PowerHighStorm_r{win}'] = power_high_roll.sum().fillna(0.0)
    for col in ROLLING_BASE_COLS:
        series = pd.to_numeric(out[col], errors='coerce') if col in out.columns else pd.Series(0.0, index=out.index)
        extras[f'{col}_d1'] = series.diff().fillna(0.0)
        extras[f'{col}_exp_range'] = (series.expanding().max() - series.expanding().min()).fillna(0.0)
        for win in ROLL_WINDOWS:
            roll = series.rolling(window=win, min_periods=1)
            mean = roll.mean()
            extras[f'{col}_r{win}_mean'] = mean
            extras[f'{col}_r{win}_std'] = roll.std().fillna(0.0)
            extras[f'{col}_r{win}_delta'] = series - mean
    return pd.concat([out, pd.DataFrame(extras, index=out.index)], axis=1)

def _clean_features(frame: pd.DataFrame, cfg: CompatCfg) -> np.ndarray:
    names = compat_feature_names(cfg)
    for name in names:
        if name not in frame.columns:
            frame[name] = NA_DEFAULT
    arr = frame[names].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(arr, nan=NA_DEFAULT, posinf=NA_DEFAULT, neginf=NA_DEFAULT).astype(np.float32)

def make_features(raw_df: pd.DataFrame, with_label: bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    renamed = raw_df.rename(columns=OFP_NAME_MAP).copy()
    out = pd.DataFrame(index=renamed.index)
    raw_ts = pd.to_numeric(renamed.get('Ts', pd.Series(np.arange(len(renamed)), index=renamed.index)), errors='coerce')
    ts_valid = pd.Series(np.isfinite(raw_ts.to_numpy(dtype=float)), index=renamed.index)
    ts = raw_ts.where(ts_valid).ffill().fillna(0.0).astype(float)
    out['Ts'] = ts
    out['TsValidMask'] = ts_valid.astype(np.float32)
    sensor_missing = pd.Series(0.0, index=renamed.index, dtype=float)
    for name in RAW_SENSOR_FEATURES:
        source = pd.to_numeric(renamed.get(name, pd.Series(np.nan, index=renamed.index)), errors='coerce')
        source_values = source.to_numpy(dtype=float)
        valid = np.isfinite(source_values) & ~np.isclose(source_values, float(NA_DEFAULT))
        if name == 'Temp':
            valid &= ~np.isclose(source_values, float(TEMP_OUTLIER))
        valid_series = pd.Series(valid, index=renamed.index)
        out[name] = source.where(valid_series).ffill().fillna(0.0).astype(float)
        out[f'{name}ValidMask'] = valid_series.astype(np.float32)
        sensor_missing += (~valid_series).astype(float)
    raw_delta = raw_ts.diff()
    delta_valid = ts_valid & ts_valid.shift(1, fill_value=False) & raw_delta.ge(0.0)
    delta_seconds = raw_delta.where(delta_valid, 0.0).fillna(0.0).clip(lower=0.0)
    positive_delta = delta_seconds[delta_seconds > 0.0]
    nominal_delta = float(positive_delta.median()) if len(positive_delta) else 0.0
    gap_threshold = nominal_delta * 1.5 if nominal_delta > 0.0 else float('inf')
    out['DeltaSeconds'] = delta_seconds.astype(float)
    out['DeltaSecondsValidMask'] = delta_valid.astype(np.float32)
    out['TsDelta'] = delta_seconds.astype(float)
    out['SamplingGapFlag'] = delta_seconds.gt(gap_threshold).astype(np.float32)
    out['SensorMissingCount'] = sensor_missing.astype(np.float32)
    out['AnySensorMissingFlag'] = sensor_missing.gt(0.0).astype(np.float32)
    out['TempInvalidFlag'] = (out['TempValidMask'] <= 0.0).astype(np.float32)
    out['FeCoCurrTemp'] = rolling_corr(out['Curr'], out['Temp'], fill=99.0)
    out['FeCoCurrTxP0'] = rolling_corr(out['Curr'], out['TxP0'], fill=-99.0)
    out['FeCoCurrRxP0'] = rolling_corr(out['Curr'], out['RxP0'], fill=-99.0)
    out['FeCoTxP0RxP0'] = rolling_corr(out['TxP0'], out['RxP0'], fill=99.0)
    out['FeTxP0-Max'] = out['TxP0'] - out[['TxP1', 'TxP2', 'TxP3', 'TxP4']].max(axis=1)
    out['FeTxP0-Min'] = out['TxP0'] - out[['TxP1', 'TxP2', 'TxP3', 'TxP4']].min(axis=1)
    out['FeRxP0-Max'] = out['RxP0'] - out[['RxP1', 'RxP2', 'RxP3', 'RxP4']].max(axis=1)
    out['FeRxP0-Min'] = out['RxP0'] - out[['RxP1', 'RxP2', 'RxP3', 'RxP4']].min(axis=1)
    if with_label and 'Ano' in renamed.columns:
        out['Ano'] = pd.to_numeric(renamed['Ano'], errors='coerce').fillna(0.0)
    preserved_ts = out['Ts'].to_numpy(dtype=np.float64, copy=True)
    out = out.astype(np.float32)
    out['Ts'] = preserved_ts
    return (out, pd.DataFrame(columns=out.columns, dtype=np.float32))

def _module_label(file_name: str, label_by_file: dict[str, int]) -> int:
    return int(label_by_file.get(file_name, 0))

def _normal_rule_predict(frame: pd.DataFrame, rule_mode: str) -> np.ndarray:
    mode = str(rule_mode).lower()
    if mode in {'none', 'off', 'false', '0', 'temp'} or len(frame) == 0:
        return np.zeros(len(frame), dtype=np.int8)
    if mode == 'ofp_rules':
        available = [name for name in RULE_FEATURES if name in frame.columns]
        if not available:
            raise ValueError("rule_mode='ofp_rules' requires feature_mode='ofp' or 'ofp_plus'")
        values = frame[available].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return values.gt(0.0).any(axis=1).astype(np.int8).to_numpy()
    if mode != 'model2_simple':
        raise ValueError(f'Unknown rule_mode={rule_mode!r}')
    temp = pd.to_numeric(frame.get('Temp', 0.0), errors='coerce').fillna(NA_DEFAULT)
    curr = pd.to_numeric(frame.get('Curr', 0.0), errors='coerce').fillna(NA_DEFAULT)
    tx_min = pd.to_numeric(frame.get('FeTxP0-Min', 0.0), errors='coerce').fillna(0.0)
    rx_min = pd.to_numeric(frame.get('FeRxP0-Min', 0.0), errors='coerce').fillna(0.0)
    tx_max = pd.to_numeric(frame.get('FeTxP0-Max', 0.0), errors='coerce').fillna(0.0)
    rx_max = pd.to_numeric(frame.get('FeRxP0-Max', 0.0), errors='coerce').fillna(0.0)
    pred = (temp < 0.0) | (curr < 5000.0) | (temp.expanding().max() - temp.expanding().min() > 100.0) | (curr.expanding().max() - curr.expanding().min() > 6000.0) | temp.expanding().std().fillna(0).gt(10.0) | curr.expanding().std().fillna(0).gt(1500.0) | (tx_min < 0.0) | (rx_min < 0.0) | (tx_max > 1000.0) | (rx_max > 1000.0)
    for col in ['TxP0', 'RxP0', 'RxP1', 'RxP2', 'RxP3', 'RxP4', 'TxP1', 'TxP2', 'TxP3', 'TxP4']:
        if col in frame.columns:
            pred = pred | pd.to_numeric(frame[col], errors='coerce').fillna(0.0).lt(0.0)
    return pred.to_numpy(dtype=np.int8)

def _make_targets(raw: pd.DataFrame, normal: pd.DataFrame, raw_idx: np.ndarray, file_name: str, cfg: CompatCfg, label_by_file: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_timestamps = pd.to_numeric(raw.get('timestamp', pd.Series(np.nan, index=raw.index)), errors='coerce').to_numpy(dtype=float)
    finite_raw = np.isfinite(raw_timestamps)
    finite = finite_raw[raw_idx] if len(raw_idx) else np.zeros(0, dtype=bool)
    anomaly_labels = pd.to_numeric(normal['Ano'], errors='coerce').fillna(0.0).to_numpy(dtype=float) > 0 if 'Ano' in normal.columns else np.zeros(len(normal), dtype=bool)
    mode = str(cfg.target_mode).lower()
    if mode in {'ahead120', 'ahead_horizon'}:
        valid_mask = first_event_valid_mask(raw)[raw_idx] if len(raw_idx) else np.zeros(0, dtype=bool)
        horizon_hours = 120.0 if mode == 'ahead120' else float(cfg.target_horizon_hours)
        labels = first_anomaly_ahead_label(raw, horizon_seconds=horizon_hours * 3600.0)[raw_idx] if len(raw_idx) else np.zeros(0, dtype=np.int8)
    elif mode == 'pre_event':
        anomaly_source = raw['anomaly'] if 'anomaly' in raw.columns else pd.Series(0.0, index=raw.index)
        anomaly_raw = pd.to_numeric(anomaly_source, errors='coerce').fillna(0.0).to_numpy(dtype=float) > 0.0
        first_ts = float(np.min(raw_timestamps[anomaly_raw & finite_raw])) if np.any(anomaly_raw & finite_raw) else None
        if first_ts is None:
            valid_mask = finite
            labels = np.zeros(len(normal), dtype=np.int8)
        else:
            valid_mask = finite & (raw_timestamps[raw_idx] < first_ts)
            labels = np.full(len(normal), _module_label(file_name, label_by_file), dtype=np.int8)
    elif mode == 'anomaly':
        valid_mask = finite
        labels = anomaly_labels.astype(np.int8)
    elif mode == 'module_fault':
        valid_mask = finite
        labels = np.full(len(normal), _module_label(file_name, label_by_file), dtype=np.int8)
    else:
        raise ValueError(f'Unknown target_mode={cfg.target_mode!r}; use pre_event, ahead_horizon, ahead120, anomaly, or module_fault')
    return (valid_mask.astype(bool), labels.astype(np.int8), anomaly_labels.astype(np.int8))

def _read_feature_parts(data_dir: Path, file_name: str, cfg: CompatCfg, label_by_file: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(data_dir / file_name)
    if bool(cfg.preserve_timepoints):
        normal, extra = make_features(raw, with_label=True)
    else:
        normal, extra = make_model2_features(raw, with_label=True)
    feature_mode = str(cfg.feature_mode).lower()
    if feature_mode in {'ofp', 'ofp_expert_stat', 'ofp_plus', 'ofp_expert_stat_plus'}:
        normal = add_ofp_expert_stat_features(normal)
    if feature_mode in {'model2_plus', 'plus', 'ofp_plus', 'ofp_expert_stat_plus'}:
        normal = add_model2_plus_features(normal)
    raw_idx = normal.index.to_numpy(dtype=int)
    valid_mask, labels, anomaly_labels = _make_targets(raw, normal, raw_idx, file_name, cfg, label_by_file)
    rule_pred = _normal_rule_predict(normal, cfg.rule_mode)
    return (normal, extra, valid_mask.astype(bool), labels.astype(np.int8), anomaly_labels.astype(np.int8), rule_pred)

def _module_feature_cache_path(cfg: CompatCfg, file_name: str) -> Path | None:
    if not str(cfg.module_cache_dir).strip():
        return None
    namespace = '__'.join([FEATURE_CACHE_VERSION, str(cfg.feature_mode).lower(), str(cfg.target_mode).lower(), f'h{float(cfg.target_horizon_hours):g}', 'preserve' if bool(cfg.preserve_timepoints) else 'drop_timepoints', str(cfg.rule_mode).lower()])
    return Path(cfg.module_cache_dir) / namespace / f'{Path(file_name).stem}.npz'

def _read_feature_arrays(data_dir: Path, file_name: str, cfg: CompatCfg, label_by_file: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = _module_feature_cache_path(cfg, file_name)
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                return (cached['timestamps'].astype(np.int64), cached['features'].astype(np.float32), cached['valid_mask'].astype(bool), cached['labels'].astype(np.int8), cached['anomaly_labels'].astype(np.int8), cached['rule_pred'].astype(np.int8), cached['extra'].astype(np.int64))
        except (OSError, ValueError, KeyError):
            cache_path.unlink(missing_ok=True)
    normal, extra, valid_mask, labels, anomaly_labels, rule_pred = _read_feature_parts(data_dir, file_name, cfg, label_by_file)
    timestamps = normal['Ts'].to_numpy(dtype=np.int64, copy=True) if len(normal) else np.zeros(0, dtype=np.int64)
    features = _clean_features(normal.copy(), cfg)
    extra_ts = extra['Ts'].to_numpy(dtype=np.int64, copy=True) if len(extra) else np.zeros(0, dtype=np.int64)
    extra_temp = pd.to_numeric(extra['Temp'], errors='coerce').to_numpy(dtype=float) if len(extra) else np.zeros(0, dtype=float)
    extra_pred = (extra_temp == TEMP_OUTLIER).astype(np.int8)
    if str(cfg.rule_mode).lower() in {'none', 'off', 'false', '0'}:
        extra_pred = np.zeros_like(extra_pred)
    extra_array = np.stack([extra_ts, extra_pred], axis=1).astype(np.int64) if len(extra_ts) else np.zeros((0, 2), dtype=np.int64)
    item = (timestamps, features.astype(np.float32), valid_mask.astype(bool), labels.astype(np.int8), anomaly_labels.astype(np.int8), rule_pred.astype(np.int8), extra_array)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_name(f'.{cache_path.name}.{os.getpid()}.tmp')
        try:
            with temp_path.open('wb') as handle:
                np.savez_compressed(handle, timestamps=item[0], features=item[1], valid_mask=item[2], labels=item[3], anomaly_labels=item[4], rule_pred=item[5], extra=item[6])
            os.replace(temp_path, cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return item

def compute_feature_norm_stats(data_dir: Path, train_files: list[str], cfg: CompatCfg, label_by_file: dict[str, int]) -> FeatureNormStats:
    feature_names = compat_feature_names(cfg)
    sums = np.zeros(len(feature_names), dtype=np.float64)
    sums_sq = np.zeros(len(feature_names), dtype=np.float64)
    count = 0
    for name in train_files:
        _timestamps, features, valid_mask, _labels, _anomaly_labels, _rule_pred, _extra = _read_feature_arrays(data_dir, name, cfg, label_by_file)
        if not len(features) or not np.any(valid_mask):
            continue
        arr = features[valid_mask].astype(np.float32)
        sums += arr.sum(axis=0)
        sums_sq += (arr.astype(np.float64) ** 2).sum(axis=0)
        count += int(arr.shape[0])
    denom = max(count, 1)
    mean = sums / denom
    var = np.maximum(sums_sq / denom - mean ** 2, 1e-06)
    for idx, name in enumerate(feature_names):
        if name.endswith('ValidMask') or name.startswith('Ru') or name.endswith('InvalidFlag') or name.endswith('MissingFlag') or name.endswith('GapFlag') or ('MissingCount' in name):
            mean[idx] = 0.0
            var[idx] = 1.0
    return FeatureNormStats(mean=mean.astype(float).tolist(), std=np.sqrt(var).astype(float).tolist(), rows_seen=int(count), files_seen=int(len(train_files)), feature_names=feature_names)

class Model2FeatureCache:

    def __init__(self, data_dir: Path, mean: np.ndarray, std: np.ndarray, cfg: CompatCfg, label_by_file: dict[str, int]) -> None:
        self.data_dir = Path(data_dir)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-06)
        self.cfg = cfg
        self.label_by_file = label_by_file
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            item = self.cache.pop(file_name)
            self.cache[file_name] = item
            return item
        timestamps, features, valid_mask, labels, anomaly_labels, rule_pred, extra_array = _read_feature_arrays(self.data_dir, file_name, self.cfg, self.label_by_file)
        features = ((features - self.mean) / self.std).astype(np.float32)
        item = (timestamps, features, valid_mask.astype(bool), labels.astype(np.int8), anomaly_labels.astype(np.int8), rule_pred.astype(np.int8), extra_array.astype(np.int64))
        self.cache[file_name] = item
        while len(self.cache) > int(self.cfg.max_cached_files):
            self.cache.popitem(last=False)
        return item

def _rank_normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float32)
    sorted_values = values[order]
    start = 0
    denominator = float(len(values) - 1)
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * float(start + end - 1) / denominator
        ranks[order[start:end]] = average_rank
        start = end
    return ranks

def row_signal_scores(features: np.ndarray, rule_pred: np.ndarray, feature_names: list[str] | None=None, mode: str='expert', temporal_fraction: float=0.5) -> np.ndarray:
    if len(features) == 0:
        return np.zeros(0, dtype=np.float32)
    finite = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    expert_score = np.mean(np.abs(finite), axis=1)
    if len(rule_pred) == len(expert_score):
        expert_score = expert_score + 5.0 * np.asarray(rule_pred, dtype=np.float32)
    names = list(feature_names or [])
    raw_indices = [names.index(name) for name in ROLLING_BASE_COLS[:12] if name in names]
    temporal_score = np.zeros(len(finite), dtype=np.float32)
    if raw_indices and len(finite) > 1:
        raw = finite[:, raw_indices]
        temporal_score[1:] = np.mean(np.abs(raw[1:] - raw[:-1]), axis=1)
    resolved = str(mode).lower()
    if resolved == 'expert':
        score = expert_score
    elif resolved in {'temporal', 'temporal_change'}:
        score = temporal_score
    elif resolved in {'mixed', 'expert_temporal'}:
        mix = float(np.clip(temporal_fraction, 0.0, 1.0))
        score = (1.0 - mix) * _rank_normalize(expert_score) + mix * _rank_normalize(temporal_score)
    else:
        raise ValueError(f'Unknown sample_signal_mode={mode!r}; use expert, temporal, or mixed')
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def choose_positions_by_signal(positions: np.ndarray, n_take: int, scores: np.ndarray, cfg: CompatCfg, seed: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    n_take = int(n_take)
    if n_take <= 0 or len(positions) == 0:
        return np.empty(0, dtype=np.int64)
    if n_take >= len(positions):
        return np.sort(positions).astype(np.int64)
    mode = str(cfg.sample_selection).lower()
    rng = np.random.default_rng(int(seed))
    if mode in {'random', 'uniform'}:
        return np.sort(rng.choice(positions, size=n_take, replace=False)).astype(np.int64)
    pos_scores = np.asarray(scores[positions], dtype=np.float32)
    order = np.argsort(pos_scores)[::-1]
    if mode in {'signal_topk', 'topk', 'hard'}:
        return np.sort(positions[order[:n_take]]).astype(np.int64)
    if mode in {'hybrid', 'topk_random'}:
        n_top = int(round(float(n_take) * float(cfg.sample_topk_fraction)))
        n_top = min(max(1, n_top), n_take)
        top = positions[order[:n_top]]
        if n_top >= n_take:
            return np.sort(top).astype(np.int64)
        rest = positions[order[n_top:]]
        n_random = n_take - n_top
        if len(rest) <= n_random:
            chosen = np.concatenate([top, rest])
        else:
            chosen = np.concatenate([top, rng.choice(rest, size=n_random, replace=False)])
        return np.sort(chosen).astype(np.int64)
    raise ValueError(f'Unknown sample_selection={cfg.sample_selection!r}; use random, signal_topk, or hybrid')

def temporal_position_weights(timestamps: np.ndarray, labels: np.ndarray, anomaly_labels: np.ndarray, positions: np.ndarray, cfg: CompatCfg) -> np.ndarray:
    weights = np.ones(len(positions), dtype=np.float32)
    max_extra = float(cfg.temporal_positive_weight)
    if max_extra <= 0.0 or len(positions) == 0:
        return weights
    pos_mask = labels[positions] > 0
    if not np.any(pos_mask) or not np.any(anomaly_labels > 0):
        return weights
    first_ts = float(timestamps[np.flatnonzero(anomaly_labels > 0)[0]])
    horizon = max(float(cfg.temporal_weight_horizon_hours), 1e-06)
    lead_hours = (first_ts - timestamps[positions].astype(float)) / 3600.0
    proximity = np.where(lead_hours >= 0.0, 1.0 - np.clip(lead_hours / horizon, 0.0, 1.0), 1.0)
    weights[pos_mask] += (max_extra * proximity[pos_mask]).astype(np.float32)
    return weights.astype(np.float32)

class CompatBatchedDataset(IterableDataset):

    def __init__(self, file_names: list[str], cache: Model2FeatureCache, cfg: CompatCfg) -> None:
        self.file_names = list(file_names)
        self.cache = cache
        self.cfg = cfg
        self.feature_names = compat_feature_names(cfg)
        self.selected_positions: list[np.ndarray] = []
        self.selected_labels: list[np.ndarray] = []
        self.selected_weights: list[np.ndarray] = []
        self.source_total_rows = 0
        self.source_pos_rows = 0
        self.source_neg_rows = 0
        pos_positions: list[np.ndarray] = []
        neg_positions: list[np.ndarray] = []
        neg_counts: list[int] = []
        for name in self.file_names:
            _ts, _features, valid_mask, labels, _anomaly_labels, _rule_pred, _extra = self.cache.get(name)
            positions = np.flatnonzero(valid_mask).astype(np.int64)
            pos = positions[labels[positions] > 0].astype(np.int64) if len(positions) else np.empty(0, dtype=np.int64)
            neg = positions[labels[positions] <= 0].astype(np.int64) if len(positions) else np.empty(0, dtype=np.int64)
            pos_positions.append(pos)
            neg_positions.append(neg)
            neg_counts.append(int(len(neg)))
            self.source_total_rows += int(len(positions))
            self.source_pos_rows += int(len(pos))
            self.source_neg_rows += int(len(neg))
        sampling_mode = str(cfg.sampling_mode).lower()
        if sampling_mode in {'row_ratio', 'pos_all_neg_ratio'}:
            target_neg = int(round(float(self.source_pos_rows) * float(cfg.negative_ratio)))
            alloc = allocate_stratified_negative_counts(np.asarray(neg_counts, dtype=np.int64), target_neg)
        elif sampling_mode in {'module_balanced', 'module'}:
            alloc = np.asarray([min(len(neg), int(cfg.negative_windows_per_faulty_module) if len(pos) > 0 else int(cfg.normal_windows_per_module)) for pos, neg in zip(pos_positions, neg_positions)], dtype=np.int64)
        else:
            raise ValueError(f'Unknown sampling_mode={cfg.sampling_mode!r}')
        self.pos_rows = 0
        self.neg_rows = int(alloc.sum())
        self.total_rows = int(self.pos_rows + self.neg_rows)
        self.total_batches = 0
        for name, pos, neg, n_neg in zip(self.file_names, pos_positions, neg_positions, alloc):
            timestamps, features, _valid_mask, labels, anomaly_labels, rule_pred, _extra = self.cache.get(name)
            signal_scores = row_signal_scores(features, rule_pred, self.feature_names, cfg.sample_signal_mode, cfg.sample_signal_temporal_fraction)
            if sampling_mode in {'module_balanced', 'module'} and len(pos) > int(cfg.positive_windows_per_module) > 0:
                stable = zlib.crc32((name + '::pos').encode('utf-8')) & 4294967295
                pos = choose_positions_by_signal(pos, int(cfg.positive_windows_per_module), signal_scores, cfg, int(cfg.seed) + int(stable))
            if int(n_neg) >= len(neg):
                chosen_neg = neg
            elif int(n_neg) > 0:
                stable = zlib.crc32(name.encode('utf-8')) & 4294967295
                chosen_neg = choose_positions_by_signal(neg, int(n_neg), signal_scores, cfg, int(cfg.seed) + int(stable))
            else:
                chosen_neg = np.empty(0, dtype=np.int64)
            selected = np.sort(np.concatenate([pos, chosen_neg])).astype(np.int64)
            self.selected_positions.append(selected)
            self.selected_labels.append(labels[selected].astype(np.int8))
            self.selected_weights.append(temporal_position_weights(timestamps, labels, anomaly_labels, selected, cfg))
            self.pos_rows += int(len(pos))
            self.total_batches += int(math.ceil(len(selected) / max(1, int(cfg.batch_size)))) if len(selected) else 0
        self.total_rows = int(self.pos_rows + self.neg_rows)
        print(f'[compat dataset] files={len(self.file_names)} source_rows={self.source_total_rows} source_pos={self.source_pos_rows} source_neg={self.source_neg_rows} sampled_rows={self.total_rows} sampled_pos={self.pos_rows} sampled_neg={self.neg_rows} sampling={cfg.sampling_mode} negative_ratio={cfg.negative_ratio} per_module pos={cfg.positive_windows_per_module} faulty_neg={cfg.negative_windows_per_faulty_module} normal_neg={cfg.normal_windows_per_module} selection={cfg.sample_selection} signal={cfg.sample_signal_mode} temporal_mix={cfg.sample_signal_temporal_fraction} temporal_pos_w={cfg.temporal_positive_weight} features={len(self.feature_names)} batches={self.total_batches}')

    def __len__(self) -> int:
        return int(self.total_batches)

    def apply_adaptive_negative_weights(self, scores_by_file: dict[str, np.ndarray], max_extra: float) -> dict[str, float]:
        max_extra = float(max_extra)
        if max_extra <= 0.0:
            return {'updated_negative_rows': 0.0, 'min_score': 0.0, 'max_score': 0.0}
        neg_scores: list[np.ndarray] = []
        for name, labels, _weights in zip(self.file_names, self.selected_labels, self.selected_weights):
            scores = np.asarray(scores_by_file.get(name, np.zeros(len(labels), dtype=np.float32)), dtype=np.float32)
            if len(scores) != len(labels):
                continue
            neg_scores.append(scores[labels <= 0])
        all_neg = np.concatenate([x for x in neg_scores if len(x)]) if any((len(x) for x in neg_scores)) else np.zeros(0, dtype=np.float32)
        if len(all_neg) == 0:
            return {'updated_negative_rows': 0.0, 'min_score': 0.0, 'max_score': 0.0}
        lo = float(np.nanmin(all_neg))
        hi = float(np.nanmax(all_neg))
        denom = max(hi - lo, 1e-08)
        updated = 0
        for idx, (name, labels, weights) in enumerate(zip(self.file_names, self.selected_labels, self.selected_weights)):
            scores = np.asarray(scores_by_file.get(name, np.zeros(len(labels), dtype=np.float32)), dtype=np.float32)
            if len(scores) != len(labels):
                continue
            neg_mask = labels <= 0
            scaled = np.clip((scores - lo) / denom, 0.0, 1.0).astype(np.float32)
            weights = weights.copy()
            weights[neg_mask] = 1.0 + max_extra * scaled[neg_mask]
            self.selected_weights[idx] = weights.astype(np.float32)
            updated += int(np.sum(neg_mask))
        return {'updated_negative_rows': float(updated), 'min_score': lo, 'max_score': hi}

    def __iter__(self):
        seq_len = int(self.cfg.seq_len)
        batch_size = int(self.cfg.batch_size)
        n_features = len(self.feature_names)
        pairs = list(zip(self.file_names, self.selected_positions))
        worker = get_worker_info()
        if worker is not None:
            pairs = pairs[worker.id::worker.num_workers]
        weight_by_name = {name: weights for name, weights in zip(self.file_names, self.selected_weights)}
        for name, selected in pairs:
            if len(selected) <= 0:
                continue
            _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = self.cache.get(name)
            selected_weights = weight_by_name.get(name, np.ones(len(selected), dtype=np.float32))
            pad_x = np.zeros((seq_len - 1, n_features), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, features.astype(np.float32)], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, np.ones_like(features, dtype=np.float32)], axis=0))
            x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            ys = torch.from_numpy(labels[selected].astype(np.float32))
            for start in range(0, len(selected), batch_size):
                end = min(len(selected), start + batch_size)
                idx = torch.from_numpy(selected[start:end])
                yield (x_windows.index_select(0, idx).contiguous(), m_windows.index_select(0, idx).contiguous(), ys[start:end].contiguous(), torch.from_numpy(selected_weights[start:end].astype(np.float32)).contiguous())

def cuda_autocast(enabled: bool):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=bool(enabled))
    return torch.cuda.amp.autocast(enabled=bool(enabled))

def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f'{hours}h{minutes:02d}m{secs:02d}s'
    if minutes > 0:
        return f'{minutes}m{secs:02d}s'
    return f'{secs}s'

def format_rate(rows: float, seconds: float) -> str:
    if float(seconds) <= 0.0:
        return '0.0 rows/s'
    return f'{float(rows) / max(float(seconds), 1e-09):.1f} rows/s'

def progress_bar(current: int, total: int, width: int=24) -> str:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    filled = int(round(width * current / total))
    return '[' + '#' * filled + '-' * (width - filled) + f'] {100.0 * current / total:5.1f}%'

def log_run_header(title: str, fields: dict[str, object]) -> None:
    line = '=' * 88
    print(line)
    print(f'{title}')
    print('-' * 88)
    for key, value in fields.items():
        print(f'{key:>22}: {value}')
    print(line)

def format_epoch_status(tag: str, model_name: str, fold: int | None, epoch: int, total_epochs: int, loss: float, rows: int, epoch_seconds: float, elapsed_seconds: float, eta_seconds: float) -> str:
    fold_text = f' fold={fold}' if fold is not None else ''
    return f'[{tag}] model={model_name}{fold_text} ep={epoch:02d}/{total_epochs:02d} {progress_bar(epoch, total_epochs, width=18)} loss={loss:.5f} rows={rows} rate={format_rate(rows, epoch_seconds)} epoch={format_duration(epoch_seconds)} elapsed={format_duration(elapsed_seconds)} eta={format_duration(eta_seconds)}'

def effective_pos_weight(dataset: CompatBatchedDataset, cfg: CompatCfg) -> float:
    value = max(1.0, float(dataset.neg_rows) / max(float(dataset.pos_rows), 1.0))
    if float(cfg.pos_weight_cap) > 0:
        value = min(value, float(cfg.pos_weight_cap))
    return float(value)

def apply_threshold(frame: pd.DataFrame, threshold: float, confirm_k: int=1, confirm_m: int=1) -> pd.DataFrame:
    out = frame.copy()
    score = pd.to_numeric(out.get('score'), errors='coerce')
    deep_hit = (score >= float(threshold)).fillna(False).astype(int)
    k = max(1, int(confirm_k))
    m = max(k, int(confirm_m))
    if k > 1 or m > 1:
        deep_pred = deep_hit.rolling(window=m, min_periods=k).sum().ge(k).astype(int)
    else:
        deep_pred = deep_hit
    rule_pred = pd.to_numeric(out.get('rule_predict', 0), errors='coerce').fillna(0).astype(int)
    out['predict'] = ((deep_pred > 0) | (rule_pred > 0)).astype(int)
    return out[['timestamp', 'predict', 'score', 'rule_predict', 'source']]

def file_label_map(index_df: pd.DataFrame) -> dict[str, int]:
    return {str(row['file_name']): int(row['Label']) for _, row in index_df[['file_name', 'Label']].drop_duplicates('file_name').iterrows()}

def parse_threshold_grid(text: str) -> list[float]:
    raw = str(text).strip()
    if not raw:
        return []
    if ':' in raw:
        parts = [float(x) for x in raw.split(':')]
        if len(parts) != 3:
            raise ValueError('threshold_grid range must be start:end:step')
        start, end, step = parts
        if step <= 0:
            raise ValueError('threshold_grid step must be positive')
        values = []
        cur = start
        while cur <= end + 1e-12:
            values.append(round(float(cur), 10))
            cur += step
        return values
    values = [float(x) for x in raw.replace(';', ',').split(',') if x.strip()]
    return sorted(set(values))

def _first_positive_timestamp_frame(frame: pd.DataFrame, predict_column: str, before_ts: float | None=None) -> tuple[int, float | None]:
    if frame.empty or predict_column not in frame.columns or 'timestamp' not in frame.columns:
        return (0, None)
    pred = pd.to_numeric(frame[predict_column], errors='coerce').fillna(0.0)
    timestamps = pd.to_numeric(frame['timestamp'], errors='coerce')
    valid = (pred > 0) & timestamps.notna()
    if before_ts is not None:
        valid = valid & (timestamps.astype(float) < float(before_ts))
    positive = pd.to_numeric(frame.loc[valid, 'timestamp'], errors='coerce').dropna()
    if positive.empty:
        return (0, None)
    return (1, float(positive.min()))

def evaluate_prediction_frames(prediction_frames: dict[str, pd.DataFrame], label_dir: Path, predict_column: str='predict', min_hit_lead_hours: float=0.0, truth_by_name: dict[str, tuple[int, float | None]] | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in sorted(prediction_frames):
        label_path = Path(label_dir) / name
        if not label_path.exists():
            continue
        if truth_by_name is not None and name in truth_by_name:
            true_label, true_ts = truth_by_name[name]
        else:
            true_label, true_ts = first_positive_timestamp(label_path, 'anomaly')
        candidate_timestamps = pd.to_numeric(prediction_frames[name].get('timestamp', pd.Series(dtype=float)), errors='coerce').dropna()
        warnable_positive = 0
        if true_label > 0 and true_ts is not None:
            latest_legal_ts = float(true_ts) - float(min_hit_lead_hours) * 3600.0
            candidate_values = candidate_timestamps.astype(float)
            warnable_positive = int(bool(((candidate_values < float(true_ts)) & (candidate_values <= latest_legal_ts)).any()))
        predict_label, predict_ts = _first_positive_timestamp_frame(prediction_frames[name], predict_column, before_ts=true_ts if true_label > 0 else None)
        valid_predict_positive = 0
        hit = 0
        lead_hour = None
        if predict_label > 0:
            if true_label == 0:
                valid_predict_positive = 1
            elif true_ts is not None and predict_ts is not None and (true_ts > predict_ts):
                lead_hour = abs(true_ts - predict_ts) / 3600.0
                if lead_hour >= float(min_hit_lead_hours):
                    valid_predict_positive = 1
                    hit = 1
        rows.append({'file_name': name, 'true_label': int(true_label), 'predict_label': int(predict_label), 'true_ts': true_ts, 'predict_ts': predict_ts, 'valid_predict_positive': int(valid_predict_positive), 'hit': int(hit), 'lead_hour': lead_hour, 'warnable_positive': int(warnable_positive), 'non_warnable_positive': int(true_label > 0 and (not warnable_positive)), 'min_hit_lead_hours': float(min_hit_lead_hours)})
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise ValueError('No matching validation prediction/label frames')
    true_pos = set(detail_df.loc[detail_df['true_label'] > 0, 'file_name'])
    pred_pos = set(detail_df.loc[detail_df['valid_predict_positive'] > 0, 'file_name'])
    hit_pos = true_pos & pred_pos
    tp = len(hit_pos)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = len(detail_df) - tp - fp - fn
    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(detail_df) if len(detail_df) else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    false_alarm_rate = fp / (fp + tn) if fp + tn > 0 else 0.0
    false_alarms_per_1000_normal = 1000.0 * false_alarm_rate
    predicted_positive_rate = len(pred_pos) / len(detail_df) if len(detail_df) else 0.0
    warnable_positive_cnt = int(detail_df['warnable_positive'].sum())
    non_warnable_positive_cnt = int(detail_df['non_warnable_positive'].sum())
    warnable_recall = tp / warnable_positive_cnt if warnable_positive_cnt > 0 else 0.0
    first_warning_upper_bound = warnable_positive_cnt / len(true_pos) if true_pos else 0.0
    lead_hours = detail_df.loc[detail_df['hit'] > 0, 'lead_hour'].dropna().astype(float)
    avg_lead_hour = float(lead_hours.mean()) if not lead_hours.empty else 0.0
    median_lead_hour = float(lead_hours.median()) if not lead_hours.empty else 0.0
    p25_lead_hour = float(lead_hours.quantile(0.25)) if not lead_hours.empty else 0.0
    p75_lead_hour = float(lead_hours.quantile(0.75)) if not lead_hours.empty else 0.0
    min_lead_hour = float(lead_hours.min()) if not lead_hours.empty else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    summary_df = pd.DataFrame({'Item': ['final_score', 'f1_score', 'precision', 'recall', 'all_hit_cnt', 'all_predict_pos_cnt', 'all_true_pos_cnt', 'avg_lead_score', 'avg_lead_hour', 'min_lead_score', 'min_lead_hour', 'lead_pread_cnt', 'accuracy', 'balanced_accuracy', 'specificity', 'false_alarm_rate', 'false_alarms_per_1000_normal', 'predicted_positive_rate', 'recall_all_positive', 'warnable_recall', 'warnable_positive_cnt', 'non_warnable_positive_cnt', 'first_warning_upper_bound', 'median_lead_hour', 'p25_lead_hour', 'p75_lead_hour', 'tp', 'fp', 'fn', 'tn', 'evaluated_module_cnt', 'min_hit_lead_hours'], 'Value': [final_score, f1, precision, recall, tp, len(pred_pos), len(true_pos), avg_lead_score, avg_lead_hour, min_lead_score, min_lead_hour, int(detail_df['hit'].sum()), accuracy, balanced_accuracy, specificity, false_alarm_rate, false_alarms_per_1000_normal, predicted_positive_rate, recall, warnable_recall, warnable_positive_cnt, non_warnable_positive_cnt, first_warning_upper_bound, median_lead_hour, p25_lead_hour, p75_lead_hour, tp, fp, fn, tn, len(detail_df), float(min_hit_lead_hours)]})
    return (summary_df, detail_df)

def evaluate_prediction_dir_compat(prediction_dir: Path, label_dir: Path, predict_column: str='predict', min_hit_lead_hours: float=0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(prediction_dir).glob('*.csv')):
        frames[path.name] = pd.read_csv(path)
    return evaluate_prediction_frames(frames, label_dir, predict_column=predict_column, min_hit_lead_hours=float(min_hit_lead_hours))
