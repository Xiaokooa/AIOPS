from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.FITS.model import build_model as build_fits
from OFP_DL_official.FTEformer.model import build_model as build_fteformer
from OFP_DL_official.ModernTCN.model import build_model as build_moderntcn
from OFP_DL_official.PatchTST.model import build_model as build_patchtst
from OFP_DL_official.common.formal_data import allocate_stratified_negative_counts
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.common.losses import primary_logit, split_alarm_aux_logits
from OFP_DL_official.common.trainer import format_metric_summary, resolve_runtime_device, runtime_device_summary
from OFP_DL_official.iTransformer.model import build_model as build_itransformer
from OFP_DL_official.ofp_protocol.evaluator import first_positive_timestamp, evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
    MODEL2_FEATURES,
    NA_DEFAULT,
    TEMP_OUTLIER,
    first_anomaly_ahead_label,
    first_event_valid_mask,
    make_model2_features,
)


ModelBuilder = Callable[[int, int], tuple[nn.Module, dict, bool]]

BUILDERS: dict[str, ModelBuilder] = {
    "patchtst": build_patchtst,
    "itransformer": build_itransformer,
    "fteformer": build_fteformer,
    "moderntcn": build_moderntcn,
    "fits": build_fits,
}


@dataclass
class CompatCfg:
    seq_len: int = 64
    epochs: int = 8
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    negative_ratio: float = 10.0
    pos_weight_cap: float = 20.0
    fixed_threshold: float = 0.5
    target_mode: str = "module_fault"
    feature_mode: str = "model2_plus"
    sampling_mode: str = "module_balanced"
    positive_windows_per_module: int = 96
    negative_windows_per_faulty_module: int = 24
    normal_windows_per_module: int = 24
    val_fraction: float = 0.2
    threshold_search: bool = True
    threshold_grid: str = "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50"
    threshold_metric: str = "f1_score"
    rule_mode: str = "model2_simple"
    sample_selection: str = "random"
    sample_topk_fraction: float = 0.5
    temporal_positive_weight: float = 0.0
    temporal_weight_horizon_hours: float = 120.0
    adaptive_negative_weight: float = 0.0
    adaptive_warmup_epochs: int = 1
    max_cached_files: int = 512
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    amp: bool = False
    allow_tf32: bool = True
    log_batches: int = 1000
    module_cache_dir: str = ""
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
        return np.asarray(self.mean, dtype=np.float32), np.maximum(np.asarray(self.std, dtype=np.float32), 1e-6)


ROLLING_BASE_COLS = [
    "Temp",
    "Curr",
    "TxP0",
    "RxP0",
    "RxP1",
    "RxP2",
    "RxP3",
    "RxP4",
    "TxP1",
    "TxP2",
    "TxP3",
    "TxP4",
    "FeTxP0-Max",
    "FeTxP0-Min",
    "FeRxP0-Max",
    "FeRxP0-Min",
]
ROLL_WINDOWS = (3, 6, 12, 24, 64)
LANE_PLUS_FEATURES = [
    "TxLaneRange",
    "RxLaneRange",
    "TxLaneStd",
    "RxLaneStd",
    "TxP0MinusLaneMean",
    "RxP0MinusLaneMean",
]
ROLLING_PLUS_FEATURES = []
for _col in ROLLING_BASE_COLS:
    ROLLING_PLUS_FEATURES.append(f"{_col}_d1")
    ROLLING_PLUS_FEATURES.append(f"{_col}_exp_range")
    for _win in ROLL_WINDOWS:
        ROLLING_PLUS_FEATURES.extend(
            [
                f"{_col}_r{_win}_mean",
                f"{_col}_r{_win}_std",
                f"{_col}_r{_win}_delta",
            ]
        )
DRAM_PLUS_FEATURES = [
    "DeltaSeconds",
    "ElapsedHours",
    "TempInvalidFlag",
    "CurrLowFlag",
    "TxNegativeCount",
    "RxNegativeCount",
    "PowerNegativeCount",
    "PowerHighCount",
    "RuleLikeAbnormalCount",
    "AnyRuleLikeAbnormal",
]
for _win in ROLL_WINDOWS:
    DRAM_PLUS_FEATURES.extend(
        [
            f"RuleLikeStorm_r{_win}",
            f"RuleLikeRate_r{_win}",
            f"PowerNegStorm_r{_win}",
            f"PowerHighStorm_r{_win}",
        ]
    )
COMPAT_PLUS_FEATURES = list(MODEL2_FEATURES) + LANE_PLUS_FEATURES + ROLLING_PLUS_FEATURES + DRAM_PLUS_FEATURES


def compat_feature_names(cfg: CompatCfg) -> list[str]:
    if str(cfg.feature_mode).lower() in {"model2", "base"}:
        return list(MODEL2_FEATURES)
    if str(cfg.feature_mode).lower() in {"model2_plus", "plus"}:
        return list(COMPAT_PLUS_FEATURES)
    raise ValueError(f"Unknown feature_mode={cfg.feature_mode!r}")


def add_model2_plus_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    extras: dict[str, object] = {}
    tx_cols = ["TxP1", "TxP2", "TxP3", "TxP4"]
    rx_cols = ["RxP1", "RxP2", "RxP3", "RxP4"]
    tx = out[tx_cols].apply(pd.to_numeric, errors="coerce") if set(tx_cols) <= set(out.columns) else pd.DataFrame(index=out.index)
    rx = out[rx_cols].apply(pd.to_numeric, errors="coerce") if set(rx_cols) <= set(out.columns) else pd.DataFrame(index=out.index)
    if len(tx.columns):
        extras["TxLaneRange"] = tx.max(axis=1) - tx.min(axis=1)
        extras["TxLaneStd"] = tx.std(axis=1).fillna(0.0)
        extras["TxP0MinusLaneMean"] = pd.to_numeric(out.get("TxP0", 0.0), errors="coerce") - tx.mean(axis=1)
    else:
        extras["TxLaneRange"] = 0.0
        extras["TxLaneStd"] = 0.0
        extras["TxP0MinusLaneMean"] = 0.0
    if len(rx.columns):
        extras["RxLaneRange"] = rx.max(axis=1) - rx.min(axis=1)
        extras["RxLaneStd"] = rx.std(axis=1).fillna(0.0)
        extras["RxP0MinusLaneMean"] = pd.to_numeric(out.get("RxP0", 0.0), errors="coerce") - rx.mean(axis=1)
    else:
        extras["RxLaneRange"] = 0.0
        extras["RxLaneStd"] = 0.0
        extras["RxP0MinusLaneMean"] = 0.0

    ts = pd.to_numeric(out.get("Ts", pd.Series(np.arange(len(out)), index=out.index)), errors="coerce").ffill().fillna(0.0)
    extras["DeltaSeconds"] = ts.diff().fillna(0.0).clip(lower=0.0)
    extras["ElapsedHours"] = ((ts - ts.iloc[0]) / 3600.0).fillna(0.0) if len(ts) else 0.0
    temp = pd.to_numeric(out.get("Temp", 0.0), errors="coerce").fillna(NA_DEFAULT)
    curr = pd.to_numeric(out.get("Curr", 0.0), errors="coerce").fillna(NA_DEFAULT)
    extras["TempInvalidFlag"] = temp.le(-254.0).astype(float)
    extras["CurrLowFlag"] = curr.lt(5000.0).astype(float)

    tx_all_cols = ["TxP0", "TxP1", "TxP2", "TxP3", "TxP4"]
    rx_all_cols = ["RxP0", "RxP1", "RxP2", "RxP3", "RxP4"]
    tx_all = out[[c for c in tx_all_cols if c in out.columns]].apply(pd.to_numeric, errors="coerce")
    rx_all = out[[c for c in rx_all_cols if c in out.columns]].apply(pd.to_numeric, errors="coerce")
    tx_neg = tx_all.lt(0.0).sum(axis=1) if len(tx_all.columns) else pd.Series(0.0, index=out.index)
    rx_neg = rx_all.lt(0.0).sum(axis=1) if len(rx_all.columns) else pd.Series(0.0, index=out.index)
    tx_high = tx_all.gt(1000.0).sum(axis=1) if len(tx_all.columns) else pd.Series(0.0, index=out.index)
    rx_high = rx_all.gt(1000.0).sum(axis=1) if len(rx_all.columns) else pd.Series(0.0, index=out.index)
    power_neg = tx_neg + rx_neg
    power_high = tx_high + rx_high
    rule_like = extras["TempInvalidFlag"] + extras["CurrLowFlag"] + power_neg + power_high
    extras["TxNegativeCount"] = tx_neg.astype(float)
    extras["RxNegativeCount"] = rx_neg.astype(float)
    extras["PowerNegativeCount"] = power_neg.astype(float)
    extras["PowerHighCount"] = power_high.astype(float)
    extras["RuleLikeAbnormalCount"] = rule_like.astype(float)
    extras["AnyRuleLikeAbnormal"] = pd.Series(rule_like, index=out.index).gt(0).astype(float)
    for win in ROLL_WINDOWS:
        abnormal_roll = pd.Series(rule_like, index=out.index).rolling(window=win, min_periods=1)
        power_neg_roll = pd.Series(power_neg, index=out.index).rolling(window=win, min_periods=1)
        power_high_roll = pd.Series(power_high, index=out.index).rolling(window=win, min_periods=1)
        extras[f"RuleLikeStorm_r{win}"] = abnormal_roll.sum().fillna(0.0)
        extras[f"RuleLikeRate_r{win}"] = abnormal_roll.mean().fillna(0.0)
        extras[f"PowerNegStorm_r{win}"] = power_neg_roll.sum().fillna(0.0)
        extras[f"PowerHighStorm_r{win}"] = power_high_roll.sum().fillna(0.0)

    for col in ROLLING_BASE_COLS:
        series = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(0.0, index=out.index)
        extras[f"{col}_d1"] = series.diff().fillna(0.0)
        extras[f"{col}_exp_range"] = (series.expanding().max() - series.expanding().min()).fillna(0.0)
        for win in ROLL_WINDOWS:
            roll = series.rolling(window=win, min_periods=1)
            mean = roll.mean()
            extras[f"{col}_r{win}_mean"] = mean
            extras[f"{col}_r{win}_std"] = roll.std().fillna(0.0)
            extras[f"{col}_r{win}_delta"] = series - mean
    return pd.concat([out, pd.DataFrame(extras, index=out.index)], axis=1)


def _clean_features(frame: pd.DataFrame, cfg: CompatCfg) -> np.ndarray:
    names = compat_feature_names(cfg)
    for name in names:
        if name not in frame.columns:
            frame[name] = NA_DEFAULT
    arr = frame[names].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(arr, nan=NA_DEFAULT, posinf=NA_DEFAULT, neginf=NA_DEFAULT).astype(np.float32)


def _finite_ts_mask(frame: pd.DataFrame) -> np.ndarray:
    if "Ts" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    ts = pd.to_numeric(frame["Ts"], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(ts)


def _module_label(file_name: str, label_by_file: dict[str, int]) -> int:
    return int(label_by_file.get(file_name, 0))


def _normal_rule_predict(frame: pd.DataFrame, rule_mode: str) -> np.ndarray:
    """Small, model2-style OR rules over features available in this compat path."""
    mode = str(rule_mode).lower()
    if mode in {"none", "off", "false", "0", "temp"} or len(frame) == 0:
        return np.zeros(len(frame), dtype=np.int8)
    if mode != "model2_simple":
        raise ValueError(f"Unknown rule_mode={rule_mode!r}")

    temp = pd.to_numeric(frame.get("Temp", 0.0), errors="coerce").fillna(NA_DEFAULT)
    curr = pd.to_numeric(frame.get("Curr", 0.0), errors="coerce").fillna(NA_DEFAULT)
    tx_min = pd.to_numeric(frame.get("FeTxP0-Min", 0.0), errors="coerce").fillna(0.0)
    rx_min = pd.to_numeric(frame.get("FeRxP0-Min", 0.0), errors="coerce").fillna(0.0)
    tx_max = pd.to_numeric(frame.get("FeTxP0-Max", 0.0), errors="coerce").fillna(0.0)
    rx_max = pd.to_numeric(frame.get("FeRxP0-Max", 0.0), errors="coerce").fillna(0.0)

    pred = (
        (temp < 0.0)
        | (curr < 5000.0)
        | ((temp.expanding().max() - temp.expanding().min()) > 100.0)
        | ((curr.expanding().max() - curr.expanding().min()) > 6000.0)
        | temp.expanding().std().fillna(0).gt(10.0)
        | curr.expanding().std().fillna(0).gt(1500.0)
        | (tx_min < 0.0)
        | (rx_min < 0.0)
        | (tx_max > 1000.0)
        | (rx_max > 1000.0)
    )
    for col in ["TxP0", "RxP0", "RxP1", "RxP2", "RxP3", "RxP4", "TxP1", "TxP2", "TxP3", "TxP4"]:
        if col in frame.columns:
            pred = pred | pd.to_numeric(frame[col], errors="coerce").fillna(0.0).lt(0.0)
    return pred.to_numpy(dtype=np.int8)


def _make_targets(
    raw: pd.DataFrame,
    normal: pd.DataFrame,
    raw_idx: np.ndarray,
    file_name: str,
    cfg: CompatCfg,
    label_by_file: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = _finite_ts_mask(normal)
    anomaly_labels = (
        pd.to_numeric(normal["Ano"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0
        if "Ano" in normal.columns
        else np.zeros(len(normal), dtype=bool)
    )
    mode = str(cfg.target_mode).lower()
    if mode == "ahead120":
        valid_mask = first_event_valid_mask(raw)[raw_idx] if len(raw_idx) else np.zeros(0, dtype=bool)
        labels = first_anomaly_ahead_label(raw)[raw_idx] if len(raw_idx) else np.zeros(0, dtype=np.int8)
    elif mode == "anomaly":
        valid_mask = finite
        labels = anomaly_labels.astype(np.int8)
    elif mode == "module_fault":
        valid_mask = finite
        labels = np.full(len(normal), _module_label(file_name, label_by_file), dtype=np.int8)
    else:
        raise ValueError(f"Unknown target_mode={cfg.target_mode!r}; use ahead120, anomaly, or module_fault")
    return valid_mask.astype(bool), labels.astype(np.int8), anomaly_labels.astype(np.int8)


def _read_feature_parts(
    data_dir: Path,
    file_name: str,
    cfg: CompatCfg,
    label_by_file: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(data_dir / file_name)
    normal, extra = make_model2_features(raw, with_label=True)
    if str(cfg.feature_mode).lower() in {"model2_plus", "plus"}:
        normal = add_model2_plus_features(normal)
    raw_idx = normal.index.to_numpy(dtype=int)
    valid_mask, labels, anomaly_labels = _make_targets(raw, normal, raw_idx, file_name, cfg, label_by_file)
    rule_pred = _normal_rule_predict(normal, cfg.rule_mode)
    return normal, extra, valid_mask.astype(bool), labels.astype(np.int8), anomaly_labels.astype(np.int8), rule_pred


def compute_feature_norm_stats(
    data_dir: Path,
    train_files: list[str],
    cfg: CompatCfg,
    label_by_file: dict[str, int],
) -> FeatureNormStats:
    feature_names = compat_feature_names(cfg)
    sums = np.zeros(len(feature_names), dtype=np.float64)
    sums_sq = np.zeros(len(feature_names), dtype=np.float64)
    count = 0
    for name in train_files:
        normal, _extra, valid_mask, _labels, _anomaly_labels, _rule_pred = _read_feature_parts(data_dir, name, cfg, label_by_file)
        if not len(normal) or not np.any(valid_mask):
            continue
        arr = _clean_features(normal.loc[valid_mask].copy(), cfg)
        sums += arr.sum(axis=0)
        sums_sq += (arr.astype(np.float64) ** 2).sum(axis=0)
        count += int(arr.shape[0])
    denom = max(count, 1)
    mean = sums / denom
    var = np.maximum(sums_sq / denom - mean**2, 1e-6)
    return FeatureNormStats(
        mean=mean.astype(float).tolist(),
        std=np.sqrt(var).astype(float).tolist(),
        rows_seen=int(count),
        files_seen=int(len(train_files)),
        feature_names=feature_names,
    )


class Model2FeatureCache:
    def __init__(self, data_dir: Path, mean: np.ndarray, std: np.ndarray, cfg: CompatCfg, label_by_file: dict[str, int]) -> None:
        self.data_dir = Path(data_dir)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.cfg = cfg
        self.label_by_file = label_by_file
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            item = self.cache.pop(file_name)
            self.cache[file_name] = item
            return item
        normal, extra, valid_mask, labels, anomaly_labels, rule_pred = _read_feature_parts(
            self.data_dir, file_name, self.cfg, self.label_by_file
        )
        timestamps = normal["Ts"].to_numpy(dtype=np.int64, copy=True) if len(normal) else np.zeros(0, dtype=np.int64)
        features = _clean_features(normal.copy(), self.cfg)
        features = ((features - self.mean) / self.std).astype(np.float32)
        extra_ts = extra["Ts"].to_numpy(dtype=np.int64, copy=True) if len(extra) else np.zeros(0, dtype=np.int64)
        extra_temp = (
            pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) if len(extra) else np.zeros(0, dtype=float)
        )
        extra_pred = (extra_temp == TEMP_OUTLIER).astype(np.int8)
        if str(self.cfg.rule_mode).lower() in {"none", "off", "false", "0"}:
            extra_pred = np.zeros_like(extra_pred)
        item = (
            timestamps,
            features,
            valid_mask.astype(bool),
            labels.astype(np.int8),
            anomaly_labels.astype(np.int8),
            rule_pred.astype(np.int8),
            np.stack([extra_ts, extra_pred], axis=1) if len(extra_ts) else np.zeros((0, 2), dtype=np.int64),
        )
        self.cache[file_name] = item
        while len(self.cache) > int(self.cfg.max_cached_files):
            self.cache.popitem(last=False)
        return item


def row_signal_scores(features: np.ndarray, rule_pred: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros(0, dtype=np.float32)
    finite = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    score = np.mean(np.abs(finite), axis=1)
    if len(rule_pred) == len(score):
        score = score + 5.0 * np.asarray(rule_pred, dtype=np.float32)
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def choose_positions_by_signal(
    positions: np.ndarray,
    n_take: int,
    scores: np.ndarray,
    cfg: CompatCfg,
    seed: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    n_take = int(n_take)
    if n_take <= 0 or len(positions) == 0:
        return np.empty(0, dtype=np.int64)
    if n_take >= len(positions):
        return np.sort(positions).astype(np.int64)

    mode = str(cfg.sample_selection).lower()
    rng = np.random.default_rng(int(seed))
    if mode in {"random", "uniform"}:
        return np.sort(rng.choice(positions, size=n_take, replace=False)).astype(np.int64)

    pos_scores = np.asarray(scores[positions], dtype=np.float32)
    order = np.argsort(pos_scores)[::-1]
    if mode in {"signal_topk", "topk", "hard"}:
        return np.sort(positions[order[:n_take]]).astype(np.int64)
    if mode in {"hybrid", "topk_random"}:
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
    raise ValueError(f"Unknown sample_selection={cfg.sample_selection!r}; use random, signal_topk, or hybrid")


def temporal_position_weights(
    timestamps: np.ndarray,
    labels: np.ndarray,
    anomaly_labels: np.ndarray,
    positions: np.ndarray,
    cfg: CompatCfg,
) -> np.ndarray:
    weights = np.ones(len(positions), dtype=np.float32)
    max_extra = float(cfg.temporal_positive_weight)
    if max_extra <= 0.0 or len(positions) == 0:
        return weights
    pos_mask = labels[positions] > 0
    if not np.any(pos_mask) or not np.any(anomaly_labels > 0):
        return weights
    first_ts = float(timestamps[np.flatnonzero(anomaly_labels > 0)[0]])
    horizon = max(float(cfg.temporal_weight_horizon_hours), 1e-6)
    lead_hours = (first_ts - timestamps[positions].astype(float)) / 3600.0
    proximity = np.where(lead_hours >= 0.0, 1.0 - np.clip(lead_hours / horizon, 0.0, 1.0), 1.0)
    weights[pos_mask] += (max_extra * proximity[pos_mask]).astype(np.float32)
    return weights.astype(np.float32)


class CompatBatchedDataset(IterableDataset):
    def __init__(
        self,
        file_names: list[str],
        cache: Model2FeatureCache,
        cfg: CompatCfg,
    ) -> None:
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
        if sampling_mode in {"row_ratio", "pos_all_neg_ratio"}:
            target_neg = int(round(float(self.source_pos_rows) * float(cfg.negative_ratio)))
            alloc = allocate_stratified_negative_counts(np.asarray(neg_counts, dtype=np.int64), target_neg)
        elif sampling_mode in {"module_balanced", "module"}:
            alloc = np.asarray(
                [
                    min(
                        len(neg),
                        int(cfg.negative_windows_per_faulty_module) if len(pos) > 0 else int(cfg.normal_windows_per_module),
                    )
                    for pos, neg in zip(pos_positions, neg_positions)
                ],
                dtype=np.int64,
            )
        else:
            raise ValueError(f"Unknown sampling_mode={cfg.sampling_mode!r}")
        self.pos_rows = 0
        self.neg_rows = int(alloc.sum())
        self.total_rows = int(self.pos_rows + self.neg_rows)
        self.total_batches = 0

        for name, pos, neg, n_neg in zip(self.file_names, pos_positions, neg_positions, alloc):
            timestamps, features, _valid_mask, labels, anomaly_labels, rule_pred, _extra = self.cache.get(name)
            signal_scores = row_signal_scores(features, rule_pred)
            if sampling_mode in {"module_balanced", "module"} and len(pos) > int(cfg.positive_windows_per_module) > 0:
                stable = zlib.crc32((name + "::pos").encode("utf-8")) & 0xFFFFFFFF
                pos = choose_positions_by_signal(
                    pos,
                    int(cfg.positive_windows_per_module),
                    signal_scores,
                    cfg,
                    int(cfg.seed) + int(stable),
                )
            if int(n_neg) >= len(neg):
                chosen_neg = neg
            elif int(n_neg) > 0:
                stable = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
                chosen_neg = choose_positions_by_signal(
                    neg,
                    int(n_neg),
                    signal_scores,
                    cfg,
                    int(cfg.seed) + int(stable),
                )
            else:
                chosen_neg = np.empty(0, dtype=np.int64)
            selected = np.sort(np.concatenate([pos, chosen_neg])).astype(np.int64)
            self.selected_positions.append(selected)
            self.selected_labels.append(labels[selected].astype(np.int8))
            self.selected_weights.append(temporal_position_weights(timestamps, labels, anomaly_labels, selected, cfg))
            self.pos_rows += int(len(pos))
            self.total_batches += int(math.ceil(len(selected) / max(1, int(cfg.batch_size)))) if len(selected) else 0
        self.total_rows = int(self.pos_rows + self.neg_rows)

        print(
            f"[compat dataset] files={len(self.file_names)} source_rows={self.source_total_rows} "
            f"source_pos={self.source_pos_rows} source_neg={self.source_neg_rows} "
            f"sampled_rows={self.total_rows} sampled_pos={self.pos_rows} sampled_neg={self.neg_rows} "
            f"sampling={cfg.sampling_mode} negative_ratio={cfg.negative_ratio} "
            f"per_module pos={cfg.positive_windows_per_module} faulty_neg={cfg.negative_windows_per_faulty_module} "
            f"normal_neg={cfg.normal_windows_per_module} selection={cfg.sample_selection} "
            f"temporal_pos_w={cfg.temporal_positive_weight} features={len(self.feature_names)} batches={self.total_batches}"
        )

    def __len__(self) -> int:
        return int(self.total_batches)

    def apply_adaptive_negative_weights(self, scores_by_file: dict[str, np.ndarray], max_extra: float) -> dict[str, float]:
        max_extra = float(max_extra)
        if max_extra <= 0.0:
            return {"updated_negative_rows": 0.0, "min_score": 0.0, "max_score": 0.0}
        neg_scores: list[np.ndarray] = []
        for name, labels, _weights in zip(self.file_names, self.selected_labels, self.selected_weights):
            scores = np.asarray(scores_by_file.get(name, np.zeros(len(labels), dtype=np.float32)), dtype=np.float32)
            if len(scores) != len(labels):
                continue
            neg_scores.append(scores[labels <= 0])
        all_neg = np.concatenate([x for x in neg_scores if len(x)]) if any(len(x) for x in neg_scores) else np.zeros(0, dtype=np.float32)
        if len(all_neg) == 0:
            return {"updated_negative_rows": 0.0, "min_score": 0.0, "max_score": 0.0}
        lo = float(np.nanmin(all_neg))
        hi = float(np.nanmax(all_neg))
        denom = max(hi - lo, 1e-8)
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
        return {"updated_negative_rows": float(updated), "min_score": lo, "max_score": hi}

    def __iter__(self):
        seq_len = int(self.cfg.seq_len)
        batch_size = int(self.cfg.batch_size)
        n_features = len(self.feature_names)
        pairs = list(zip(self.file_names, self.selected_positions))
        worker = get_worker_info()
        if worker is not None:
            pairs = pairs[worker.id :: worker.num_workers]
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
                yield (
                    x_windows.index_select(0, idx).contiguous(),
                    m_windows.index_select(0, idx).contiguous(),
                    ys[start:end].contiguous(),
                    torch.from_numpy(selected_weights[start:end].astype(np.float32)).contiguous(),
                )


def cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=bool(enabled))
    return torch.cuda.amp.autocast(enabled=bool(enabled))


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_rate(rows: float, seconds: float) -> str:
    if float(seconds) <= 0.0:
        return "0.0 rows/s"
    return f"{float(rows) / max(float(seconds), 1e-9):.1f} rows/s"


def progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {100.0 * current / total:5.1f}%"


def log_run_header(title: str, fields: dict[str, object]) -> None:
    line = "=" * 88
    print(line)
    print(f"{title}")
    print("-" * 88)
    for key, value in fields.items():
        print(f"{key:>22}: {value}")
    print(line)


def format_epoch_status(
    tag: str,
    model_name: str,
    fold: int | None,
    epoch: int,
    total_epochs: int,
    loss: float,
    rows: int,
    epoch_seconds: float,
    elapsed_seconds: float,
    eta_seconds: float,
) -> str:
    fold_text = f" fold={fold}" if fold is not None else ""
    return (
        f"[{tag}] model={model_name}{fold_text} ep={epoch:02d}/{total_epochs:02d} "
        f"{progress_bar(epoch, total_epochs, width=18)} "
        f"loss={loss:.5f} rows={rows} rate={format_rate(rows, epoch_seconds)} "
        f"epoch={format_duration(epoch_seconds)} elapsed={format_duration(elapsed_seconds)} "
        f"eta={format_duration(eta_seconds)}"
    )


def compat_alarm_logit(outputs: torch.Tensor | tuple) -> torch.Tensor:
    logits_all = primary_logit(outputs)
    alarm_logits, _aux = split_alarm_aux_logits(logits_all, aux_dim=4)
    if alarm_logits.ndim > 1:
        alarm_logits = alarm_logits[..., 0]
    return alarm_logits


def effective_pos_weight(dataset: CompatBatchedDataset, cfg: CompatCfg) -> float:
    value = max(1.0, float(dataset.neg_rows) / max(float(dataset.pos_rows), 1.0))
    if float(cfg.pos_weight_cap) > 0:
        value = min(value, float(cfg.pos_weight_cap))
    return float(value)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn, cfg: CompatCfg, epoch: int) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_seen = 0
    started = time.time()
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
    for batch_idx, batch in enumerate(loader, 1):
        if len(batch) == 3:
            xs, masks, ys = batch
            ws = torch.ones_like(ys, dtype=torch.float32)
        else:
            xs, masks, ys, ws = batch
        xs = xs.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        ys = ys.to(cfg.device, non_blocking=True).float()
        ws = ws.to(cfg.device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(amp_enabled):
            outputs = model(xs, masks)
            alarm_logits = compat_alarm_logit(outputs)
            loss_raw = loss_fn(alarm_logits, ys)
            loss = (loss_raw * ws).sum() / torch.clamp(ws.sum(), min=1.0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={batch_idx}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"non-finite gradient in {name} at epoch={epoch} batch={batch_idx}")
        optimizer.step()
        batch_n = int(xs.size(0))
        total_loss += float(loss.detach().cpu()) * batch_n
        n_seen += batch_n
        if int(cfg.log_batches) > 0 and batch_idx % int(cfg.log_batches) == 0:
            elapsed = time.time() - started
            print(
                f"[batch] ep={epoch:02d} batch={batch_idx:>5}/{len(loader):<5} "
                f"{progress_bar(batch_idx, len(loader), width=16)} "
                f"loss={total_loss / max(n_seen, 1):.5f} rows={n_seen} "
                f"rate={format_rate(n_seen, elapsed)} elapsed={format_duration(elapsed)}"
            )
    return {
        "loss": total_loss / max(n_seen, 1),
        "rows_seen": float(n_seen),
        "elapsed_seconds": float(time.time() - started),
    }


@torch.no_grad()
def score_dataset_selected_positions(
    model: nn.Module,
    dataset: CompatBatchedDataset,
    cache: Model2FeatureCache,
    cfg: CompatCfg,
) -> dict[str, np.ndarray]:
    model.eval()
    seq_len = int(cfg.seq_len)
    n_features = len(compat_feature_names(cfg))
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
    scores_by_file: dict[str, np.ndarray] = {}
    for file_name, selected in zip(dataset.file_names, dataset.selected_positions):
        if len(selected) == 0:
            scores_by_file[file_name] = np.zeros(0, dtype=np.float32)
            continue
        _timestamps, features, _valid_mask, _labels, _anomaly_labels, _rule_pred, _extra = cache.get(file_name)
        pad_x = np.zeros((seq_len - 1, n_features), dtype=np.float32)
        pad_m = np.zeros_like(pad_x)
        x_pad = torch.from_numpy(np.concatenate([pad_x, features.astype(np.float32)], axis=0))
        m_pad = torch.from_numpy(np.concatenate([pad_m, np.ones_like(features, dtype=np.float32)], axis=0))
        x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
        m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
        parts: list[np.ndarray] = []
        for start in range(0, len(selected), int(cfg.batch_size)):
            idx = torch.from_numpy(selected[start : start + int(cfg.batch_size)])
            xb = x_windows.index_select(0, idx).contiguous().to(cfg.device)
            mb = m_windows.index_select(0, idx).contiguous().to(cfg.device)
            with cuda_autocast(amp_enabled):
                outputs = model(xb, mb)
                alarm_logits = compat_alarm_logit(outputs)
            parts.append(torch.sigmoid(alarm_logits).float().detach().cpu().numpy().reshape(-1))
        scores_by_file[file_name] = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    return scores_by_file


@torch.no_grad()
def score_one_file(model: nn.Module, cache: Model2FeatureCache, file_name: str, cfg: CompatCfg) -> pd.DataFrame:
    model.eval()
    timestamps, features, _valid_mask, _labels, _anomaly_labels, rule_pred, extra = cache.get(file_name)
    seq_len = int(cfg.seq_len)
    n_features = len(compat_feature_names(cfg))
    frames: list[pd.DataFrame] = []
    if len(features):
        pad_x = np.zeros((seq_len - 1, n_features), dtype=np.float32)
        pad_m = np.zeros_like(pad_x)
        x_pad = torch.from_numpy(np.concatenate([pad_x, features.astype(np.float32)], axis=0))
        m_pad = torch.from_numpy(np.concatenate([pad_m, np.ones_like(features, dtype=np.float32)], axis=0))
        x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
        m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
        scores: list[np.ndarray] = []
        for start in range(0, len(features), int(cfg.batch_size)):
            end = min(len(features), start + int(cfg.batch_size))
            xb = x_windows[start:end].contiguous().to(cfg.device)
            mb = m_windows[start:end].contiguous().to(cfg.device)
            outputs = model(xb, mb)
            alarm_logits = compat_alarm_logit(outputs)
            scores.append(torch.sigmoid(alarm_logits).float().detach().cpu().numpy())
        score_arr = np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, dtype=np.float32)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps.astype(np.int64),
                    "score": score_arr,
                    "rule_predict": rule_pred.astype(int),
                    "source": "model2_feature_deep",
                }
            )
        )
    if len(extra):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": extra[:, 0].astype(np.int64),
                    "score": np.nan,
                    "rule_predict": extra[:, 1].astype(int),
                    "source": "model2_extra_rule",
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["timestamp", "score", "rule_predict", "source"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def apply_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = frame.copy()
    score = pd.to_numeric(out.get("score"), errors="coerce")
    deep_pred = (score >= float(threshold)).fillna(False).astype(int)
    rule_pred = pd.to_numeric(out.get("rule_predict", 0), errors="coerce").fillna(0).astype(int)
    out["predict"] = ((deep_pred > 0) | (rule_pred > 0)).astype(int)
    return out[["timestamp", "predict", "score", "rule_predict", "source"]]


def write_predictions(
    model: nn.Module,
    cache: Model2FeatureCache,
    file_names: list[str],
    out_dir: Path,
    cfg: CompatCfg,
    threshold: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for idx, name in enumerate(file_names, 1):
        pred = apply_threshold(score_one_file(model, cache, name, cfg), threshold)
        pred.to_csv(out_dir / name, index=False)
        total_rows += int(len(pred))
        if idx % 200 == 0:
            print(f"[predict] {idx}/{len(file_names)} files")
    return total_rows


def file_label_map(index_df: pd.DataFrame) -> dict[str, int]:
    return {
        str(row["file_name"]): int(row["Label"])
        for _, row in index_df[["file_name", "Label"]].drop_duplicates("file_name").iterrows()
    }


def cap_files_stratified(
    file_names: list[str],
    label_by_file: dict[str, int],
    max_files: int,
    seed: int,
) -> list[str]:
    if int(max_files) <= 0 or len(file_names) <= int(max_files):
        return list(file_names)
    original_order = {name: idx for idx, name in enumerate(file_names)}
    groups: dict[int, list[str]] = {}
    for name in file_names:
        groups.setdefault(int(label_by_file.get(name, 0)), []).append(name)
    labels = sorted(groups)
    counts = np.asarray([len(groups[label]) for label in labels], dtype=float)
    raw = counts / max(float(counts.sum()), 1.0) * float(max_files)
    alloc = np.floor(raw).astype(int)
    if int(max_files) >= len(labels):
        for idx, label in enumerate(labels):
            if len(groups[label]) > 0 and alloc[idx] == 0:
                alloc[idx] = 1
    while int(alloc.sum()) > int(max_files):
        idx = int(np.argmax(alloc))
        alloc[idx] -= 1
    remainder = int(max_files) - int(alloc.sum())
    if remainder > 0:
        order = np.argsort(raw - np.floor(raw))[::-1]
        for idx in order:
            if remainder <= 0:
                break
            room = len(groups[labels[int(idx)]]) - int(alloc[int(idx)])
            if room > 0:
                alloc[int(idx)] += 1
                remainder -= 1
    rng = np.random.default_rng(int(seed))
    selected: list[str] = []
    for label, n_take in zip(labels, alloc):
        group = np.asarray(groups[label], dtype=object)
        rng.shuffle(group)
        selected.extend(str(x) for x in group[: int(n_take)])
    return sorted(selected, key=lambda name: original_order.get(name, len(original_order)))


def split_train_val_files(
    train_files: list[str],
    label_by_file: dict[str, int],
    cfg: CompatCfg,
    fold: int,
) -> tuple[list[str], list[str]]:
    frac = float(cfg.val_fraction)
    if frac <= 0.0 or len(train_files) < 4:
        return list(train_files), []
    rng = np.random.default_rng(int(cfg.seed) + 1009 * int(fold))
    train_out: list[str] = []
    val_out: list[str] = []
    for label in sorted({int(label_by_file.get(name, 0)) for name in train_files}):
        group = [name for name in train_files if int(label_by_file.get(name, 0)) == label]
        order = np.asarray(group, dtype=object)
        rng.shuffle(order)
        n_val = int(round(len(order) * frac))
        if len(order) > 1:
            n_val = min(max(1, n_val), len(order) - 1)
        else:
            n_val = 0
        val_out.extend(str(x) for x in order[:n_val])
        train_out.extend(str(x) for x in order[n_val:])
    train_out.sort()
    val_out.sort()
    if not train_out:
        return list(train_files), []
    return train_out, val_out


def parse_threshold_grid(text: str) -> list[float]:
    raw = str(text).strip()
    if not raw:
        return []
    if ":" in raw:
        parts = [float(x) for x in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("threshold_grid range must be start:end:step")
        start, end, step = parts
        if step <= 0:
            raise ValueError("threshold_grid step must be positive")
        values = []
        cur = start
        while cur <= end + 1e-12:
            values.append(round(float(cur), 10))
            cur += step
        return values
    values = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
    return sorted(set(values))


def _first_positive_timestamp_frame(
    frame: pd.DataFrame,
    predict_column: str,
    before_ts: float | None = None,
) -> tuple[int, float | None]:
    if frame.empty or predict_column not in frame.columns or "timestamp" not in frame.columns:
        return 0, None
    pred = pd.to_numeric(frame[predict_column], errors="coerce").fillna(0.0)
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    valid = (pred > 0) & timestamps.notna()
    if before_ts is not None:
        valid = valid & (timestamps.astype(float) < float(before_ts))
    positive = pd.to_numeric(frame.loc[valid, "timestamp"], errors="coerce").dropna()
    if positive.empty:
        return 0, None
    return 1, float(positive.min())


def evaluate_prediction_frames(
    prediction_frames: dict[str, pd.DataFrame],
    label_dir: Path,
    predict_column: str = "predict",
    min_hit_lead_hours: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in sorted(prediction_frames):
        label_path = Path(label_dir) / name
        if not label_path.exists():
            continue
        true_label, true_ts = first_positive_timestamp(label_path, "anomaly")
        predict_label, predict_ts = _first_positive_timestamp_frame(
            prediction_frames[name],
            predict_column,
            before_ts=true_ts if true_label > 0 else None,
        )
        valid_predict_positive = 0
        hit = 0
        lead_hour = None
        if predict_label > 0:
            if true_label == 0:
                valid_predict_positive = 1
            elif true_ts is not None and predict_ts is not None and true_ts > predict_ts:
                lead_hour = abs(true_ts - predict_ts) / 3600.0
                if lead_hour >= float(min_hit_lead_hours):
                    valid_predict_positive = 1
                    hit = 1
        rows.append(
            {
                "file_name": name,
                "true_label": int(true_label),
                "predict_label": int(predict_label),
                "true_ts": true_ts,
                "predict_ts": predict_ts,
                "valid_predict_positive": int(valid_predict_positive),
                "hit": int(hit),
                "lead_hour": lead_hour,
                "min_hit_lead_hours": float(min_hit_lead_hours),
            }
        )
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise ValueError("No matching validation prediction/label frames")
    true_pos = set(detail_df.loc[detail_df["true_label"] > 0, "file_name"])
    pred_pos = set(detail_df.loc[detail_df["valid_predict_positive"] > 0, "file_name"])
    hit_pos = true_pos & pred_pos
    tp = len(hit_pos)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = len(detail_df) - tp - fp - fn
    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(detail_df) if len(detail_df) else 0.0
    lead_hours = detail_df.loc[detail_df["hit"] > 0, "lead_hour"].dropna().astype(float)
    avg_lead_hour = float(lead_hours.mean()) if not lead_hours.empty else 0.0
    min_lead_hour = float(lead_hours.min()) if not lead_hours.empty else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    summary_df = pd.DataFrame(
        {
            "Item": [
                "final_score",
                "f1_score",
                "precision",
                "recall",
                "all_hit_cnt",
                "all_predict_pos_cnt",
                "all_true_pos_cnt",
                "avg_lead_score",
                "avg_lead_hour",
                "min_lead_score",
                "min_lead_hour",
                "lead_pread_cnt",
                "accuracy",
                "tp",
                "fp",
                "fn",
                "tn",
                "evaluated_module_cnt",
                "min_hit_lead_hours",
            ],
            "Value": [
                final_score,
                f1,
                precision,
                recall,
                tp,
                len(pred_pos),
                len(true_pos),
                avg_lead_score,
                avg_lead_hour,
                min_lead_score,
                min_lead_hour,
                int(detail_df["hit"].sum()),
                accuracy,
                tp,
                fp,
                fn,
                tn,
                len(detail_df),
                float(min_hit_lead_hours),
            ],
        }
    )
    return summary_df, detail_df


def evaluate_prediction_dir_compat(
    prediction_dir: Path,
    label_dir: Path,
    predict_column: str = "predict",
    min_hit_lead_hours: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(prediction_dir).glob("*.csv")):
        frames[path.name] = pd.read_csv(path)
    return evaluate_prediction_frames(
        frames,
        label_dir,
        predict_column=predict_column,
        min_hit_lead_hours=float(min_hit_lead_hours),
    )


def score_files_to_memory(
    model: nn.Module,
    cache: Model2FeatureCache,
    file_names: list[str],
    cfg: CompatCfg,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for idx, name in enumerate(file_names, 1):
        frames[name] = score_one_file(model, cache, name, cfg)
        if idx % 200 == 0:
            print(f"[val-score] {idx}/{len(file_names)} files")
    return frames


def select_threshold(
    model: nn.Module,
    cache: Model2FeatureCache,
    val_files: list[str],
    data_dir: Path,
    run_dir: Path,
    cfg: CompatCfg,
) -> tuple[float, dict[str, float]]:
    candidates = parse_threshold_grid(cfg.threshold_grid)
    if not bool(cfg.threshold_search) or not val_files or not candidates:
        return float(cfg.fixed_threshold), {}
    print(
        f"[val-select] files={len(val_files)} metric={cfg.threshold_metric} "
        f"candidates={len(candidates)} rule_mode={cfg.rule_mode}"
    )
    score_frames = score_files_to_memory(model, cache, val_files, cfg)
    search_rows: list[dict] = []
    best_threshold = float(cfg.fixed_threshold)
    best_metrics: dict[str, float] = {}
    best_detail: pd.DataFrame | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None
    metric_key = str(cfg.threshold_metric)
    for threshold in candidates:
        pred_frames = {name: apply_threshold(frame, threshold) for name, frame in score_frames.items()}
        summary_df, detail_df = evaluate_prediction_frames(
            pred_frames,
            data_dir,
            min_hit_lead_hours=float(cfg.min_hit_lead_hours),
        )
        metrics = {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}
        value = float(metrics.get(metric_key, metrics.get("f1_score", 0.0)))
        row = {"threshold": float(threshold)}
        row.update(metrics)
        search_rows.append(row)
        key = (
            value,
            float(metrics.get("final_score", 0.0)),
            float(metrics.get("precision", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            float(metrics.get("recall", 0.0)),
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
            best_detail = detail_df
    val_dir = run_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search_rows).to_csv(val_dir / "threshold_search.csv", index=False)
    pd.DataFrame({"Item": list(best_metrics.keys()), "Value": list(best_metrics.values())}).to_csv(
        val_dir / "best_evaluate_result.csv",
        index=False,
    )
    if best_detail is not None:
        best_detail.to_csv(val_dir / "best_module_decisions.csv", index=False)
    print(
        f"[val-select] threshold={best_threshold:.4f} {metric_key}={best_metrics.get(metric_key, 0.0):.5f} "
        f"{format_metric_summary(best_metrics)}"
    )
    return best_threshold, best_metrics


def run_fold(model_name: str, model_builder: ModelBuilder, fold: int, data_dir: Path, index_df: pd.DataFrame, out_root: Path, cfg: CompatCfg) -> dict:
    torch.manual_seed(int(cfg.seed) + int(fold))
    np.random.seed(int(cfg.seed) + int(fold))
    cfg.device = resolve_runtime_device(cfg.device)
    if str(cfg.device).startswith("cuda") and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"[compat-init] model={model_name} fold={fold} {runtime_device_summary(cfg.device)}")
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    label_by_file = file_label_map(index_df)
    if int(cfg.max_train_files) > 0:
        train_files = cap_files_stratified(train_files, label_by_file, int(cfg.max_train_files), int(cfg.seed) + int(fold))
    if int(cfg.max_test_files) > 0:
        test_files = cap_files_stratified(test_files, label_by_file, int(cfg.max_test_files), int(cfg.seed) + 7919 + int(fold))
    train_files, val_files = split_train_val_files(train_files, label_by_file, cfg, int(fold))
    run_dir = Path(out_root) / model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_run_header(
        "OFP MODEL2-COMPAT DEEP TRAIN",
        {
            "model": model_name,
            "fold": fold,
            "train/val/test": f"{len(train_files)}/{len(val_files)}/{len(test_files)} modules",
            "target": cfg.target_mode,
            "features": cfg.feature_mode,
            "sampling": cfg.sampling_mode,
            "sample selection": cfg.sample_selection,
            "rule mode": cfg.rule_mode,
            "seq_len": cfg.seq_len,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "device": cfg.device,
            "out_dir": run_dir,
        },
    )
    stats = compute_feature_norm_stats(data_dir, train_files, cfg, label_by_file)
    mean, std = stats.arrays()
    cache = Model2FeatureCache(data_dir, mean, std, cfg, label_by_file)
    dataset = CompatBatchedDataset(train_files, cache, cfg)
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=int(cfg.num_workers))
    pos_weight = effective_pos_weight(dataset, cfg)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=cfg.device),
        reduction="none",
    )
    model, model_cfg, _use_special_losses = model_builder(int(cfg.seq_len), len(compat_feature_names(cfg)))
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history: list[dict] = []
    adaptive_weight_meta: dict[str, float] = {}
    adaptive_applied = False
    t0 = time.time()
    total_epochs = int(cfg.epochs)
    for epoch in range(1, total_epochs + 1):
        parts = train_one_epoch(model, loader, optimizer, loss_fn, cfg, epoch)
        history.append({"epoch": epoch, **parts})
        elapsed_total = time.time() - t0
        eta = (elapsed_total / max(epoch, 1)) * max(total_epochs - epoch, 0)
        print(
            format_epoch_status(
                "compat-train",
                model_name,
                int(fold),
                epoch,
                total_epochs,
                float(parts["loss"]),
                int(parts["rows_seen"]),
                float(parts["elapsed_seconds"]),
                elapsed_total,
                eta,
            )
        )
        if (
            float(cfg.adaptive_negative_weight) > 0.0
            and not adaptive_applied
            and epoch >= int(cfg.adaptive_warmup_epochs)
            and epoch < int(cfg.epochs)
        ):
            scores_by_file = score_dataset_selected_positions(model, dataset, cache, cfg)
            adaptive_weight_meta = dataset.apply_adaptive_negative_weights(scores_by_file, float(cfg.adaptive_negative_weight))
            adaptive_weight_meta["applied_after_epoch"] = float(epoch)
            adaptive_applied = True
            print(
                f"[adaptive-neg] rows={int(adaptive_weight_meta.get('updated_negative_rows', 0))} "
                f"score_min={adaptive_weight_meta.get('min_score', 0.0):.5f} "
                f"score_max={adaptive_weight_meta.get('max_score', 0.0):.5f} "
                f"max_extra={cfg.adaptive_negative_weight}"
            )

    selected_threshold, val_metrics = select_threshold(model, cache, val_files, data_dir, run_dir, cfg)

    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    test_rows = write_predictions(model, cache, test_files, pred_dir, cfg, selected_threshold)
    if float(cfg.min_hit_lead_hours) > 0.0:
        summary_df, detail_df = evaluate_prediction_dir_compat(
            pred_dir,
            data_dir,
            min_hit_lead_hours=float(cfg.min_hit_lead_hours),
        )
    else:
        summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}
    result = {
        "model": model_name,
        "fold": int(fold),
        "protocol": f"model2_feature_legacy_timestamp_deep_{cfg.target_mode}_{cfg.feature_mode}_{cfg.sampling_mode}_row_bce",
        "train_modules": len(train_files),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "train_rows": int(dataset.total_rows),
        "train_source_rows": int(dataset.source_total_rows),
        "train_pos_rows": int(dataset.pos_rows),
        "train_neg_rows": int(dataset.neg_rows),
        "sampling_mode": cfg.sampling_mode,
        "feature_mode": cfg.feature_mode,
        "positive_windows_per_module": int(cfg.positive_windows_per_module),
        "negative_windows_per_faulty_module": int(cfg.negative_windows_per_faulty_module),
        "normal_windows_per_module": int(cfg.normal_windows_per_module),
        "test_rows": int(test_rows),
        "threshold": float(selected_threshold),
        "fixed_threshold": float(cfg.fixed_threshold),
        "val_metrics": val_metrics,
        "pos_weight": float(pos_weight),
        "pos_weight_cap": float(cfg.pos_weight_cap),
        "adaptive_negative_weight_meta": adaptive_weight_meta,
        "min_hit_lead_hours": float(cfg.min_hit_lead_hours),
        "seconds": time.time() - t0,
        "metrics": metrics,
        "cfg": asdict(cfg),
        "feature_norm_stats": asdict(stats),
        "model_cfg": model_cfg,
        "history": history,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "model_cfg": model_cfg}, run_dir / "model.pt")
    print(f"[compat-done] model={model_name} fold={fold} {format_metric_summary(metrics)}")
    del model, optimizer, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def aggregate_results(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "target_mode": item["cfg"].get("target_mode", ""),
            "feature_mode": item["cfg"].get("feature_mode", ""),
            "sampling_mode": item["cfg"].get("sampling_mode", ""),
            "rule_mode": item["cfg"].get("rule_mode", ""),
            "min_hit_lead_hours": item["cfg"].get("min_hit_lead_hours", 0.0),
            "threshold": item["threshold"],
            "train_rows": item["train_rows"],
            "test_rows": item["test_rows"],
            "seconds": item["seconds"],
        }
        row.update(item["metrics"])
        rows.append(row)
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new.drop_duplicates(
            subset=[
                "model",
                "fold",
                "target_mode",
                "feature_mode",
                "sampling_mode",
                "rule_mode",
                "min_hit_lead_hours",
            ],
            keep="last",
            inplace=True,
        )
    new.sort_values(["model", "target_mode", "feature_mode", "sampling_mode", "rule_mode", "min_hit_lead_hours", "fold"], inplace=True)
    new.to_csv(path, index=False)
    group_cols = ["model", "target_mode", "feature_mode", "sampling_mode", "rule_mode", "min_hit_lead_hours"]
    non_metric_cols = set(group_cols) | {"fold"}
    numeric = [
        c
        for c in new.columns
        if c not in non_metric_cols and pd.api.types.is_numeric_dtype(new[c])
    ]
    summary = new.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep row classifiers over OFP model2 engineered features.")
    parser.add_argument("--models", nargs="+", default=list(BUILDERS), choices=sorted(BUILDERS))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_model2_compat_results/model2_feature_deep"))
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--target_mode", choices=["ahead120", "anomaly", "module_fault"], default="module_fault")
    parser.add_argument("--feature_mode", choices=["model2", "model2_plus"], default="model2_plus")
    parser.add_argument("--sampling_mode", choices=["row_ratio", "module_balanced"], default="module_balanced")
    parser.add_argument("--positive_windows_per_module", type=int, default=96)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=24)
    parser.add_argument("--normal_windows_per_module", type=int, default=24)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_grid", default="0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--rule_mode", choices=["none", "temp", "model2_simple"], default="model2_simple")
    parser.add_argument("--sample_selection", choices=["random", "signal_topk", "hybrid"], default="random")
    parser.add_argument("--sample_topk_fraction", type=float, default=0.5)
    parser.add_argument("--temporal_positive_weight", type=float, default=0.0)
    parser.add_argument("--temporal_weight_horizon_hours", type=float, default=120.0)
    parser.add_argument("--adaptive_negative_weight", type=float, default=0.0)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=1)
    parser.add_argument("--no_threshold_search", action="store_true")
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="enable CUDA autocast mixed precision")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--log_batches", type=int, default=1000)
    parser.add_argument("--module_cache_dir", default="")
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_test_files", type=int, default=0)
    parser.add_argument(
        "--min_hit_lead_hours",
        type=float,
        default=0.0,
        help="minimum lead time, in hours, required for a faulty module alarm to count as a hit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    cfg = CompatCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        fixed_threshold=args.fixed_threshold,
        target_mode=args.target_mode,
        feature_mode=args.feature_mode,
        sampling_mode=args.sampling_mode,
        positive_windows_per_module=args.positive_windows_per_module,
        negative_windows_per_faulty_module=args.negative_windows_per_faulty_module,
        normal_windows_per_module=args.normal_windows_per_module,
        val_fraction=args.val_fraction,
        threshold_search=not args.no_threshold_search,
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        rule_mode=args.rule_mode,
        sample_selection=args.sample_selection,
        sample_topk_fraction=args.sample_topk_fraction,
        temporal_positive_weight=args.temporal_positive_weight,
        temporal_weight_horizon_hours=args.temporal_weight_horizon_hours,
        adaptive_negative_weight=args.adaptive_negative_weight,
        adaptive_warmup_epochs=args.adaptive_warmup_epochs,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        amp=bool(args.amp),
        allow_tf32=not args.no_tf32,
        log_batches=args.log_batches,
        module_cache_dir=args.module_cache_dir,
        max_train_files=args.max_train_files,
        max_test_files=args.max_test_files,
        min_hit_lead_hours=args.min_hit_lead_hours,
    )
    results: list[dict] = []
    for model_name in args.models:
        for fold in args.folds:
            result = run_fold(model_name, BUILDERS[model_name], int(fold), args.data_dir, index_df, args.out_root, cfg)
            results.append(result)
            aggregate_results([result], args.out_root)
    aggregate_results(results, args.out_root)


if __name__ == "__main__":
    main()
