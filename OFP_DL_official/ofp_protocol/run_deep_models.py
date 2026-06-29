"""Run deep-learning models under the OFP module-level protocol.

This file intentionally lives outside ``OFP/``.  It adapts the existing deep
models to the teacher protocol:

1. split by module using ``train_test_set_index(in).csv``;
2. train point-level windows with ``anomaly_ahead_120_hours`` labels;
3. export per-module ``timestamp,predict`` CSVs;
4. evaluate by ``ofp_protocol.evaluator``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_formal_protocol.protocol import (
    DEFAULT_HORIZON_HOURS,
    ahead_labels_and_valid_mask_multi_first_event,
    primary_horizon_index,
    valid_time_mask_first_event,
)
from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
    SENSORS,
    read_index,
)
from OFP_DL_official.common.formal_data import (
    choose_alarm_strategy_by_final_score,
    write_predictions as write_alarm_predictions,
)


HORIZON_HOURS = DEFAULT_HORIZON_HOURS
PRIMARY_HORIZON_INDEX = primary_horizon_index(HORIZON_HOURS)


def primary_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 1:
        return arr.astype(np.int8, copy=False)
    return arr[:, PRIMARY_HORIZON_INDEX].astype(np.int8, copy=False)


@dataclass
class DeepRunCfg:
    seq_len: int = 288
    epochs: int = 8
    batch_size: int = 96
    lr: float = 3e-4
    weight_decay: float = 1e-2
    patience: int = 3
    grad_clip: float = 1.0
    train_pos_per_file: int = 96
    train_neg_per_file: int = 24
    val_fraction: float = 0.1
    threshold_grid_size: int = 99
    max_cached_files: int = 2048
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    horizon_hours: tuple[float, ...] = HORIZON_HOURS
    alarm_smoothing: str = "ema"
    alarm_smooth_window: int = 1
    alarm_consecutive_k: int = 1
    alarm_search_smooth_windows: tuple[int, ...] = (1, 3, 6, 12)
    alarm_search_consecutive_ks: tuple[int, ...] = (1, 2, 3)
    first_alarm_only: bool = True


class ModuleArrayCache:
    def __init__(self, data_dir: Path, max_cached_files: int = 2048) -> None:
        self.data_dir = Path(data_dir)
        self.max_cached_files = int(max_cached_files)
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            item = self.cache.pop(file_name)
            self.cache[file_name] = item
            return item

        df = pd.read_csv(self.data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        values = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        labels, valid_mask = ahead_labels_and_valid_mask_multi_first_event(
            timestamps.astype(float),
            anomaly,
            HORIZON_HOURS,
        )
        item = (timestamps, values, anomaly, labels, valid_mask)

        self.cache[file_name] = item
        if len(self.cache) > self.max_cached_files:
            self.cache.popitem(last=False)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_norm_stats(data_dir: Path, train_files: list[str], max_files: int = 512) -> tuple[np.ndarray, np.ndarray]:
    files = train_files[:]
    if len(files) > max_files:
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, size=max_files, replace=False))
    s = np.zeros(len(SENSORS), dtype=np.float64)
    ss = np.zeros(len(SENSORS), dtype=np.float64)
    n = np.zeros(len(SENSORS), dtype=np.float64)
    for name in files:
        df = pd.read_csv(data_dir / name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        valid_rows = valid_time_mask_first_event(timestamps, anomaly)
        arr = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)[valid_rows]
        mask = np.isfinite(arr)
        arr0 = np.where(mask, arr, 0.0)
        s += arr0.sum(axis=0)
        ss += (arr0 * arr0).sum(axis=0)
        n += mask.sum(axis=0)
    n = np.maximum(n, 1.0)
    mean = (s / n).astype(np.float32)
    var = np.maximum(ss / n - mean.astype(np.float64) ** 2, 1e-6)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def sample_training_windows(
    data_dir: Path,
    file_names: list[str],
    pos_per_file: int,
    neg_per_file: int,
    seed: int,
    max_files: int | None = None,
) -> list[tuple[str, int, tuple[float, ...]]]:
    rng = np.random.default_rng(seed)
    files = file_names[: max_files if max_files is not None else None]
    samples: list[tuple[str, int, int]] = []
    for name in files:
        df = pd.read_csv(data_dir / name, usecols=lambda c: c in {"timestamp", "anomaly"})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        labels, valid_mask = ahead_labels_and_valid_mask_multi_first_event(timestamps, anomaly, HORIZON_HOURS)
        primary = primary_labels(labels)
        pos_idx = np.where((primary > 0) & valid_mask)[0]
        neg_idx = np.where((primary == 0) & valid_mask)[0]
        if len(pos_idx):
            take = min(int(pos_per_file), len(pos_idx))
            chosen = rng.choice(pos_idx, size=take, replace=False)
            samples.extend((name, int(i), tuple(labels[int(i)].astype(float).tolist())) for i in chosen)
        if len(neg_idx):
            take = min(int(neg_per_file), len(neg_idx))
            chosen = rng.choice(neg_idx, size=take, replace=False)
            samples.extend((name, int(i), tuple(labels[int(i)].astype(float).tolist())) for i in chosen)
    rng.shuffle(samples)
    return samples


class OFPWindowDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int, tuple[float, ...]]],
        cache: ModuleArrayCache,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.samples = samples
        self.cache = cache
        self.seq_len = int(seq_len)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_name, row_idx, label = self.samples[idx]
        item = self.cache.get(file_name)
        _ts, values, _anom, _labels = item[:4]
        x, mask = make_window(values, row_idx, self.seq_len, self.mean, self.std)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(label, dtype=torch.float32)


def make_window(
    values: np.ndarray,
    row_idx: int,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    end = int(row_idx) + 1
    start = max(0, end - int(seq_len))
    seq = values[start:end]
    out = np.full((seq_len, values.shape[1]), np.nan, dtype=np.float32)
    out[-len(seq):] = seq
    mask = np.isfinite(out).astype(np.float32)
    out = np.where(mask > 0, out, 0.0).astype(np.float32)
    out = (out - mean) / std
    out = out * mask
    return out.astype(np.float32), mask.astype(np.float32)


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ms = torch.stack([b[1] for b in batch])
    ys = torch.stack([b[2] for b in batch])
    return xs, ms, ys


def make_model(model_name: str, seq_len: int, n_sensors: int) -> tuple[nn.Module, dict]:
    if model_name == "itransformer":
        from model.Optical_prediction_model.deep_learning.models import ITransformerCfg, ITransformerClassifier

        cfg = ITransformerCfg(seq_len=seq_len, n_sensors=n_sensors, d_model=96, n_heads=4, e_layers=2, d_ff=192, output_dim=len(HORIZON_HOURS))
        return ITransformerClassifier(cfg), asdict(cfg)
    if model_name == "itransformer_ofp":
        from model.Optical_prediction_model.deep_learning.models import ITransformerOFPCfg, ITransformerOFPClassifier

        cfg = ITransformerOFPCfg(seq_len=seq_len, n_sensors=n_sensors, d_model=96, n_heads=4, e_layers=2, d_ff=192, output_dim=len(HORIZON_HOURS))
        return ITransformerOFPClassifier(cfg), asdict(cfg)
    if model_name == "patchtst":
        from model.Optical_prediction_model.deep_learning.patchtst_wrapper import PatchTSTCfg, PatchTSTClassifier

        cfg = PatchTSTCfg(seq_len=seq_len, n_sensors=n_sensors, d_model=96, n_heads=4, e_layers=2, d_ff=192, output_dim=len(HORIZON_HOURS))
        return PatchTSTClassifier(cfg), asdict(cfg)
    if model_name == "patchtst_ofp":
        from model.Optical_prediction_model.deep_learning.ofp_stat_fusion import wrap_with_ofp_stat_fusion
        from model.Optical_prediction_model.deep_learning.patchtst_wrapper import PatchTSTCfg, PatchTSTClassifier

        cfg = PatchTSTCfg(seq_len=seq_len, n_sensors=n_sensors, d_model=96, n_heads=4, e_layers=2, d_ff=192, output_dim=len(HORIZON_HOURS))
        model, fusion_cfg = wrap_with_ofp_stat_fusion(PatchTSTClassifier(cfg), n_sensors=n_sensors, d_hidden=96)
        return model, {"backbone": asdict(cfg), "ofp_fusion": fusion_cfg}
    if model_name == "moderntcn":
        from model.Optical_prediction_model.deep_learning.moderntcn_wrapper import ModernTCNCfg, ModernTCNClassifier

        cfg = ModernTCNCfg(seq_len=seq_len, n_sensors=n_sensors, dims=(64,), dw_dims=(64,), num_blocks=(2,), output_dim=len(HORIZON_HOURS))
        return ModernTCNClassifier(cfg), asdict(cfg)
    if model_name == "moderntcn_ofp":
        from model.Optical_prediction_model.deep_learning.moderntcn_wrapper import ModernTCNCfg, ModernTCNClassifier
        from model.Optical_prediction_model.deep_learning.ofp_stat_fusion import wrap_with_ofp_stat_fusion

        cfg = ModernTCNCfg(seq_len=seq_len, n_sensors=n_sensors, dims=(64,), dw_dims=(64,), num_blocks=(2,), output_dim=len(HORIZON_HOURS))
        model, fusion_cfg = wrap_with_ofp_stat_fusion(ModernTCNClassifier(cfg), n_sensors=n_sensors, d_hidden=96)
        return model, {"backbone": asdict(cfg), "ofp_fusion": fusion_cfg}
    if model_name == "fits":
        from model.Optical_prediction_model.deep_learning.interpretable_fits import InterpFITSCfg, InterpFITSClassifier

        cfg = InterpFITSCfg(seq_len=seq_len, n_sensors=n_sensors, cut_freq=min(30, seq_len // 2 + 1), d_hidden=96, output_dim=len(HORIZON_HOURS))
        return InterpFITSClassifier(cfg), asdict(cfg)
    if model_name == "fits_ofp":
        from model.Optical_prediction_model.deep_learning.interpretable_fits import InterpFITSCfg, InterpFITSClassifier
        from model.Optical_prediction_model.deep_learning.ofp_stat_fusion import wrap_with_ofp_stat_fusion

        cfg = InterpFITSCfg(seq_len=seq_len, n_sensors=n_sensors, cut_freq=min(30, seq_len // 2 + 1), d_hidden=96, output_dim=len(HORIZON_HOURS))
        model, fusion_cfg = wrap_with_ofp_stat_fusion(InterpFITSClassifier(cfg), n_sensors=n_sensors, d_hidden=96)
        return model, {"backbone": asdict(cfg), "ofp_fusion": fusion_cfg}
    if model_name == "fteformer":
        from model.Optical_prediction_model.deep_learning.mf_transformer_v3 import MFTransformerV3, MFTransformerV3Cfg

        cfg = MFTransformerV3Cfg(
            seq_len=seq_len,
            n_sensors=n_sensors,
            d_model=96,
            n_heads=4,
            e_layers=2,
            d_ff=192,
            n_mfp_layers=2,
            time_pool_size=max(1, seq_len // 48),
            use_dynamic_sensor_mask=True,
            sensor_mask_mode="soft",
            sensor_contrastive_weight=0.03,
            sensor_mask_temperature=0.5,
            output_dim=len(HORIZON_HOURS),
        )
        return MFTransformerV3(cfg), asdict(cfg)
    if model_name == "fteformer_ofp":
        from model.Optical_prediction_model.deep_learning.mf_transformer_v3 import MFTransformerV3, MFTransformerV3Cfg
        from model.Optical_prediction_model.deep_learning.ofp_stat_fusion import wrap_with_ofp_stat_fusion

        cfg = MFTransformerV3Cfg(
            seq_len=seq_len,
            n_sensors=n_sensors,
            d_model=96,
            n_heads=4,
            e_layers=2,
            d_ff=192,
            n_mfp_layers=2,
            time_pool_size=max(1, seq_len // 48),
            use_dynamic_sensor_mask=True,
            sensor_mask_mode="soft",
            sensor_contrastive_weight=0.03,
            sensor_mask_temperature=0.5,
            output_dim=len(HORIZON_HOURS),
        )
        model, fusion_cfg = wrap_with_ofp_stat_fusion(MFTransformerV3(cfg), n_sensors=n_sensors, d_hidden=96)
        return model, {"backbone": asdict(cfg), "ofp_fusion": fusion_cfg}
    raise ValueError(model_name)


def model_logits(model: nn.Module, xs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    out = model(xs, masks)
    if isinstance(out, tuple):
        return out[0]
    return out


def primary_score_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1:
        return arr[:, PRIMARY_HORIZON_INDEX]
    return arr


def train_epoch(model, loader, optimizer, loss_fn, device: str, grad_clip: float) -> tuple[float, np.ndarray, np.ndarray]:
    model.train()
    losses, scores, labels = [], [], []
    for xs, masks, ys in loader:
        xs = xs.to(device)
        masks = masks.to(device)
        ys = ys.to(device)
        optimizer.zero_grad()
        logits = model_logits(model, xs, masks)
        if ys.ndim > 1 and (logits.ndim <= 1 or logits.shape[-1] != ys.shape[-1]):
            raise ValueError(f"multi-horizon labels require logits shape (batch,{ys.shape[-1]}), got {tuple(logits.shape)}")
        loss = loss_fn(logits, ys)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()) * xs.size(0))
        scores.append(primary_score_matrix(torch.sigmoid(logits).detach().cpu().numpy()))
        labels.append(primary_score_matrix(ys.detach().cpu().numpy()).astype(int))
    denom = max(len(loader.dataset), 1)
    return sum(losses) / denom, np.concatenate(scores), np.concatenate(labels)


@torch.no_grad()
def infer_samples(model, loader, loss_fn, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, scores, labels = [], [], []
    for xs, masks, ys in loader:
        xs = xs.to(device)
        masks = masks.to(device)
        ys = ys.to(device)
        logits = model_logits(model, xs, masks)
        if ys.ndim > 1 and (logits.ndim <= 1 or logits.shape[-1] != ys.shape[-1]):
            raise ValueError(f"multi-horizon labels require logits shape (batch,{ys.shape[-1]}), got {tuple(logits.shape)}")
        loss = loss_fn(logits, ys)
        losses.append(float(loss.item()) * xs.size(0))
        scores.append(primary_score_matrix(torch.sigmoid(logits).cpu().numpy()))
        labels.append(primary_score_matrix(ys.cpu().numpy()).astype(int))
    denom = max(len(loader.dataset), 1)
    return sum(losses) / denom, np.concatenate(scores), np.concatenate(labels)


def point_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    out = {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


@torch.no_grad()
def score_one_module(
    model: nn.Module,
    values: np.ndarray,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    valid = np.isfinite(values).astype(np.float32)
    x = np.where(valid > 0, values, 0.0).astype(np.float32)
    x = (x - mean) / std
    x = x * valid
    pad_x = np.zeros((seq_len - 1, values.shape[1]), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = np.concatenate([pad_x, x], axis=0)
    m_pad = np.concatenate([pad_m, valid], axis=0)
    # (T, N, L) -> (T, L, N), one causal window per timestamp.
    x_windows = np.lib.stride_tricks.sliding_window_view(
        x_pad, window_shape=seq_len, axis=0
    ).transpose(0, 2, 1)
    m_windows = np.lib.stride_tricks.sliding_window_view(
        m_pad, window_shape=seq_len, axis=0
    ).transpose(0, 2, 1)

    scores = []
    for start in range(0, len(values), batch_size):
        end = min(len(values), start + batch_size)
        xb = torch.from_numpy(np.array(x_windows[start:end], copy=True)).to(device)
        mb = torch.from_numpy(np.array(m_windows[start:end], copy=True)).to(device)
        logits = model_logits(model, xb, mb)
        score = torch.sigmoid(logits).cpu().numpy()
        scores.append(primary_score_matrix(score))
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def collect_module_scores(
    model: nn.Module,
    data_dir: Path,
    file_names: list[str],
    cache: ModuleArrayCache,
    cfg: DeepRunCfg,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[dict]:
    rows = []
    for name in file_names:
        item = cache.get(name)
        timestamps, values, anomaly, _labels = item[:4]
        scores = score_one_module(model, values, cfg.seq_len, mean, std, cfg.batch_size, cfg.device)
        true_pos = anomaly > 0
        true_label = int(true_pos.any())
        true_ts = float(timestamps[np.argmax(true_pos)]) if true_label else None
        if len(item) > 4:
            valid_mask = np.asarray(item[4], dtype=bool)
        else:
            valid_mask = np.isfinite(timestamps.astype(float))
            if true_ts is not None:
                valid_mask = valid_mask & (timestamps.astype(float) < float(true_ts))
        rows.append(
            {
                "file_name": name,
                "timestamps": timestamps,
                "scores": scores,
                "true_label": true_label,
                "true_ts": true_ts,
                "valid_mask": valid_mask,
            }
        )
    return rows


def module_metrics_from_scores(module_rows: list[dict], threshold: float) -> dict[str, float]:
    true_pos, pred_pos = set(), set()
    for row in module_rows:
        name = row["file_name"]
        if row["true_label"]:
            true_pos.add(name)
        scores = row["scores"]
        valid_mask = np.asarray(row.get("valid_mask", np.ones_like(scores, dtype=bool)), dtype=bool)
        idx = np.where((scores >= threshold) & valid_mask)[0]
        if len(idx) == 0:
            continue
        pred_ts = float(row["timestamps"][int(idx[0])])
        if row["true_label"] == 0 or (row["true_ts"] is not None and pred_ts < row["true_ts"]):
            pred_pos.add(name)
    hit = true_pos & pred_pos
    tp = len(hit)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def choose_module_threshold(module_rows: list[dict], grid_size: int = 99) -> tuple[float, dict[str, float]]:
    best_thr, best = 0.5, {"f1": -1.0}
    for thr in np.linspace(0.01, 0.99, int(grid_size)):
        metrics = module_metrics_from_scores(module_rows, float(thr))
        if metrics["f1"] > best["f1"]:
            best_thr, best = float(thr), metrics
    best = dict(best)
    best["threshold"] = best_thr
    return best_thr, best


def write_predictions(
    module_rows: list[dict],
    out_dir: Path,
    threshold: float,
    cfg: DeepRunCfg | None = None,
    alarm_strategy: dict[str, object] | None = None,
) -> None:
    strategy = dict(alarm_strategy or {})
    write_alarm_predictions(
        module_rows,
        out_dir,
        threshold,
        alarm_smoothing=str(strategy.get("alarm_smoothing", cfg.alarm_smoothing if cfg is not None else "none")),
        alarm_smooth_window=int(strategy.get("alarm_smooth_window", cfg.alarm_smooth_window if cfg is not None else 1)),
        alarm_consecutive_k=int(strategy.get("alarm_consecutive_k", cfg.alarm_consecutive_k if cfg is not None else 1)),
        first_alarm_only=bool(cfg.first_alarm_only) if cfg is not None else False,
    )


def split_train_val(index_df: pd.DataFrame, fold: int, val_fraction: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    train_df = index_df[index_df["folder_index"] != fold].copy()
    test_df = index_df[index_df["folder_index"] == fold].copy()
    tr, va = train_test_split(
        train_df,
        test_size=float(val_fraction),
        random_state=seed + fold,
        stratify=train_df["Label"],
    )
    return tr["file_name"].tolist(), va["file_name"].tolist(), test_df["file_name"].tolist()


def run_model_fold(
    model_name: str,
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    cfg: DeepRunCfg,
    max_train_files: int | None,
    max_val_files: int | None,
    max_test_files: int | None,
) -> dict:
    t0 = time.time()
    set_seed(cfg.seed + fold)
    train_files, val_files, test_files = split_train_val(index_df, fold, cfg.val_fraction, cfg.seed)
    if max_val_files is not None:
        val_files = val_files[:max_val_files]
    if max_test_files is not None:
        test_files = test_files[:max_test_files]

    run_dir = out_root / model_name / f"fold_{fold}"
    cache = ModuleArrayCache(data_dir, max_cached_files=cfg.max_cached_files)
    mean, std = compute_norm_stats(data_dir, train_files)
    samples = sample_training_windows(
        data_dir,
        train_files,
        cfg.train_pos_per_file,
        cfg.train_neg_per_file,
        seed=cfg.seed + fold,
        max_files=max_train_files,
    )
    sample_labels = np.asarray([label for *_rest, label in samples], dtype=np.float32) if samples else np.empty((0, len(HORIZON_HOURS)), dtype=np.float32)
    primary_sample_labels = sample_labels[:, PRIMARY_HORIZON_INDEX] if sample_labels.ndim == 2 and len(sample_labels) else np.empty((0,), dtype=np.float32)
    pos = int(primary_sample_labels.sum())
    neg = int(len(samples) - pos)
    ds = OFPWindowDataset(samples, cache, cfg.seq_len, mean, std)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate, drop_last=False)

    model, model_cfg = make_model(model_name, cfg.seq_len, len(SENSORS))
    model = model.to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    pos_by_horizon = sample_labels.sum(axis=0) if len(sample_labels) else np.ones(len(HORIZON_HOURS), dtype=np.float32)
    neg_by_horizon = len(sample_labels) - pos_by_horizon
    pos_weight = np.maximum(1.0, neg_by_horizon / np.maximum(pos_by_horizon, 1.0)).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state, best_f1, best_epoch = None, -1.0, 0
    history = []
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        ep0 = time.time()
        tr_loss, tr_scores, tr_y = train_epoch(model, loader, optimizer, loss_fn, cfg.device, cfg.grad_clip)
        train_metrics = point_metrics(tr_y, tr_scores, threshold=0.5)
        improved = train_metrics["f1"] > best_f1 + 1e-5
        if improved:
            best_f1 = train_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": tr_loss, **{f"train_{k}": v for k, v in train_metrics.items()}, "seconds": time.time() - ep0})
        print(f"[{model_name} fold={fold}] ep{epoch:02d} loss={tr_loss:.4f} train_F1={train_metrics['f1']:.4f} time={time.time()-ep0:.1f}s")
        if stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[{model_name} fold={fold}] scoring val modules={len(val_files)}")
    val_rows = collect_module_scores(model, data_dir, val_files, cache, cfg, mean, std)
    threshold, alarm_strategy, val_module_metrics = choose_alarm_strategy_by_final_score(
        val_rows,
        grid_size=cfg.threshold_grid_size,
        smoothing=cfg.alarm_smoothing,
        smooth_windows=cfg.alarm_search_smooth_windows,
        consecutive_ks=cfg.alarm_search_consecutive_ks,
    )
    print(
        f"[{model_name} fold={fold}] val final={val_module_metrics['final_score']:.4f} "
        f"F1={val_module_metrics['f1_score']:.4f} thr={threshold:.3f} alarm={alarm_strategy}"
    )

    print(f"[{model_name} fold={fold}] scoring test modules={len(test_files)}")
    test_rows = collect_module_scores(model, data_dir, test_files, cache, cfg, mean, std)
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(test_rows, pred_dir, threshold, cfg=cfg, alarm_strategy=alarm_strategy)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    test_metrics = {str(r["Item"]): float(r["Value"]) for _, r in summary_df.iterrows()}

    result = {
        "model": model_name,
        "fold": fold,
        "threshold": threshold,
        "alarm_strategy": alarm_strategy,
        "first_alarm_only": bool(cfg.first_alarm_only),
        "best_train_f1": best_f1,
        "best_epoch": best_epoch,
        "train_modules": len(train_files),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "train_samples": len(samples),
        "train_pos_samples": int(pos),
        "train_neg_samples": int(neg),
        "train_pos_samples_by_horizon": pos_by_horizon.astype(float).tolist(),
        "train_neg_samples_by_horizon": neg_by_horizon.astype(float).tolist(),
        "horizon_hours": list(HORIZON_HOURS),
        "primary_horizon_index": int(PRIMARY_HORIZON_INDEX),
        "pos_weight": pos_weight.astype(float).tolist(),
        "n_params": int(n_params),
        "seconds": time.time() - t0,
        "model_cfg": model_cfg,
        "run_cfg": asdict(cfg),
        "val_module_metrics": val_module_metrics,
        "metrics": test_metrics,
        "history": history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "run_cfg": asdict(cfg)}, run_dir / "model.pt")
    return result


def aggregate(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "threshold": item["threshold"],
            "alarm_smoothing": item.get("alarm_strategy", {}).get("alarm_smoothing", ""),
            "alarm_smooth_window": item.get("alarm_strategy", {}).get("alarm_smooth_window", 1),
            "alarm_consecutive_k": item.get("alarm_strategy", {}).get("alarm_consecutive_k", 1),
            "first_alarm_only": item.get("first_alarm_only", False),
            "train_samples": item["train_samples"],
            "train_pos_samples": item["train_pos_samples"],
            "train_neg_samples": item["train_neg_samples"],
            "seconds": item["seconds"],
        }
        row.update(item["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "fold_metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df.sort_values(["model", "fold"], inplace=True)
    df.to_csv(metrics_path, index=False)
    numeric = [c for c in df.columns if c not in {"model", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_protocol/deep_models"))
    parser.add_argument("--models", nargs="+", default=["itransformer", "patchtst", "moderntcn", "fits", "fteformer"],
                        choices=[
                            "itransformer", "itransformer_ofp",
                            "patchtst", "patchtst_ofp",
                            "moderntcn", "moderntcn_ofp",
                            "fits", "fits_ofp",
                            "fteformer", "fteformer_ofp",
                        ])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train_pos_per_file", type=int, default=96)
    parser.add_argument("--train_neg_per_file", type=int, default=24)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--alarm_search_smooth_windows", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--alarm_search_consecutive_ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no_first_alarm_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    cfg = DeepRunCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        train_pos_per_file=args.train_pos_per_file,
        train_neg_per_file=args.train_neg_per_file,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        max_cached_files=args.max_cached_files,
        alarm_smoothing=args.alarm_smoothing,
        alarm_smooth_window=args.alarm_smooth_window,
        alarm_consecutive_k=args.alarm_consecutive_k,
        alarm_search_smooth_windows=tuple(int(x) for x in args.alarm_search_smooth_windows),
        alarm_search_consecutive_ks=tuple(int(x) for x in args.alarm_search_consecutive_ks),
        first_alarm_only=not args.no_first_alarm_only,
    )
    all_results: list[dict] = []
    for model_name in args.models:
        for fold in args.folds:
            print(f"[run] model={model_name} fold={fold} device={cfg.device}")
            result = run_model_fold(
                model_name,
                fold,
                args.data_dir,
                index_df,
                args.out_root,
                cfg,
                args.max_train_files,
                args.max_val_files,
                args.max_test_files,
            )
            all_results.append(result)
            aggregate([result], args.out_root)
            print(
                f"[done] {model_name} fold={fold} "
                f"F1={result['metrics'].get('f1_score', 0):.4f} "
                f"P={result['metrics'].get('precision', 0):.4f} "
                f"R={result['metrics'].get('recall', 0):.4f} "
                f"time={result['seconds']:.1f}s"
            )
    aggregate(all_results, args.out_root)


if __name__ == "__main__":
    main()

