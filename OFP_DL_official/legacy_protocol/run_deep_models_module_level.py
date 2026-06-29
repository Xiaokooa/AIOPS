from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_deep_models import (
    make_model,
    make_window,
    model_logits,
    set_seed,
    split_train_val,
)
from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
    SENSORS,
    first_anomaly_ahead_label,
    first_event_valid_mask,
    read_index,
)


SEC_IN_HOUR = 3600.0


@dataclass
class ModuleLevelCfg:
    seq_len: int = 288
    epochs: int = 3
    module_batch_size: int = 8
    score_batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    windows_per_module: int = 32
    positive_windows_per_faulty: int = 16
    val_fraction: float = 0.1
    threshold_grid_size: int = 99
    max_cached_files: int = 512
    seed: int = 42
    point_loss_weight: float = 0.5
    bag_loss_weight: float = 1.0
    healthy_bag_weight: float = 1.0
    positive_horizon_hours: float = 120.0
    horizon_hours: tuple[float, ...] = (16.0, 24.0, 72.0, 120.0)
    primary_horizon_hours: float = 120.0
    score_fusion: str = "primary"
    horizon_fusion_weights: tuple[float, ...] = ()
    sampling_strategy: str = "random"
    lead_time_bins: tuple[float, ...] = (0.0, 16.0, 24.0, 72.0, 120.0)
    exclude_post_event_train: bool = False
    bag_pooling: str = "max"
    bag_topk: int = 3
    bag_lse_tau: float = 0.5
    trigger_mode: str = "point"
    trigger_k: int = 1
    smooth_window: int = 1
    feature_mode: str = "raw"
    rolling_windows: tuple[int, ...] = (12, 36)
    norm_max_files: int = 512
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_metric: str = "final"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


KEY_ROLLING_FEATURES = [
    "temperature",
    "current",
    "currentTXPower",
    "currentRXPower",
    "rx_lane_mean",
    "tx_lane_mean",
]


def feature_names(feature_mode: str, rolling_windows: tuple[int, ...]) -> list[str]:
    names = list(SENSORS)
    if feature_mode == "raw":
        return names
    derived = [
        "rx_lane_mean",
        "rx_lane_std",
        "rx_lane_range",
        "tx_lane_mean",
        "tx_lane_std",
        "tx_lane_range",
        "rx0_lane_mean_gap",
        "tx0_lane_mean_gap",
        "tx_rx_gap",
        "abs_tx_rx_gap",
    ]
    names.extend(derived)
    for col in KEY_ROLLING_FEATURES:
        names.append(f"{col}_delta")
    for win in rolling_windows:
        for col in KEY_ROLLING_FEATURES:
            names.append(f"{col}_rmean{win}")
            names.append(f"{col}_rstd{win}")
    return names


def build_feature_frame(raw_df: pd.DataFrame, feature_mode: str, rolling_windows: tuple[int, ...]) -> pd.DataFrame:
    base = raw_df[SENSORS].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    if feature_mode == "raw":
        return base

    out = base.copy()
    rx_cols = [
        "currentMultiRXPower1",
        "currentMultiRXPower2",
        "currentMultiRXPower3",
        "currentMultiRXPower4",
    ]
    tx_cols = [
        "currentMultiTXPower1",
        "currentMultiTXPower2",
        "currentMultiTXPower3",
        "currentMultiTXPower4",
    ]
    rx_lanes = base[rx_cols]
    tx_lanes = base[tx_cols]
    out["rx_lane_mean"] = rx_lanes.mean(axis=1)
    out["rx_lane_std"] = rx_lanes.std(axis=1).fillna(0.0)
    out["rx_lane_range"] = rx_lanes.max(axis=1) - rx_lanes.min(axis=1)
    out["tx_lane_mean"] = tx_lanes.mean(axis=1)
    out["tx_lane_std"] = tx_lanes.std(axis=1).fillna(0.0)
    out["tx_lane_range"] = tx_lanes.max(axis=1) - tx_lanes.min(axis=1)
    out["rx0_lane_mean_gap"] = base["currentRXPower"] - out["rx_lane_mean"]
    out["tx0_lane_mean_gap"] = base["currentTXPower"] - out["tx_lane_mean"]
    out["tx_rx_gap"] = base["currentTXPower"] - base["currentRXPower"]
    out["abs_tx_rx_gap"] = out["tx_rx_gap"].abs()

    for col in KEY_ROLLING_FEATURES:
        series = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_delta"] = series.diff().fillna(0.0)
        for win in rolling_windows:
            roll = series.rolling(int(win), min_periods=2)
            out[f"{col}_rmean{win}"] = roll.mean().fillna(series)
            out[f"{col}_rstd{win}"] = roll.std().fillna(0.0)

    return out[feature_names(feature_mode, rolling_windows)].replace([np.inf, -np.inf], np.nan).astype(np.float32)


class FeatureModuleArrayCache:
    def __init__(
        self,
        data_dir: Path,
        max_cached_files: int,
        feature_mode: str,
        rolling_windows: tuple[int, ...],
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_cached_files = int(max_cached_files)
        self.feature_mode = feature_mode
        self.rolling_windows = rolling_windows
        self.feature_names = feature_names(feature_mode, rolling_windows)
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.order: list[str] = []

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            if file_name in self.order:
                self.order.remove(file_name)
            self.order.append(file_name)
            return self.cache[file_name]

        df = pd.read_csv(self.data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        values = build_feature_frame(df, self.feature_mode, self.rolling_windows).to_numpy(dtype=np.float32)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        labels = first_anomaly_ahead_label(df)
        valid_mask = first_event_valid_mask(df)
        item = (timestamps, values, anomaly, labels, valid_mask)

        self.cache[file_name] = item
        self.order.append(file_name)
        while len(self.order) > self.max_cached_files:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return item


def compute_feature_norm_stats(
    data_dir: Path,
    train_files: list[str],
    feature_mode: str,
    rolling_windows: tuple[int, ...],
    max_files: int,
) -> tuple[np.ndarray, np.ndarray]:
    files = train_files[:]
    if len(files) > int(max_files):
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, size=int(max_files), replace=False))
    n_features = len(feature_names(feature_mode, rolling_windows))
    total = np.zeros(n_features, dtype=np.float64)
    total_sq = np.zeros(n_features, dtype=np.float64)
    counts = np.zeros(n_features, dtype=np.float64)
    for name in files:
        df = pd.read_csv(data_dir / name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        arr = build_feature_frame(df, feature_mode, rolling_windows).to_numpy(dtype=np.float64)
        valid_mask = first_event_valid_mask(df)
        arr = arr[valid_mask]
        if len(arr) == 0:
            continue
        mask = np.isfinite(arr)
        arr0 = np.where(mask, arr, 0.0)
        total += arr0.sum(axis=0)
        total_sq += (arr0 * arr0).sum(axis=0)
        counts += mask.sum(axis=0)
    counts = np.maximum(counts, 1.0)
    mean = (total / counts).astype(np.float32)
    var = np.maximum(total_sq / counts - mean.astype(np.float64) ** 2, 1e-6)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


class ModuleBagDataset(Dataset):
    """One item is one module represented as a bag of timestamp windows."""

    def __init__(
        self,
        file_names: list[str],
        cache: FeatureModuleArrayCache,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        windows_per_module: int,
        positive_windows_per_faulty: int,
        positive_horizon_hours: float,
        horizon_hours: tuple[float, ...],
        primary_horizon_hours: float,
        sampling_strategy: str,
        lead_time_bins: tuple[float, ...],
        exclude_post_event_train: bool,
        seed: int,
    ) -> None:
        self.cache = cache
        self.seq_len = int(seq_len)
        self.mean = mean
        self.std = std
        self.windows_per_module = int(windows_per_module)
        self.positive_windows_per_faulty = int(positive_windows_per_faulty)
        self.positive_horizon_seconds = max(1.0, float(positive_horizon_hours) * SEC_IN_HOUR)
        self.horizon_hours = tuple(float(h) for h in horizon_hours)
        if not self.horizon_hours:
            self.horizon_hours = (float(positive_horizon_hours),)
        self.horizon_seconds = np.asarray([max(1.0, h * SEC_IN_HOUR) for h in self.horizon_hours], dtype=np.float64)
        self.max_horizon_seconds = float(self.horizon_seconds.max())
        primary = float(primary_horizon_hours)
        self.primary_horizon_idx = int(np.argmin(np.abs(np.asarray(self.horizon_hours, dtype=np.float64) - primary)))
        self.sampling_strategy = str(sampling_strategy)
        self.lead_time_bins = tuple(float(x) for x in lead_time_bins)
        self.exclude_post_event_train = bool(exclude_post_event_train)
        self.seed = int(seed)
        self.epoch = 0
        self.file_names = [name for name in file_names if np.any(self.cache.get(name)[4])]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.file_names)

    @staticmethod
    def _sample_indices(rng: np.random.Generator, candidates: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(candidates) == 0:
            return np.empty((0,), dtype=np.int64)
        replace = len(candidates) < count
        return rng.choice(candidates, size=count, replace=replace).astype(np.int64)

    def _sample_lead_bins(
        self,
        rng: np.random.Generator,
        timestamps: np.ndarray,
        first_ts: int,
        total_count: int,
        fallback: np.ndarray,
    ) -> np.ndarray:
        if total_count <= 0:
            return np.empty((0,), dtype=np.int64)
        bins = sorted(set(float(x) for x in self.lead_time_bins))
        if len(bins) < 2:
            return self._sample_indices(rng, fallback, total_count)

        lead_hours = (float(first_ts) - timestamps.astype(np.float64)) / SEC_IN_HOUR
        sampled: list[np.ndarray] = []
        n_bins = len(bins) - 1
        base = total_count // n_bins
        extra = total_count % n_bins
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            quota = base + (1 if i < extra else 0)
            candidates = np.where((lead_hours > lo) & (lead_hours <= hi))[0].astype(np.int64)
            part = self._sample_indices(rng, candidates, quota)
            if len(part):
                sampled.append(part)
        out = np.concatenate(sampled) if sampled else np.empty((0,), dtype=np.int64)
        if len(out) < total_count:
            fill = self._sample_indices(rng, fallback, total_count - len(out))
            out = np.concatenate([out, fill])
        if len(out) > total_count:
            out = rng.choice(out, size=total_count, replace=False).astype(np.int64)
        return out.astype(np.int64)

    def __getitem__(self, idx: int):
        file_name = self.file_names[idx]
        timestamps, values, anomaly, _legacy_labels, valid_mask = self.cache.get(file_name)
        all_idx = np.flatnonzero(valid_mask).astype(np.int64)
        rng = np.random.default_rng(self.seed + idx + self.epoch * 1_000_003)

        anomaly_idx = np.where(anomaly > 0)[0]
        module_label = int(len(anomaly_idx) > 0)
        pos_mask = np.zeros(self.windows_per_module, dtype=np.float32)
        targets = np.zeros((self.windows_per_module, len(self.horizon_hours)), dtype=np.float32)

        if module_label:
            first_ts = int(timestamps[int(anomaly_idx[0])])
            pos_candidates = all_idx[(timestamps[all_idx] >= first_ts - self.max_horizon_seconds) & (timestamps[all_idx] < first_ts)]
            if len(pos_candidates) == 0:
                pos_candidates = all_idx[all_idx < int(anomaly_idx[0])]
            n_pos = min(self.positive_windows_per_faulty, self.windows_per_module)
            if self.sampling_strategy == "lead_bins":
                pos_idx = self._sample_lead_bins(rng, timestamps, first_ts, n_pos, pos_candidates)
            else:
                pos_idx = self._sample_indices(rng, pos_candidates, n_pos)
            if self.exclude_post_event_train:
                train_base = all_idx[timestamps[all_idx] < first_ts]
                if len(train_base) == 0:
                    train_base = all_idx[all_idx < int(anomaly_idx[0])]
            else:
                train_base = all_idx
            neg_base = train_base
            neg_candidates = np.setdiff1d(neg_base, pos_candidates, assume_unique=False)
            if len(neg_candidates) == 0:
                neg_candidates = np.setdiff1d(train_base, pos_candidates, assume_unique=False)
            neg_idx = self._sample_indices(rng, neg_candidates, self.windows_per_module - len(pos_idx))
            chosen = np.concatenate([pos_idx, neg_idx])
            rng.shuffle(chosen)
            if len(chosen) < self.windows_per_module:
                fill_base = train_base if len(train_base) else all_idx
                fill = self._sample_indices(rng, fill_base, self.windows_per_module - len(chosen))
                chosen = np.concatenate([chosen, fill])
            for j, row_idx in enumerate(chosen[: self.windows_per_module]):
                lead_seconds = float(first_ts - int(timestamps[int(row_idx)]))
                if lead_seconds > 0:
                    labels = (lead_seconds <= self.horizon_seconds).astype(np.float32)
                    targets[j, :] = labels
                    pos_mask[j] = labels[self.primary_horizon_idx]
        else:
            chosen = self._sample_indices(rng, all_idx, self.windows_per_module)

        xs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for row_idx in chosen[: self.windows_per_module]:
            x, mask = make_window(values, int(row_idx), self.seq_len, self.mean, self.std)
            xs.append(x)
            masks.append(mask)

        return (
            torch.from_numpy(np.stack(xs).astype(np.float32)),
            torch.from_numpy(np.stack(masks).astype(np.float32)),
            torch.from_numpy(targets),
            torch.tensor(module_label, dtype=torch.float32),
            torch.from_numpy(pos_mask),
        )


def collate_module_bags(batch):
    xs = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    targets = torch.stack([item[2] for item in batch])
    module_labels = torch.stack([item[3] for item in batch])
    pos_masks = torch.stack([item[4] for item in batch]).bool()
    return xs, masks, targets, module_labels, pos_masks


def primary_horizon_index(cfg: ModuleLevelCfg) -> int:
    horizons = np.asarray(cfg.horizon_hours if cfg.horizon_hours else (cfg.positive_horizon_hours,), dtype=np.float64)
    return int(np.argmin(np.abs(horizons - float(cfg.primary_horizon_hours))))


def bag_pool_logits(values: torch.Tensor, cfg: ModuleLevelCfg, mask: torch.Tensor | None = None) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"bag pooling expects (batch, windows), got {tuple(values.shape)}")
    if mask is not None:
        values = values.masked_fill(~mask, -1.0e9)

    mode = cfg.bag_pooling
    if mode == "max":
        return values.max(dim=1).values
    if mode == "topk":
        k = max(1, min(int(cfg.bag_topk), values.shape[1]))
        top_vals = torch.topk(values, k=k, dim=1).values
        if mask is None:
            return top_vals.mean(dim=1)
        valid = top_vals > -1.0e8
        denom = valid.sum(dim=1).clamp(min=1).to(values.dtype)
        return top_vals.masked_fill(~valid, 0.0).sum(dim=1) / denom
    if mode == "lse":
        tau = max(float(cfg.bag_lse_tau), 1.0e-4)
        pooled = tau * torch.logsumexp(values / tau, dim=1)
        if mask is None:
            count = torch.full((values.shape[0],), values.shape[1], dtype=values.dtype, device=values.device)
        else:
            count = mask.sum(dim=1).clamp(min=1).to(values.dtype)
        return pooled - tau * torch.log(count)
    raise ValueError(f"unknown bag_pooling: {mode}")


def module_level_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    module_labels: torch.Tensor,
    pos_masks: torch.Tensor,
    point_loss_fn: nn.Module,
    cfg: ModuleLevelCfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits_all = logits.unsqueeze(-1) if logits.ndim == 2 else logits
    targets_all = targets.unsqueeze(-1) if targets.ndim == 2 else targets
    point_loss = point_loss_fn(logits_all, targets_all)
    loss = cfg.point_loss_weight * point_loss
    primary_idx = primary_horizon_index(cfg)
    primary_logits = logits_all[..., primary_idx]

    healthy = module_labels < 0.5
    healthy_loss = logits.new_tensor(0.0)
    if healthy.any():
        healthy_max = bag_pool_logits(primary_logits[healthy], cfg)
        healthy_loss = nn.functional.binary_cross_entropy_with_logits(
            healthy_max,
            torch.zeros_like(healthy_max),
        )
        loss = loss + cfg.healthy_bag_weight * healthy_loss

    faulty_with_pos = (module_labels > 0.5) & pos_masks.any(dim=1)
    positive_bag_loss = logits.new_tensor(0.0)
    if faulty_with_pos.any():
        pos_bag_logit = bag_pool_logits(
            primary_logits[faulty_with_pos],
            cfg,
            mask=pos_masks[faulty_with_pos],
        )
        positive_bag_loss = nn.functional.binary_cross_entropy_with_logits(
            pos_bag_logit,
            torch.ones_like(pos_bag_logit),
        )
        loss = loss + cfg.bag_loss_weight * positive_bag_loss

    with torch.no_grad():
        module_pred = (torch.sigmoid(primary_logits).max(dim=1).values >= 0.5).float()
        tp = ((module_pred > 0.5) & (module_labels > 0.5)).sum().item()
        pred_pos = (module_pred > 0.5).sum().item()
        true_pos = (module_labels > 0.5).sum().item()
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / true_pos if true_pos else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return loss, {
        "point_loss": float(point_loss.item()),
        "healthy_loss": float(healthy_loss.item()),
        "positive_bag_loss": float(positive_bag_loss.item()),
        "module_f1": f1,
        "module_precision": precision,
        "module_recall": recall,
    }


def train_epoch_module_level(
    model: nn.Module,
    dataset: ModuleBagDataset,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    point_loss_fn: nn.Module,
    cfg: ModuleLevelCfg,
    epoch: int,
) -> dict[str, float]:
    dataset.set_epoch(epoch)
    model.train()
    totals: dict[str, float] = {
        "loss": 0.0,
        "point_loss": 0.0,
        "healthy_loss": 0.0,
        "positive_bag_loss": 0.0,
        "module_f1": 0.0,
        "module_precision": 0.0,
        "module_recall": 0.0,
    }
    n_batches = 0
    n_modules = 0
    for xs, masks, targets, module_labels, pos_masks in loader:
        bsz, n_windows, seq_len, n_sensors = xs.shape
        xs = xs.to(cfg.device).reshape(bsz * n_windows, seq_len, n_sensors)
        masks = masks.to(cfg.device).reshape(bsz * n_windows, seq_len, n_sensors)
        targets = targets.to(cfg.device)
        module_labels = module_labels.to(cfg.device)
        pos_masks = pos_masks.to(cfg.device)

        flat_logits = model_logits(model, xs, masks)
        if flat_logits.ndim == 1:
            logits = flat_logits.reshape(bsz, n_windows)
        else:
            logits = flat_logits.reshape(bsz, n_windows, -1)
        loss, parts = module_level_loss(
            logits=logits,
            targets=targets,
            module_labels=module_labels,
            pos_masks=pos_masks,
            point_loss_fn=point_loss_fn,
            cfg=cfg,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        n_batches += 1
        n_modules += int(bsz)
        totals["loss"] += float(loss.item())
        for key, value in parts.items():
            totals[key] += value

    denom = max(n_batches, 1)
    out = {key: value / denom for key, value in totals.items()}
    out["modules"] = float(n_modules)
    return out


def reduce_logits_to_scores(logits: torch.Tensor, cfg: ModuleLevelCfg) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if probs.ndim == 1:
        return probs
    mode = cfg.score_fusion
    if mode == "primary":
        return probs[:, primary_horizon_index(cfg)]
    if mode == "mean":
        return probs.mean(dim=1)
    if mode == "weighted":
        if cfg.horizon_fusion_weights:
            weights = torch.tensor(cfg.horizon_fusion_weights, dtype=probs.dtype, device=probs.device)
        else:
            weights = torch.ones(probs.shape[1], dtype=probs.dtype, device=probs.device)
        if weights.numel() != probs.shape[1]:
            raise ValueError("horizon_fusion_weights must match the number of horizon heads")
        weights = weights / weights.sum().clamp(min=1.0e-8)
        return (probs * weights.view(1, -1)).sum(dim=1)
    raise ValueError(f"unknown score_fusion: {mode}")


@torch.no_grad()
def score_one_module_module_level(
    model: nn.Module,
    values: np.ndarray,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: str,
    cfg: ModuleLevelCfg,
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
    x_windows = np.lib.stride_tricks.sliding_window_view(
        x_pad, window_shape=seq_len, axis=0
    ).transpose(0, 2, 1)
    m_windows = np.lib.stride_tricks.sliding_window_view(
        m_pad, window_shape=seq_len, axis=0
    ).transpose(0, 2, 1)

    scores: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        end = min(len(values), start + batch_size)
        xb = torch.from_numpy(np.array(x_windows[start:end], copy=True)).to(device)
        mb = torch.from_numpy(np.array(m_windows[start:end], copy=True)).to(device)
        logits = model_logits(model, xb, mb)
        scores.append(reduce_logits_to_scores(logits, cfg).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def collect_module_scores_module_level(
    model: nn.Module,
    data_dir: Path,
    file_names: list[str],
    cache: FeatureModuleArrayCache,
    cfg: ModuleLevelCfg,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[dict]:
    rows = []
    for name in file_names:
        timestamps, values, anomaly, _labels, valid_mask = cache.get(name)
        scores = score_one_module_module_level(
            model=model,
            values=values,
            seq_len=cfg.seq_len,
            mean=mean,
            std=std,
            batch_size=cfg.score_batch_size,
            device=cfg.device,
            cfg=cfg,
        )
        true_pos = anomaly > 0
        true_label = int(true_pos.any())
        true_ts = float(timestamps[np.argmax(true_pos)]) if true_label else None
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


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or len(scores) == 0:
        return scores
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(scores.astype(np.float64), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def first_trigger_index(
    scores: np.ndarray,
    threshold: float,
    cfg: ModuleLevelCfg,
    valid_mask: np.ndarray | None = None,
) -> int | None:
    work = smooth_scores(scores, cfg.smooth_window) if cfg.trigger_mode in {"smooth", "smooth_consecutive"} else scores
    active = work >= float(threshold)
    if valid_mask is not None:
        active = active & valid_mask.astype(bool)
    if cfg.trigger_mode in {"point", "smooth"}:
        idx = np.where(active)[0]
        return int(idx[0]) if len(idx) else None
    if cfg.trigger_mode in {"consecutive", "smooth_consecutive"}:
        need = max(1, int(cfg.trigger_k))
        run = 0
        for i, flag in enumerate(active):
            run = run + 1 if bool(flag) else 0
            if run >= need:
                return int(i)
        return None
    raise ValueError(f"unknown trigger_mode: {cfg.trigger_mode}")


def write_predictions_with_trigger(module_rows: list[dict], out_dir: Path, threshold: float, cfg: ModuleLevelCfg) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in module_rows:
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            true_ts = row.get("true_ts")
            valid_mask = np.ones(len(row["scores"]), dtype=bool) if true_ts is None else row["timestamps"] < float(true_ts)
        pred = np.zeros(len(row["scores"]), dtype=int)
        idx = first_trigger_index(row["scores"], threshold, cfg, valid_mask)
        if idx is not None:
            pred[idx:] = 1
        pred = pred & valid_mask.astype(int)
        pd.DataFrame(
            {
                "timestamp": row["timestamps"],
                "predict": pred,
                "score": row["scores"],
                "valid_for_eval": valid_mask.astype(int),
            }
        ).to_csv(
            out_dir / row["file_name"], index=False
        )


def evaluate_detail_from_scores(module_rows: list[dict], threshold: float, cfg: ModuleLevelCfg) -> dict[str, float]:
    detail_rows: list[dict] = []
    for row in module_rows:
        scores = row["scores"]
        timestamps = row["timestamps"]
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            true_ts = row.get("true_ts")
            valid_mask = np.ones(len(scores), dtype=bool) if true_ts is None else timestamps < float(true_ts)
        trigger_idx = first_trigger_index(scores, threshold, cfg, valid_mask)
        predict_label = int(trigger_idx is not None)
        predict_ts = float(timestamps[int(trigger_idx)]) if trigger_idx is not None else None
        true_label = int(row["true_label"])
        true_ts = row["true_ts"]

        valid_predict_positive = 0
        hit = 0
        lead_hour = None
        if predict_label > 0:
            if true_label == 0:
                valid_predict_positive = 1
            elif true_ts is not None and predict_ts is not None and true_ts > predict_ts:
                valid_predict_positive = 1
                hit = 1
                lead_hour = abs(true_ts - predict_ts) / SEC_IN_HOUR

        detail_rows.append(
            {
                "file_name": row["file_name"],
                "true_label": true_label,
                "predict_label": predict_label,
                "true_ts": true_ts,
                "predict_ts": predict_ts,
                "valid_predict_positive": valid_predict_positive,
                "hit": hit,
                "lead_hour": lead_hour,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    true_pos = set(detail_df.loc[detail_df["true_label"] > 0, "file_name"])
    pred_pos = set(detail_df.loc[detail_df["valid_predict_positive"] > 0, "file_name"])
    hit_pos = true_pos & pred_pos
    tp = len(hit_pos)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = len(detail_df) - tp - fp - fn
    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(detail_df) if len(detail_df) else 0.0
    lead_hours = detail_df.loc[detail_df["hit"] > 0, "lead_hour"].dropna().astype(float)
    avg_lead_hour = float(lead_hours.mean()) if not lead_hours.empty else 0.0
    min_lead_hour = float(lead_hours.min()) if not lead_hours.empty else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    return {
        "final_score": final_score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "all_hit_cnt": tp,
        "all_predict_pos_cnt": len(pred_pos),
        "all_true_pos_cnt": len(true_pos),
        "avg_lead_score": avg_lead_score,
        "avg_lead_hour": avg_lead_hour,
        "min_lead_score": min_lead_score,
        "min_lead_hour": min_lead_hour,
        "lead_pread_cnt": int(detail_df["hit"].sum()),
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_module_cnt": len(detail_df),
    }


def choose_ofp_threshold(module_rows: list[dict], cfg: ModuleLevelCfg) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in np.linspace(float(cfg.threshold_min), float(cfg.threshold_max), int(cfg.threshold_grid_size)):
        metrics = evaluate_detail_from_scores(module_rows, float(threshold), cfg)
        if best_metrics is None:
            best_threshold, best_metrics = float(threshold), metrics
            continue
        if cfg.threshold_metric == "f1":
            current_key = (
                metrics["f1_score"],
                metrics["precision"],
                metrics["recall"],
                -metrics["all_predict_pos_cnt"],
            )
            best_key = (
                best_metrics["f1_score"],
                best_metrics["precision"],
                best_metrics["recall"],
                -best_metrics["all_predict_pos_cnt"],
            )
        else:
            current_key = (
                metrics["final_score"],
                metrics["f1_score"],
                metrics["precision"],
                -metrics["all_predict_pos_cnt"],
            )
            best_key = (
                best_metrics["final_score"],
                best_metrics["f1_score"],
                best_metrics["precision"],
                -best_metrics["all_predict_pos_cnt"],
            )
        if current_key > best_key:
            best_threshold, best_metrics = float(threshold), metrics
    assert best_metrics is not None
    best_metrics = dict(best_metrics)
    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def make_module_model(model_name: str, seq_len: int, n_sensors: int, cfg: ModuleLevelCfg) -> tuple[nn.Module, dict]:
    n_outputs = len(cfg.horizon_hours if cfg.horizon_hours else (cfg.positive_horizon_hours,))
    if model_name == "itransformer_ofp" and n_outputs > 1:
        from model.Optical_prediction_model.deep_learning.models import ITransformerOFPCfg, ITransformerOFPClassifier

        model_cfg = ITransformerOFPCfg(
            seq_len=seq_len,
            n_sensors=n_sensors,
            d_model=96,
            n_heads=4,
            e_layers=2,
            d_ff=192,
            output_dim=n_outputs,
        )
        return ITransformerOFPClassifier(model_cfg), asdict(model_cfg)
    return make_model(model_name, seq_len, n_sensors)


def run_model_fold(
    model_name: str,
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    cfg: ModuleLevelCfg,
    max_train_files: int | None,
    max_val_files: int | None,
    max_test_files: int | None,
) -> dict:
    t0 = time.time()
    set_seed(cfg.seed + fold)
    train_files, val_files, test_files = split_train_val(index_df, fold, cfg.val_fraction, cfg.seed)
    if max_train_files is not None:
        train_files = train_files[: int(max_train_files)]
    if max_val_files is not None:
        val_files = val_files[: int(max_val_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]

    run_dir = out_root / model_name / f"fold_{fold}"
    rolling_windows = tuple(int(win) for win in cfg.rolling_windows)
    cache = FeatureModuleArrayCache(
        data_dir,
        max_cached_files=cfg.max_cached_files,
        feature_mode=cfg.feature_mode,
        rolling_windows=rolling_windows,
    )
    mean, std = compute_feature_norm_stats(
        data_dir,
        train_files,
        feature_mode=cfg.feature_mode,
        rolling_windows=rolling_windows,
        max_files=cfg.norm_max_files,
    )
    train_dataset = ModuleBagDataset(
        file_names=train_files,
        cache=cache,
        seq_len=cfg.seq_len,
        mean=mean,
        std=std,
        windows_per_module=cfg.windows_per_module,
        positive_windows_per_faulty=cfg.positive_windows_per_faulty,
        positive_horizon_hours=cfg.positive_horizon_hours,
        horizon_hours=cfg.horizon_hours,
        primary_horizon_hours=cfg.primary_horizon_hours,
        sampling_strategy=cfg.sampling_strategy,
        lead_time_bins=cfg.lead_time_bins,
        exclude_post_event_train=cfg.exclude_post_event_train,
        seed=cfg.seed + fold,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.module_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_module_bags,
        drop_last=False,
    )

    model, model_cfg = make_module_model(model_name, cfg.seq_len, len(cache.feature_names), cfg)
    model = model.to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    point_pos_weight = max(1.0, (cfg.windows_per_module - cfg.positive_windows_per_faulty) / max(cfg.positive_windows_per_faulty, 1))
    point_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(point_pos_weight, device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        ep0 = time.time()
        train_metrics = train_epoch_module_level(
            model=model,
            dataset=train_dataset,
            loader=train_loader,
            optimizer=optimizer,
            point_loss_fn=point_loss_fn,
            cfg=cfg,
            epoch=epoch,
        )
        train_metrics["epoch"] = epoch
        train_metrics["seconds"] = time.time() - ep0
        history.append(train_metrics)
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[module {model_name} fold={fold}] ep{epoch:02d} "
            f"loss={train_metrics['loss']:.4f} "
            f"bagF1={train_metrics['module_f1']:.4f} "
            f"time={train_metrics['seconds']:.1f}s"
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[module {model_name} fold={fold}] scoring val modules={len(val_files)}")
    val_rows = collect_module_scores_module_level(model, data_dir, val_files, cache, cfg, mean, std)
    threshold, val_module_metrics = choose_ofp_threshold(val_rows, cfg)
    print(
        f"[module {model_name} fold={fold}] val final={val_module_metrics['final_score']:.4f} "
        f"F1={val_module_metrics['f1_score']:.4f} thr={threshold:.3f}"
    )

    print(f"[module {model_name} fold={fold}] scoring test modules={len(test_files)}")
    test_rows = collect_module_scores_module_level(model, data_dir, test_files, cache, cfg, mean, std)
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions_with_trigger(test_rows, pred_dir, threshold, cfg)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    test_metrics = metrics_to_dict(summary_df)

    result = {
        "model": model_name,
        "fold": int(fold),
        "training_mode": "module_level_mil",
        "adapter_type": "pure_ofp_decision_adapter",
        "threshold_selection": f"validation_ofp_{cfg.threshold_metric}",
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_loss),
        "train_modules": len(train_files),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "n_params": int(n_params),
        "seconds": time.time() - t0,
        "model_cfg": model_cfg,
        "run_cfg": asdict(cfg),
        "feature_names": cache.feature_names,
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
            "training_mode": item["training_mode"],
            "threshold": item["threshold"],
            "best_epoch": item["best_epoch"],
            "train_modules": item["train_modules"],
            "val_modules": item["val_modules"],
            "test_modules": item["test_modules"],
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
    numeric = [col for col in df.columns if col not in {"model", "fold", "training_mode"} and pd.api.types.is_numeric_dtype(df[col])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run module-level OFP deep models.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_legacy_protocol/deep_module_level"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["itransformer", "patchtst", "moderntcn", "fits", "fteformer"],
        choices=[
            "itransformer", "itransformer_ofp",
            "patchtst", "patchtst_ofp",
            "moderntcn", "moderntcn_ofp",
            "fits", "fits_ofp",
            "fteformer", "fteformer_ofp",
        ],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--module_batch_size", type=int, default=8)
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--windows_per_module", type=int, default=32)
    parser.add_argument("--positive_windows_per_faulty", type=int, default=16)
    parser.add_argument("--point_loss_weight", type=float, default=0.5)
    parser.add_argument("--bag_loss_weight", type=float, default=1.0)
    parser.add_argument("--healthy_bag_weight", type=float, default=1.0)
    parser.add_argument("--positive_horizon_hours", type=float, default=120.0)
    parser.add_argument("--horizon_hours", nargs="+", type=float, default=[16.0, 24.0, 72.0, 120.0])
    parser.add_argument("--primary_horizon_hours", type=float, default=None)
    parser.add_argument("--score_fusion", choices=["primary", "mean", "weighted"], default="primary")
    parser.add_argument("--horizon_fusion_weights", nargs="+", type=float, default=[])
    parser.add_argument("--sampling_strategy", choices=["random", "lead_bins"], default="random")
    parser.add_argument("--lead_time_bins", nargs="+", type=float, default=[0.0, 16.0, 24.0, 72.0, 120.0])
    parser.add_argument("--exclude_post_event_train", action="store_true")
    parser.add_argument("--bag_pooling", choices=["max", "topk", "lse"], default="max")
    parser.add_argument("--bag_topk", type=int, default=3)
    parser.add_argument("--bag_lse_tau", type=float, default=0.5)
    parser.add_argument("--trigger_mode", choices=["point", "smooth", "consecutive", "smooth_consecutive"], default="point")
    parser.add_argument("--trigger_k", type=int, default=1)
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--feature_mode", choices=["raw", "agg"], default="raw")
    parser.add_argument("--rolling_windows", nargs="+", type=int, default=[12, 36])
    parser.add_argument("--norm_max_files", type=int, default=512)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_metric", choices=["final", "f1"], default="final")
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_cached_files", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    horizon_hours = tuple(float(x) for x in (args.horizon_hours if args.horizon_hours is not None else [args.positive_horizon_hours]))
    primary_horizon_hours = float(args.primary_horizon_hours if args.primary_horizon_hours is not None else horizon_hours[-1])
    cfg = ModuleLevelCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        module_batch_size=args.module_batch_size,
        score_batch_size=args.score_batch_size,
        lr=args.lr,
        windows_per_module=args.windows_per_module,
        positive_windows_per_faulty=args.positive_windows_per_faulty,
        val_fraction=args.val_fraction,
        threshold_grid_size=args.threshold_grid_size,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        point_loss_weight=args.point_loss_weight,
        bag_loss_weight=args.bag_loss_weight,
        healthy_bag_weight=args.healthy_bag_weight,
        positive_horizon_hours=args.positive_horizon_hours,
        horizon_hours=horizon_hours,
        primary_horizon_hours=primary_horizon_hours,
        score_fusion=args.score_fusion,
        horizon_fusion_weights=tuple(float(x) for x in args.horizon_fusion_weights),
        sampling_strategy=args.sampling_strategy,
        lead_time_bins=tuple(float(x) for x in args.lead_time_bins),
        exclude_post_event_train=bool(args.exclude_post_event_train),
        bag_pooling=args.bag_pooling,
        bag_topk=args.bag_topk,
        bag_lse_tau=args.bag_lse_tau,
        trigger_mode=args.trigger_mode,
        trigger_k=args.trigger_k,
        smooth_window=args.smooth_window,
        feature_mode=args.feature_mode,
        rolling_windows=tuple(int(win) for win in args.rolling_windows),
        norm_max_files=args.norm_max_files,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_metric=args.threshold_metric,
        device=args.device,
    )
    all_results: list[dict] = []
    for model_name in args.models:
        for fold in args.folds:
            print(f"[run-module] model={model_name} fold={fold} device={cfg.device}")
            result = run_model_fold(
                model_name=model_name,
                fold=fold,
                data_dir=args.data_dir,
                index_df=index_df,
                out_root=args.out_root,
                cfg=cfg,
                max_train_files=args.max_train_files,
                max_val_files=args.max_val_files,
                max_test_files=args.max_test_files,
            )
            all_results.append(result)
            aggregate([result], args.out_root)
            print(
                f"[done-module] {model_name} fold={fold} "
                f"Final={result['metrics'].get('final_score', 0):.4f} "
                f"F1={result['metrics'].get('f1_score', 0):.4f} "
                f"P={result['metrics'].get('precision', 0):.4f} "
                f"R={result['metrics'].get('recall', 0):.4f} "
                f"time={result['seconds']:.1f}s"
            )
    aggregate(all_results, args.out_root)


if __name__ == "__main__":
    main()

