"""Time-series window dataset for deep-learning failure prediction.

Re-uses the *exact* train/val/test frames built for R4 (so XGB and DL share
windows, labels, splits, TPS-synthetic rows and NegDS healthy sampling).
The only difference vs ML pipeline: instead of 108 aggregated statistics
per window, we return the raw resampled time series sliced from
`_shared_cache/resampled_modules/`.

Output of __getitem__:
    x         : float32 tensor (L, N)    -- z-score normalized, NaN -> 0
    mask      : float32 tensor (L, N)    -- 1 where original value was valid
    y         : float32 tensor           -- scalar label or multi-horizon labels
    meta      : dict   -- file_name, label, event_label, time_to_first_minutes,
                          obs_start_ts, tps_is_synthetic, module_label
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.common import base_utils as base
from model.Optical_prediction_model.common import cache_utils


@dataclass
class NormStats:
    """Per-sensor mean / std computed from training raw values."""
    mean: np.ndarray   # (N,)
    std: np.ndarray    # (N,)
    sensors: list[str]


class WindowSliceCache:
    """Lazy loader: resampled module dataframes are cached per process."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    def get(self, file_name: str) -> tuple[np.ndarray, list[str]]:
        if file_name not in self._cache:
            df, sensors, _ = cache_utils.load_resampled_module(file_name)
            arr = df[sensors].to_numpy(dtype=np.float32)
            self._cache[file_name] = (arr, sensors)
        return self._cache[file_name]


def compute_norm_stats(
    frame: pd.DataFrame,
    slice_cache: WindowSliceCache,
    obs_steps: int,
    max_modules: int | None = 400,
    random_state: int = 42,
) -> NormStats:
    """Compute mean/std per sensor over a sample of train windows' raw values."""
    sampled_files = frame["file_name"].drop_duplicates()
    if max_modules is not None and len(sampled_files) > max_modules:
        sampled_files = sampled_files.sample(n=max_modules, random_state=random_state)
    sensors_ref: list[str] | None = None
    running_sum = None
    running_sq = None
    running_n = None
    for fname in sampled_files:
        arr, sensors = slice_cache.get(fname)
        if sensors_ref is None:
            sensors_ref = sensors
            running_sum = np.zeros(len(sensors), dtype=np.float64)
            running_sq = np.zeros(len(sensors), dtype=np.float64)
            running_n = np.zeros(len(sensors), dtype=np.float64)
        valid = np.isfinite(arr)
        arr0 = np.where(valid, arr, 0.0).astype(np.float64)
        running_sum += arr0.sum(axis=0)
        running_sq += (arr0 ** 2).sum(axis=0)
        running_n += valid.sum(axis=0)
    running_n = np.maximum(running_n, 1.0)
    mean = (running_sum / running_n).astype(np.float32)
    var = (running_sq / running_n) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None)).astype(np.float32)
    return NormStats(mean=mean, std=std, sensors=sensors_ref or [])


class FailureWindowDataset(Dataset):
    """A single split's windows as (raw time series, label) tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        obs_steps: int,
        slice_cache: WindowSliceCache,
        norm: NormStats,
        target_columns: list[str] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.obs_steps = int(obs_steps)
        self.slice_cache = slice_cache
        self.norm = norm
        self.sensor_count = len(norm.sensors)
        self.target_columns = target_columns or ["label"]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        fname = str(row["file_name"])
        start = int(row["obs_start_idx"])
        end = int(row["obs_end_idx"])  # inclusive
        arr, sensors = self.slice_cache.get(fname)
        seq = arr[start : end + 1]                       # (L, N)
        # Pad or crop in case of edge differences.
        if seq.shape[0] != self.obs_steps:
            pad = np.full(
                (self.obs_steps, seq.shape[1]),
                np.nan,
                dtype=np.float32,
            )
            n = min(seq.shape[0], self.obs_steps)
            pad[:n] = seq[:n]
            seq = pad
        mask = np.isfinite(seq).astype(np.float32)
        seq = np.where(mask > 0, seq, 0.0).astype(np.float32)
        # z-score per sensor, only where valid
        seq = (seq - self.norm.mean) / self.norm.std
        seq = seq * mask                                  # keep 0 where masked
        x = torch.from_numpy(seq)                         # (L, N)
        m = torch.from_numpy(mask)                        # (L, N)
        targets = [float(row[col]) for col in self.target_columns]
        y = torch.tensor(targets, dtype=torch.float32)
        if len(self.target_columns) == 1:
            y = y.squeeze(0)
        meta = {
            "file_name": fname,
            "label": int(row["label"]),
            "event_label": int(row.get("event_label", 0)),
            "time_to_first_minutes": float(row.get("time_to_first_minutes", float("nan"))),
            "obs_start_ts": int(row["obs_start_ts"]),
            "obs_end_ts": int(row.get("obs_end_ts", row["obs_start_ts"])),
            "pred_ts": int(row.get("pred_ts", row.get("obs_end_ts", row["obs_start_ts"]))),
            "first_failure_ts": int(row.get("first_failure_ts", -1)),
            "tps_is_synthetic": int(row.get("tps_is_synthetic", 0)),
            "module_label": int(row.get("module_label", 0)),
            "row_idx": int(idx),
        }
        return x, m, y, meta


def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ms = torch.stack([b[1] for b in batch], dim=0)
    ys = torch.stack([b[2] for b in batch], dim=0)
    metas = [b[3] for b in batch]
    return xs, ms, ys, metas
