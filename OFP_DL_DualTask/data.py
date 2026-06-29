"""Dual-task OFP window construction for deep-learning models.

The frame produced here is deliberately model2-compatible:

* it keeps one row per module prediction timestamp;
* it supports prefix-padded windows so a model can alert at the first row;
* it exposes both current abnormal labels and future-ahead labels.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.common import base_utils as base
from model.Optical_prediction_model.common import cache_utils
from model.Optical_prediction_model.deep_learning.data import (
    NormStats,
    WindowSliceCache,
    compute_norm_stats,
)


SplitRole = Literal["train", "val", "test"]


def _short_sensor_name(name: str) -> str:
    mapping = {
        "temperature": "temp",
        "current": "curr",
        "currentTXPower": "tx0",
        "currentRXPower": "rx0",
        "currentMultiRXPower1": "rx1",
        "currentMultiRXPower2": "rx2",
        "currentMultiRXPower3": "rx3",
        "currentMultiRXPower4": "rx4",
        "currentMultiTXPower1": "tx1",
        "currentMultiTXPower2": "tx2",
        "currentMultiTXPower3": "tx3",
        "currentMultiTXPower4": "tx4",
    }
    return mapping.get(name, "".join(ch for ch in name if ch.isalnum())[:16] or "sensor")


def sensor_columns_tag(sensor_columns: tuple[str, ...] | list[str] | None) -> str:
    if not sensor_columns:
        return "allraw"
    return f"raw{len(sensor_columns)}-" + "-".join(_short_sensor_name(str(name)) for name in sensor_columns)


def subset_norm_stats(norm: NormStats, sensor_columns: tuple[str, ...] | list[str] | None) -> NormStats:
    """Return a NormStats object restricted to the requested raw sensor columns."""
    if not sensor_columns:
        return norm
    selected = [str(name) for name in sensor_columns]
    missing = [name for name in selected if name not in norm.sensors]
    if missing:
        raise ValueError(
            "unknown sensor_columns: "
            + ", ".join(missing)
            + f"; available sensors: {', '.join(norm.sensors)}"
        )
    idx = np.asarray([norm.sensors.index(name) for name in selected], dtype=int)
    return NormStats(mean=norm.mean[idx], std=norm.std[idx], sensors=selected)


@dataclass(frozen=True)
class DualTaskConfig:
    """Window and label configuration for the OFP dual-task adapter."""

    obs_minutes: int = 1440
    ahead_hours: int = 120
    train_step_minutes: int = 60
    eval_step_minutes: int = 5
    train_faulty_max_windows: int = 256
    train_healthy_max_windows: int = 32
    min_history_steps: int = 1
    include_prefix_windows: bool = True
    current_label_mode: Literal["end", "any"] = "end"
    exclude_start_fault_modules: bool = False
    pre_event_only: bool = False
    sensor_columns: tuple[str, ...] = ()
    random_state: int = 42

    def __post_init__(self) -> None:
        for name in ("obs_minutes", "train_step_minutes", "eval_step_minutes"):
            value = int(getattr(self, name))
            if value <= 0 or value % 5 != 0:
                raise ValueError(f"{name} must be a positive multiple of 5 minutes, got {value}")
        if self.ahead_hours <= 0:
            raise ValueError("ahead_hours must be positive")
        if self.min_history_steps <= 0:
            raise ValueError("min_history_steps must be positive")
        cleaned = tuple(str(name).strip() for name in self.sensor_columns if str(name).strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"sensor_columns contains duplicates: {cleaned}")
        object.__setattr__(self, "sensor_columns", cleaned)

    @property
    def obs_steps(self) -> int:
        return self.obs_minutes // 5

    @property
    def ahead_seconds(self) -> int:
        return self.ahead_hours * 3600

    @property
    def tag(self) -> str:
        prefix = "prefix" if self.include_prefix_windows else "fullobs"
        scope = "transition" if self.exclude_start_fault_modules else "allpos"
        validity = "prefault" if self.pre_event_only else "allrows"
        sensors = sensor_columns_tag(self.sensor_columns)
        return (
            f"dual_ofp_obs{self.obs_minutes}m_ahead{self.ahead_hours}h_"
            f"tr{self.train_step_minutes}m_ev{self.eval_step_minutes}m_"
            f"{prefix}_cur-{self.current_label_mode}_{scope}_{validity}_{sensors}"
        )

    @property
    def cache_tag(self) -> str:
        return (
            f"{self.tag}_maxf{self.train_faulty_max_windows}_"
            f"maxh{self.train_healthy_max_windows}_hist{self.min_history_steps}"
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update({"obs_steps": self.obs_steps, "ahead_seconds": self.ahead_seconds, "tag": self.tag})
        return payload


def _limit_split(split_df: pd.DataFrame, limit: int | None, random_state: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(split_df) <= limit:
        return split_df.reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    remaining = int(limit)
    groups = list(split_df.groupby("Label", sort=True))
    for idx, (_, group) in enumerate(groups):
        if idx == len(groups) - 1:
            take = min(len(group), remaining)
        else:
            take = int(round(limit * len(group) / len(split_df)))
            take = max(1, min(len(group), take, remaining - (len(groups) - idx - 1)))
        pieces.append(group.sample(n=take, random_state=random_state + idx))
        remaining -= take
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def get_split_map(
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
    max_train_modules: int | None = None,
    max_val_modules: int | None = None,
    max_test_modules: int | None = None,
) -> dict[str, pd.DataFrame]:
    split_map = base.split_modules_stratified(
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
    )
    limits = {
        "train": max_train_modules,
        "val": max_val_modules,
        "test": max_test_modules,
    }
    return {
        name: _limit_split(split_df, limits[name], random_state)
        for name, split_df in split_map.items()
    }


def _uniform_take(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[idx].copy()


def _sample_train_rows(rows: list[dict[str, object]], cfg: DualTaskConfig) -> list[dict[str, object]]:
    if not rows:
        return rows
    df = pd.DataFrame(rows).sort_values("pred_ts").reset_index(drop=True)
    event_label = int(df["event_label"].max()) if "event_label" in df.columns else 0
    max_windows = cfg.train_faulty_max_windows if event_label == 1 else cfg.train_healthy_max_windows
    if len(df) <= max_windows:
        return rows

    pos_df = df[(df["current_label"] > 0) | (df["ahead_label"] > 0)]
    neg_df = df[(df["current_label"] <= 0) & (df["ahead_label"] <= 0)]
    if len(pos_df) >= max_windows:
        return _uniform_take(pos_df, max_windows).to_dict("records")
    neg_take = max_windows - len(pos_df)
    sampled = pd.concat([pos_df, _uniform_take(neg_df, neg_take)], ignore_index=True)
    sampled = sampled.sort_values("pred_ts").reset_index(drop=True)
    return sampled.to_dict("records")


def _prediction_indices(n_rows: int, cfg: DualTaskConfig, split_role: SplitRole) -> range:
    step_minutes = cfg.train_step_minutes if split_role == "train" else cfg.eval_step_minutes
    stride = max(1, step_minutes // 5)
    first = 0 if cfg.include_prefix_windows else cfg.obs_steps - 1
    first = max(0, first)
    if first >= n_rows:
        return range(0, 0, stride)
    return range(first, n_rows, stride)


def _extract_module_rows(
    df: pd.DataFrame,
    file_name: str,
    module_label: int,
    cfg: DualTaskConfig,
    split_role: SplitRole,
) -> list[dict[str, object]]:
    if df.empty:
        return []
    t_first = base.first_failure_timestamp(df)
    event_label = int(t_first is not None)
    anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0
    observed = pd.to_numeric(df["observed"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64").to_numpy()
    finite_ts = timestamps[np.isfinite(timestamps.astype(float))]
    start_ts = int(finite_ts[0]) if len(finite_ts) else int(timestamps[0])
    start_fault = bool(event_label and t_first is not None and int(t_first) <= start_ts)
    if cfg.exclude_start_fault_modules and start_fault:
        return []

    rows: list[dict[str, object]] = []
    for pred_idx in _prediction_indices(len(df), cfg, split_role):
        raw_start_idx = pred_idx - cfg.obs_steps + 1
        obs_start_idx = max(0, raw_start_idx)
        obs_end_idx = pred_idx
        history_steps = obs_end_idx - obs_start_idx + 1
        if history_steps < cfg.min_history_steps:
            continue

        obs_anomaly = anomaly[obs_start_idx : obs_end_idx + 1]
        obs_observed = observed[obs_start_idx : obs_end_idx + 1]
        pred_ts = int(timestamps[pred_idx])
        if cfg.current_label_mode == "any":
            current_label = int(bool(obs_anomaly.any()))
        else:
            current_label = int(bool(anomaly[pred_idx]))
        window_current_label = int(bool(obs_anomaly.any()))

        lead_seconds = int(t_first - pred_ts) if t_first is not None else -1
        if cfg.pre_event_only and event_label and lead_seconds <= 0:
            continue
        ahead_label = int(t_first is not None and 0 < lead_seconds <= cfg.ahead_seconds)
        obs_coverage = float(obs_observed.sum() / max(cfg.obs_steps, 1))
        payload = {
            "file_name": str(file_name),
            "module_label": int(module_label),
            "split_role": split_role,
            "task_family": "ofp_dual_current_ahead",
            "obs_start_idx": int(obs_start_idx),
            "obs_end_idx": int(obs_end_idx),
            "raw_obs_start_idx": int(raw_start_idx),
            "prefix_pad_steps": int(max(0, -raw_start_idx)),
            "history_steps": int(history_steps),
            "obs_start_ts": int(timestamps[obs_start_idx]),
            "obs_end_ts": pred_ts,
            "pred_ts": pred_ts,
            "first_failure_ts": int(t_first) if t_first is not None else -1,
            "event_label": event_label,
            "start_fault": int(start_fault),
            "current_label": current_label,
            "window_current_label": window_current_label,
            "ahead_label": ahead_label,
            "label": int(current_label or ahead_label),
            "lead_seconds": lead_seconds,
            "time_to_first_minutes": float(lead_seconds / 60.0) if t_first is not None else np.nan,
            "obs_coverage_ratio": obs_coverage,
            "ahead_hours": int(cfg.ahead_hours),
        }
        rows.append(payload)

    if split_role == "train":
        rows = _sample_train_rows(rows, cfg)
    return rows


def _frame_cache_path(split_name: str, split_df: pd.DataFrame, cfg: DualTaskConfig) -> Path:
    split_sig = cache_utils.split_fingerprint(split_df)
    return cache_utils.FEATURE_FRAME_CACHE_DIR / f"{split_name}_{split_sig}_{cfg.cache_tag}_dual_task.pkl"


def build_dual_task_frame(
    split_df: pd.DataFrame,
    cfg: DualTaskConfig,
    split_role: SplitRole,
    cache_path: Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if cache_path and cache_path.exists() and not force_rebuild:
        return pd.read_pickle(cache_path)

    rows: list[dict[str, object]] = []
    for row in split_df.itertuples(index=False):
        file_name = str(getattr(row, "file_name"))
        module_label = int(getattr(row, "Label"))
        if not (base.TRAINING_DIR / file_name).exists():
            continue
        resampled, _, _ = cache_utils.load_resampled_module(file_name)
        rows.extend(_extract_module_rows(resampled, file_name, module_label, cfg, split_role))
    frame = pd.DataFrame(rows)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache_path)
    return frame


def load_dual_task_frames(
    cfg: DualTaskConfig | None = None,
    force_rebuild: bool = False,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
    max_train_modules: int | None = None,
    max_val_modules: int | None = None,
    max_test_modules: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cfg = cfg or DualTaskConfig(random_state=random_state)
    split_map = get_split_map(
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
        max_train_modules=max_train_modules,
        max_val_modules=max_val_modules,
        max_test_modules=max_test_modules,
    )
    frames: dict[str, pd.DataFrame] = {}
    for split_name, split_df in split_map.items():
        cache_path = _frame_cache_path(split_name, split_df, cfg)
        frames[split_name] = build_dual_task_frame(
            split_df,
            cfg,
            split_role=split_name,  # type: ignore[arg-type]
            cache_path=cache_path,
            force_rebuild=force_rebuild,
        )
        if cfg.exclude_start_fault_modules or cfg.pre_event_only:
            present = set(frames[split_name].get("file_name", pd.Series(dtype=str)).astype(str).unique())
            split_map[split_name] = split_df[split_df["file_name"].astype(str).isin(present)].reset_index(drop=True)
    return frames, split_map


class DualTaskWindowDataset(Dataset):
    """Raw time-series windows with prefix padding and two supervision heads."""

    def __init__(
        self,
        frame: pd.DataFrame,
        obs_steps: int,
        slice_cache: WindowSliceCache,
        norm: NormStats,
        target_columns: tuple[str, str] = ("current_label", "ahead_label"),
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.obs_steps = int(obs_steps)
        self.slice_cache = slice_cache
        self.norm = norm
        self.target_columns = target_columns

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        file_name = str(row["file_name"])
        arr, sensors = self.slice_cache.get(file_name)
        if list(self.norm.sensors) != list(sensors):
            missing = [name for name in self.norm.sensors if name not in sensors]
            if missing:
                raise ValueError(f"{file_name} is missing selected sensors: {', '.join(missing)}")
            sensor_idx = [sensors.index(name) for name in self.norm.sensors]
            arr = arr[:, sensor_idx]
        raw_start = int(row.get("raw_obs_start_idx", int(row["obs_start_idx"])))
        end = int(row["obs_end_idx"])
        start = max(0, raw_start)
        seq = arr[start : end + 1]

        out = np.full((self.obs_steps, arr.shape[1]), np.nan, dtype=np.float32)
        pad_left = max(0, -raw_start)
        n = min(seq.shape[0], self.obs_steps - pad_left)
        if n > 0:
            out[pad_left : pad_left + n] = seq[:n]
        mask = np.isfinite(out).astype(np.float32)
        values = np.where(mask > 0, out, 0.0).astype(np.float32)
        values = (values - self.norm.mean) / self.norm.std
        values = values * mask

        targets = torch.tensor([float(row[col]) for col in self.target_columns], dtype=torch.float32)
        meta = {
            "file_name": file_name,
            "pred_ts": int(row["pred_ts"]),
            "first_failure_ts": int(row.get("first_failure_ts", -1)),
            "event_label": int(row.get("event_label", 0)),
            "current_label": int(row.get("current_label", 0)),
            "ahead_label": int(row.get("ahead_label", 0)),
            "label": int(row.get("label", 0)),
            "row_idx": int(idx),
        }
        return torch.from_numpy(values), torch.from_numpy(mask), targets, meta


def collate_dual(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    ys = torch.stack([item[2] for item in batch], dim=0)
    metas = [item[3] for item in batch]
    return xs, masks, ys, metas


def frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"windows": 0, "modules": 0}
    return {
        "windows": int(len(frame)),
        "modules": int(frame["file_name"].nunique()),
        "current_pos": int(frame["current_label"].sum()),
        "ahead_pos": int(frame["ahead_label"].sum()),
        "union_pos": int(frame["label"].sum()),
        "start_fault_modules": int(frame.groupby("file_name")["start_fault"].max().sum()) if "start_fault" in frame else 0,
        "current_ratio": float(frame["current_label"].mean()),
        "ahead_ratio": float(frame["ahead_label"].mean()),
        "union_ratio": float(frame["label"].mean()),
    }


__all__ = [
    "DualTaskConfig",
    "DualTaskWindowDataset",
    "collate_dual",
    "compute_norm_stats",
    "frame_summary",
    "load_dual_task_frames",
    "WindowSliceCache",
]
