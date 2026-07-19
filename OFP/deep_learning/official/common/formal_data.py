from __future__ import annotations

import bisect
import json
import math
import random
import sys
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from OFP.deep_learning.official.ofp_formal_protocol.protocol import (
    DEFAULT_HORIZON_HOURS,
    PRIMARY_HORIZON_SECONDS,
    PRIMARY_HORIZON_HOURS,
    SENSORS,
    ahead_labels_and_valid_mask_multi_first_event,
    ahead_labels_multi_first_event,
    first_anomaly_timestamp,
    primary_horizon_index,
    valid_time_mask_first_event,
)
from OFP.deep_learning.official.common.losses import split_alarm_aux_logits

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    _tqdm = None


TYPE_NAMES = [
    "thermal_anomaly",
    "current_bias_anomaly",
    "tx_power_anomaly",
    "rx_power_anomaly",
    "lane_imbalance",
]

HORIZON_HOURS = DEFAULT_HORIZON_HOURS
PRIMARY_HORIZON_INDEX = primary_horizon_index(HORIZON_HOURS, PRIMARY_HORIZON_HOURS)
TASK_HORIZON_TAG = "h" + "_".join(f"{float(h):g}" for h in HORIZON_HOURS)


def primary_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 1:
        return arr.astype(np.int8, copy=False)
    return arr[:, PRIMARY_HORIZON_INDEX].astype(np.int8, copy=False)


def horizon_label_counts(labels: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(labels)
    if arr.ndim == 1:
        selected = arr[np.asarray(positions, dtype=np.int64)].reshape(-1, 1)
        if len(HORIZON_HOURS) > 1:
            selected = np.repeat(selected, len(HORIZON_HOURS), axis=1)
    else:
        selected = arr[np.asarray(positions, dtype=np.int64)]
    pos = selected.astype(np.int64).sum(axis=0)
    neg = selected.shape[0] - pos
    return pos.astype(np.int64), neg.astype(np.int64)


@dataclass
class NormStats:
    mean: list[float]
    std: list[float]
    rows_seen: int
    files_seen: int

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.mean, dtype=np.float32),
            np.asarray(self.std, dtype=np.float32),
        )


@dataclass
class WeakTypeReference:
    sensor_mean: list[float]
    sensor_std: list[float]
    tx_lane_mean: float
    tx_lane_std: float
    rx_lane_mean: float
    rx_lane_std: float
    z_threshold: float = 3.0
    fallback_z_threshold: float = 2.0

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.sensor_mean, dtype=np.float32),
            np.asarray(self.sensor_std, dtype=np.float32),
        )

    def targets_for_window(self, values: np.ndarray, mask: np.ndarray, label: int) -> tuple[np.ndarray, float]:
        targets = np.zeros(len(TYPE_NAMES), dtype=np.float32)
        if int(label) <= 0 or len(values) == 0:
            return targets, 0.0

        sensor_mean, sensor_std = self.arrays()
        valid = mask > 0
        safe_values = np.where(valid, values, np.nan)
        count = np.maximum(valid.sum(axis=0), 1)
        win_mean = np.nan_to_num(np.nansum(safe_values, axis=0) / count, nan=0.0)
        win_last = np.zeros(values.shape[1], dtype=np.float32)
        win_first = np.zeros(values.shape[1], dtype=np.float32)
        win_min = np.zeros(values.shape[1], dtype=np.float32)
        win_max = np.zeros(values.shape[1], dtype=np.float32)

        for col in range(values.shape[1]):
            idx = np.flatnonzero(valid[:, col])
            if len(idx) == 0:
                continue
            col_values = values[idx, col]
            win_first[col] = float(col_values[0])
            win_last[col] = float(col_values[-1])
            win_min[col] = float(np.nanmin(col_values))
            win_max[col] = float(np.nanmax(col_values))

        def sensor_score(indices: list[int]) -> float:
            pieces = []
            for arr in [win_mean, win_last, win_min, win_max]:
                z = np.abs((arr[indices] - sensor_mean[indices]) / np.maximum(sensor_std[indices], 1e-6))
                pieces.append(np.nanmax(z) if len(z) else 0.0)
            return float(np.nanmax(pieces)) if pieces else 0.0

        tx_lane = lane_dispersion(win_last, "currentMultiTXPower")
        rx_lane = lane_dispersion(win_last, "currentMultiRXPower")
        lane_score = max(
            abs((tx_lane - self.tx_lane_mean) / max(self.tx_lane_std, 1e-6)),
            abs((rx_lane - self.rx_lane_mean) / max(self.rx_lane_std, 1e-6)),
        )

        groups = [
            sensor_score([SENSORS.index("temperature")]),
            sensor_score([SENSORS.index("current")]),
            sensor_score([SENSORS.index("currentTXPower")] + sensor_indices("currentMultiTXPower")),
            sensor_score([SENSORS.index("currentRXPower")] + sensor_indices("currentMultiRXPower")),
            float(lane_score),
        ]
        score_arr = np.asarray(groups, dtype=np.float32)
        targets[score_arr >= float(self.z_threshold)] = 1.0
        if targets.sum() <= 0 and float(score_arr.max()) >= float(self.fallback_z_threshold):
            targets[int(score_arr.argmax())] = 1.0
        return targets, float(targets.sum() > 0)


class FormalModuleCache:
    """LRU cache with first-event OFP labels."""

    def __init__(
        self,
        data_dir: Path,
        max_cached_files: int = 512,
        timestamp_mode: str = "legacy_float32",
        module_cache_dir: Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_cached_files = int(max_cached_files)
        self.timestamp_mode = str(timestamp_mode)
        self.module_cache_dir = Path(module_cache_dir) / self.timestamp_mode / TASK_HORIZON_TAG if module_cache_dir else None
        if self.module_cache_dir is not None:
            self.module_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def _module_cache_path(self, file_name: str) -> Path | None:
        if self.module_cache_dir is None:
            return None
        return self.module_cache_dir / f"{Path(file_name).stem}.npz"

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            item = self.cache.pop(file_name)
            self.cache[file_name] = item
            return item

        cache_path = self._module_cache_path(file_name)
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as payload:
                timestamps = payload["timestamps"].astype(np.int64, copy=False)
                values = payload["values"].astype(np.float32, copy=False)
                anomaly = payload["anomaly"].astype(np.int8, copy=False)
                labels = payload["labels"].astype(np.int8, copy=False)
                if labels.ndim == 1 or labels.shape[-1] != len(HORIZON_HOURS):
                    labels = ahead_labels_multi_first_event(timestamps.astype(float), anomaly, HORIZON_HOURS)
                if "valid_mask" in payload.files:
                    valid_mask = payload["valid_mask"].astype(bool, copy=False)
                else:
                    valid_mask = valid_time_mask_first_event(timestamps.astype(float), anomaly)
        else:
            df = pd.read_csv(self.data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
            timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
            timestamps = apply_timestamp_mode(timestamps, self.timestamp_mode)
            values = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
            labels, valid_mask = ahead_labels_and_valid_mask_multi_first_event(
                timestamps.astype(float),
                anomaly,
                HORIZON_HOURS,
            )
            if cache_path is not None:
                np.savez(
                    cache_path,
                    timestamps=timestamps,
                    values=values,
                    anomaly=anomaly,
                    labels=labels.astype(np.int8),
                    valid_mask=valid_mask.astype(np.int8),
                )
        item = (timestamps, values, anomaly, labels, valid_mask.astype(bool))

        self.cache[file_name] = item
        if len(self.cache) > self.max_cached_files:
            self.cache.popitem(last=False)
        return item


def valid_row_positions(valid_mask: np.ndarray, row_stride: int = 1) -> np.ndarray:
    positions = np.flatnonzero(np.asarray(valid_mask, dtype=bool)).astype(np.int64)
    return positions[:: int(row_stride)]


class FullRowWindowDataset(Dataset):
    """One causal window per raw timestamp.

    This dataset intentionally indexes every row by default. A stride other
    than 1 is treated as a development-only setting by the trainer.
    """

    def __init__(
        self,
        file_names: list[str],
        cache: FormalModuleCache,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        type_reference: WeakTypeReference | None = None,
        row_stride: int = 1,
    ) -> None:
        self.file_names = list(file_names)
        self.cache = cache
        self.seq_len = int(seq_len)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.type_reference = type_reference
        self.row_stride = int(row_stride)
        if self.row_stride <= 0:
            raise ValueError("row_stride must be positive")

        self.starts: list[int] = [0]
        self.lengths: list[int] = []
        self.positions_by_file: list[np.ndarray] = []
        self.pos_rows = 0
        self.neg_rows = 0
        self.pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        for idx, name in enumerate(self.file_names, 1):
            _ts, _values, _anomaly, labels, valid_mask = self.cache.get(name)
            positions = valid_row_positions(valid_mask, self.row_stride)
            n = int(len(positions))
            self.lengths.append(n)
            self.positions_by_file.append(positions)
            sampled_primary = primary_labels(labels)[positions]
            pos_h, neg_h = horizon_label_counts(labels, positions)
            self.pos_rows_by_horizon += pos_h
            self.neg_rows_by_horizon += neg_h
            self.pos_rows += int(sampled_primary.sum())
            self.neg_rows += int(len(sampled_primary) - sampled_primary.sum())
            self.starts.append(self.starts[-1] + n)
        print(
            f"[official dataset] indexed done files={len(self.file_names)} "
            f"rows={self.starts[-1]} pos={self.pos_rows} neg={self.neg_rows}"
        )

    def __len__(self) -> int:
        return self.starts[-1]

    def __getitem__(self, index: int):
        file_idx = bisect.bisect_right(self.starts, int(index)) - 1
        local = int(index) - self.starts[file_idx]
        row_idx = int(self.positions_by_file[file_idx][local])
        file_name = self.file_names[file_idx]
        _ts, values, _anomaly, labels, _valid_mask = self.cache.get(file_name)
        label = np.asarray(labels[row_idx], dtype=np.float32)
        primary_label = int(label[PRIMARY_HORIZON_INDEX]) if label.ndim > 0 else int(label)
        x, mask, raw, raw_mask = make_window(values, row_idx, self.seq_len, self.mean, self.std, return_raw=True)
        if self.type_reference is None:
            type_target = np.zeros(len(TYPE_NAMES), dtype=np.float32)
            type_mask = 0.0
        else:
            type_target, type_mask = self.type_reference.targets_for_window(raw, raw_mask, primary_label)
        return (
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.from_numpy(np.atleast_1d(label).astype(np.float32)),
            torch.from_numpy(type_target),
            torch.tensor(float(type_mask), dtype=torch.float32),
        )


class BatchedModuleWindowDataset(IterableDataset):
    """Stream full row windows in module-sized batches.

    This keeps the official row-level sample set intact while avoiding one
    Python `__getitem__` call per timestamp. Each yielded item is already a
    batch of causal windows.
    """

    def __init__(
        self,
        file_names: list[str],
        cache: FormalModuleCache,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        type_reference: WeakTypeReference | None = None,
        row_stride: int = 1,
        batch_rows: int = 4096,
    ) -> None:
        self.file_names = list(file_names)
        self.cache = cache
        self.seq_len = int(seq_len)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.type_reference = type_reference
        self.row_stride = int(row_stride)
        self.batch_rows = int(batch_rows)
        if self.row_stride <= 0:
            raise ValueError("row_stride must be positive")
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive")

        self.lengths: list[int] = []
        self.pos_rows = 0
        self.neg_rows = 0
        self.pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.total_rows = 0
        self.total_batches = 0
        for idx, name in enumerate(self.file_names, 1):
            _ts, _values, _anomaly, labels, valid_mask = self.cache.get(name)
            positions = valid_row_positions(valid_mask, self.row_stride)
            sampled_primary = primary_labels(labels)[positions]
            n = int(len(sampled_primary))
            self.lengths.append(n)
            self.total_rows += n
            self.total_batches += int(math.ceil(n / self.batch_rows))
            pos_h, neg_h = horizon_label_counts(labels, positions)
            self.pos_rows_by_horizon += pos_h
            self.neg_rows_by_horizon += neg_h
            self.pos_rows += int(sampled_primary.sum())
            self.neg_rows += int(n - sampled_primary.sum())
        print(
            f"[official batched dataset] indexed done files={len(self.file_names)} "
            f"rows={self.total_rows} pos={self.pos_rows} neg={self.neg_rows} "
            f"batches={self.total_batches} batch_rows={self.batch_rows}"
        )
        self.cache.cache.clear()

    def __len__(self) -> int:
        return int(self.total_batches)

    def _worker_files(self) -> list[str]:
        worker = get_worker_info()
        if worker is None:
            return self.file_names
        return self.file_names[worker.id :: worker.num_workers]

    def __iter__(self):
        for name in self._worker_files():
            _ts, values, _anomaly, labels, valid_mask = self.cache.get(name)
            row_positions = valid_row_positions(valid_mask, self.row_stride)
            n_rows = len(row_positions)
            if n_rows <= 0:
                continue

            valid = np.isfinite(values).astype(np.float32)
            raw = np.where(valid > 0, values, 0.0).astype(np.float32)
            x = ((raw - self.mean) / self.std) * valid
            pad_x = np.zeros((self.seq_len - 1, values.shape[1]), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, x.astype(np.float32)], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, valid], axis=0))
            x_windows = x_pad.unfold(0, self.seq_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, self.seq_len, 1).permute(0, 2, 1)

            if self.type_reference is not None:
                raw_pad = np.concatenate([pad_x, raw], axis=0)
                mask_pad = np.concatenate([pad_m, valid], axis=0)
            else:
                raw_pad = None
                mask_pad = None

            sampled_labels = np.asarray(labels[row_positions], dtype=np.float32)
            sampled_primary = primary_labels(labels)[row_positions].astype(np.float32)
            for start in range(0, n_rows, self.batch_rows):
                end = min(n_rows, start + self.batch_rows)
                positions = row_positions[start:end]
                index = torch.from_numpy(positions)
                xs = x_windows.index_select(0, index).contiguous()
                masks = m_windows.index_select(0, index).contiguous()
                ys_np = sampled_labels[start:end]
                ys = torch.from_numpy(ys_np.copy())
                type_targets = torch.zeros((len(ys_np), len(TYPE_NAMES)), dtype=torch.float32)
                type_masks = torch.zeros(len(ys_np), dtype=torch.float32)

                if self.type_reference is not None and raw_pad is not None and mask_pad is not None:
                    primary_np = sampled_primary[start:end]
                    positive = np.flatnonzero(primary_np > 0)
                    for local_idx in positive:
                        row_idx = int(positions[local_idx])
                        raw_window = raw_pad[row_idx : row_idx + self.seq_len]
                        mask_window = mask_pad[row_idx : row_idx + self.seq_len]
                        target, target_mask = self.type_reference.targets_for_window(raw_window, mask_window, 1)
                        type_targets[int(local_idx)] = torch.from_numpy(target)
                        type_masks[int(local_idx)] = float(target_mask)

                yield xs, masks, ys, type_targets, type_masks


def allocate_stratified_negative_counts(neg_counts: np.ndarray, target_total: int) -> np.ndarray:
    neg_counts = np.asarray(neg_counts, dtype=np.int64)
    target_total = int(max(0, min(int(target_total), int(neg_counts.sum()))))
    alloc = np.zeros_like(neg_counts, dtype=np.int64)
    if target_total <= 0 or int(neg_counts.sum()) <= 0:
        return alloc

    positive_modules = neg_counts > 0
    if target_total >= int(positive_modules.sum()):
        alloc[positive_modules] = 1
    else:
        order = np.argsort(-neg_counts, kind="mergesort")
        chosen = order[:target_total]
        alloc[chosen] = 1
        return alloc

    remaining = target_total - int(alloc.sum())
    capacity = neg_counts - alloc
    if remaining <= 0 or int(capacity.sum()) <= 0:
        return alloc

    raw = capacity.astype(np.float64) * (float(remaining) / float(capacity.sum()))
    extra = np.floor(raw).astype(np.int64)
    extra = np.minimum(extra, capacity)
    alloc += extra
    remaining = target_total - int(alloc.sum())
    if remaining > 0:
        fractions = raw - np.floor(raw)
        order = np.lexsort((-capacity, -fractions))
        for idx in order:
            if remaining <= 0:
                break
            if alloc[idx] < neg_counts[idx]:
                alloc[idx] += 1
                remaining -= 1
    return alloc


class SampledBatchedModuleWindowDataset(IterableDataset):
    """Official sampled DL protocol: all positives + stratified negative ratio."""

    def __init__(
        self,
        file_names: list[str],
        cache: FormalModuleCache,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        type_reference: WeakTypeReference | None = None,
        batch_rows: int = 4096,
        negative_ratio: float = 10.0,
        seed: int = 42,
    ) -> None:
        self.file_names = list(file_names)
        self.cache = cache
        self.seq_len = int(seq_len)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.type_reference = type_reference
        self.batch_rows = int(batch_rows)
        self.negative_ratio = float(negative_ratio)
        self.seed = int(seed)
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if self.negative_ratio < 0:
            raise ValueError("negative_ratio must be non-negative")

        self.selected_positions: list[np.ndarray] = []
        self.source_total_rows = 0
        self.source_pos_rows = 0
        self.source_neg_rows = 0
        self.source_pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.source_neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        pos_positions: list[np.ndarray] = []
        neg_positions: list[np.ndarray] = []
        neg_counts: list[int] = []

        for idx, name in enumerate(self.file_names, 1):
            _ts, _values, _anomaly, labels, valid_mask = self.cache.get(name)
            labels = np.asarray(labels, dtype=np.int8)
            primary = primary_labels(labels)
            positions = valid_row_positions(valid_mask)
            pos = positions[primary[positions] > 0].astype(np.int64)
            neg = positions[primary[positions] <= 0].astype(np.int64)
            pos_positions.append(pos)
            neg_positions.append(neg)
            neg_counts.append(int(len(neg)))
            self.source_total_rows += int(len(positions))
            self.source_pos_rows += int(len(pos))
            self.source_neg_rows += int(len(neg))
            pos_h, neg_h = horizon_label_counts(labels, positions)
            self.source_pos_rows_by_horizon += pos_h
            self.source_neg_rows_by_horizon += neg_h

        target_neg = int(round(float(self.source_pos_rows) * self.negative_ratio))
        alloc = allocate_stratified_negative_counts(np.asarray(neg_counts, dtype=np.int64), target_neg)
        self.pos_rows = int(self.source_pos_rows)
        self.neg_rows = int(alloc.sum())
        self.pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.total_rows = int(self.pos_rows + self.neg_rows)
        self.total_batches = 0

        for name, pos, neg, n_neg in zip(self.file_names, pos_positions, neg_positions, alloc):
            if int(n_neg) >= len(neg):
                chosen_neg = neg
            elif int(n_neg) > 0:
                stable = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
                rng = np.random.default_rng(self.seed + int(stable))
                chosen_neg = np.sort(rng.choice(neg, size=int(n_neg), replace=False)).astype(np.int64)
            else:
                chosen_neg = np.empty(0, dtype=np.int64)
            selected = np.sort(np.concatenate([pos, chosen_neg])).astype(np.int64)
            self.selected_positions.append(selected)
            self.total_batches += int(math.ceil(len(selected) / self.batch_rows)) if len(selected) else 0
            if len(selected):
                _ts, _values, _anomaly, labels, _valid_mask = self.cache.get(name)
                pos_h, neg_h = horizon_label_counts(labels, selected)
                self.pos_rows_by_horizon += pos_h
                self.neg_rows_by_horizon += neg_h

        print(
            f"[official sampled dataset] indexed done files={len(self.file_names)} "
            f"source_rows={self.source_total_rows} source_pos={self.source_pos_rows} "
            f"source_neg={self.source_neg_rows} sampled_rows={self.total_rows} "
            f"sampled_pos={self.pos_rows} sampled_neg={self.neg_rows} "
            f"negative_ratio={self.negative_ratio} batches={self.total_batches} "
            f"batch_rows={self.batch_rows}"
        )
        self.cache.cache.clear()

    def __len__(self) -> int:
        return int(self.total_batches)

    def _worker_items(self) -> list[tuple[str, np.ndarray]]:
        worker = get_worker_info()
        pairs = list(zip(self.file_names, self.selected_positions))
        if worker is None:
            return pairs
        return pairs[worker.id :: worker.num_workers]

    def __iter__(self):
        for name, selected_positions in self._worker_items():
            if len(selected_positions) <= 0:
                continue
            _ts, values, _anomaly, labels, _valid_mask = self.cache.get(name)
            valid = np.isfinite(values).astype(np.float32)
            raw = np.where(valid > 0, values, 0.0).astype(np.float32)
            x = ((raw - self.mean) / self.std) * valid
            pad_x = np.zeros((self.seq_len - 1, values.shape[1]), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, x.astype(np.float32)], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, valid], axis=0))
            x_windows = x_pad.unfold(0, self.seq_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, self.seq_len, 1).permute(0, 2, 1)

            if self.type_reference is not None:
                raw_pad = np.concatenate([pad_x, raw], axis=0)
                mask_pad = np.concatenate([pad_m, valid], axis=0)
            else:
                raw_pad = None
                mask_pad = None

            for start in range(0, len(selected_positions), self.batch_rows):
                positions = selected_positions[start : start + self.batch_rows]
                index = torch.from_numpy(positions)
                xs = x_windows.index_select(0, index).contiguous()
                masks = m_windows.index_select(0, index).contiguous()
                ys_np = np.asarray(labels[positions], dtype=np.float32)
                ys = torch.from_numpy(ys_np.copy())
                type_targets = torch.zeros((len(ys_np), len(TYPE_NAMES)), dtype=torch.float32)
                type_masks = torch.zeros(len(ys_np), dtype=torch.float32)

                if self.type_reference is not None and raw_pad is not None and mask_pad is not None:
                    primary_np = primary_labels(labels)[positions].astype(np.float32)
                    positive = np.flatnonzero(primary_np > 0)
                    for local_idx in positive:
                        row_idx = int(positions[local_idx])
                        raw_window = raw_pad[row_idx : row_idx + self.seq_len]
                        mask_window = mask_pad[row_idx : row_idx + self.seq_len]
                        target, target_mask = self.type_reference.targets_for_window(raw_window, mask_window, 1)
                        type_targets[int(local_idx)] = torch.from_numpy(target)
                        type_masks[int(local_idx)] = float(target_mask)

                yield xs, masks, ys, type_targets, type_masks


def _stable_rng(seed: int, name: str, salt: int = 0) -> np.random.Generator:
    stable = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(int(seed) + int(stable) + int(salt))


def _sample_positions(
    positions: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.int64)
    if len(arr) <= 0 or int(max_count) == 0:
        return np.empty(0, dtype=np.int64)
    if int(max_count) < 0 or int(max_count) >= len(arr):
        return np.sort(arr).astype(np.int64)
    return np.sort(rng.choice(arr, size=int(max_count), replace=False)).astype(np.int64)


class ModuleBalancedBatchedModuleWindowDataset(IterableDataset):
    """Module-balanced windows for module-level first-warning training.

    The previous sampled protocol balanced by row count: all positive rows and
    a global negative ratio.  This protocol first caps each module's
    contribution, then samples windows inside the module.  Faulty modules with
    no valid primary pre-fault target are skipped for ahead-warning training.
    """

    def __init__(
        self,
        file_names: list[str],
        cache: FormalModuleCache,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        type_reference: WeakTypeReference | None = None,
        batch_rows: int = 4096,
        positive_windows_per_module: int = 96,
        negative_windows_per_faulty_module: int = 24,
        normal_windows_per_module: int = 12,
        seed: int = 42,
    ) -> None:
        self.file_names = list(file_names)
        self.cache = cache
        self.seq_len = int(seq_len)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.type_reference = type_reference
        self.batch_rows = int(batch_rows)
        self.positive_windows_per_module = int(positive_windows_per_module)
        self.negative_windows_per_faulty_module = int(negative_windows_per_faulty_module)
        self.normal_windows_per_module = int(normal_windows_per_module)
        self.seed = int(seed)
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive")

        self.selected_positions: list[np.ndarray] = []
        self.source_total_rows = 0
        self.source_pos_rows = 0
        self.source_neg_rows = 0
        self.source_pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.source_neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.pos_rows = 0
        self.neg_rows = 0
        self.pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.total_rows = 0
        self.total_batches = 0
        self.normal_modules = 0
        self.trainable_faulty_modules = 0
        self.true_transition_modules = 0
        self.start_in_window_modules = 0
        self.left_censored_modules = 0
        self.faulty_no_primary_window_modules = 0

        for name in self.file_names:
            _ts, _values, anomaly, labels, valid_mask = self.cache.get(name)
            labels = np.asarray(labels, dtype=np.int8)
            primary = primary_labels(labels)
            positions = valid_row_positions(valid_mask)
            has_fault = bool(np.any(np.asarray(anomaly) > 0))
            if len(positions):
                source_primary = primary[positions]
                self.source_total_rows += int(len(positions))
                self.source_pos_rows += int(source_primary.sum())
                self.source_neg_rows += int(len(source_primary) - source_primary.sum())
                pos_h, neg_h = horizon_label_counts(labels, positions)
                self.source_pos_rows_by_horizon += pos_h
                self.source_neg_rows_by_horizon += neg_h

            if has_fault and len(positions) <= 0:
                self.left_censored_modules += 1
                self.selected_positions.append(np.empty(0, dtype=np.int64))
                continue

            pos = positions[primary[positions] > 0].astype(np.int64) if len(positions) else np.empty(0, dtype=np.int64)
            neg = positions[primary[positions] <= 0].astype(np.int64) if len(positions) else np.empty(0, dtype=np.int64)
            rng = _stable_rng(self.seed, name, salt=113)

            if has_fault:
                if len(pos) <= 0:
                    self.faulty_no_primary_window_modules += 1
                    self.selected_positions.append(np.empty(0, dtype=np.int64))
                    continue
                self.trainable_faulty_modules += 1
                if len(neg) > 0:
                    self.true_transition_modules += 1
                else:
                    self.start_in_window_modules += 1
                chosen_pos = _sample_positions(pos, self.positive_windows_per_module, rng)
                chosen_neg = _sample_positions(neg, self.negative_windows_per_faulty_module, rng)
            else:
                self.normal_modules += 1
                chosen_pos = np.empty(0, dtype=np.int64)
                chosen_neg = _sample_positions(neg, self.normal_windows_per_module, rng)

            selected = np.sort(np.concatenate([chosen_pos, chosen_neg])).astype(np.int64)
            self.selected_positions.append(selected)
            if len(selected):
                selected_primary = primary[selected]
                self.pos_rows += int(selected_primary.sum())
                self.neg_rows += int(len(selected_primary) - selected_primary.sum())
                pos_h, neg_h = horizon_label_counts(labels, selected)
                self.pos_rows_by_horizon += pos_h
                self.neg_rows_by_horizon += neg_h
                self.total_rows += int(len(selected))
                self.total_batches += int(math.ceil(len(selected) / self.batch_rows))

        print(
            f"[official module-balanced dataset] indexed done files={len(self.file_names)} "
            f"source_rows={self.source_total_rows} source_pos={self.source_pos_rows} "
            f"source_neg={self.source_neg_rows} sampled_rows={self.total_rows} "
            f"sampled_pos={self.pos_rows} sampled_neg={self.neg_rows} "
            f"modules normal={self.normal_modules} trainable_faulty={self.trainable_faulty_modules} "
            f"true_transition={self.true_transition_modules} start_in_window={self.start_in_window_modules} "
            f"left_censored={self.left_censored_modules} no_primary_window={self.faulty_no_primary_window_modules} "
            f"per_module pos={self.positive_windows_per_module} faulty_neg={self.negative_windows_per_faulty_module} "
            f"normal_neg={self.normal_windows_per_module} batches={self.total_batches} batch_rows={self.batch_rows}"
        )
        self.cache.cache.clear()

    def __len__(self) -> int:
        return int(self.total_batches)

    def _worker_items(self) -> list[tuple[str, np.ndarray]]:
        worker = get_worker_info()
        pairs = list(zip(self.file_names, self.selected_positions))
        if worker is None:
            return pairs
        return pairs[worker.id :: worker.num_workers]

    def __iter__(self):
        for name, selected_positions in self._worker_items():
            if len(selected_positions) <= 0:
                continue
            _ts, values, _anomaly, labels, _valid_mask = self.cache.get(name)
            valid = np.isfinite(values).astype(np.float32)
            raw = np.where(valid > 0, values, 0.0).astype(np.float32)
            x = ((raw - self.mean) / self.std) * valid
            pad_x = np.zeros((self.seq_len - 1, values.shape[1]), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, x.astype(np.float32)], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, valid], axis=0))
            x_windows = x_pad.unfold(0, self.seq_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, self.seq_len, 1).permute(0, 2, 1)

            if self.type_reference is not None:
                raw_pad = np.concatenate([pad_x, raw], axis=0)
                mask_pad = np.concatenate([pad_m, valid], axis=0)
            else:
                raw_pad = None
                mask_pad = None

            for start in range(0, len(selected_positions), self.batch_rows):
                positions = selected_positions[start : start + self.batch_rows]
                index = torch.from_numpy(positions)
                xs = x_windows.index_select(0, index).contiguous()
                masks = m_windows.index_select(0, index).contiguous()
                ys_np = np.asarray(labels[positions], dtype=np.float32)
                ys = torch.from_numpy(ys_np.copy())
                type_targets = torch.zeros((len(ys_np), len(TYPE_NAMES)), dtype=torch.float32)
                type_masks = torch.zeros(len(ys_np), dtype=torch.float32)

                if self.type_reference is not None and raw_pad is not None and mask_pad is not None:
                    primary_np = primary_labels(labels)[positions].astype(np.float32)
                    positive = np.flatnonzero(primary_np > 0)
                    for local_idx in positive:
                        row_idx = int(positions[local_idx])
                        raw_window = raw_pad[row_idx : row_idx + self.seq_len]
                        mask_window = mask_pad[row_idx : row_idx + self.seq_len]
                        target, target_mask = self.type_reference.targets_for_window(raw_window, mask_window, 1)
                        type_targets[int(local_idx)] = torch.from_numpy(target)
                        type_masks[int(local_idx)] = float(target_mask)

                yield xs, masks, ys, type_targets, type_masks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_timestamp_mode(timestamps: np.ndarray, mode: str) -> np.ndarray:
    """Return timestamps under the selected OFP compatibility mode.

    `legacy_float32` intentionally reproduces the OFP/model2 feature pipeline:
    large UNIX timestamps are stored in a float32 feature table and later cast
    back to int64. This can round timestamps backward/forward by tens of
    seconds, and is required to reproduce the OFP README numbers.
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if mode in {"strict_int64", "strict"}:
        return timestamps
    if mode in {"legacy_float32", "ofp_legacy"}:
        return timestamps.astype(np.float32).astype(np.int64)
    raise ValueError(f"unknown timestamp_mode: {mode}")


def sensor_indices(prefix: str) -> list[int]:
    return [SENSORS.index(f"{prefix}{idx}") for idx in range(1, 5)]


def lane_dispersion(values: np.ndarray, prefix: str) -> float:
    idx = sensor_indices(prefix)
    arr = np.asarray(values, dtype=float)[idx]
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    return float(np.nanmax(arr) - np.nanmin(arr))


def lane_dispersion_rows(values: np.ndarray, prefix: str) -> np.ndarray:
    idx = sensor_indices(prefix)
    arr = np.asarray(values, dtype=np.float64)[:, idx]
    finite = np.isfinite(arr)
    valid_rows = finite.any(axis=1)
    if not valid_rows.any():
        return np.empty(0, dtype=np.float64)
    valid_arr = arr[valid_rows]
    row_max = np.where(np.isfinite(valid_arr), valid_arr, -np.inf).max(axis=1)
    row_min = np.where(np.isfinite(valid_arr), valid_arr, np.inf).min(axis=1)
    lane = row_max - row_min
    return lane[np.isfinite(lane)]


def make_window(
    values: np.ndarray,
    row_idx: int,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    return_raw: bool = False,
):
    end = int(row_idx) + 1
    start = max(0, end - int(seq_len))
    seq = values[start:end]
    out = np.full((seq_len, values.shape[1]), np.nan, dtype=np.float32)
    out[-len(seq):] = seq
    mask = np.isfinite(out).astype(np.float32)
    raw = np.where(mask > 0, out, 0.0).astype(np.float32)
    x = (raw - mean) / np.maximum(std, 1e-6)
    x = x * mask
    if return_raw:
        return x.astype(np.float32), mask.astype(np.float32), raw, mask.astype(np.float32)
    return x.astype(np.float32), mask.astype(np.float32)


def official_collate(batch):
    xs = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    ys = torch.stack([item[2] for item in batch])
    type_targets = torch.stack([item[3] for item in batch])
    type_masks = torch.stack([item[4] for item in batch])
    return xs, masks, ys, type_targets, type_masks


def cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=bool(enabled))
    return torch.cuda.amp.autocast(enabled=bool(enabled))


def compute_norm_stats_all_rows(
    data_dir: Path,
    train_files: list[str],
    timestamp_mode: str = "legacy_float32",
) -> NormStats:
    sums = np.zeros(len(SENSORS), dtype=np.float64)
    sums_sq = np.zeros(len(SENSORS), dtype=np.float64)
    counts = np.zeros(len(SENSORS), dtype=np.float64)
    rows_seen = 0
    for idx, name in enumerate(train_files, 1):
        frame = pd.read_csv(data_dir / name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        anomaly = pd.to_numeric(frame["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        valid_rows = valid_time_mask_first_event(
            apply_timestamp_mode(timestamps, timestamp_mode).astype(float),
            anomaly,
        )
        arr = frame[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        arr = arr[valid_rows]
        mask = np.isfinite(arr)
        arr0 = np.where(mask, arr, 0.0)
        sums += arr0.sum(axis=0)
        sums_sq += (arr0 * arr0).sum(axis=0)
        counts += mask.sum(axis=0)
        rows_seen += int(len(arr))
    counts = np.maximum(counts, 1.0)
    mean = sums / counts
    var = np.maximum(sums_sq / counts - mean**2, 1e-6)
    return NormStats(
        mean=mean.astype(float).tolist(),
        std=np.sqrt(var).astype(float).tolist(),
        rows_seen=int(rows_seen),
        files_seen=int(len(train_files)),
    )


def compute_type_reference_all_train_negatives(
    data_dir: Path,
    train_files: list[str],
    z_threshold: float = 3.0,
    fallback_z_threshold: float = 2.0,
    timestamp_mode: str = "legacy_float32",
) -> WeakTypeReference:
    sums = np.zeros(len(SENSORS), dtype=np.float64)
    sums_sq = np.zeros(len(SENSORS), dtype=np.float64)
    counts = np.zeros(len(SENSORS), dtype=np.float64)
    tx_sum = tx_sq = tx_count = 0.0
    rx_sum = rx_sq = rx_count = 0.0

    for idx, name in enumerate(train_files, 1):
        df = pd.read_csv(data_dir / name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        timestamps = apply_timestamp_mode(timestamps.astype(np.int64), timestamp_mode).astype(float)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        labels, valid_mask = ahead_labels_and_valid_mask_multi_first_event(timestamps, anomaly, HORIZON_HOURS)
        neg_mask = (primary_labels(labels) == 0) & valid_mask
        arr = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        arr = arr[neg_mask]
        if len(arr):
            finite = np.isfinite(arr)
            arr0 = np.where(finite, arr, 0.0)
            sums += arr0.sum(axis=0)
            sums_sq += (arr0 * arr0).sum(axis=0)
            counts += finite.sum(axis=0)
            for prefix, acc in [("currentMultiTXPower", "tx"), ("currentMultiRXPower", "rx")]:
                lane = lane_dispersion_rows(arr, prefix)
                if len(lane):
                    if acc == "tx":
                        tx_sum += float(lane.sum())
                        tx_sq += float((lane * lane).sum())
                        tx_count += float(len(lane))
                    else:
                        rx_sum += float(lane.sum())
                        rx_sq += float((lane * lane).sum())
                        rx_count += float(len(lane))

    counts = np.maximum(counts, 1.0)
    mean = sums / counts
    var = np.maximum(sums_sq / counts - mean**2, 1e-6)

    def lane_mean_std(total: float, total_sq: float, count: float) -> tuple[float, float]:
        count = max(float(count), 1.0)
        center = total / count
        std = math.sqrt(max(total_sq / count - center * center, 1e-6))
        return float(center), float(std)

    tx_mean, tx_std = lane_mean_std(tx_sum, tx_sq, tx_count)
    rx_mean, rx_std = lane_mean_std(rx_sum, rx_sq, rx_count)
    return WeakTypeReference(
        sensor_mean=mean.astype(float).tolist(),
        sensor_std=np.sqrt(var).astype(float).tolist(),
        tx_lane_mean=tx_mean,
        tx_lane_std=tx_std,
        rx_lane_mean=rx_mean,
        rx_lane_std=rx_std,
        z_threshold=float(z_threshold),
        fallback_z_threshold=float(fallback_z_threshold),
    )


@torch.no_grad()
def score_one_module(
    model: torch.nn.Module,
    values: np.ndarray,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: str,
    use_amp: bool = False,
) -> np.ndarray:
    model.eval()
    valid = np.isfinite(values).astype(np.float32)
    raw = np.where(valid > 0, values, 0.0).astype(np.float32)
    x = ((raw - mean) / np.maximum(std, 1e-6)) * valid
    pad_x = np.zeros((seq_len - 1, values.shape[1]), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = np.concatenate([pad_x, x], axis=0)
    m_pad = np.concatenate([pad_m, valid], axis=0)
    x_windows = np.lib.stride_tricks.sliding_window_view(x_pad, window_shape=seq_len, axis=0).transpose(0, 2, 1)
    m_windows = np.lib.stride_tricks.sliding_window_view(m_pad, window_shape=seq_len, axis=0).transpose(0, 2, 1)

    scores: list[np.ndarray] = []
    for start in range(0, len(values), int(batch_size)):
        end = min(len(values), start + int(batch_size))
        xb = torch.from_numpy(np.array(x_windows[start:end], copy=True)).to(device)
        mb = torch.from_numpy(np.array(m_windows[start:end], copy=True)).to(device)
        amp_enabled = bool(use_amp) and str(device).startswith("cuda")
        with cuda_autocast(amp_enabled):
            out = model(xb, mb)
        logits = out[0] if isinstance(out, tuple) else out
        alarm_logits, _aux_logits = split_alarm_aux_logits(logits, aux_dim=len(HORIZON_HOURS))
        scores.append(torch.sigmoid(alarm_logits).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, dtype=np.float32)


def collect_module_scores(
    model: torch.nn.Module,
    cache: FormalModuleCache,
    file_names: list[str],
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: str,
    use_amp: bool = False,
    use_tqdm: bool = True,
    force_tqdm: bool = False,
    progress_desc: str = "official score",
) -> list[dict]:
    rows: list[dict] = []
    iterator = enumerate(file_names, 1)
    progress_bar = None
    tqdm_active = bool(use_tqdm) and _tqdm is not None and (bool(force_tqdm) or sys.stdout.isatty())
    if tqdm_active:
        progress_bar = _tqdm(
            iterator,
            total=len(file_names),
            desc=progress_desc,
            unit="module",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
            file=sys.stdout,
            ascii=True,
        )
        iterator = progress_bar
    for idx, name in iterator:
        timestamps, values, anomaly, _labels, valid_mask = cache.get(name)
        scores = score_one_module(model, values, seq_len, mean, std, batch_size, device, use_amp=use_amp)
        true_ts = first_anomaly_timestamp(timestamps.astype(float), anomaly)
        rows.append(
            {
                "file_name": name,
                "timestamps": timestamps,
                "scores": scores,
                "true_label": int(true_ts is not None),
                "true_ts": true_ts,
                "valid_mask": valid_mask,
            }
        )
        if bool(use_tqdm) and progress_bar is None and idx % 200 == 0:
            print(f"[official score] {idx}/{len(file_names)} modules")
    if progress_bar is not None:
        progress_bar.close()
    return rows


def smooth_scores_causal(
    scores: np.ndarray,
    method: str = "none",
    window: int = 1,
) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    method = str(method or "none").lower()
    window = int(max(1, window))
    if method in {"none", "point"} or window <= 1:
        return arr.astype(np.float64, copy=True)
    if method != "ema":
        raise ValueError(f"unknown alarm smoothing method: {method}")
    alpha = 2.0 / (float(window) + 1.0)
    out = np.empty_like(arr, dtype=np.float64)
    if len(arr) <= 0:
        return out
    prev = float(arr[0])
    out[0] = prev
    for idx in range(1, len(arr)):
        prev = alpha * float(arr[idx]) + (1.0 - alpha) * prev
        out[idx] = prev
    return out


def first_alarm_index(
    scores: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
    smoothing: str = "none",
    smooth_window: int = 1,
    consecutive_k: int = 1,
) -> int | None:
    work = smooth_scores_causal(scores, method=smoothing, window=smooth_window)
    active = work >= float(threshold)
    if valid_mask is not None:
        active = active & np.asarray(valid_mask, dtype=bool)
    need = max(1, int(consecutive_k))
    run = 0
    for idx, flag in enumerate(active):
        run = run + 1 if bool(flag) else 0
        if run >= need:
            return int(idx)
    return None


def alarm_prediction_mask(
    scores: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
    smoothing: str = "none",
    smooth_window: int = 1,
    consecutive_k: int = 1,
    first_alarm_only: bool = False,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    valid = np.ones(len(scores), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    pred = np.zeros(len(scores), dtype=np.int8)
    idx = first_alarm_index(
        scores,
        threshold,
        valid_mask=valid,
        smoothing=smoothing,
        smooth_window=smooth_window,
        consecutive_k=consecutive_k,
    )
    if idx is not None:
        if bool(first_alarm_only):
            pred[int(idx)] = 1
        else:
            pred[int(idx) :] = 1
    return (pred & valid.astype(np.int8)).astype(np.int8)


def metrics_from_module_scores_with_alarm(
    module_rows: list[dict],
    threshold: float,
    smoothing: str = "none",
    smooth_window: int = 1,
    consecutive_k: int = 1,
) -> dict[str, float]:
    decisions: list[dict[str, object]] = []
    for row in module_rows:
        scores = np.asarray(row["scores"], dtype=float)
        timestamps = np.asarray(row["timestamps"], dtype=float)
        valid_mask = np.asarray(row.get("valid_mask", np.ones(len(scores), dtype=bool)), dtype=bool)
        alarm_idx = first_alarm_index(
            scores,
            threshold,
            valid_mask=valid_mask,
            smoothing=smoothing,
            smooth_window=smooth_window,
            consecutive_k=consecutive_k,
        )
        predict_label = int(alarm_idx is not None)
        predict_ts = float(timestamps[int(alarm_idx)]) if alarm_idx is not None else None
        true_label = int(row.get("true_label", 0))
        true_ts = row.get("true_ts")
        true_ts = float(true_ts) if true_ts is not None and np.isfinite(float(true_ts)) else None

        valid_predict_positive = 0
        hit = 0
        lead_hour = None
        if predict_label > 0:
            if true_label == 0:
                valid_predict_positive = 1
            elif true_ts is not None and predict_ts is not None and true_ts > predict_ts:
                valid_predict_positive = 1
                hit = 1
                lead_hour = abs(float(true_ts) - float(predict_ts)) / 3600.0
        decisions.append(
            {
                "file_name": row["file_name"],
                "true_label": true_label,
                "valid_predict_positive": valid_predict_positive,
                "hit": hit,
                "lead_hour": lead_hour,
            }
        )

    detail_df = pd.DataFrame(decisions)
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
        "final_score": float(final_score),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "avg_lead_hour": float(avg_lead_hour),
        "min_lead_hour": float(min_lead_hour),
        "all_hit_cnt": int(tp),
        "all_predict_pos_cnt": int(len(pred_pos)),
        "all_true_pos_cnt": int(len(true_pos)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "evaluated_module_cnt": int(len(detail_df)),
    }


def choose_alarm_strategy_by_final_score(
    module_rows: list[dict],
    grid_size: int = 99,
    smoothing: str = "ema",
    smooth_windows: tuple[int, ...] | list[int] = (1, 3, 6, 12),
    consecutive_ks: tuple[int, ...] | list[int] = (1, 2, 3),
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    selection_metric: str = "final_score",
) -> tuple[float, dict[str, object], dict[str, float]]:
    smoothing = str(smoothing or "none").lower()
    selection_metric = str(selection_metric or "final_score")
    windows = [1] if smoothing in {"none", "point"} else sorted({max(1, int(w)) for w in smooth_windows})
    ks = sorted({max(1, int(k)) for k in consecutive_ks})
    if not windows:
        windows = [1]
    if not ks:
        ks = [1]

    best_threshold = 0.5
    best_strategy: dict[str, object] = {
        "alarm_smoothing": smoothing,
        "alarm_smooth_window": 1,
        "alarm_consecutive_k": 1,
    }
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float, float, int] | None = None
    thresholds = np.linspace(float(min_threshold), float(max_threshold), int(grid_size))
    for window in windows:
        for k in ks:
            for threshold in thresholds:
                metrics = metrics_from_module_scores_with_alarm(
                    module_rows,
                    float(threshold),
                    smoothing=smoothing,
                    smooth_window=int(window),
                    consecutive_k=int(k),
                )
                if selection_metric not in metrics:
                    raise ValueError(
                        f"unknown selection_metric={selection_metric!r}; "
                        f"available={sorted(metrics)}"
                    )
                key = (
                    float(metrics[selection_metric]),
                    float(metrics["f1_score"]),
                    float(metrics["recall"]),
                    float(metrics["precision"]),
                    float(metrics["final_score"]),
                    -int(metrics["all_predict_pos_cnt"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_threshold = float(threshold)
                    best_strategy = {
                        "alarm_smoothing": smoothing,
                        "alarm_smooth_window": int(window),
                        "alarm_consecutive_k": int(k),
                    }
                    best_metrics = dict(metrics)
    assert best_metrics is not None
    best_metrics["threshold"] = float(best_threshold)
    best_metrics["selection_metric"] = selection_metric
    best_metrics.update(best_strategy)
    return float(best_threshold), best_strategy, best_metrics


def write_predictions(
    module_rows: list[dict],
    out_dir: Path,
    threshold: float,
    alarm_smoothing: str = "none",
    alarm_smooth_window: int = 1,
    alarm_consecutive_k: int = 1,
    first_alarm_only: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in module_rows:
        scores = np.asarray(row["scores"], dtype=float)
        valid_mask = np.asarray(row.get("valid_mask", np.ones_like(scores, dtype=bool)), dtype=bool)
        processed_scores = smooth_scores_causal(scores, method=alarm_smoothing, window=alarm_smooth_window)
        pred = alarm_prediction_mask(
            scores,
            threshold,
            valid_mask=valid_mask,
            smoothing=alarm_smoothing,
            smooth_window=alarm_smooth_window,
            consecutive_k=alarm_consecutive_k,
            first_alarm_only=first_alarm_only,
        )
        pd.DataFrame(
            {
                "timestamp": np.asarray(row["timestamps"], dtype=np.int64),
                "predict": pred.astype(int),
                "score": processed_scores.astype(np.float32),
                "raw_score": scores.astype(np.float32),
                "valid_for_eval": valid_mask.astype(int),
            }
        ).to_csv(out_dir / row["file_name"], index=False)


def assert_prediction_coverage(pred_dir: Path, expected_files: list[str]) -> None:
    expected = set(expected_files)
    actual = {path.name for path in Path(pred_dir).glob("*.csv")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise AssertionError(f"prediction coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
