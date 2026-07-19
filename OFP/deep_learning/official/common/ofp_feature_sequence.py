from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from OFP.deep_learning.official.ofp_formal_protocol.protocol import (
    DEFAULT_HORIZON_HOURS,
    ahead_labels_and_valid_mask_multi_first_event,
    ahead_labels_multi_first_event,
    valid_time_mask_first_event,
)
from OFP.deep_learning.official.ofp_protocol.run_ofp_baselines import MODEL2_FEATURES, make_model2_features

from OFP.deep_learning.official.common.formal_data import apply_timestamp_mode


TASK_HORIZON_TAG = "h" + "_".join(f"{float(h):g}" for h in DEFAULT_HORIZON_HOURS)


@dataclass
class FeatureNormStats:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    rows_seen: int
    files_seen: int
    timestamp_mode: str

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.mean, dtype=np.float32),
            np.asarray(self.std, dtype=np.float32),
        )


class OFPFeatureModuleCache:
    """LRU cache for OFP model2 feature sequences.

    The returned sequence has one row per original timestamp. Normal and extra
    OFP rows are both placed back onto the original module index; rows that the
    OFP feature extractor drops keep NaN features and are masked by the dataset.
    """

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
        self.module_cache_dir = (
            Path(module_cache_dir) / "ofp_model2_features" / self.timestamp_mode / TASK_HORIZON_TAG
            if module_cache_dir
            else None
        )
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
                if labels.ndim == 1 or labels.shape[-1] != len(DEFAULT_HORIZON_HOURS):
                    labels = ahead_labels_multi_first_event(
                        timestamps.astype(float),
                        anomaly,
                        DEFAULT_HORIZON_HOURS,
                    ).astype(np.int8, copy=False)
                if "valid_mask" in payload.files:
                    valid_mask = payload["valid_mask"].astype(bool, copy=False)
                else:
                    valid_mask = valid_time_mask_first_event(timestamps.astype(float), anomaly)
        else:
            timestamps, values, anomaly, labels, valid_mask = build_ofp_feature_arrays(
                self.data_dir / file_name,
                timestamp_mode=self.timestamp_mode,
            )
            if cache_path is not None:
                np.savez(
                    cache_path,
                    timestamps=timestamps,
                    values=values,
                    anomaly=anomaly,
                    labels=labels,
                    valid_mask=valid_mask.astype(np.int8),
                )

        item = (timestamps, values, anomaly, labels, valid_mask.astype(bool))
        self.cache[file_name] = item
        if len(self.cache) > self.max_cached_files:
            self.cache.popitem(last=False)
        return item


def build_ofp_feature_arrays(
    csv_path: Path,
    timestamp_mode: str = "legacy_float32",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(csv_path)
    timestamps_raw = pd.to_numeric(raw["timestamp"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    timestamps = apply_timestamp_mode(timestamps_raw, timestamp_mode).astype(np.int64)
    raw = raw.copy()
    raw["timestamp"] = timestamps

    anomaly = pd.to_numeric(raw["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    labels, valid_mask = ahead_labels_and_valid_mask_multi_first_event(
        timestamps.astype(float),
        anomaly,
        DEFAULT_HORIZON_HOURS,
    )
    values = np.full((len(raw), len(MODEL2_FEATURES)), np.nan, dtype=np.float32)

    normal, extra = make_model2_features(raw, with_label="anomaly" in raw.columns)
    for frame in (normal, extra):
        if len(frame) <= 0:
            continue
        index = frame.index.to_numpy(dtype=np.int64)
        valid = (index >= 0) & (index < len(raw))
        if not valid.any():
            continue
        arr = frame.loc[frame.index[valid], MODEL2_FEATURES].to_numpy(dtype=np.float32)
        values[index[valid]] = arr

    return timestamps, values, anomaly, labels.astype(np.int8), valid_mask.astype(bool)


def compute_ofp_feature_norm_stats(
    cache: OFPFeatureModuleCache,
    file_names: list[str],
) -> FeatureNormStats:
    sums = np.zeros(len(MODEL2_FEATURES), dtype=np.float64)
    sums_sq = np.zeros(len(MODEL2_FEATURES), dtype=np.float64)
    counts = np.zeros(len(MODEL2_FEATURES), dtype=np.float64)
    rows_seen = 0

    for idx, name in enumerate(file_names, 1):
        _timestamps, values, _anomaly, _labels, valid_mask = cache.get(name)
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.asarray(valid_mask, dtype=bool)]
        mask = np.isfinite(arr)
        arr0 = np.where(mask, arr, 0.0)
        sums += arr0.sum(axis=0)
        sums_sq += (arr0 * arr0).sum(axis=0)
        counts += mask.sum(axis=0)
        rows_seen += int(len(arr))
    counts = np.maximum(counts, 1.0)
    mean = sums / counts
    var = np.maximum(sums_sq / counts - mean**2, 1e-6)
    return FeatureNormStats(
        feature_names=list(MODEL2_FEATURES),
        mean=mean.astype(float).tolist(),
        std=np.sqrt(var).astype(float).tolist(),
        rows_seen=int(rows_seen),
        files_seen=int(len(file_names)),
        timestamp_mode=cache.timestamp_mode,
    )


def save_feature_norm_stats(norm: FeatureNormStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(norm), indent=2), encoding="utf-8")
