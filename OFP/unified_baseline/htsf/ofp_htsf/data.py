"""Causal feature views, window construction, train-only normalization and streams."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

import ofp_unified.features as unified_features
from ofp_unified.features import (
    EXPERT_FEATURES,
    MISSING_SENTINEL,
    RAW_FEATURES,
    STATISTICAL_FEATURES,
    TEMPERATURE_OUTLIER_SENTINEL,
    build_feature_frame,
    feature_names,
    groups_for_features,
)
from ofp_unified.labels import validate_time_order

from .config import HTSFConfig, fingerprint
from .endpoints import endpoint_fingerprint


ENGINEERED_FEATURES = STATISTICAL_FEATURES + EXPERT_FEATURES
B2_FEATURES = feature_names("b2")
TRAINING_TENSOR_CACHE_VERSION = "htsf-training-tensors-v1"


def preprocessing_implementation_fingerprint() -> str:
    paths = [Path(__file__).resolve(), Path(unified_features.__file__).resolve()]
    return fingerprint(
        [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ]
    )


@dataclass
class ModuleViews:
    timestamps: np.ndarray
    raw: np.ndarray
    raw_valid: np.ndarray
    engineered: np.ndarray
    engineered_valid: np.ndarray
    b2: np.ndarray


@dataclass
class Batch:
    raw: np.ndarray
    raw_mask: np.ndarray
    engineered: np.ndarray
    engineered_mask: np.ndarray
    b2: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    file_names: list[str]
    row_indices: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray

    def transform(
        self,
        values: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=np.float32)
        mask = np.isfinite(array) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(array))
        normalized = (array - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        normalized = np.where(mask, normalized, 0.0).astype(np.float32)
        return normalized, mask.astype(np.float32)

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "count": self.count.astype(int).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Standardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            count=np.asarray(payload["count"], dtype=np.int64),
        )


class _Moments:
    def __init__(self, dimensions: int) -> None:
        self.count = np.zeros(dimensions, dtype=np.int64)
        self.total = np.zeros(dimensions, dtype=np.float64)
        self.square_total = np.zeros(dimensions, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        flat_values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        flat_valid = np.asarray(valid, dtype=bool).reshape(-1, valid.shape[-1])
        safe = np.where(flat_valid, flat_values, 0.0)
        self.count += flat_valid.sum(axis=0, dtype=np.int64)
        self.total += safe.sum(axis=0, dtype=np.float64)
        self.square_total += np.square(safe, dtype=np.float64).sum(axis=0, dtype=np.float64)

    def finish(self) -> Standardizer:
        divisor = np.maximum(self.count, 1)
        mean = self.total / divisor
        variance = self.square_total / divisor - np.square(mean)
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        std[(self.count < 2) | (~np.isfinite(std)) | (std < 1e-8)] = 1.0
        mean[(self.count == 0) | (~np.isfinite(mean))] = 0.0
        return Standardizer(mean=mean, std=std, count=self.count.copy())


def read_module_views(path: Path) -> ModuleViews:
    needed = {"timestamp", *RAW_FEATURES}
    frame = pd.read_csv(path, usecols=lambda name: name in needed)
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    timestamps = validate_time_order(frame).astype(np.int64)
    b2_frame = build_feature_frame(frame, "b2")
    raw = b2_frame.loc[:, RAW_FEATURES].to_numpy(dtype=np.float32)
    raw_valid = np.isfinite(raw) & (raw != float(MISSING_SENTINEL))
    raw_valid[:, 0] &= raw[:, 0] != float(TEMPERATURE_OUTLIER_SENTINEL)
    engineered = b2_frame.loc[:, ENGINEERED_FEATURES].to_numpy(dtype=np.float32)
    engineered_valid = np.isfinite(engineered)
    return ModuleViews(
        timestamps=timestamps,
        raw=raw,
        raw_valid=raw_valid,
        engineered=engineered,
        engineered_valid=engineered_valid,
        b2=b2_frame.loc[:, B2_FEATURES].to_numpy(dtype=np.float32),
    )


def causal_windows(
    raw: np.ndarray,
    raw_valid: np.ndarray,
    endpoints: Sequence[int],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left-padded windows ending at each endpoint, never after it."""

    raw = np.asarray(raw, dtype=np.float32)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if raw.ndim != 2 or raw_valid.shape != raw.shape:
        raise ValueError("raw and raw_valid must be aligned [time, channel] arrays")
    if np.any(endpoints < 0) or np.any(endpoints >= len(raw)):
        raise ValueError("window endpoint is outside the module")
    length = int(sequence_length)
    windows = np.zeros((len(endpoints), length, raw.shape[1]), dtype=np.float32)
    masks = np.zeros_like(windows, dtype=bool)
    for output_index, endpoint in enumerate(endpoints):
        start = max(0, int(endpoint) - length + 1)
        source = raw[start : int(endpoint) + 1]
        source_mask = raw_valid[start : int(endpoint) + 1]
        windows[output_index, -len(source) :] = source
        masks[output_index, -len(source) :] = source_mask
    return windows, masks


def fit_standardizers(
    data_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
) -> tuple[Standardizer, Standardizer]:
    raw_moments = _Moments(len(RAW_FEATURES))
    engineered_moments = _Moments(len(ENGINEERED_FEATURES))
    for file_name, group in manifest.groupby("file_name", sort=False):
        views = read_module_views(Path(data_dir) / str(file_name))
        endpoints = group["row_index"].to_numpy(dtype=np.int64)
        windows, window_valid = causal_windows(
            views.raw,
            views.raw_valid,
            endpoints,
            config.window.sequence_length,
        )
        raw_moments.update(windows, window_valid)
        engineered_moments.update(
            views.engineered[endpoints],
            views.engineered_valid[endpoints],
        )
    return raw_moments.finish(), engineered_moments.finish()


def load_or_fit_standardizers(
    output_path: Path,
    data_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
    overwrite: bool = False,
) -> tuple[Standardizer, Standardizer]:
    output_path = Path(output_path)
    expected_endpoint = endpoint_fingerprint(manifest)
    source_fingerprint = fingerprint(
        {
            "endpoint_fingerprint": expected_endpoint,
            "window": config.to_dict()["window"],
            "feature_schema_version": config.feature_schema_version,
            "fit_scope": "sampled causal raw windows and engineered endpoints",
            "preprocessing_implementation_fingerprint": preprocessing_implementation_fingerprint(),
        }
    )
    if output_path.exists() and not overwrite:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("source_fingerprint") != source_fingerprint:
            raise ValueError(
                "normalizer was fit with a different endpoint/window/feature schema; "
                "use --overwrite or another artifacts directory"
            )
        return (
            Standardizer.from_dict(payload["raw"]),
            Standardizer.from_dict(payload["engineered"]),
        )
    raw, engineered = fit_standardizers(data_dir, manifest, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "endpoint_fingerprint": expected_endpoint,
                "source_fingerprint": source_fingerprint,
                "fit_scope": "training fold endpoints and their causal raw windows only",
                "raw_feature_names": list(RAW_FEATURES),
                "engineered_feature_names": list(ENGINEERED_FEATURES),
                "raw": raw.to_dict(),
                "engineered": engineered.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw, engineered


def _batch_from_views(
    file_name: str,
    views: ModuleViews,
    rows: pd.DataFrame,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
) -> Batch:
    endpoints = rows["row_index"].to_numpy(dtype=np.int64)
    windows, window_valid = causal_windows(
        views.raw,
        views.raw_valid,
        endpoints,
        config.window.sequence_length,
    )
    raw, raw_mask = raw_standardizer.transform(windows, window_valid)
    engineered, engineered_mask = engineered_standardizer.transform(
        views.engineered[endpoints],
        views.engineered_valid[endpoints],
    )
    target = (
        rows["target"].to_numpy(dtype=np.float32)
        if "target" in rows
        else np.zeros(len(rows), dtype=np.float32)
    )
    weight = (
        rows["sample_weight"].to_numpy(dtype=np.float32)
        if "sample_weight" in rows
        else np.ones(len(rows), dtype=np.float32)
    )
    return Batch(
        raw=raw,
        raw_mask=raw_mask,
        engineered=engineered,
        engineered_mask=engineered_mask,
        b2=views.b2[endpoints],
        target=target,
        weight=weight,
        file_names=[str(file_name)] * len(rows),
        row_indices=endpoints,
        timestamps=views.timestamps[endpoints],
    )


def _concatenate_batches(batches: Sequence[Batch]) -> Batch:
    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    return Batch(
        raw=np.concatenate([batch.raw for batch in batches], axis=0),
        raw_mask=np.concatenate([batch.raw_mask for batch in batches], axis=0),
        engineered=np.concatenate([batch.engineered for batch in batches], axis=0),
        engineered_mask=np.concatenate([batch.engineered_mask for batch in batches], axis=0),
        b2=np.concatenate([batch.b2 for batch in batches], axis=0),
        target=np.concatenate([batch.target for batch in batches], axis=0),
        weight=np.concatenate([batch.weight for batch in batches], axis=0),
        file_names=[name for batch in batches for name in batch.file_names],
        row_indices=np.concatenate([batch.row_indices for batch in batches], axis=0),
        timestamps=np.concatenate([batch.timestamps for batch in batches], axis=0),
    )


def _take_batch(batch: Batch, positions: np.ndarray) -> Batch:
    positions = np.asarray(positions, dtype=np.int64)
    return Batch(
        raw=batch.raw[positions],
        raw_mask=batch.raw_mask[positions],
        engineered=batch.engineered[positions],
        engineered_mask=batch.engineered_mask[positions],
        b2=batch.b2[positions],
        target=batch.target[positions],
        weight=batch.weight[positions],
        file_names=[batch.file_names[int(index)] for index in positions],
        row_indices=batch.row_indices[positions],
        timestamps=batch.timestamps[positions],
    )


def training_tensor_cache_fingerprint(
    manifest: pd.DataFrame,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
) -> str:
    return fingerprint(
        {
            "cache_version": TRAINING_TENSOR_CACHE_VERSION,
            "endpoint_fingerprint": endpoint_fingerprint(manifest),
            "window": config.to_dict()["window"],
            "feature_schema_version": config.feature_schema_version,
            "raw_standardizer": raw_standardizer.to_dict(),
            "engineered_standardizer": engineered_standardizer.to_dict(),
            "preprocessing_implementation_fingerprint": preprocessing_implementation_fingerprint(),
        }
    )


def materialize_training_tensor_cache(
    cache_dir: Path,
    data_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    *,
    overwrite: bool = False,
    shard_rows: int = 4096,
) -> dict[str, object]:
    """Materialize selected windows once; all variants/epochs reuse the shards."""

    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "cache_manifest.json"
    source = training_tensor_cache_fingerprint(
        manifest,
        config,
        raw_standardizer,
        engineered_standardizer,
    )
    if meta_path.exists() and not overwrite:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("source_fingerprint") != source:
            raise ValueError("training tensor cache provenance mismatch")
        missing = [name for name in metadata.get("shards", []) if not (cache_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"training tensor cache is incomplete; first={missing[0]}")
        return metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    pending: list[Batch] = []
    pending_rows = 0
    shard_names: list[str] = []
    feature_count = len(B2_FEATURES)
    finite_count = np.zeros(feature_count, dtype=np.int64)
    nonzero_count = np.zeros(feature_count, dtype=np.int64)
    unique_values: list[set[float]] = [set() for _ in range(feature_count)]
    unique_cap = 1001

    def flush() -> None:
        nonlocal pending, pending_rows
        if not pending:
            return
        batch = _concatenate_batches(pending)
        shard_name = f"shard_{len(shard_names):05d}.npz"
        np.savez(
            cache_dir / shard_name,
            raw=batch.raw.astype(np.float32),
            raw_mask=batch.raw_mask.astype(np.uint8),
            engineered=batch.engineered.astype(np.float32),
            engineered_mask=batch.engineered_mask.astype(np.uint8),
            b2=batch.b2.astype(np.float32),
            target=batch.target.astype(np.float32),
            file_names=np.asarray(batch.file_names, dtype=str),
            row_indices=batch.row_indices.astype(np.int64),
            timestamps=batch.timestamps.astype(np.int64),
        )
        shard_names.append(shard_name)
        pending = []
        pending_rows = 0

    for file_name, group in manifest.groupby("file_name", sort=False):
        views = read_module_views(Path(data_dir) / str(file_name))
        batch = _batch_from_views(
            str(file_name),
            views,
            group,
            config,
            raw_standardizer,
            engineered_standardizer,
        )
        pending.append(batch)
        pending_rows += len(batch.target)
        finite = np.isfinite(batch.b2)
        finite_count += finite.sum(axis=0, dtype=np.int64)
        nonzero_count += (finite & (batch.b2 != 0.0)).sum(axis=0, dtype=np.int64)
        for index in range(feature_count):
            if len(unique_values[index]) >= unique_cap:
                continue
            values = np.unique(batch.b2[finite[:, index], index])
            room = unique_cap - len(unique_values[index])
            unique_values[index].update(float(value) for value in values[:room])
        if pending_rows >= int(shard_rows):
            flush()
    flush()

    rows = int(len(manifest))
    groups = groups_for_features(B2_FEATURES)
    activity = pd.DataFrame(
        {
            "feature": list(B2_FEATURES),
            "group": groups,
            "rows": rows,
            "non_missing_count": finite_count,
            "non_missing_rate": finite_count / max(rows, 1),
            "nonzero_count": nonzero_count,
            "nonzero_rate": nonzero_count / max(rows, 1),
            "unique_count_capped": [min(len(values), unique_cap) for values in unique_values],
            "unique_count_is_capped": [len(values) >= unique_cap for values in unique_values],
        }
    )
    activity.to_csv(cache_dir / "sampled_feature_activity.csv", index=False)
    metadata: dict[str, object] = {
        "cache_version": TRAINING_TENSOR_CACHE_VERSION,
        "source_fingerprint": source,
        "endpoint_fingerprint": endpoint_fingerprint(manifest),
        "rows": rows,
        "modules": int(manifest["file_name"].nunique()),
        "shard_rows_target": int(shard_rows),
        "shards": shard_names,
        "feature_activity": "sampled_feature_activity.csv",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _iter_cached_batches(
    cache_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[Batch]:
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    expected = training_tensor_cache_fingerprint(
        manifest,
        config,
        raw_standardizer,
        engineered_standardizer,
    )
    if metadata.get("source_fingerprint") != expected:
        raise ValueError("training tensor cache does not match this manifest/config")
    lookup = manifest.set_index(["file_name", "row_index"], verify_integrity=True)
    rng = np.random.default_rng(int(seed))
    shard_names = list(metadata["shards"])
    if shuffle:
        rng.shuffle(shard_names)
    batch_size = int(config.training.batch_size)
    for shard_name in shard_names:
        with np.load(cache_dir / str(shard_name), allow_pickle=False) as shard:
            file_names = shard["file_names"].astype(str).tolist()
            row_indices = shard["row_indices"].astype(np.int64)
            keys = pd.MultiIndex.from_arrays([file_names, row_indices])
            lookup_positions = lookup.index.get_indexer(keys)
            if np.any(lookup_positions < 0):
                raise ValueError("cached shard contains an endpoint absent from the manifest")
            selected = lookup.iloc[lookup_positions]
            cached_target = shard["target"].astype(np.float32)
            target = selected["target"].to_numpy(dtype=np.float32)
            if not np.array_equal(cached_target, target):
                raise ValueError("cached target differs from the endpoint manifest")
            weight = (
                selected["sample_weight"].to_numpy(dtype=np.float32)
                if "sample_weight" in selected
                else np.ones(len(selected), dtype=np.float32)
            )
            full = Batch(
                raw=shard["raw"].astype(np.float32),
                raw_mask=shard["raw_mask"].astype(np.float32),
                engineered=shard["engineered"].astype(np.float32),
                engineered_mask=shard["engineered_mask"].astype(np.float32),
                b2=shard["b2"].astype(np.float32),
                target=target,
                weight=weight,
                file_names=file_names,
                row_indices=row_indices,
                timestamps=shard["timestamps"].astype(np.int64),
            )
        order = rng.permutation(len(full.target)) if shuffle else np.arange(len(full.target))
        for start in range(0, len(order), batch_size):
            yield _take_batch(full, order[start : start + batch_size])


def iter_manifest_batches(
    data_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    *,
    shuffle: bool,
    seed: int,
    tensor_cache_dir: Path | None = None,
) -> Iterator[Batch]:
    if tensor_cache_dir is not None:
        yield from _iter_cached_batches(
            tensor_cache_dir,
            manifest,
            config,
            raw_standardizer,
            engineered_standardizer,
            shuffle=shuffle,
            seed=seed,
        )
        return
    groups = [(str(name), group.copy()) for name, group in manifest.groupby("file_name", sort=False)]
    rng = np.random.default_rng(int(seed))
    if shuffle:
        rng.shuffle(groups)
    batch_size = int(config.training.batch_size)
    pending: list[Batch] = []
    pending_rows = 0
    for file_name, group in groups:
        if shuffle and len(group) > 1:
            group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        views = read_module_views(Path(data_dir) / file_name)
        start = 0
        while start < len(group):
            take = min(batch_size - pending_rows, len(group) - start)
            pending.append(_batch_from_views(
                file_name,
                views,
                group.iloc[start : start + take],
                config,
                raw_standardizer,
                engineered_standardizer,
            ))
            pending_rows += take
            start += take
            if pending_rows == batch_size:
                yield _concatenate_batches(pending)
                pending = []
                pending_rows = 0
    if pending:
        yield _concatenate_batches(pending)


def iter_full_module_batches(
    path: Path,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
) -> Iterator[Batch]:
    views = read_module_views(path)
    batch_size = int(config.training.batch_size)
    all_rows = pd.DataFrame(
        {
            "row_index": np.arange(len(views.timestamps), dtype=np.int64),
            "timestamp": views.timestamps,
        }
    )
    for start in range(0, len(all_rows), batch_size):
        yield _batch_from_views(
            path.name,
            views,
            all_rows.iloc[start : start + batch_size],
            config,
            raw_standardizer,
            engineered_standardizer,
        )
