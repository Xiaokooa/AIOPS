"""Dataset protocol for Deep-Rule optical-module failure prediction.

This module fixes the comparable single split, creates cumulative
multi-horizon failure labels, and exposes causal endpoint samples.  All
normalization is train-only and mask-aware; all post-failure endpoints are
marked invalid for training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from ofp_unified.features import (
    MISSING_SENTINEL,
    RAW_FEATURES,
    STATISTICAL_FEATURES,
    TEMPERATURE_OUTLIER_SENTINEL,
)
from ofp_unified.labels import first_anomaly_timestamp, validate_time_order

from .features import (
    RULE_SIGNAL_DIM,
    build_causal_statistics,
    build_rule_margins,
    causal_hourly_window,
    endpoint_rule_signals,
)


HORIZONS_HOURS = (16.0, 24.0, 72.0, 120.0)
FIXED_TEST_FOLD = 3


@dataclass
class ModuleRecord:
    file_name: str
    timestamps: np.ndarray
    raw: np.ndarray
    raw_valid: np.ndarray
    statistics: np.ndarray
    stat_valid: np.ndarray
    anomaly: np.ndarray
    first_fault_ts: float | None
    rule_margin: np.ndarray
    rule_valid: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.timestamps)
        if np.asarray(self.timestamps).shape != (rows,):
            raise ValueError("timestamps must be one-dimensional")
        if np.asarray(self.raw).shape != (rows, len(RAW_FEATURES)):
            raise ValueError("raw has an unexpected shape")
        if np.asarray(self.raw_valid).shape != np.asarray(self.raw).shape:
            raise ValueError("raw_valid must align with raw")
        if np.asarray(self.statistics).shape != (rows, len(STATISTICAL_FEATURES)):
            raise ValueError("statistics has an unexpected shape")
        if np.asarray(self.stat_valid).shape != np.asarray(self.statistics).shape:
            raise ValueError("stat_valid must align with statistics")
        if np.asarray(self.anomaly).shape != (rows,):
            raise ValueError("anomaly must align with timestamps")
        if np.asarray(self.rule_margin).shape[0] != rows:
            raise ValueError("rule_margin must align with timestamps")
        if np.asarray(self.rule_valid).shape != np.asarray(self.rule_margin).shape:
            raise ValueError("rule_valid must align with rule_margin")
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if not np.isfinite(timestamps).all() or (
            rows > 1 and np.any(np.diff(timestamps) < 0)
        ):
            raise ValueError("timestamps must be finite and non-decreasing")


def read_module_record(path: Path) -> ModuleRecord:
    """Read one module and build only causal raw/statistical/rule histories."""

    path = Path(path)
    needed = {"timestamp", "anomaly", *RAW_FEATURES}
    frame = pd.read_csv(path, usecols=lambda name: name in needed)
    required = {"timestamp", *RAW_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "anomaly" not in frame:
        frame["anomaly"] = 0.0
    timestamps = validate_time_order(frame).astype(np.int64)
    anomaly = (
        pd.to_numeric(frame["anomaly"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    numeric_raw = frame.loc[:, RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    raw = numeric_raw.to_numpy(dtype=np.float32)
    raw_valid = np.isfinite(raw) & (raw != float(MISSING_SENTINEL))
    raw_valid[:, 0] &= raw[:, 0] != float(TEMPERATURE_OUTLIER_SENTINEL)
    statistics, stat_valid = build_causal_statistics(
        numeric_raw.to_numpy(dtype=np.float64), raw_valid
    )
    rule_margin, rule_valid = build_rule_margins(statistics, stat_valid)
    fault_ts = first_anomaly_timestamp(frame)
    return ModuleRecord(
        file_name=path.name,
        timestamps=timestamps,
        raw=raw,
        raw_valid=raw_valid,
        statistics=statistics,
        stat_valid=stat_valid,
        anomaly=anomaly,
        first_fault_ts=fault_ts,
        rule_margin=rule_margin,
        rule_valid=rule_valid,
    )


def load_module_records(
    data_dir: Path,
    file_names: Sequence[str],
    progress: bool = False,
) -> list[ModuleRecord]:
    """Load an ordered module list; ordering follows ``file_names`` exactly."""

    data_dir = Path(data_dir)
    names = [str(name) for name in file_names]
    records: list[ModuleRecord] = []
    for index, name in enumerate(names, 1):
        records.append(read_module_record(data_dir / name))
        if progress and (index == len(names) or index % 100 == 0):
            print(f"  [data] loaded {index}/{len(names)} modules")
    return records


def multi_horizon_targets(
    timestamps: np.ndarray,
    first_fault_ts: float | None,
    horizons_hours: Sequence[float] = HORIZONS_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative failure labels and the valid pre-failure mask.

    A label at horizon ``H`` is one exactly when the first failure occurs in
    ``(t, t+H]``.  Endpoints at or after the first failure are invalid.  All
    endpoints of a healthy module are valid negatives, matching the original
    OFP label policy.
    """

    times = np.asarray(timestamps, dtype=np.float64)
    horizons = np.asarray(tuple(float(value) for value in horizons_hours), dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("timestamps must be a finite one-dimensional array")
    if len(times) > 1 and np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be non-decreasing")
    if horizons.ndim != 1 or not len(horizons) or np.any(horizons <= 0):
        raise ValueError("horizons_hours must contain positive values")
    if np.any(np.diff(horizons) <= 0):
        raise ValueError("horizons_hours must be strictly increasing")
    targets = np.zeros((len(times), len(horizons)), dtype=np.float32)
    valid = np.ones(len(times), dtype=bool)
    if first_fault_ts is None:
        return targets, valid
    fault_ts = float(first_fault_ts)
    valid = times < fault_ts
    remaining_hours = (fault_ts - times) / 3600.0
    targets = (
        valid[:, None]
        & (remaining_hours[:, None] > 0.0)
        & (remaining_hours[:, None] <= horizons[None, :])
    ).astype(np.float32)
    return targets, valid


@dataclass(frozen=True)
class SplitManifest:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def all_file_names(self) -> tuple[str, ...]:
        return tuple(
            pd.concat((self.train, self.validation, self.test), ignore_index=True)[
                "file_name"
            ].astype(str)
        )


def _validate_index(index_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"file_name", "folder_index", "Label"}
    missing = required - set(index_frame.columns)
    if missing:
        raise ValueError(f"index file missing columns: {sorted(missing)}")
    frame = index_frame.copy()
    frame["file_name"] = frame["file_name"].astype(str)
    frame["folder_index"] = pd.to_numeric(
        frame["folder_index"], errors="raise"
    ).astype(int)
    frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int)
    if frame["file_name"].duplicated().any():
        raise ValueError("index contains duplicate file names")
    observed_folds = set(frame["folder_index"].unique())
    if observed_folds != {1, 2, 3}:
        raise ValueError(f"fixed protocol requires folds 1, 2, 3; got {sorted(observed_folds)}")
    return frame


def _stratified_validation_indices(
    development: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    binary_label = development["Label"].to_numpy(dtype=int) > 0
    groups = [np.flatnonzero(binary_label == value) for value in (False, True)]
    if any(len(group) < 2 for group in groups):
        raise ValueError("both label strata need at least two development modules")
    target_total = int(math.ceil(len(development) * float(validation_fraction)))
    ideals = np.asarray([len(group) * float(validation_fraction) for group in groups])
    allocation = np.floor(ideals).astype(int)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, np.asarray([len(group) - 1 for group in groups]))
    while int(allocation.sum()) < target_total:
        candidates = [
            index
            for index, group in enumerate(groups)
            if allocation[index] < len(group) - 1
        ]
        if not candidates:
            break
        chosen = max(candidates, key=lambda index: (ideals[index] - allocation[index], -index))
        allocation[chosen] += 1
    while int(allocation.sum()) > target_total:
        candidates = [index for index in range(len(groups)) if allocation[index] > 1]
        if not candidates:
            break
        chosen = min(candidates, key=lambda index: (ideals[index] - allocation[index], index))
        allocation[chosen] -= 1
    if int(allocation.sum()) != target_total:
        raise RuntimeError("unable to allocate the requested stratified validation size")

    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for positions, count in zip(groups, allocation):
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        selected.append(shuffled[: int(count)])
    return np.sort(np.concatenate(selected)).astype(np.int64)


def build_fixed_split(
    index: Path | pd.DataFrame,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    test_fold: int = FIXED_TEST_FOLD,
) -> SplitManifest:
    """Build folds 1+2 train/validation and reserve all of fold 3 for test."""

    if int(test_fold) != FIXED_TEST_FOLD:
        raise ValueError("the comparable DRFP protocol fixes test_fold=3")
    source = pd.read_csv(index) if isinstance(index, (str, Path)) else index
    frame = _validate_index(source)
    development = frame.loc[frame["folder_index"].isin((1, 2))].copy()
    test = frame.loc[frame["folder_index"] == FIXED_TEST_FOLD].copy()
    validation_positions = _stratified_validation_indices(
        development, validation_fraction, seed
    )
    validation_mask = np.zeros(len(development), dtype=bool)
    validation_mask[validation_positions] = True
    train = development.iloc[np.flatnonzero(~validation_mask)].copy()
    validation = development.iloc[validation_positions].copy()
    result = SplitManifest(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
    names = [set(part["file_name"].astype(str)) for part in (result.train, result.validation, result.test)]
    if names[0] & names[1] or names[0] & names[2] or names[1] & names[2]:
        raise RuntimeError("module leakage across fixed split")
    return result


@dataclass(frozen=True)
class EndpointRef:
    file_name: str
    module_index: int
    endpoint_index: int


def _record_sequence(
    records: Sequence[ModuleRecord] | Mapping[str, ModuleRecord],
) -> list[ModuleRecord]:
    if isinstance(records, Mapping):
        ordered = list(records.values())
        for key, record in zip(records.keys(), ordered):
            if str(key) != record.file_name:
                raise ValueError("record mapping keys must equal ModuleRecord.file_name")
        return ordered
    return list(records)


def _evenly_spaced(positions: np.ndarray, count: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if count <= 0 or not len(positions):
        return np.empty(0, dtype=np.int64)
    offsets = np.rint(np.linspace(0, len(positions) - 1, int(count))).astype(np.int64)
    selected = positions[offsets]
    if count <= len(positions) and len(np.unique(selected)) != len(selected):
        raise RuntimeError("even endpoint selection produced duplicates")
    return selected


def build_endpoint_refs(
    records: Sequence[ModuleRecord] | Mapping[str, ModuleRecord],
    *,
    endpoints_per_module: int = 24,
    horizons_hours: Sequence[float] = HORIZONS_HOURS,
    include_invalid: bool = False,
) -> list[EndpointRef]:
    """Select the same endpoint quota per module without class rebalancing.

    Very short modules repeat evenly spaced valid endpoints so every module
    still contributes exactly the configured quota and class weights reflect
    the actual module-uniform training distribution.
    """

    if endpoints_per_module <= 0:
        raise ValueError("endpoints_per_module must be positive")
    refs: list[EndpointRef] = []
    for module_index, record in enumerate(_record_sequence(records)):
        _, valid = multi_horizon_targets(
            record.timestamps, record.first_fault_ts, horizons_hours
        )
        positions = np.arange(len(record.timestamps), dtype=np.int64)
        if not include_invalid:
            positions = positions[valid]
        selected = _evenly_spaced(positions, int(endpoints_per_module))
        refs.extend(
            EndpointRef(record.file_name, module_index, int(endpoint))
            for endpoint in selected
        )
    return refs


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray

    @staticmethod
    def _finish(count: np.ndarray, total: np.ndarray, square_total: np.ndarray) -> "Standardizer":
        divisor = np.maximum(count, 1)
        mean = total / divisor
        variance = np.maximum(square_total / divisor - np.square(mean), 0.0)
        std = np.sqrt(variance)
        mean[(count == 0) | ~np.isfinite(mean)] = 0.0
        std[(count < 2) | ~np.isfinite(std) | (std < 1e-8)] = 1.0
        return Standardizer(mean.astype(np.float64), std.astype(np.float64), count.astype(np.int64))

    @classmethod
    def fit_values(
        cls,
        values: np.ndarray | Iterable[tuple[np.ndarray, np.ndarray]],
        valid: np.ndarray | None = None,
    ) -> "Standardizer":
        """Fit feature-wise moments on arrays whose final axis is feature."""

        if isinstance(values, np.ndarray):
            pairs: Iterable[tuple[np.ndarray, np.ndarray]] = [
                (
                    values,
                    np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool),
                )
            ]
        else:
            if valid is not None:
                raise ValueError("valid is only accepted with one ndarray")
            pairs = values
        count: np.ndarray | None = None
        total: np.ndarray | None = None
        square_total: np.ndarray | None = None
        for item_values, item_valid in pairs:
            array = np.asarray(item_values, dtype=np.float64)
            mask = np.asarray(item_valid, dtype=bool) & np.isfinite(array)
            if array.ndim < 1 or mask.shape != array.shape:
                raise ValueError("standardizer values and masks must be aligned")
            dimensions = array.shape[-1]
            if count is None:
                count = np.zeros(dimensions, dtype=np.int64)
                total = np.zeros(dimensions, dtype=np.float64)
                square_total = np.zeros(dimensions, dtype=np.float64)
            elif len(count) != dimensions:
                raise ValueError("standardizer feature dimensions changed between arrays")
            flat_values = array.reshape(-1, dimensions)
            flat_mask = mask.reshape(-1, dimensions)
            safe = np.where(flat_mask, flat_values, 0.0)
            count += flat_mask.sum(axis=0, dtype=np.int64)
            total += safe.sum(axis=0, dtype=np.float64)
            square_total += np.square(safe).sum(axis=0, dtype=np.float64)
        if count is None or total is None or square_total is None:
            raise ValueError("cannot fit a Standardizer on no arrays")
        return cls._finish(count, total, square_total)

    @classmethod
    def fit(
        cls,
        records: Sequence[ModuleRecord] | Mapping[str, ModuleRecord],
        *,
        field: str = "raw",
        only_pre_failure: bool = True,
    ) -> "Standardizer":
        """Fit raw/statistical/rule-margin moments on training records only."""

        aliases = {"stat": "statistics", "rule": "rule_margin"}
        field = aliases.get(field, field)
        valid_field = {
            "raw": "raw_valid",
            "statistics": "stat_valid",
            "rule_margin": "rule_valid",
        }.get(field)
        if valid_field is None:
            raise ValueError("field must be raw, statistics/stat, or rule_margin/rule")

        def pairs() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for record in _record_sequence(records):
                values = np.asarray(getattr(record, field))
                mask = np.asarray(getattr(record, valid_field), dtype=bool).copy()
                if only_pre_failure and record.first_fault_ts is not None:
                    row_valid = record.timestamps < float(record.first_fault_ts)
                    mask &= row_valid[:, None]
                yield values, mask

        return cls.fit_values(pairs())

    def transform(
        self,
        values: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.mean):
            raise ValueError("standardizer dimension does not match values")
        mask = np.isfinite(array)
        if valid is not None:
            valid_array = np.asarray(valid, dtype=bool)
            if valid_array.shape != array.shape:
                raise ValueError("valid must have the same shape as values")
            mask &= valid_array
        normalized = (
            array - self.mean.astype(np.float32)
        ) / self.std.astype(np.float32)
        return np.where(mask, normalized, 0.0).astype(np.float32), mask

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "count": self.count.astype(int).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]]) -> "Standardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            count=np.asarray(payload["count"], dtype=np.int64),
        )


def _safe_values(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(array)
    return np.where(mask, array, 0.0).astype(np.float32), mask


class FailurePredictionDataset(Dataset[dict[str, object]]):
    """Torch dataset that materializes only the requested causal endpoint."""

    def __init__(
        self,
        records: Sequence[ModuleRecord] | Mapping[str, ModuleRecord],
        endpoints: Sequence[EndpointRef] | None = None,
        *,
        horizons_hours: Sequence[float] = HORIZONS_HOURS,
        history_hours: int = 168,
        grid_hours: int = 1,
        rule_approach_hours: float = 12.0,
        rule_persistence_hours: float = 6.0,
        endpoints_per_module: int = 24,
        input_variant: str = "gated_fusion",
        raw_standardizer: Standardizer | None = None,
        stat_standardizer: Standardizer | None = None,
        rule_standardizer: Standardizer | None = None,
    ) -> None:
        self.records = _record_sequence(records)
        self.horizons_hours = tuple(float(value) for value in horizons_hours)
        self.history_hours = int(history_hours)
        self.grid_hours = int(grid_hours)
        self.rule_approach_hours = float(rule_approach_hours)
        self.rule_persistence_hours = float(rule_persistence_hours)
        allowed_variants = {
            "temporal_only",
            "deep_raw_stat",
            "rule_only",
            "fixed_fusion",
            "gated_fusion",
        }
        if input_variant not in allowed_variants:
            raise ValueError(f"unknown input_variant={input_variant!r}")
        self.input_variant = str(input_variant)
        self.need_raw = input_variant != "rule_only"
        self.need_statistics = input_variant in {
            "deep_raw_stat",
            "fixed_fusion",
            "gated_fusion",
        }
        self.need_rule = input_variant in {
            "rule_only",
            "fixed_fusion",
            "gated_fusion",
        }
        self.raw_standardizer = raw_standardizer
        self.stat_standardizer = stat_standardizer
        self.rule_standardizer = rule_standardizer
        self.endpoints = list(endpoints) if endpoints is not None else build_endpoint_refs(
            self.records,
            endpoints_per_module=endpoints_per_module,
            horizons_hours=self.horizons_hours,
        )
        label_cache = [
            multi_horizon_targets(
                record.timestamps, record.first_fault_ts, self.horizons_hours
            )
            for record in self.records
        ]
        targets: list[np.ndarray] = []
        endpoint_valid: list[bool] = []
        for ref in self.endpoints:
            if not 0 <= ref.module_index < len(self.records):
                raise ValueError("EndpointRef module_index is outside records")
            record = self.records[ref.module_index]
            if ref.file_name != record.file_name:
                raise ValueError("EndpointRef file_name disagrees with its module_index")
            if not 0 <= ref.endpoint_index < len(record.timestamps):
                raise ValueError("EndpointRef endpoint_index is outside its module")
            module_targets, module_valid = label_cache[ref.module_index]
            targets.append(module_targets[ref.endpoint_index])
            endpoint_valid.append(bool(module_valid[ref.endpoint_index]))
        self.target_matrix = (
            np.stack(targets).astype(np.float32)
            if targets
            else np.empty((0, len(self.horizons_hours)), dtype=np.float32)
        )
        self.valid_endpoint_mask = np.asarray(endpoint_valid, dtype=bool)

    def __len__(self) -> int:
        return len(self.endpoints)

    def targets_array(self) -> np.ndarray:
        return self.target_matrix.copy()

    def __getitem__(self, item: int) -> dict[str, object]:
        ref = self.endpoints[int(item)]
        record = self.records[ref.module_index]
        if self.need_raw:
            raw, raw_mask, delta_hours = causal_hourly_window(
                record,
                ref.endpoint_index,
                history_hours=self.history_hours,
                grid_hours=self.grid_hours,
            )
            if self.raw_standardizer is None:
                raw, raw_mask = _safe_values(raw, raw_mask)
            else:
                raw, raw_mask = self.raw_standardizer.transform(raw, raw_mask)
        else:
            raw = np.zeros((1, 1), dtype=np.float32)
            raw_mask = np.zeros_like(raw, dtype=bool)
            delta_hours = np.zeros((1, 1), dtype=np.float32)
        if self.need_statistics:
            statistics = record.statistics[ref.endpoint_index]
            stat_mask = record.stat_valid[ref.endpoint_index]
            if self.stat_standardizer is None:
                statistics, stat_mask = _safe_values(statistics, stat_mask)
            else:
                statistics, stat_mask = self.stat_standardizer.transform(
                    statistics, stat_mask
                )
        else:
            statistics = np.zeros(1, dtype=np.float32)
            stat_mask = np.zeros(1, dtype=bool)
        # Signed rule margins already carry fixed physical normalization.  A
        # rule standardizer is optional and never fit implicitly.
        if self.need_rule:
            rule, rule_mask = endpoint_rule_signals(
                record,
                ref.endpoint_index,
                approach_hours=self.rule_approach_hours,
                persistence_hours=self.rule_persistence_hours,
            )
        else:
            rule = np.zeros(1, dtype=np.float32)
            rule_mask = np.zeros(1, dtype=bool)
        if self.rule_standardizer is None:
            rule, rule_mask = _safe_values(rule, rule_mask)
        else:
            rule, rule_mask = self.rule_standardizer.transform(rule, rule_mask)
        if self.need_rule and rule.shape != (RULE_SIGNAL_DIM,):
            raise RuntimeError("rule signal dimension changed unexpectedly")
        return {
            "raw": torch.from_numpy(raw.astype(np.float32)),
            "raw_mask": torch.from_numpy(raw_mask.astype(np.float32)),
            "delta_hours": torch.from_numpy(delta_hours.astype(np.float32)),
            "statistics": torch.from_numpy(statistics.astype(np.float32)),
            "stat_mask": torch.from_numpy(stat_mask.astype(np.float32)),
            "rule": torch.from_numpy(rule.astype(np.float32)),
            "rule_mask": torch.from_numpy(rule_mask.astype(np.float32)),
            "targets": torch.from_numpy(self.target_matrix[int(item)]),
            "module_index": int(ref.module_index),
            "endpoint_index": int(ref.endpoint_index),
            "file_name": ref.file_name,
            "timestamp": int(record.timestamps[ref.endpoint_index]),
            "valid_training_endpoint": bool(self.valid_endpoint_mask[int(item)]),
        }


class ModuleUniformEndpointSampler(Sampler[int]):
    """Sample modules uniformly, then choose one of that module's endpoints."""

    def __init__(
        self,
        dataset: FailurePredictionDataset,
        *,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for dataset_index, ref in enumerate(dataset.endpoints):
            if not dataset.valid_endpoint_mask[dataset_index]:
                continue
            groups.setdefault(ref.module_index, []).append(dataset_index)
        if not groups:
            raise ValueError("sampler has no valid training endpoints")
        self._groups = {
            module_index: np.asarray(indices, dtype=np.int64)
            for module_index, indices in groups.items()
        }
        self._modules = np.asarray(sorted(self._groups), dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        emitted = 0
        while emitted < self.num_samples:
            modules = self._modules.copy()
            rng.shuffle(modules)
            for module_index in modules:
                candidates = self._groups[int(module_index)]
                yield int(candidates[rng.integers(0, len(candidates))])
                emitted += 1
                if emitted >= self.num_samples:
                    break


__all__ = [
    "HORIZONS_HOURS",
    "FIXED_TEST_FOLD",
    "ModuleRecord",
    "read_module_record",
    "load_module_records",
    "multi_horizon_targets",
    "SplitManifest",
    "build_fixed_split",
    "EndpointRef",
    "build_endpoint_refs",
    "Standardizer",
    "FailurePredictionDataset",
    "ModuleUniformEndpointSampler",
]
