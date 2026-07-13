"""Native-row sequence data protocol for Future-Guided OFP.

The module deliberately does *not* resample, aggregate, interpolate, or pad
inside an optical-module record.  One CSV row becomes one sequence step.  A
batch collator may pad different modules, but padded steps are always masked.

All timestamps remain signed ``int64`` Unix seconds.  Legacy-Inclusive means
that the first anomalous row is both a valid training step and a valid alarm
time.  Rows after that positional first failure are excluded, including rows
that repeat the same timestamp.
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
    TEMPERATURE_OUTLIER_SENTINEL,
)


FIXED_TEST_FOLD = 3
DEFAULT_STUDENT_HORIZON_HOURS = 120.0
DEFAULT_TEACHER_HORIZON_HOURS = 114.0
DEFAULT_FUTURE_OFFSET_HOURS = 6.0
DEFAULT_EXPECTED_CADENCE_SECONDS = 300
DEFAULT_ALIGNMENT_TOLERANCE_SECONDS = 300


def _integer_seconds(value: float, *, name: str, positive: bool = False) -> int:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-9:
        raise ValueError(f"{name} must resolve to an integer number of seconds")
    result = int(rounded)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def hours_to_seconds(hours: float, *, name: str = "hours") -> int:
    """Convert a duration to exact integer seconds without silent rounding."""

    return _integer_seconds(float(hours) * 3600.0, name=name, positive=True)


def _parse_timestamps(series: pd.Series, source: Path | str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise ValueError(f"{source} contains missing or non-numeric timestamps")
    rounded = np.rint(numeric)
    if np.any(numeric != rounded):
        raise ValueError(f"{source} timestamps must be integer seconds")
    limits = np.iinfo(np.int64)
    if np.any(rounded < limits.min) or np.any(rounded > limits.max):
        raise ValueError(f"{source} timestamp lies outside int64")
    timestamps = rounded.astype(np.int64)
    if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{source} timestamps must be non-decreasing")
    return timestamps


@dataclass
class ModuleRecord:
    """One native-cadence module sequence with no row transformation."""

    file_name: str
    timestamps: np.ndarray
    raw: np.ndarray
    raw_mask: np.ndarray
    anomaly: np.ndarray
    first_fault_index: int | None

    def __post_init__(self) -> None:
        self.file_name = str(self.file_name)
        self.timestamps = np.asarray(self.timestamps)
        self.raw = np.asarray(self.raw, dtype=np.float32)
        self.raw_mask = np.asarray(self.raw_mask, dtype=bool)
        self.anomaly = np.asarray(self.anomaly, dtype=np.float32)
        rows = len(self.timestamps)
        if self.timestamps.dtype != np.int64:
            raise ValueError("ModuleRecord.timestamps must have dtype int64")
        if self.timestamps.shape != (rows,):
            raise ValueError("timestamps must be one-dimensional")
        if rows == 0:
            raise ValueError("a module sequence cannot be empty")
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError("timestamps must be non-decreasing")
        if self.raw.shape != (rows, len(RAW_FEATURES)):
            raise ValueError("raw must have shape [rows, 12]")
        if self.raw_mask.shape != self.raw.shape:
            raise ValueError("raw_mask must align with raw")
        if self.anomaly.shape != (rows,) or not np.isfinite(self.anomaly).all():
            raise ValueError("anomaly must be a finite one-dimensional row array")
        expected = np.flatnonzero(self.anomaly > 0)
        expected_first = int(expected[0]) if len(expected) else None
        if self.first_fault_index is None:
            if expected_first is not None:
                raise ValueError("first_fault_index is missing despite anomaly > 0")
        else:
            index = int(self.first_fault_index)
            if index != expected_first:
                raise ValueError("first_fault_index must be the first anomaly > 0 row")
            self.first_fault_index = index

    @property
    def first_fault_timestamp(self) -> int | None:
        if self.first_fault_index is None:
            return None
        return int(self.timestamps[self.first_fault_index])

    @property
    def length(self) -> int:
        return len(self.timestamps)


def read_module_record(path: Path | str) -> ModuleRecord:
    """Read one CSV while preserving every original row and its order."""

    path = Path(path)
    needed = {"timestamp", "anomaly", *RAW_FEATURES}
    frame = pd.read_csv(path, usecols=lambda name: name in needed)
    required = {"timestamp", *RAW_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    timestamps = _parse_timestamps(frame["timestamp"], path)
    if "anomaly" in frame:
        anomaly = (
            pd.to_numeric(frame["anomaly"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    else:
        anomaly = np.zeros(len(frame), dtype=np.float32)
    numeric_raw = frame.loc[:, RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    raw = numeric_raw.to_numpy(dtype=np.float32)
    raw_mask = np.isfinite(raw) & (raw != float(MISSING_SENTINEL))
    temperature_index = RAW_FEATURES.index("temperature")
    raw_mask[:, temperature_index] &= (
        raw[:, temperature_index] != float(TEMPERATURE_OUTLIER_SENTINEL)
    )
    failures = np.flatnonzero(anomaly > 0)
    first_fault_index = int(failures[0]) if len(failures) else None
    return ModuleRecord(
        file_name=path.name,
        timestamps=timestamps,
        raw=raw,
        raw_mask=raw_mask,
        anomaly=anomaly,
        first_fault_index=first_fault_index,
    )


def load_module_records(
    data_dir: Path | str,
    file_names: Sequence[str],
    *,
    progress: bool = False,
) -> list[ModuleRecord]:
    """Load records in exactly the supplied module order."""

    directory = Path(data_dir)
    records: list[ModuleRecord] = []
    for position, name in enumerate((str(item) for item in file_names), 1):
        records.append(read_module_record(directory / name))
        if progress and (position % 100 == 0 or position == len(file_names)):
            print(f"  [data] loaded {position}/{len(file_names)} modules")
    return records


def legacy_inclusive_targets(
    timestamps: np.ndarray,
    first_fault_index: int | None,
    horizon_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one future-window target and its Legacy-Inclusive loss mask.

    For a faulty module, row ``i`` is positive iff
    ``0 <= timestamp[first_fault_index] - timestamp[i] <= horizon`` and the
    row is not positionally after the first failure.  Positional masking is
    intentional: a duplicate timestamp after the failure row stays invalid.
    Healthy modules contain valid negatives at all observed rows.
    """

    times = np.asarray(timestamps)
    if times.dtype != np.int64 or times.ndim != 1 or len(times) == 0:
        raise ValueError("timestamps must be a non-empty int64 vector")
    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be non-decreasing")
    horizon_seconds = hours_to_seconds(horizon_hours, name="horizon_hours")
    targets = np.zeros(len(times), dtype=np.int64)
    loss_mask = np.ones(len(times), dtype=bool)
    if first_fault_index is None:
        return targets, loss_mask
    fault_index = int(first_fault_index)
    if not 0 <= fault_index < len(times):
        raise ValueError("first_fault_index lies outside timestamps")
    loss_mask[fault_index + 1 :] = False
    remaining = int(times[fault_index]) - times
    targets[
        loss_mask & (remaining >= 0) & (remaining <= horizon_seconds)
    ] = 1
    return targets, loss_mask


def native_delta_steps(
    timestamps: np.ndarray,
    *,
    expected_cadence_seconds: int = DEFAULT_EXPECTED_CADENCE_SECONDS,
    clip_steps: float | None = None,
) -> np.ndarray:
    """Encode observed time gaps without inserting or deleting sequence rows."""

    times = np.asarray(timestamps)
    if times.dtype != np.int64 or times.ndim != 1 or len(times) == 0:
        raise ValueError("timestamps must be a non-empty int64 vector")
    cadence = int(expected_cadence_seconds)
    if cadence <= 0:
        raise ValueError("expected_cadence_seconds must be positive")
    differences = np.diff(times)
    if np.any(differences < 0):
        raise ValueError("timestamps must be non-decreasing")
    delta = np.zeros(len(times), dtype=np.float32)
    delta[1:] = differences.astype(np.float64) / float(cadence)
    if clip_steps is not None:
        clip = float(clip_steps)
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("clip_steps must be positive and finite")
        np.minimum(delta, clip, out=delta)
    return delta[:, None]


@dataclass(frozen=True)
class Standardizer:
    """Mask-aware raw-feature moments that must be fit on train records only."""

    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray

    def __post_init__(self) -> None:
        dimensions = len(RAW_FEATURES)
        if np.asarray(self.mean).shape != (dimensions,):
            raise ValueError("standardizer mean has an unexpected shape")
        if np.asarray(self.std).shape != (dimensions,):
            raise ValueError("standardizer std has an unexpected shape")
        if np.asarray(self.count).shape != (dimensions,):
            raise ValueError("standardizer count has an unexpected shape")
        if np.any(np.asarray(self.std) <= 0):
            raise ValueError("standardizer std must be positive")

    @classmethod
    def fit(cls, train_records: Sequence[ModuleRecord]) -> "Standardizer":
        """Fit on the caller-supplied *training* modules, never implicitly."""

        records = list(train_records)
        if not records:
            raise ValueError("cannot fit Standardizer on no training records")
        dimensions = len(RAW_FEATURES)
        count = np.zeros(dimensions, dtype=np.int64)
        total = np.zeros(dimensions, dtype=np.float64)
        square_total = np.zeros(dimensions, dtype=np.float64)
        for record in records:
            row_mask = np.ones(record.length, dtype=bool)
            if record.first_fault_index is not None:
                row_mask[record.first_fault_index + 1 :] = False
            valid = record.raw_mask & row_mask[:, None]
            values = record.raw.astype(np.float64)
            safe = np.where(valid, values, 0.0)
            count += valid.sum(axis=0, dtype=np.int64)
            total += safe.sum(axis=0, dtype=np.float64)
            square_total += np.square(safe).sum(axis=0, dtype=np.float64)
        divisor = np.maximum(count, 1)
        mean = total / divisor
        variance = np.maximum(square_total / divisor - np.square(mean), 0.0)
        std = np.sqrt(variance)
        mean[(count == 0) | ~np.isfinite(mean)] = 0.0
        std[(count < 2) | ~np.isfinite(std) | (std < 1e-8)] = 1.0
        return cls(mean=mean, std=std, count=count)

    def transform(
        self,
        raw: np.ndarray,
        raw_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(raw, dtype=np.float32)
        mask = np.asarray(raw_mask, dtype=bool) & np.isfinite(values)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("raw has an unexpected shape")
        if mask.shape != values.shape:
            raise ValueError("raw_mask must align with raw")
        normalized = (
            values - np.asarray(self.mean, dtype=np.float32)
        ) / np.asarray(self.std, dtype=np.float32)
        return np.where(mask, normalized, 0.0).astype(np.float32), mask

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "mean": np.asarray(self.mean, dtype=float).tolist(),
            "std": np.asarray(self.std, dtype=float).tolist(),
            "count": np.asarray(self.count, dtype=int).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]]) -> "Standardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            count=np.asarray(payload["count"], dtype=np.int64),
        )


def build_fgl_alignment(
    timestamps: np.ndarray,
    *,
    student_targets: np.ndarray,
    teacher_targets: np.ndarray,
    loss_mask: np.ndarray,
    first_fault_index: int | None,
    offset_hours: float,
    tolerance_seconds: int = DEFAULT_ALIGNMENT_TOLERANCE_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each student row to the first observed row at/after ``t+offset``.

    A KL pair is retained only if the future row is within the allowed time
    error, is no later than the first failure row, is loss-valid, and its
    teacher label equals the student's future-window label.  Thus the
    distillation target is semantically aligned rather than merely shifted by
    a fixed row count.
    """

    times = np.asarray(timestamps)
    student = np.asarray(student_targets, dtype=np.int64)
    teacher = np.asarray(teacher_targets, dtype=np.int64)
    valid = np.asarray(loss_mask, dtype=bool)
    rows = len(times)
    if times.dtype != np.int64 or times.shape != (rows,):
        raise ValueError("timestamps must be an int64 vector")
    if student.shape != (rows,) or teacher.shape != (rows,) or valid.shape != (rows,):
        raise ValueError("targets and loss_mask must align with timestamps")
    offset_seconds = hours_to_seconds(offset_hours, name="offset_hours")
    tolerance = _integer_seconds(
        tolerance_seconds, name="tolerance_seconds", positive=False
    )
    teacher_index = np.full(rows, -1, dtype=np.int64)
    pair_mask = np.zeros(rows, dtype=bool)
    if first_fault_index is not None and int(first_fault_index) == 0:
        return teacher_index, pair_mask
    if len(times) == 0:
        return teacher_index, pair_mask
    max_int = np.iinfo(np.int64).max
    safe_student = times <= max_int - offset_seconds
    desired = np.full(rows, max_int, dtype=np.int64)
    desired[safe_student] = times[safe_student] + np.int64(offset_seconds)
    candidate = np.searchsorted(times, desired, side="left").astype(np.int64)
    candidate_in_range = safe_student & (candidate < rows) & valid
    source_indices = np.flatnonzero(candidate_in_range)
    if not len(source_indices):
        return teacher_index, pair_mask
    target_indices = candidate[source_indices]
    time_error = times[target_indices] - desired[source_indices]
    keep = time_error <= tolerance
    keep &= valid[target_indices]
    if first_fault_index is not None:
        keep &= target_indices <= int(first_fault_index)
    keep &= teacher[target_indices] == student[source_indices]
    kept_sources = source_indices[keep]
    teacher_index[kept_sources] = target_indices[keep]
    pair_mask[kept_sources] = True
    return teacher_index, pair_mask


def _record_sequence(
    records: Sequence[ModuleRecord] | Mapping[str, ModuleRecord],
) -> list[ModuleRecord]:
    if isinstance(records, Mapping):
        sequence = list(records.values())
        for key, record in zip(records.keys(), sequence):
            if str(key) != record.file_name:
                raise ValueError("record mapping keys must equal file_name")
        return sequence
    return list(records)


class NativeSequenceDataset(Dataset[dict[str, object]]):
    """One native observed-row sequence per module for sequence-to-sequence CE."""

    def __init__(
        self,
        records: Sequence[ModuleRecord] | Mapping[str, ModuleRecord],
        *,
        standardizer: Standardizer | None = None,
        student_horizon_hours: float = DEFAULT_STUDENT_HORIZON_HOURS,
        teacher_horizon_hours: float = DEFAULT_TEACHER_HORIZON_HOURS,
        offset_hours: float = DEFAULT_FUTURE_OFFSET_HOURS,
        alignment_tolerance_seconds: int = DEFAULT_ALIGNMENT_TOLERANCE_SECONDS,
        expected_cadence_seconds: int = DEFAULT_EXPECTED_CADENCE_SECONDS,
        delta_clip_steps: float | None = None,
        truncate_after_first_fault: bool = False,
        include_supervision: bool = True,
    ) -> None:
        self.records = _record_sequence(records)
        if not self.records:
            raise ValueError("NativeSequenceDataset requires at least one module")
        names = [record.file_name for record in self.records]
        if len(names) != len(set(names)):
            raise ValueError("NativeSequenceDataset contains duplicate file names")
        self.standardizer = standardizer
        self.student_horizon_hours = float(student_horizon_hours)
        self.teacher_horizon_hours = float(teacher_horizon_hours)
        self.offset_hours = float(offset_hours)
        # Fail early if any duration cannot be represented by integer seconds.
        student_seconds = hours_to_seconds(
            self.student_horizon_hours, name="student_horizon_hours"
        )
        teacher_seconds = hours_to_seconds(
            self.teacher_horizon_hours, name="teacher_horizon_hours"
        )
        offset_seconds = hours_to_seconds(self.offset_hours, name="offset_hours")
        if teacher_seconds + offset_seconds != student_seconds:
            raise ValueError(
                "teacher_horizon_hours + offset_hours must equal "
                "student_horizon_hours"
            )
        self.alignment_tolerance_seconds = _integer_seconds(
            alignment_tolerance_seconds,
            name="alignment_tolerance_seconds",
            positive=False,
        )
        self.expected_cadence_seconds = int(expected_cadence_seconds)
        if self.expected_cadence_seconds <= 0:
            raise ValueError("expected_cadence_seconds must be positive")
        self.delta_clip_steps = delta_clip_steps
        self.truncate_after_first_fault = bool(truncate_after_first_fault)
        self.include_supervision = bool(include_supervision)
        if self.truncate_after_first_fault and not self.include_supervision:
            raise ValueError(
                "inference-only sequences cannot be truncated using failure labels"
            )
        self.lengths = np.asarray(
            [
                (
                    int(record.first_fault_index) + 1
                    if self.truncate_after_first_fault
                    and record.first_fault_index is not None
                    else record.length
                )
                for record in self.records
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, object]:
        record = self.records[int(item)]
        length = int(self.lengths[int(item)])
        timestamps = record.timestamps[:length]
        raw_values = record.raw[:length]
        raw_validity = record.raw_mask[:length]
        anomaly = record.anomaly[:length]
        if self.include_supervision:
            student_targets, loss_mask = legacy_inclusive_targets(
                timestamps,
                record.first_fault_index,
                self.student_horizon_hours,
            )
            teacher_targets, teacher_loss_mask = legacy_inclusive_targets(
                timestamps,
                record.first_fault_index,
                self.teacher_horizon_hours,
            )
            if not np.array_equal(loss_mask, teacher_loss_mask):
                raise RuntimeError("teacher and student Legacy-Inclusive masks diverged")
            teacher_index, fgl_mask = build_fgl_alignment(
                timestamps,
                student_targets=student_targets,
                teacher_targets=teacher_targets,
                loss_mask=loss_mask,
                first_fault_index=record.first_fault_index,
                offset_hours=self.offset_hours,
                tolerance_seconds=self.alignment_tolerance_seconds,
            )
        else:
            student_targets = np.zeros(length, dtype=np.int64)
            teacher_targets = np.zeros(length, dtype=np.int64)
            loss_mask = np.zeros(length, dtype=bool)
            teacher_index = np.full(length, -1, dtype=np.int64)
            fgl_mask = np.zeros(length, dtype=bool)
        if self.standardizer is None:
            raw_mask = raw_validity & np.isfinite(raw_values)
            raw = np.where(raw_mask, raw_values, 0.0).astype(np.float32)
        else:
            raw, raw_mask = self.standardizer.transform(raw_values, raw_validity)
        delta_steps = native_delta_steps(
            timestamps,
            expected_cadence_seconds=self.expected_cadence_seconds,
            clip_steps=self.delta_clip_steps,
        )
        return {
            "raw": torch.from_numpy(raw),
            "raw_mask": torch.from_numpy(raw_mask),
            "delta_steps": torch.from_numpy(delta_steps),
            "targets_teacher": torch.from_numpy(teacher_targets),
            "targets_student": torch.from_numpy(student_targets),
            "loss_mask": torch.from_numpy(loss_mask),
            "teacher_index": torch.from_numpy(teacher_index),
            "fgl_mask": torch.from_numpy(fgl_mask),
            "timestamps": torch.from_numpy(timestamps.copy()),
            "anomaly": torch.from_numpy(anomaly.copy()),
            "length": length,
            "file_name": record.file_name,
            "first_fault_index": (
                -1 if record.first_fault_index is None else int(record.first_fault_index)
            ),
        }


def collate_native_sequences(batch: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Pad variable module lengths while keeping every real row loss-visible."""

    samples = list(batch)
    if not samples:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.as_tensor([int(sample["length"]) for sample in samples], dtype=torch.long)
    batch_size = len(samples)
    max_length = int(lengths.max().item())
    feature_count = len(RAW_FEATURES)
    raw = torch.zeros((batch_size, max_length, feature_count), dtype=torch.float32)
    raw_mask = torch.zeros((batch_size, max_length, feature_count), dtype=torch.bool)
    delta_steps = torch.zeros((batch_size, max_length, 1), dtype=torch.float32)
    targets_teacher = torch.zeros((batch_size, max_length), dtype=torch.long)
    targets_student = torch.zeros((batch_size, max_length), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    teacher_index = torch.full((batch_size, max_length), -1, dtype=torch.long)
    fgl_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    timestamps = torch.full((batch_size, max_length), -1, dtype=torch.long)
    anomaly = torch.zeros((batch_size, max_length), dtype=torch.float32)
    file_names: list[str] = []
    first_fault_indices = torch.full((batch_size,), -1, dtype=torch.long)
    for batch_index, sample in enumerate(samples):
        length = int(lengths[batch_index])
        raw[batch_index, :length] = torch.as_tensor(sample["raw"], dtype=torch.float32)
        raw_mask[batch_index, :length] = torch.as_tensor(sample["raw_mask"], dtype=torch.bool)
        delta_steps[batch_index, :length] = torch.as_tensor(
            sample["delta_steps"], dtype=torch.float32
        )
        targets_teacher[batch_index, :length] = torch.as_tensor(
            sample["targets_teacher"], dtype=torch.long
        )
        targets_student[batch_index, :length] = torch.as_tensor(
            sample["targets_student"], dtype=torch.long
        )
        loss_mask[batch_index, :length] = torch.as_tensor(
            sample["loss_mask"], dtype=torch.bool
        )
        teacher_index[batch_index, :length] = torch.as_tensor(
            sample["teacher_index"], dtype=torch.long
        )
        fgl_mask[batch_index, :length] = torch.as_tensor(
            sample["fgl_mask"], dtype=torch.bool
        )
        timestamps[batch_index, :length] = torch.as_tensor(
            sample["timestamps"], dtype=torch.long
        )
        anomaly[batch_index, :length] = torch.as_tensor(
            sample["anomaly"], dtype=torch.float32
        )
        file_names.append(str(sample["file_name"]))
        first_fault_indices[batch_index] = int(sample["first_fault_index"])
    return {
        "raw": raw,
        "raw_mask": raw_mask,
        "delta_steps": delta_steps,
        "targets_teacher": targets_teacher,
        "targets_student": targets_student,
        "loss_mask": loss_mask,
        "teacher_index": teacher_index,
        "fgl_mask": fgl_mask,
        "timestamps": timestamps,
        "anomaly": anomaly,
        "lengths": lengths,
        "file_names": file_names,
        "first_fault_indices": first_fault_indices,
    }


@dataclass(frozen=True)
class SplitManifest:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


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
    folds = set(frame["folder_index"].unique())
    if folds != {1, 2, 3}:
        raise ValueError(f"fixed protocol requires folds 1, 2, 3; got {sorted(folds)}")
    return frame


def _stratified_validation_indices(
    development: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    binary = development["Label"].to_numpy(dtype=int) > 0
    groups = [np.flatnonzero(binary == value) for value in (False, True)]
    if any(len(group) < 2 for group in groups):
        raise ValueError("both label strata need at least two development modules")
    target_total = int(math.ceil(len(development) * float(validation_fraction)))
    ideals = np.asarray([len(group) * float(validation_fraction) for group in groups])
    allocation = np.maximum(np.floor(ideals).astype(int), 1)
    allocation = np.minimum(allocation, np.asarray([len(group) - 1 for group in groups]))
    while int(allocation.sum()) < target_total:
        candidates = [
            index
            for index, group in enumerate(groups)
            if allocation[index] < len(group) - 1
        ]
        if not candidates:
            break
        selected = max(
            candidates, key=lambda index: (ideals[index] - allocation[index], -index)
        )
        allocation[selected] += 1
    while int(allocation.sum()) > target_total:
        candidates = [index for index in range(2) if allocation[index] > 1]
        if not candidates:
            break
        selected = min(
            candidates, key=lambda index: (ideals[index] - allocation[index], index)
        )
        allocation[selected] -= 1
    if int(allocation.sum()) != target_total:
        raise RuntimeError("unable to allocate stratified validation modules")
    rng = np.random.default_rng(int(seed))
    chosen: list[np.ndarray] = []
    for positions, count in zip(groups, allocation):
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        chosen.append(shuffled[: int(count)])
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def build_fixed_split(
    index: Path | str | pd.DataFrame,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    test_fold: int = FIXED_TEST_FOLD,
) -> SplitManifest:
    """Use fold 3 as test and stratify 10% of folds 1+2 for validation."""

    if int(test_fold) != FIXED_TEST_FOLD:
        raise ValueError("the comparable OFP protocol fixes test_fold=3")
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
    sets = [
        set(part["file_name"].astype(str))
        for part in (result.train, result.validation, result.test)
    ]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("module leakage across train/validation/test")
    return result


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch similarly sized module sequences to reduce padding overhead."""

    def __init__(
        self,
        dataset_or_lengths: NativeSequenceDataset | Sequence[int],
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        bucket_size_multiplier: int = 20,
    ) -> None:
        if isinstance(dataset_or_lengths, NativeSequenceDataset):
            lengths = dataset_or_lengths.lengths
        else:
            lengths = np.asarray(tuple(dataset_or_lengths), dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if self.lengths.ndim != 1 or not len(self.lengths) or np.any(self.lengths <= 0):
            raise ValueError("lengths must be a non-empty positive vector")
        self.batch_size = int(batch_size)
        self.bucket_size_multiplier = int(bucket_size_multiplier)
        if self.batch_size <= 0 or self.bucket_size_multiplier <= 0:
            raise ValueError("batch and bucket sizes must be positive")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return int(math.ceil(len(self.lengths) / self.batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        sorted_indices = np.argsort(self.lengths, kind="stable")
        bucket_size = self.batch_size * self.bucket_size_multiplier
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for start in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[start : start + bucket_size].copy()
            if self.shuffle:
                rng.shuffle(bucket)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch.astype(int).tolist())
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


__all__ = [
    "DEFAULT_ALIGNMENT_TOLERANCE_SECONDS",
    "DEFAULT_EXPECTED_CADENCE_SECONDS",
    "DEFAULT_FUTURE_OFFSET_HOURS",
    "DEFAULT_STUDENT_HORIZON_HOURS",
    "DEFAULT_TEACHER_HORIZON_HOURS",
    "FIXED_TEST_FOLD",
    "LengthBucketBatchSampler",
    "ModuleRecord",
    "NativeSequenceDataset",
    "SplitManifest",
    "Standardizer",
    "build_fgl_alignment",
    "build_fixed_split",
    "collate_native_sequences",
    "hours_to_seconds",
    "legacy_inclusive_targets",
    "load_module_records",
    "native_delta_steps",
    "read_module_record",
]
