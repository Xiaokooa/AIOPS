from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fgofp.data import (
    LengthBucketBatchSampler,
    ModuleRecord,
    NativeSequenceDataset,
    Standardizer,
    build_fgl_alignment,
    build_fixed_split,
    collate_native_sequences,
    legacy_inclusive_targets,
    native_delta_steps,
    read_module_record,
)
from fgofp.features import RAW_FEATURES


def _frame(timestamps: list[float], anomaly: list[float], base: float = 1.0) -> pd.DataFrame:
    values: dict[str, object] = {
        "timestamp": timestamps,
        "anomaly": anomaly,
    }
    for index, name in enumerate(RAW_FEATURES):
        values[name] = np.arange(len(timestamps), dtype=float) + base + index
    return pd.DataFrame(values)


def _record(
    name: str,
    timestamps: list[int],
    anomaly: list[float] | None = None,
    base: float = 1.0,
) -> ModuleRecord:
    anomaly_values = anomaly if anomaly is not None else [0.0] * len(timestamps)
    frame = _frame(timestamps, anomaly_values, base=base)
    raw = frame.loc[:, RAW_FEATURES].to_numpy(dtype=np.float32)
    failures = np.flatnonzero(np.asarray(anomaly_values) > 0)
    return ModuleRecord(
        file_name=name,
        timestamps=np.asarray(timestamps, dtype=np.int64),
        raw=raw,
        raw_mask=np.ones_like(raw, dtype=bool),
        anomaly=np.asarray(anomaly_values, dtype=np.float32),
        first_fault_index=int(failures[0]) if len(failures) else None,
    )


def test_reader_preserves_every_native_row_and_integer_timestamps(tmp_path: Path) -> None:
    frame = _frame([100, 400, 1000, 1000], [0, 0, 1, 0])
    frame.loc[1, "current"] = -999.0
    frame.loc[2, "temperature"] = -255.0
    path = tmp_path / "module.csv"
    frame.to_csv(path, index=False)

    record = read_module_record(path)

    assert record.length == len(frame)
    assert record.timestamps.dtype == np.int64
    assert record.timestamps.tolist() == [100, 400, 1000, 1000]
    assert record.first_fault_index == 2
    assert record.raw.shape == (4, 12)
    assert not record.raw_mask[1, RAW_FEATURES.index("current")]
    assert not record.raw_mask[2, RAW_FEATURES.index("temperature")]
    assert native_delta_steps(record.timestamps).ravel().tolist() == [0.0, 1.0, 2.0, 0.0]


def test_reader_rejects_fractional_or_decreasing_timestamp(tmp_path: Path) -> None:
    fractional = tmp_path / "fractional.csv"
    _frame([100.0, 400.5], [0, 0]).to_csv(fractional, index=False)
    with pytest.raises(ValueError, match="integer seconds"):
        read_module_record(fractional)

    decreasing = tmp_path / "decreasing.csv"
    _frame([400, 100], [0, 0]).to_csv(decreasing, index=False)
    with pytest.raises(ValueError, match="non-decreasing"):
        read_module_record(decreasing)


def test_legacy_inclusive_uses_first_fault_row_position_with_duplicate_time() -> None:
    times = np.asarray([0, 300, 600, 600, 600, 900], dtype=np.int64)

    targets, valid = legacy_inclusive_targets(times, first_fault_index=3, horizon_hours=1.0)

    assert targets.tolist() == [1, 1, 1, 1, 0, 0]
    assert valid.tolist() == [True, True, True, True, False, False]
    # The repeated timestamp at row 4 is after the positional fault row and
    # must not leak into CE despite having zero timestamp lead.
    assert not valid[4]


def test_fgl_alignment_is_timestamp_based_label_matched_and_future_only() -> None:
    times = np.arange(0, 2100, 300, dtype=np.int64)
    student, valid = legacy_inclusive_targets(times, 6, horizon_hours=0.5)
    teacher, _ = legacy_inclusive_targets(times, 6, horizon_hours=0.25)

    teacher_index, pair_mask = build_fgl_alignment(
        times,
        student_targets=student,
        teacher_targets=teacher,
        loss_mask=valid,
        first_fault_index=6,
        offset_hours=0.25,
        tolerance_seconds=0,
    )

    assert teacher_index[:4].tolist() == [3, 4, 5, 6]
    assert pair_mask.tolist() == [True, True, True, True, False, False, False]
    paired_source = np.flatnonzero(pair_mask)
    assert np.all(times[teacher_index[paired_source]] > times[paired_source])
    assert np.array_equal(teacher[teacher_index[paired_source]], student[paired_source])


def test_fgl_first_row_failure_has_no_kl_pair() -> None:
    record = _record("first.csv", [0, 300, 600], anomaly=[1, 0, 0])
    dataset = NativeSequenceDataset(
        [record],
        student_horizon_hours=0.5,
        teacher_horizon_hours=0.25,
        offset_hours=0.25,
        alignment_tolerance_seconds=0,
    )

    sample = dataset[0]

    assert sample["loss_mask"].tolist() == [True, False, False]
    assert not bool(sample["fgl_mask"].any())
    assert sample["teacher_index"].tolist() == [-1, -1, -1]


def test_alignment_requires_first_row_at_or_after_offset_within_tolerance() -> None:
    times = np.asarray([0, 300, 905, 1200], dtype=np.int64)
    targets = np.zeros(4, dtype=np.int64)
    valid = np.ones(4, dtype=bool)

    index_tight, mask_tight = build_fgl_alignment(
        times,
        student_targets=targets,
        teacher_targets=targets,
        loss_mask=valid,
        first_fault_index=None,
        offset_hours=0.25,
        tolerance_seconds=4,
    )
    index_wide, mask_wide = build_fgl_alignment(
        times,
        student_targets=targets,
        teacher_targets=targets,
        loss_mask=valid,
        first_fault_index=None,
        offset_hours=0.25,
        tolerance_seconds=5,
    )

    assert index_tight[0] == -1 and not mask_tight[0]
    assert index_wide[0] == 2 and mask_wide[0]


def test_standardizer_is_explicitly_fit_on_train_and_excludes_post_fault() -> None:
    train = _record("train.csv", [0, 300, 600], anomaly=[0, 1, 0])
    train.raw[0, :] = 1.0
    train.raw[1, :] = 3.0
    train.raw[2, :] = 1000.0  # post-failure; never enters train moments
    validation = _record("validation.csv", [0, 300], base=10000.0)

    standardizer = Standardizer.fit([train])
    transformed_validation, _ = standardizer.transform(
        validation.raw, validation.raw_mask
    )

    assert np.allclose(standardizer.mean, 2.0)
    assert np.allclose(standardizer.std, 1.0)
    assert np.all(standardizer.count == 2)
    assert transformed_validation[0, 0] > 9000.0
    restored = Standardizer.from_dict(standardizer.to_dict())
    assert np.allclose(restored.mean, standardizer.mean)


def test_native_dataset_and_collate_keep_all_rows_and_pad_only_batch() -> None:
    short = _record("short.csv", [0, 300, 600], anomaly=[0, 0, 1])
    long = _record("long.csv", [0, 300, 600, 900, 1200])
    standardizer = Standardizer.fit([short, long])
    dataset = NativeSequenceDataset(
        [short, long],
        standardizer=standardizer,
        student_horizon_hours=0.5,
        teacher_horizon_hours=0.25,
        offset_hours=0.25,
        alignment_tolerance_seconds=0,
    )

    batch = collate_native_sequences([dataset[0], dataset[1]])

    assert batch["raw"].shape == (2, 5, 12)
    assert batch["raw_mask"].dtype == torch.bool
    assert batch["timestamps"].dtype == torch.int64
    assert batch["lengths"].tolist() == [3, 5]
    assert batch["file_names"] == ["short.csv", "long.csv"]
    assert batch["timestamps"][0, :3].tolist() == [0, 300, 600]
    assert batch["timestamps"][0, 3:].tolist() == [-1, -1]
    assert not bool(batch["loss_mask"][0, 3:].any())
    assert not bool(batch["fgl_mask"][0, 3:].any())


def test_training_dataset_can_drop_only_positionally_postfault_rows() -> None:
    faulty = _record(
        "faulty.csv",
        [0, 300, 600, 600, 900],
        anomaly=[0, 0, 1, 1, 0],
    )
    dataset = NativeSequenceDataset(
        [faulty],
        student_horizon_hours=0.5,
        teacher_horizon_hours=0.25,
        offset_hours=0.25,
        alignment_tolerance_seconds=0,
        truncate_after_first_fault=True,
    )
    sample = dataset[0]
    assert dataset.lengths.tolist() == [3]
    assert sample["timestamps"].tolist() == [0, 300, 600]
    assert sample["loss_mask"].tolist() == [True, True, True]


def test_inference_only_dataset_builds_no_future_targets_or_alignment() -> None:
    faulty = _record("faulty.csv", [0, 300, 600], anomaly=[0, 0, 1])
    dataset = NativeSequenceDataset(
        [faulty],
        student_horizon_hours=0.5,
        teacher_horizon_hours=0.25,
        offset_hours=0.25,
        include_supervision=False,
    )
    sample = dataset[0]
    assert sample["length"] == 3
    assert not bool(sample["loss_mask"].any())
    assert not bool(sample["fgl_mask"].any())
    assert sample["teacher_index"].tolist() == [-1, -1, -1]
    with pytest.raises(ValueError, match="inference-only"):
        NativeSequenceDataset(
            [faulty],
            student_horizon_hours=0.5,
            teacher_horizon_hours=0.25,
            offset_hours=0.25,
            truncate_after_first_fault=True,
            include_supervision=False,
        )


def test_fixed_split_is_seeded_stratified_and_has_no_module_leakage() -> None:
    rows: list[dict[str, int | str]] = []
    for fold in (1, 2, 3):
        for label in (0, 1):
            for index in range(10):
                rows.append(
                    {
                        "file_name": f"module-{fold}-{label}-{index}.csv",
                        "folder_index": fold,
                        "Label": label,
                    }
                )
    index_frame = pd.DataFrame(rows)

    for test_fold in (1, 2, 3):
        first = build_fixed_split(
            index_frame, validation_fraction=0.1, seed=7, test_fold=test_fold
        )
        second = build_fixed_split(
            index_frame, validation_fraction=0.1, seed=7, test_fold=test_fold
        )

        assert first.validation["file_name"].tolist() == second.validation["file_name"].tolist()
        assert set(first.test["folder_index"]) == {test_fold}
        assert set(first.train["folder_index"]) == {1, 2, 3} - {test_fold}
        assert set(first.validation["folder_index"]) <= {1, 2, 3} - {test_fold}
        assert len(first.validation) == 4
        assert first.validation["Label"].value_counts().to_dict() == {0: 2, 1: 2}
        train_names = set(first.train["file_name"])
        validation_names = set(first.validation["file_name"])
        test_names = set(first.test["file_name"])
        assert not train_names & validation_names
        assert not train_names & test_names
        assert not validation_names & test_names


def test_length_bucket_batch_sampler_covers_each_index_once() -> None:
    sampler = LengthBucketBatchSampler(
        [2, 20, 3, 30, 4, 40, 5],
        batch_size=3,
        shuffle=True,
        seed=11,
        bucket_size_multiplier=2,
    )

    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert sorted(index for batch in epoch_zero for index in batch) == list(range(7))
    assert len(epoch_zero) == 3
    assert epoch_zero != epoch_one
