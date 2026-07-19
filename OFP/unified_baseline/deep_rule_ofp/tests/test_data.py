from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
UNIFIED_DIR = PROJECT_DIR.parent
for search_path in (str(UNIFIED_DIR), str(PROJECT_DIR)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from ofp_unified.features import (  # noqa: E402
    MISSING_SENTINEL,
    TEMPERATURE_OUTLIER_SENTINEL,
    build_feature_frame,
)

from drfp.data import (  # noqa: E402
    FailurePredictionDataset,
    ModuleUniformEndpointSampler,
    Standardizer,
    build_endpoint_refs,
    build_fixed_split,
    load_module_records,
    multi_horizon_targets,
    read_module_record,
)
from drfp.features import (  # noqa: E402
    RAW_FEATURES,
    RULE_SIGNAL_DIM,
    RULE_SPECS,
    STATISTICAL_FEATURES,
    build_causal_statistics,
    build_rule_margins,
    causal_hourly_window,
    endpoint_rule_signals,
)


def _module_frame(rows: int, *, fault_row: int | None = None) -> pd.DataFrame:
    timestamps = 1_700_000_000 + np.arange(rows, dtype=np.int64) * 3600
    data: dict[str, np.ndarray] = {"timestamp": timestamps}
    for feature_index, name in enumerate(RAW_FEATURES):
        data[name] = (
            1000.0 + feature_index * 100.0 + np.arange(rows, dtype=np.float64)
        )
    data["temperature"] = 20.0 + np.arange(rows, dtype=np.float64)
    data["current"] = 6000.0 + np.arange(rows, dtype=np.float64)
    anomaly = np.zeros(rows, dtype=np.int8)
    if fault_row is not None:
        anomaly[int(fault_row) :] = 1
    data["anomaly"] = anomaly
    return pd.DataFrame(data)


def test_rule_signed_margins_use_fixed_physical_scales_and_clip() -> None:
    statistics = np.zeros((2, len(STATISTICAL_FEATURES)), dtype=np.float32)
    valid = np.zeros_like(statistics, dtype=bool)
    lookup = {name: index for index, name in enumerate(STATISTICAL_FEATURES)}
    temp_min = lookup["FeTempMin"]
    curr_min = lookup["FeCurrMin"]
    statistics[:, temp_min] = (-20.0, 200.0)
    statistics[:, curr_min] = (4000.0, 6000.0)
    valid[:, [temp_min, curr_min]] = True

    margin, margin_valid = build_rule_margins(statistics, valid)

    assert margin.shape == (2, len(RULE_SPECS))
    assert margin[0, 0] == pytest.approx(2.0)  # (0 - -20) / Temp scale 10
    assert margin[1, 0] == pytest.approx(-10.0)  # -20 clipped from -20
    assert margin[0, 1] == pytest.approx(0.2)  # (5000 - 4000) / 5000
    assert margin[1, 1] == pytest.approx(-0.2)
    assert margin_valid[:, 0].all()
    assert margin_valid[:, 1].all()


def test_causal_statistics_preserve_unified_model2_semantics() -> None:
    """Lock the 76-D formulas/order while using a stricter causal engine."""

    rng = np.random.default_rng(2026)
    raw = rng.normal(
        loc=np.linspace(20.0, 2000.0, len(RAW_FEATURES)),
        scale=np.linspace(2.0, 50.0, len(RAW_FEATURES)),
        size=(32, len(RAW_FEATURES)),
    )
    frame = pd.DataFrame(raw, columns=RAW_FEATURES)
    frame.loc[2, "current"] = MISSING_SENTINEL
    frame.loc[3, "temperature"] = TEMPERATURE_OUTLIER_SENTINEL
    raw = frame.loc[:, RAW_FEATURES].to_numpy(dtype=np.float64)
    raw_valid = np.isfinite(raw) & (raw != float(MISSING_SENTINEL))
    raw_valid[:, 0] &= raw[:, 0] != float(TEMPERATURE_OUTLIER_SENTINEL)

    strict_values, strict_valid = build_causal_statistics(raw, raw_valid)
    unified_values = (
        build_feature_frame(frame, "b1")
        .loc[:, STATISTICAL_FEATURES]
        .to_numpy(dtype=np.float32)
    )

    assert strict_values.shape == (32, len(STATISTICAL_FEATURES))
    np.testing.assert_array_equal(strict_valid, np.isfinite(unified_values))
    np.testing.assert_allclose(
        strict_values,
        unified_values,
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )


def test_endpoint_rule_signals_are_causal_and_have_locked_dimension() -> None:
    timestamps = np.arange(13, dtype=np.int64) * 3600
    margins = np.zeros((13, len(RULE_SPECS)), dtype=np.float32)
    valid = np.zeros_like(margins, dtype=bool)
    margins[:, 0] = np.linspace(-1.0, 1.0, 13)
    valid[:, 0] = True
    record = SimpleNamespace(
        timestamps=timestamps,
        rule_margin=margins,
        rule_valid=valid,
    )

    signals, signal_valid = endpoint_rule_signals(record, 12)

    assert signals.shape == (RULE_SIGNAL_DIM,)
    assert signal_valid.shape == (RULE_SIGNAL_DIM,)
    assert signals[:4] == pytest.approx((1.0, 2.0, 6.0 / 7.0, 1.0))
    assert signal_valid[:4].all()
    assert signals[-4:] == pytest.approx((1.0, 1.0, 2.0, 1.0 / len(RULE_SPECS)))

    future_record = SimpleNamespace(
        timestamps=np.append(timestamps, 13 * 3600),
        rule_margin=np.vstack((margins, np.full((1, len(RULE_SPECS)), 10.0))),
        rule_valid=np.vstack((valid, np.ones((1, len(RULE_SPECS)), dtype=bool))),
    )
    future_signals, future_valid = endpoint_rule_signals(future_record, 12)
    np.testing.assert_array_equal(future_signals, signals)
    np.testing.assert_array_equal(future_valid, signal_valid)


def test_hourly_asof_window_never_uses_future_and_delta_is_single_channel() -> None:
    raw = np.vstack(
        (
            np.arange(len(RAW_FEATURES), dtype=np.float32),
            np.arange(len(RAW_FEATURES), dtype=np.float32) + 100.0,
        )
    )
    record = SimpleNamespace(
        timestamps=np.asarray([0, 2 * 3600], dtype=np.int64),
        raw=raw,
        raw_valid=np.ones_like(raw, dtype=bool),
    )

    window, mask, delta = causal_hourly_window(
        record, 1, history_hours=3, grid_hours=1
    )

    assert window.shape == (3, len(RAW_FEATURES))
    assert mask.shape == window.shape
    assert delta.shape == (3, 1)
    np.testing.assert_array_equal(window[0], raw[0])
    np.testing.assert_array_equal(window[1], raw[0])
    np.testing.assert_array_equal(window[2], raw[1])
    np.testing.assert_allclose(delta[:, 0], (0.0, 1.0, 0.0))


def test_appending_future_row_cannot_change_existing_features(tmp_path: Path) -> None:
    original = _module_frame(14)
    original_path = tmp_path / "original.csv"
    original.to_csv(original_path, index=False)
    original_record = read_module_record(original_path)

    future = _module_frame(15, fault_row=14)
    for name in RAW_FEATURES:
        future.loc[14, name] = 1_000_000.0
    future_path = tmp_path / "future.csv"
    future.to_csv(future_path, index=False)
    future_record = read_module_record(future_path)

    rows = len(original)
    np.testing.assert_array_equal(future_record.raw[:rows], original_record.raw)
    np.testing.assert_array_equal(future_record.raw_valid[:rows], original_record.raw_valid)
    np.testing.assert_allclose(
        future_record.statistics[:rows], original_record.statistics, equal_nan=True
    )
    np.testing.assert_array_equal(
        future_record.stat_valid[:rows], original_record.stat_valid
    )
    np.testing.assert_allclose(
        future_record.rule_margin[:rows], original_record.rule_margin
    )
    np.testing.assert_array_equal(
        future_record.rule_valid[:rows], original_record.rule_valid
    )
    original_window = causal_hourly_window(
        original_record, rows - 1, history_hours=12, grid_hours=1
    )
    future_window = causal_hourly_window(
        future_record, rows - 1, history_hours=12, grid_hours=1
    )
    for left, right in zip(original_window, future_window):
        np.testing.assert_array_equal(left, right)


def test_multi_horizon_labels_are_nested_and_post_failure_is_invalid() -> None:
    timestamps = np.asarray([0, 48 * 3600, 96 * 3600, 104 * 3600, 120 * 3600])
    targets, valid = multi_horizon_targets(timestamps, 120 * 3600)

    np.testing.assert_array_equal(targets[0], (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(targets[1], (0.0, 0.0, 1.0, 1.0))
    np.testing.assert_array_equal(targets[2], (0.0, 1.0, 1.0, 1.0))
    np.testing.assert_array_equal(targets[3], (1.0, 1.0, 1.0, 1.0))
    assert not valid[-1]
    np.testing.assert_array_equal(targets[-1], np.zeros(4, dtype=np.float32))
    assert np.all(np.diff(targets, axis=1) >= 0.0)


def test_fixed_split_reserves_fold3_and_stratifies_one_validation_set() -> None:
    frame = pd.DataFrame(
        {
            "file_name": [f"module_{index}.csv" for index in range(30)],
            "folder_index": np.repeat((1, 2, 3), 10),
            "Label": np.tile((0, 1), 15),
        }
    )
    split = build_fixed_split(frame, validation_fraction=0.1, seed=7)
    repeated = build_fixed_split(frame, validation_fraction=0.1, seed=7)

    assert len(split.train) == 18
    assert len(split.validation) == 2
    assert len(split.test) == 10
    assert set(split.test["folder_index"]) == {3}
    assert set(split.train["folder_index"]) == {1, 2}
    assert set(split.validation["folder_index"]) == {1, 2}
    assert split.validation["Label"].value_counts().to_dict() == {0: 1, 1: 1}
    assert split.validation["file_name"].tolist() == repeated.validation["file_name"].tolist()
    with pytest.raises(ValueError, match="test_fold=3"):
        build_fixed_split(frame, test_fold=2)


def test_standardizer_ignores_invalid_values() -> None:
    values = np.asarray([[1.0, np.nan], [3.0, 100.0]], dtype=np.float32)
    valid = np.asarray([[True, False], [True, False]])
    standardizer = Standardizer.fit_values(values, valid)
    transformed, transformed_valid = standardizer.transform(values, valid)

    np.testing.assert_allclose(standardizer.mean, (2.0, 0.0))
    np.testing.assert_array_equal(standardizer.count, (2, 0))
    np.testing.assert_allclose(transformed[:, 0], (-1.0, 1.0))
    np.testing.assert_array_equal(transformed[:, 1], (0.0, 0.0))
    assert not transformed_valid[:, 1].any()
    restored = Standardizer.from_dict(standardizer.to_dict())
    np.testing.assert_array_equal(restored.count, standardizer.count)


def test_dataset_shapes_targets_and_module_uniform_sampler(tmp_path: Path) -> None:
    first = _module_frame(16, fault_row=15)
    second = _module_frame(16)
    first.to_csv(tmp_path / "first.csv", index=False)
    second.to_csv(tmp_path / "second.csv", index=False)
    records = load_module_records(tmp_path, ["first.csv", "second.csv"])
    endpoints = build_endpoint_refs(records, endpoints_per_module=2)
    raw_standardizer = Standardizer.fit(records, field="raw")
    stat_standardizer = Standardizer.fit(records, field="statistics")
    dataset = FailurePredictionDataset(
        records,
        endpoints,
        raw_standardizer=raw_standardizer,
        stat_standardizer=stat_standardizer,
    )

    assert dataset.target_matrix.shape == (4, 4)
    np.testing.assert_array_equal(dataset.targets_array(), dataset.target_matrix)
    item = dataset[0]
    assert tuple(item["raw"].shape) == (168, len(RAW_FEATURES))
    assert tuple(item["raw_mask"].shape) == (168, len(RAW_FEATURES))
    assert tuple(item["delta_hours"].shape) == (168, 1)
    assert tuple(item["statistics"].shape) == (len(STATISTICAL_FEATURES),)
    assert tuple(item["rule"].shape) == (RULE_SIGNAL_DIM,)
    assert tuple(item["targets"].shape) == (4,)
    assert item["valid_training_endpoint"] is True
    assert item["file_name"] == "first.csv"

    sampler = ModuleUniformEndpointSampler(dataset, num_samples=4, seed=3)
    sampled = list(sampler)
    module_counts = Counter(dataset.endpoints[index].module_index for index in sampled)
    assert module_counts == {0: 2, 1: 2}
    sampler.set_epoch(1)
    assert len(list(sampler)) == 4


def test_short_modules_repeat_endpoints_to_keep_exact_module_quota(tmp_path: Path) -> None:
    _module_frame(2).to_csv(tmp_path / "short.csv", index=False)
    _module_frame(8).to_csv(tmp_path / "long.csv", index=False)
    records = load_module_records(tmp_path, ["short.csv", "long.csv"])
    endpoints = build_endpoint_refs(records, endpoints_per_module=4)
    counts = Counter(reference.module_index for reference in endpoints)
    assert counts == {0: 4, 1: 4}


def test_dataset_skips_unused_branch_inputs(tmp_path: Path) -> None:
    _module_frame(8).to_csv(tmp_path / "module.csv", index=False)
    records = load_module_records(tmp_path, ["module.csv"])
    endpoints = build_endpoint_refs(records, endpoints_per_module=1)
    rule_only = FailurePredictionDataset(
        records, endpoints, input_variant="rule_only"
    )[0]
    temporal_only = FailurePredictionDataset(
        records, endpoints, input_variant="temporal_only"
    )[0]
    assert tuple(rule_only["raw"].shape) == (1, 1)
    assert tuple(rule_only["rule"].shape) == (RULE_SIGNAL_DIM,)
    assert tuple(temporal_only["raw"].shape) == (168, len(RAW_FEATURES))
    assert tuple(temporal_only["rule"].shape) == (1,)
