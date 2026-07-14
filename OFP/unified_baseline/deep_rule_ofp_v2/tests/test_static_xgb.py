from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fgofp.data import ModuleRecord, Standardizer
from fgofp.evaluation import PROTOCOL_NAME, evaluate_legacy_inclusive
from fgofp.features import RAW_FEATURES
from fgofp.static_xgb import (
    STATIC_FEATURE_COUNT,
    StaticRows,
    aggregate_static_xgb_threefold,
    build_static_rows,
    fit_xgboost,
    module_normalized_positive_weight,
    predict_xgboost,
    row_sample_weights,
)


def _record(
    name: str,
    timestamps: list[int],
    *,
    fault_index: int | None,
    raw_offset: float = 0.0,
) -> ModuleRecord:
    rows = len(timestamps)
    raw = np.arange(rows * len(RAW_FEATURES), dtype=np.float32).reshape(
        rows, len(RAW_FEATURES)
    )
    raw += float(raw_offset)
    mask = np.ones_like(raw, dtype=bool)
    anomaly = np.zeros(rows, dtype=np.float32)
    if fault_index is not None:
        anomaly[int(fault_index)] = 1.0
    return ModuleRecord(
        file_name=name,
        timestamps=np.asarray(timestamps, dtype=np.int64),
        raw=raw,
        raw_mask=mask,
        anomaly=anomaly,
        first_fault_index=fault_index,
    )


def _identity_standardizer() -> Standardizer:
    dimensions = len(RAW_FEATURES)
    return Standardizer(
        mean=np.zeros(dimensions, dtype=np.float64),
        std=np.ones(dimensions, dtype=np.float64),
        count=np.ones(dimensions, dtype=np.int64),
    )


def test_static_rows_are_current_row_25d_and_module_balanced() -> None:
    faulty = _record("fault.csv", [0, 300, 900], fault_index=2)
    normal = _record("normal.csv", [0, 300], fault_index=None, raw_offset=100.0)
    rows = build_static_rows(
        [faulty, normal],
        _identity_standardizer(),
        horizon_hours=0.1,
        expected_cadence_seconds=300,
        delta_clip_steps=10.0,
        include_long_frame=True,
    )

    assert rows.features.shape == (5, STATIC_FEATURE_COUNT)
    assert rows.features[:, -1].tolist() == pytest.approx([0.0, 1.0, 2.0, 0.0, 1.0])
    assert rows.labels.tolist() == [0, 0, 1, 0, 0]
    assert rows.base_weights[:3].sum() == pytest.approx(1.0)
    assert rows.base_weights[3:].sum() == pytest.approx(1.0)
    assert rows.long_frame is not None
    assert rows.long_frame["file_name"].astype(str).tolist() == [
        "fault.csv",
        "fault.csv",
        "fault.csv",
        "normal.csv",
        "normal.csv",
    ]

    positive_weight = module_normalized_positive_weight(rows, maximum=5.0)
    # Equal module mass gives positive=1/3 and negative=5/3, hence a ratio of 5.
    assert positive_weight == pytest.approx(5.0)
    weights = row_sample_weights(rows, positive_weight)
    assert weights[2] == pytest.approx(5.0 / 3.0)
    assert weights[3:].sum() == pytest.approx(1.0)


def test_small_synthetic_xgboost_fit_returns_probabilities() -> None:
    rng = np.random.default_rng(7)
    train_features = rng.normal(size=(80, STATIC_FEATURE_COUNT)).astype(np.float32)
    train_labels = (train_features[:, 0] + train_features[:, 1] > 0.0).astype(np.int8)
    validation_features = rng.normal(size=(30, STATIC_FEATURE_COUNT)).astype(np.float32)
    validation_labels = (
        validation_features[:, 0] + validation_features[:, 1] > 0.0
    ).astype(np.int8)
    train = StaticRows(
        train_features,
        train_labels,
        np.full(80, 1.0 / 20.0, dtype=np.float32),
        module_count=4,
    )
    validation = StaticRows(
        validation_features,
        validation_labels,
        np.full(30, 1.0 / 10.0, dtype=np.float32),
        module_count=3,
    )

    fit = fit_xgboost(
        train,
        validation,
        positive_weight=1.0,
        device="cpu",
        seed=7,
        nthread=1,
        num_boost_round=8,
        early_stopping_rounds=3,
        max_depth=2,
        eta=0.3,
        max_bin=32,
    )
    probabilities = predict_xgboost(
        fit, validation.features, max_bin=32, nthread=1
    )
    assert probabilities.shape == (30,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def _fold_frame(fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_name": [
                f"f{fold}_fault.csv",
                f"f{fold}_fault.csv",
                f"f{fold}_normal.csv",
                f"f{fold}_normal.csv",
            ],
            "timestamp": [100, 200, 100, 200],
            "anomaly": [0, 1, 0, 0],
            "score": [0.9, 0.9, 0.1, 0.1],
        }
    )


def test_static_threefold_pooling_recomputes_oof_metrics(tmp_path: Path) -> None:
    index_rows: list[dict[str, object]] = []
    for fold in (1, 2, 3):
        fold_dir = tmp_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        evaluated = evaluate_legacy_inclusive(_fold_frame(fold), threshold=0.5)
        evaluated.detail.to_csv(fold_dir / "test_module_decisions.csv", index=False)
        pd.DataFrame(
            [
                {
                    "experiment_id": f"unit_fold{fold}",
                    "variant": "static_xgb_current_row_25d",
                    "test_fold": fold,
                    "protocol": PROTOCOL_NAME,
                    "positive_weight_mode": "module_normalized_auto",
                    "decision_threshold": 0.5,
                    "seconds_total": float(fold),
                    **evaluated.metrics,
                }
            ]
        ).to_csv(fold_dir / "result.csv", index=False)
        pd.DataFrame(
            [
                {
                    "selected": True,
                    "selection_split": "validation",
                    "threshold": 0.5,
                }
            ]
        ).to_csv(fold_dir / "validation_threshold_search.csv", index=False)
        index_rows.extend(
            (
                {"file_name": f"f{fold}_fault.csv", "folder_index": fold, "Label": 1},
                {"file_name": f"f{fold}_normal.csv", "folder_index": fold, "Label": 0},
            )
        )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    comparison = aggregate_static_xgb_threefold(
        tmp_path, index_path, experiment_name="unit_static", formal=False
    )
    assert comparison.loc[0, "evaluation_protocol"] == PROTOCOL_NAME
    assert comparison.loc[0, "module_count"] == 6
    assert comparison.loc[0, "recall"] == pytest.approx(1.0)
    pooled_path = (
        tmp_path / "pooled" / "test_module_decisions_legacy_inclusive.csv"
    )
    pooled = pd.read_csv(pooled_path)
    assert set(pooled["outer_fold"]) == {1, 2, 3}
    assert len(pooled) == 6
    assert (tmp_path / "fold_metrics.csv").is_file()
    assert (tmp_path / "comparison.csv").is_file()


def test_static_xgb_entrypoint_help() -> None:
    base = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-B", str(base / "run_static_xgb_threefold.py"), "--help"],
        cwd=base,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "static current-row XGBoost" in completed.stdout.replace("\n", " ")
