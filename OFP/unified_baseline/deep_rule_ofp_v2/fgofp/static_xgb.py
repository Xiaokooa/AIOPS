"""Leakage-safe static XGBoost control under the FORT data protocol.

The control deliberately sees only the current native observation.  Its 25
inputs are the 12 train-standardized raw values, their 12 validity masks, and
the observed time gap from the preceding row.  It shares the outer folds,
inner validation split, 120-hour Legacy-Inclusive labels, module-balanced row
weights, validation-only threshold selection, and module-level evaluation of
the deep model.  Consequently its difference from FORT is the absence of a
temporal encoder and RuleModel inputs, rather than a protocol difference.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    DEFAULT_EXPECTED_CADENCE_SECONDS,
    ModuleRecord,
    SplitManifest,
    Standardizer,
    build_fixed_split,
    legacy_inclusive_targets,
    native_delta_steps,
    read_module_record,
)
from .evaluation import (
    PROTOCOL_NAME,
    evaluate_legacy_inclusive,
    metrics_from_module_decisions,
    search_validation_threshold,
)
from .features import RAW_FEATURES


OUTER_FOLDS = (1, 2, 3)
STATIC_FEATURE_COUNT = len(RAW_FEATURES) * 2 + 1
EXPECTED_FULL_MODULES = 13_372
EXPECTED_FULL_FAULTY = 4_102
EXPECTED_FULL_NORMAL = 9_270


@dataclass(frozen=True)
class StaticRows:
    """Flattened native rows while retaining equal module mass."""

    features: np.ndarray
    labels: np.ndarray
    base_weights: np.ndarray
    module_count: int
    long_frame: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        labels = np.asarray(self.labels)
        weights = np.asarray(self.base_weights)
        if features.ndim != 2 or features.shape[1] != STATIC_FEATURE_COUNT:
            raise ValueError(
                f"features must have shape [rows, {STATIC_FEATURE_COUNT}]"
            )
        if labels.shape != (len(features),) or weights.shape != (len(features),):
            raise ValueError("labels and base_weights must align with features")
        if not np.isfinite(features).all() or not np.isfinite(weights).all():
            raise ValueError("static rows must be finite")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("static labels must be binary")
        if (weights <= 0.0).any() or not int(self.module_count) > 0:
            raise ValueError("every row and module must have positive mass")
        if self.long_frame is not None and len(self.long_frame) != len(features):
            raise ValueError("long_frame must align with flattened features")


@dataclass(frozen=True)
class XGBoostFit:
    booster: Any
    training_matrix: Any
    evaluation_history: Mapping[str, Mapping[str, Sequence[float]]]
    best_iteration: int


def _prefix_length(record: ModuleRecord) -> int:
    return (
        record.length
        if record.first_fault_index is None
        else int(record.first_fault_index) + 1
    )


def current_row_features(
    record: ModuleRecord,
    standardizer: Standardizer,
    *,
    expected_cadence_seconds: int = DEFAULT_EXPECTED_CADENCE_SECONDS,
    delta_clip_steps: float | None = 288.0,
) -> np.ndarray:
    """Return the 25 current-row inputs through the first failure row."""

    stop = _prefix_length(record)
    normalized, valid = standardizer.transform(
        record.raw[:stop], record.raw_mask[:stop]
    )
    delta = native_delta_steps(
        record.timestamps[:stop],
        expected_cadence_seconds=expected_cadence_seconds,
        clip_steps=delta_clip_steps,
    )
    features = np.concatenate(
        (normalized, valid.astype(np.float32), delta.astype(np.float32)), axis=1
    ).astype(np.float32, copy=False)
    if features.shape != (stop, STATIC_FEATURE_COUNT):
        raise RuntimeError("current-row feature construction changed shape")
    return features


def build_static_rows(
    records: Iterable[ModuleRecord],
    standardizer: Standardizer,
    *,
    horizon_hours: float = 120.0,
    expected_cadence_seconds: int = DEFAULT_EXPECTED_CADENCE_SECONDS,
    delta_clip_steps: float | None = 288.0,
    include_long_frame: bool = False,
) -> StaticRows:
    """Flatten module prefixes without allowing long modules to dominate."""

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    anomaly_parts: list[np.ndarray] = []
    module_code_parts: list[np.ndarray] = []
    module_names: list[str] = []

    for module_code, record in enumerate(records):
        if record.file_name in module_names:
            raise ValueError(f"duplicate module record: {record.file_name}")
        stop = _prefix_length(record)
        targets, loss_mask = legacy_inclusive_targets(
            record.timestamps,
            record.first_fault_index,
            horizon_hours,
        )
        if not bool(loss_mask[:stop].all()) or bool(loss_mask[stop:].any()):
            raise RuntimeError("Legacy-Inclusive prefix and target mask disagree")
        feature_parts.append(
            current_row_features(
                record,
                standardizer,
                expected_cadence_seconds=expected_cadence_seconds,
                delta_clip_steps=delta_clip_steps,
            )
        )
        label_parts.append(targets[:stop].astype(np.int8, copy=False))
        weight_parts.append(np.full(stop, 1.0 / float(stop), dtype=np.float32))
        module_names.append(record.file_name)
        if include_long_frame:
            timestamp_parts.append(record.timestamps[:stop].astype(np.int64, copy=False))
            anomaly_parts.append(record.anomaly[:stop].astype(np.float32, copy=False))
            module_code_parts.append(np.full(stop, module_code, dtype=np.int32))

    if not feature_parts:
        raise ValueError("at least one module record is required")
    features = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    base_weights = np.concatenate(weight_parts, axis=0)
    long_frame: pd.DataFrame | None = None
    if include_long_frame:
        module_codes = np.concatenate(module_code_parts, axis=0)
        long_frame = pd.DataFrame(
            {
                "file_name": pd.Categorical.from_codes(
                    module_codes, categories=module_names, ordered=False
                ),
                "timestamp": np.concatenate(timestamp_parts, axis=0),
                "anomaly": np.concatenate(anomaly_parts, axis=0),
            }
        )
    return StaticRows(
        features=features,
        labels=labels,
        base_weights=base_weights,
        module_count=len(module_names),
        long_frame=long_frame,
    )


def module_normalized_positive_weight(
    rows: StaticRows,
    *,
    maximum: float = 5.0,
) -> float:
    """Match FORT's class ratio in the equal-module loss measure."""

    if not math.isfinite(float(maximum)) or float(maximum) < 1.0:
        raise ValueError("maximum positive weight must be finite and at least one")
    positive = float(rows.base_weights[rows.labels == 1].sum(dtype=np.float64))
    negative = float(rows.base_weights[rows.labels == 0].sum(dtype=np.float64))
    if positive <= 0.0:
        return 1.0
    return float(np.clip(negative / positive, 1.0, float(maximum)))


def row_sample_weights(rows: StaticRows, positive_weight: float) -> np.ndarray:
    weight = float(positive_weight)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("positive_weight must be finite and positive")
    multiplier = np.where(rows.labels == 1, weight, 1.0)
    return (rows.base_weights.astype(np.float64) * multiplier).astype(np.float32)


def fit_streaming_standardizer(records: Iterable[ModuleRecord]) -> Standardizer:
    """Fit the same train-only moments without retaining rule caches in RAM."""

    dimensions = len(RAW_FEATURES)
    count = np.zeros(dimensions, dtype=np.int64)
    total = np.zeros(dimensions, dtype=np.float64)
    square_total = np.zeros(dimensions, dtype=np.float64)
    module_count = 0
    for record in records:
        module_count += 1
        stop = _prefix_length(record)
        valid = record.raw_mask[:stop]
        values = record.raw[:stop].astype(np.float64)
        safe = np.where(valid, values, 0.0)
        count += valid.sum(axis=0, dtype=np.int64)
        total += safe.sum(axis=0, dtype=np.float64)
        square_total += np.square(safe).sum(axis=0, dtype=np.float64)
    if module_count == 0:
        raise ValueError("cannot fit Standardizer on no training modules")
    divisor = np.maximum(count, 1)
    mean = total / divisor
    variance = np.maximum(square_total / divisor - np.square(mean), 0.0)
    std = np.sqrt(variance)
    mean[(count == 0) | ~np.isfinite(mean)] = 0.0
    std[(count < 2) | ~np.isfinite(std) | (std < 1e-8)] = 1.0
    return Standardizer(mean=mean, std=std, count=count)


def _iter_records(
    data_dir: Path,
    membership: pd.DataFrame,
    *,
    split_name: str,
    progress: bool,
) -> Iterable[ModuleRecord]:
    expected = {
        str(row.file_name): bool(int(row.Label) > 0)
        for row in membership.itertuples(index=False)
    }
    total = len(membership)
    for position, file_name in enumerate(membership["file_name"].astype(str), 1):
        path = data_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f"missing {split_name} module: {path}")
        record = read_module_record(path)
        if (record.first_fault_index is not None) != expected[file_name]:
            raise ValueError(f"{split_name} index/CSV label mismatch: {file_name}")
        yield record
        if progress and (position % 100 == 0 or position == total):
            print(f"  [{split_name}] loaded {position}/{total} modules", flush=True)


def _limit_stratified(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or int(maximum) >= len(frame):
        return frame.copy().reset_index(drop=True)
    maximum = int(maximum)
    if maximum < 2:
        raise ValueError("module limits must retain at least two modules")
    faulty = frame.loc[pd.to_numeric(frame["Label"], errors="raise") > 0]
    normal = frame.loc[pd.to_numeric(frame["Label"], errors="raise") <= 0]
    faulty_count = min(
        len(faulty), max(1, int(round(maximum * len(faulty) / len(frame))))
    )
    normal_count = min(len(normal), maximum - faulty_count)
    if faulty_count + normal_count < maximum:
        faulty_count = min(len(faulty), maximum - normal_count)
    if faulty_count <= 0 or normal_count <= 0:
        raise ValueError("limited split must retain faulty and normal modules")
    return (
        pd.concat((faulty.head(faulty_count), normal.head(normal_count)))
        .sort_index()
        .head(maximum)
        .reset_index(drop=True)
    )


def _limited_split(
    split: SplitManifest,
    *,
    max_train_modules: int | None,
    max_validation_modules: int | None,
    max_test_modules: int | None,
) -> SplitManifest:
    return SplitManifest(
        train=_limit_stratified(split.train, max_train_modules),
        validation=_limit_stratified(split.validation, max_validation_modules),
        test=_limit_stratified(split.test, max_test_modules),
    )


def resolve_xgboost_device(requested: str) -> str:
    """Resolve against XGBoost itself, including its silent CUDA fallback.

    XGBoost can be installed without CUDA even when PyTorch sees a GPU, and
    some builds warn and silently switch ``device=cuda`` back to CPU.  A tiny
    one-tree probe plus the saved booster configuration is therefore a more
    reliable capability check than consulting another framework.
    """

    requested = str(requested)
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import warnings

        import xgboost as xgb

        probe_features = np.asarray([[0.0], [1.0]], dtype=np.float32)
        probe_labels = np.asarray([0, 1], dtype=np.int8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe_matrix = xgb.QuantileDMatrix(
                probe_features, label=probe_labels, max_bin=4, nthread=1
            )
            probe = xgb.train(
                {
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    "device": "cuda",
                    "max_depth": 1,
                    "max_bin": 4,
                    "nthread": 1,
                },
                probe_matrix,
                num_boost_round=1,
                verbose_eval=False,
            )
        configured = str(
            json.loads(probe.save_config())["learner"]["generic_param"]["device"]
        ).lower()
        cuda_available = configured.startswith("cuda")
    except Exception as error:
        cuda_available = False
        probe_error = error
    else:
        probe_error = None
    if requested == "cuda" and not cuda_available:
        detail = f": {probe_error}" if probe_error is not None else ""
        raise RuntimeError(
            "--device cuda requested, but XGBoost could not activate a CUDA "
            f"device and would fall back to CPU{detail}"
        )
    return "cuda" if cuda_available else "cpu"


def fit_xgboost(
    train_rows: StaticRows,
    validation_rows: StaticRows,
    *,
    positive_weight: float,
    device: str,
    seed: int = 42,
    nthread: int = 4,
    num_boost_round: int = 100,
    early_stopping_rounds: int = 20,
    max_depth: int = 6,
    eta: float = 0.3,
    max_bin: int = 256,
) -> XGBoostFit:
    """Fit binary-logistic XGBoost with module-balanced row weights."""

    import xgboost as xgb

    if len(np.unique(train_rows.labels)) != 2:
        raise ValueError("XGBoost training rows must contain both classes")
    if int(num_boost_round) <= 0 or int(early_stopping_rounds) <= 0:
        raise ValueError("boosting and early-stopping rounds must be positive")
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": str(device),
        "max_depth": int(max_depth),
        "eta": float(eta),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "max_bin": int(max_bin),
        "seed": int(seed),
        "nthread": int(nthread),
    }
    train_matrix = xgb.QuantileDMatrix(
        train_rows.features,
        label=train_rows.labels,
        weight=row_sample_weights(train_rows, positive_weight),
        max_bin=int(max_bin),
        nthread=int(nthread),
    )
    validation_matrix = xgb.QuantileDMatrix(
        validation_rows.features,
        label=validation_rows.labels,
        weight=row_sample_weights(validation_rows, positive_weight),
        max_bin=int(max_bin),
        nthread=int(nthread),
        ref=train_matrix,
    )
    history: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        parameters,
        train_matrix,
        num_boost_round=int(num_boost_round),
        evals=((train_matrix, "train"), (validation_matrix, "validation")),
        evals_result=history,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    best = booster.attr("best_iteration")
    best_iteration = int(best) if best is not None else booster.num_boosted_rounds() - 1
    return XGBoostFit(
        booster=booster,
        training_matrix=train_matrix,
        evaluation_history=history,
        best_iteration=best_iteration,
    )


def predict_xgboost(
    fit: XGBoostFit,
    features: np.ndarray,
    *,
    max_bin: int = 256,
    nthread: int = 4,
) -> np.ndarray:
    import xgboost as xgb

    matrix = xgb.QuantileDMatrix(
        np.asarray(features, dtype=np.float32),
        max_bin=int(max_bin),
        nthread=int(nthread),
        ref=fit.training_matrix,
    )
    scores = fit.booster.predict(
        matrix, iteration_range=(0, int(fit.best_iteration) + 1)
    )
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(features),) or not np.isfinite(scores).all():
        raise RuntimeError("XGBoost returned invalid probabilities")
    return np.clip(scores, 0.0, 1.0)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _result_frame(rows: StaticRows, scores: np.ndarray) -> pd.DataFrame:
    if rows.long_frame is None:
        raise ValueError("evaluation rows require a long_frame")
    frame = rows.long_frame.copy()
    frame["score"] = np.asarray(scores, dtype=np.float64)
    return frame


def _split_manifest(split: SplitManifest) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for name, frame in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        part = frame.loc[:, ["file_name", "folder_index", "Label"]].copy()
        part["split"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_static_xgb_fold(
    *,
    data_dir: Path | str,
    index_path: Path | str,
    output_dir: Path | str,
    test_fold: int,
    experiment_name: str,
    device: str = "auto",
    seed: int = 42,
    validation_fraction: float = 0.1,
    horizon_hours: float = 120.0,
    expected_cadence_seconds: int = DEFAULT_EXPECTED_CADENCE_SECONDS,
    delta_clip_steps: float | None = 288.0,
    positive_weight_mode: str = "module_normalized_auto",
    max_positive_weight: float = 5.0,
    fixed_threshold: float = 0.5,
    threshold_grid_size: int = 201,
    threshold_quantile_count: int = 201,
    num_boost_round: int = 100,
    early_stopping_rounds: int = 20,
    max_depth: int = 6,
    eta: float = 0.3,
    max_bin: int = 256,
    nthread: int = 4,
    max_train_modules: int | None = None,
    max_validation_modules: int | None = None,
    max_test_modules: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Train/evaluate one outer fold without any RuleModel fallback."""

    start = time.perf_counter()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; pass --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = build_fixed_split(
        index_path,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
        test_fold=int(test_fold),
    )
    split = _limited_split(
        split,
        max_train_modules=max_train_modules,
        max_validation_modules=max_validation_modules,
        max_test_modules=max_test_modules,
    )
    resolved_device = resolve_xgboost_device(device)

    print(f"[static-xgb fold {test_fold}] fitting train-only normalizer", flush=True)
    standardizer = fit_streaming_standardizer(
        _iter_records(data_dir, split.train, split_name="normalizer", progress=progress)
    )
    print(f"[static-xgb fold {test_fold}] building train rows", flush=True)
    train_rows = build_static_rows(
        _iter_records(data_dir, split.train, split_name="train", progress=progress),
        standardizer,
        horizon_hours=horizon_hours,
        expected_cadence_seconds=expected_cadence_seconds,
        delta_clip_steps=delta_clip_steps,
    )
    validation_rows = build_static_rows(
        _iter_records(
            data_dir, split.validation, split_name="validation", progress=progress
        ),
        standardizer,
        horizon_hours=horizon_hours,
        expected_cadence_seconds=expected_cadence_seconds,
        delta_clip_steps=delta_clip_steps,
        include_long_frame=True,
    )
    if positive_weight_mode == "module_normalized_auto":
        positive_weight = module_normalized_positive_weight(
            train_rows, maximum=max_positive_weight
        )
    elif positive_weight_mode == "none":
        positive_weight = 1.0
    else:
        raise ValueError("positive_weight_mode must be none or module_normalized_auto")

    fit = fit_xgboost(
        train_rows,
        validation_rows,
        positive_weight=positive_weight,
        device=resolved_device,
        seed=seed,
        nthread=nthread,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        max_depth=max_depth,
        eta=eta,
        max_bin=max_bin,
    )
    validation_scores = predict_xgboost(
        fit, validation_rows.features, max_bin=max_bin, nthread=nthread
    )
    validation_frame = _result_frame(validation_rows, validation_scores)
    threshold_search = search_validation_threshold(
        validation_frame,
        score_col="score",
        fixed_threshold=fixed_threshold,
        grid_size=threshold_grid_size,
        quantile_count=threshold_quantile_count,
        split_name="validation",
    )
    selected_thresholds = threshold_search.table.loc[
        threshold_search.table["selected"].astype(bool), "threshold"
    ]
    if len(selected_thresholds) != 1 or not np.isclose(
        float(selected_thresholds.iloc[0]),
        float(threshold_search.threshold),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("validation threshold audit table and selection disagree")

    print(f"[static-xgb fold {test_fold}] scoring outer test fold", flush=True)
    test_rows = build_static_rows(
        _iter_records(data_dir, split.test, split_name="test", progress=progress),
        standardizer,
        horizon_hours=horizon_hours,
        expected_cadence_seconds=expected_cadence_seconds,
        delta_clip_steps=delta_clip_steps,
        include_long_frame=True,
    )
    test_scores = predict_xgboost(
        fit, test_rows.features, max_bin=max_bin, nthread=nthread
    )
    test_frame = _result_frame(test_rows, test_scores)
    test_result = evaluate_legacy_inclusive(
        test_frame, threshold=threshold_search.threshold, score_col="score"
    )
    seconds_total = float(time.perf_counter() - start)

    result: dict[str, Any] = {
        "experiment_id": experiment_name,
        "variant": "static_xgb_current_row_25d",
        "changed_axis": "temporal_encoder",
        "test_fold": int(test_fold),
        "protocol": PROTOCOL_NAME,
        "native_rows": True,
        "sequence_to_sequence": False,
        "teacher_used_at_inference": False,
        "decision_feature_count": STATIC_FEATURE_COUNT,
        "model_architecture": "xgboost_current_row",
        "safe_rule_fallback": False,
        "decision_source": "static_xgb_current_row",
        "training_rows": int(len(train_rows.features)),
        "training_positive_rows": int(train_rows.labels.sum()),
        "training_modules": int(train_rows.module_count),
        "positive_weight_mode": positive_weight_mode,
        "positive_weight": float(positive_weight),
        "model_rounds": int(fit.booster.num_boosted_rounds()),
        "best_iteration": int(fit.best_iteration),
        "decision_threshold": float(threshold_search.threshold),
        "threshold_policy": "validation_selected",
        "seconds_total": seconds_total,
        **test_result.metrics,
    }
    # The evaluator also reports the decision threshold; keep the frozen
    # validation value explicit after merging its metrics.
    result["decision_threshold"] = float(threshold_search.threshold)
    result["decision_source"] = "static_xgb_current_row"

    _split_manifest(split).to_csv(output_dir / "split_manifest.csv", index=False)
    threshold_search.table.to_csv(
        output_dir / "validation_threshold_search.csv", index=False
    )
    threshold_search.detail.to_csv(
        output_dir / "validation_module_decisions.csv", index=False
    )
    test_result.detail.to_csv(output_dir / "test_module_decisions.csv", index=False)
    pd.DataFrame([result]).to_csv(output_dir / "result.csv", index=False)
    history_columns = {
        f"{dataset}_{metric}": list(values)
        for dataset, metrics in fit.evaluation_history.items()
        for metric, values in metrics.items()
    }
    pd.DataFrame(history_columns).to_csv(
        output_dir / "training_history.csv", index_label="iteration"
    )
    fit.booster.save_model(str(output_dir / "model.json"))
    _write_json(output_dir / "normalizer.json", standardizer.to_dict())
    _write_json(
        output_dir / "run_manifest.json",
        {
            "experiment_id": experiment_name,
            "protocol": PROTOCOL_NAME,
            "test_fold": int(test_fold),
            "current_row_only": True,
            "static_feature_count": STATIC_FEATURE_COUNT,
            "static_feature_order": [
                *[f"raw_z::{name}" for name in RAW_FEATURES],
                *[f"valid::{name}" for name in RAW_FEATURES],
                "delta_steps",
            ],
            "horizon_hours": float(horizon_hours),
            "positive_weight_mode": positive_weight_mode,
            "positive_weight": float(positive_weight),
            "threshold_selection": "inner_validation_only",
            "safe_rule_fallback": False,
            "device": resolved_device,
            "xgboost": {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "max_depth": int(max_depth),
                "eta": float(eta),
                "max_bin": int(max_bin),
                "num_boost_round": int(num_boost_round),
                "early_stopping_rounds": int(early_stopping_rounds),
                "best_iteration": int(fit.best_iteration),
            },
        },
    )
    return result


def _one_row(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"expected exactly one result row in {path}")
    return frame.iloc[0].to_dict()


def aggregate_static_xgb_threefold(
    output_dir: Path | str,
    index_path: Path | str,
    *,
    experiment_name: str,
    formal: bool,
) -> pd.DataFrame:
    """Concatenate three outer-fold decisions and recompute pooled metrics."""

    output_dir = Path(output_dir)
    index = pd.read_csv(index_path)
    required = {"file_name", "folder_index", "Label"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"index missing columns: {sorted(missing)}")
    index = index.copy()
    index["file_name"] = index["file_name"].astype(str)
    index["folder_index"] = pd.to_numeric(
        index["folder_index"], errors="raise"
    ).astype(int)
    if index["file_name"].duplicated().any():
        raise ValueError("index contains duplicate module names")
    if set(index["folder_index"]) != set(OUTER_FOLDS):
        raise ValueError("static XGBoost aggregation requires outer folds 1, 2, 3")

    pooled_parts: list[pd.DataFrame] = []
    fold_results: list[dict[str, Any]] = []
    thresholds: dict[int, float] = {}
    for fold in OUTER_FOLDS:
        fold_dir = output_dir / f"fold_{fold}"
        detail = pd.read_csv(fold_dir / "test_module_decisions.csv")
        if "file_name" not in detail or detail["file_name"].astype(str).duplicated().any():
            raise ValueError(f"invalid module coverage in {fold_dir}")
        expected = set(
            index.loc[index["folder_index"] == fold, "file_name"].astype(str)
        )
        actual = set(detail["file_name"].astype(str))
        if formal and actual != expected:
            raise ValueError(f"fold {fold} does not cover its complete outer test fold")
        result = _one_row(fold_dir / "result.csv")
        if int(result.get("test_fold", -1)) != fold:
            raise ValueError(f"fold identity mismatch in {fold_dir / 'result.csv'}")
        if str(result.get("protocol")) != PROTOCOL_NAME:
            raise ValueError(f"protocol mismatch in {fold_dir / 'result.csv'}")
        threshold_table = pd.read_csv(
            fold_dir / "validation_threshold_search.csv"
        )
        if "selected" not in threshold_table or "selection_split" not in threshold_table:
            raise ValueError(f"threshold provenance missing in {fold_dir}")
        selected_mask = threshold_table["selected"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        selected = threshold_table.loc[selected_mask]
        if (
            len(selected) != 1
            or str(selected.iloc[0]["selection_split"]) != "validation"
            or not np.isclose(
                float(selected.iloc[0]["threshold"]),
                float(result["decision_threshold"]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"invalid validation threshold provenance in {fold_dir}")
        recomputed = metrics_from_module_decisions(detail, protocol=PROTOCOL_NAME)
        for metric in (
            "final_score",
            "f1_score",
            "precision",
            "recall",
            "accuracy",
            "avg_lead_hour",
        ):
            if not np.isclose(
                float(result[metric]), float(recomputed[metric]), rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"result/detail mismatch for {metric} in fold {fold}")
        result["outer_fold"] = fold
        fold_results.append(result)
        thresholds[fold] = float(result["decision_threshold"])
        detail.insert(0, "outer_fold", fold)
        pooled_parts.append(detail)

    pooled = pd.concat(pooled_parts, ignore_index=True)
    if pooled["file_name"].astype(str).duplicated().any():
        raise ValueError("outer test folds overlap in pooled static decisions")
    if formal:
        all_names = set(index["file_name"].astype(str))
        faulty = int((pd.to_numeric(index["Label"], errors="raise") > 0).sum())
        if set(pooled["file_name"].astype(str)) != all_names:
            raise ValueError("pooled static decisions do not cover the full index")
        if (
            len(index) != EXPECTED_FULL_MODULES
            or faulty != EXPECTED_FULL_FAULTY
            or len(index) - faulty != EXPECTED_FULL_NORMAL
        ):
            raise ValueError("formal index does not match the frozen OFP universe")

    metrics = metrics_from_module_decisions(pooled, protocol=PROTOCOL_NAME)
    folds = pd.DataFrame(fold_results).sort_values("outer_fold").reset_index(drop=True)
    if folds["positive_weight_mode"].astype(str).nunique() != 1:
        raise ValueError("static XGBoost folds disagree on positive_weight_mode")
    comparison_row: dict[str, Any] = {
        "experiment_id": experiment_name,
        "variant": "static_xgb_current_row_25d",
        "changed_axis": "temporal_encoder",
        "training_protocol": PROTOCOL_NAME,
        "evaluation_protocol": PROTOCOL_NAME,
        "outer_folds": "1,2,3",
        "training_runs": 3,
        "formal_full_oof": bool(formal),
        "decision_threshold": np.nan,
        "threshold_policy": "per_fold_validation_selected",
        "fold_1_threshold": thresholds[1],
        "fold_2_threshold": thresholds[2],
        "fold_3_threshold": thresholds[3],
        "native_rows": True,
        "sequence_to_sequence": False,
        "teacher_used_at_inference": False,
        "decision_feature_count": STATIC_FEATURE_COUNT,
        "model_architecture": "xgboost_current_row",
        "decision_source": "static_xgb_current_row",
        "safe_rule_fallback": False,
        "positive_weight_mode": str(folds["positive_weight_mode"].iloc[0]),
        "seconds_total": float(
            pd.to_numeric(folds["seconds_total"], errors="raise").sum()
        ),
        **metrics,
    }
    comparison = pd.DataFrame([comparison_row])
    pooled_dir = output_dir / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(
        pooled_dir / "test_module_decisions_legacy_inclusive.csv", index=False
    )
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    index_payload = index.loc[:, ["file_name", "folder_index", "Label"]].sort_values(
        "file_name"
    ).to_csv(index=False)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "experiment_id": experiment_name,
            "formal_full_oof": bool(formal),
            "outer_folds": list(OUTER_FOLDS),
            "index_sha256": hashlib.sha256(
                index_payload.encode("utf-8")
            ).hexdigest(),
            "module_count": int(metrics["module_count"]),
            "threshold_by_fold": {
                str(fold): float(value) for fold, value in thresholds.items()
            },
            "aggregation": "concatenate_oof_module_decisions_then_recompute",
            "fold_metrics_averaged": False,
            "safe_rule_fallback": False,
            "result": dict(metrics),
        },
    )
    return comparison


def run_static_xgb_threefold(
    *,
    data_dir: Path | str,
    index_path: Path | str,
    output_dir: Path | str,
    experiment_name: str = "S0_static_xgb_current_row_3fold",
    device: str = "auto",
    smoke: bool = False,
    overwrite: bool = False,
    seed: int = 42,
    nthread: int = 4,
    num_boost_round: int = 100,
    early_stopping_rounds: int = 20,
    max_train_modules: int | None = None,
    max_validation_modules: int | None = None,
    max_test_modules: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; pass --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        max_train_modules = 256 if max_train_modules is None else max_train_modules
        max_validation_modules = (
            64 if max_validation_modules is None else max_validation_modules
        )
        max_test_modules = 64 if max_test_modules is None else max_test_modules
        num_boost_round = min(int(num_boost_round), 20)
        early_stopping_rounds = min(int(early_stopping_rounds), 5)

    for fold in OUTER_FOLDS:
        print(f"[static-xgb] running outer fold {fold}", flush=True)
        run_static_xgb_fold(
            data_dir=data_dir,
            index_path=index_path,
            output_dir=output_dir / f"fold_{fold}",
            test_fold=fold,
            experiment_name=f"{experiment_name}_fold{fold}",
            device=device,
            seed=seed,
            nthread=nthread,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            max_train_modules=max_train_modules,
            max_validation_modules=max_validation_modules,
            max_test_modules=max_test_modules,
            overwrite=False,
            progress=progress,
            threshold_grid_size=21 if smoke else 201,
            threshold_quantile_count=21 if smoke else 201,
        )
    return aggregate_static_xgb_threefold(
        output_dir,
        index_path,
        experiment_name=experiment_name,
        formal=not smoke,
    )


__all__ = [
    "EXPECTED_FULL_FAULTY",
    "EXPECTED_FULL_MODULES",
    "EXPECTED_FULL_NORMAL",
    "OUTER_FOLDS",
    "STATIC_FEATURE_COUNT",
    "StaticRows",
    "XGBoostFit",
    "aggregate_static_xgb_threefold",
    "build_static_rows",
    "current_row_features",
    "fit_streaming_standardizer",
    "fit_xgboost",
    "module_normalized_positive_weight",
    "predict_xgboost",
    "resolve_xgboost_device",
    "row_sample_weights",
    "run_static_xgb_fold",
    "run_static_xgb_threefold",
]
