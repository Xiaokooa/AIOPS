"""Two-stage clean HTSF fold pipeline and pooled official evaluation."""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from ofp_unified.b0 import booster_effective_device, config_fingerprint, resolve_device
from ofp_unified.evaluation import (
    evaluate_prediction_dir,
    metrics_from_decisions,
    write_evaluation,
)

from .config import HTSFConfig, fingerprint
from .data import Standardizer, iter_full_module_batches
from .endpoints import endpoint_fingerprint, weight_fingerprint
from .model import HTSFEncoder
from .training import (
    decision_matrix_for_batch,
    extract_training_table,
    resolve_torch_device,
    train_encoder,
)
from .variants import VariantSpec


def _finite_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(matrix), matrix, np.nan).astype(np.float32)


def implementation_fingerprint() -> str:
    records = []
    package_dir = Path(__file__).resolve().parent
    paths = list(package_dir.glob("*.py"))
    runner = package_dir.parent / "run_htsf.py"
    if runner.is_file():
        paths.append(runner)
    for path in sorted(paths):
        records.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return fingerprint(records)


def runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "xgboost": xgb.__version__,
    }


def _train_xgboost(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    feature_names: Sequence[str],
    config: HTSFConfig,
) -> tuple[xgb.Booster, str]:
    positive = int(np.sum(target > 0))
    if positive <= 0 or positive >= len(target):
        raise ValueError(
            f"XGBoost requires both classes; rows={len(target)} positive={positive}. "
            "Increase smoke modules or use HSS."
        )
    xgb_config = dict(config.decision.xgboost)
    rounds = int(xgb_config.pop("num_boost_round"))
    requested_device = str(xgb_config.get("device", "auto"))
    device = resolve_device(requested_device)
    xgb_config["device"] = device
    sample_weight = weight if config.decision.use_sample_weight else None
    dtrain = xgb.QuantileDMatrix(
        _finite_matrix(matrix),
        label=target,
        weight=sample_weight,
        feature_names=list(feature_names),
        nthread=int(xgb_config.get("nthread", 4)),
        max_bin=int(xgb_config.get("max_bin", 256)),
    )
    booster = xgb.train(xgb_config, dtrain, num_boost_round=rounds, verbose_eval=False)
    actual_device = booster_effective_device(booster)
    if actual_device != device:
        raise RuntimeError(f"XGBoost silently changed device from {device} to {actual_device}")
    return booster, actual_device


def _write_importance(
    booster: xgb.Booster,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    gain = booster.get_score(importance_type="total_gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for name in feature_names:
        if name.startswith("temporal_"):
            component = "temporal_representation"
        elif name.startswith("engineered_"):
            component = "engineered_representation"
        elif name.startswith("fused_"):
            component = "fused_representation"
        else:
            component = "b2_direct"
        rows.append(
            {
                "feature": name,
                "decision_component": component,
                "total_gain": float(gain.get(name, 0.0)),
                "weight": float(weight.get(name, 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    total = float(frame["total_gain"].sum())
    frame["total_gain_share"] = frame["total_gain"] / total if total > 0 else 0.0
    frame.sort_values(["total_gain", "feature"], ascending=[False, True]).to_csv(
        output_path,
        index=False,
    )


def _predict_validation_files(
    booster: xgb.Booster,
    model: HTSFEncoder | None,
    variant: VariantSpec,
    validation_files: Sequence[str],
    data_dir: Path,
    prediction_dir: Path,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    torch_device: torch.device,
) -> int:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for stale_prediction in prediction_dir.glob("*.csv"):
        stale_prediction.unlink()
    if model is not None:
        model.eval()
    total_rows = 0
    for index, file_name in enumerate(validation_files, 1):
        score_parts: list[np.ndarray] = []
        timestamp_parts: list[np.ndarray] = []
        for batch in iter_full_module_batches(
            Path(data_dir) / str(file_name),
            config,
            raw_standardizer,
            engineered_standardizer,
        ):
            matrix = decision_matrix_for_batch(batch, variant, model, torch_device)
            score_parts.append(booster.inplace_predict(_finite_matrix(matrix)).astype(np.float32))
            timestamp_parts.append(batch.timestamps.astype(np.int64))
        scores = np.concatenate(score_parts) if score_parts else np.zeros(0, dtype=np.float32)
        timestamps = np.concatenate(timestamp_parts) if timestamp_parts else np.zeros(0, dtype=np.int64)
        prediction = (scores >= float(config.decision.threshold)).astype(np.int8)
        pd.DataFrame(
            {"timestamp": timestamps, "predict": prediction, "score": scores}
        ).to_csv(prediction_dir / str(file_name), index=False)
        total_rows += len(scores)
        if index % 500 == 0:
            print(f"  [predict] {index}/{len(validation_files)} modules", flush=True)
    return total_rows


def run_fold_variant(
    *,
    fold: int,
    variant: VariantSpec,
    config: HTSFConfig,
    data_dir: Path,
    train_manifest: pd.DataFrame,
    endpoint_meta: dict[str, object],
    validation_files: Sequence[str],
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    output_dir: Path,
    split_fingerprint: str,
    dataset_fingerprint: str,
    overwrite: bool,
    tensor_cache_dir: Path | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "fold_summary.json"
    run_identity = fingerprint(
        {
            "fold": int(fold),
            "variant": variant.name,
            "config": config.to_dict(),
            "endpoint_fingerprint": endpoint_meta["endpoint_fingerprint"],
            "split_fingerprint": split_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "validation_files": list(validation_files),
            "implementation_fingerprint": implementation_fingerprint(),
            "runtime": runtime_identity(),
        }
    )
    if summary_path.exists() and not overwrite:
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        if stored.get("run_fingerprint") != run_identity:
            raise ValueError(f"{output_dir} was produced by a different experiment; use --overwrite")
        return stored

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    weighted_manifest = train_manifest.copy()
    if "sample_weight" not in weighted_manifest:
        raise ValueError("pipeline requires the weighting axis to attach sample_weight")
    model: HTSFEncoder | None = None
    representation_manifest: dict[str, object] | None = None
    if variant.needs_encoder:
        model, weighted_manifest, representation_manifest, torch_device = train_encoder(
            data_dir,
            weighted_manifest,
            config,
            variant,
            raw_standardizer,
            engineered_standardizer,
            output_dir,
            tensor_cache_dir,
        )
    else:
        torch_device = resolve_torch_device("cpu")

    matrix, target, weight, identity, feature_names = extract_training_table(
        data_dir,
        weighted_manifest,
        config,
        variant,
        raw_standardizer,
        engineered_standardizer,
        model,
        torch_device,
        tensor_cache_dir,
    )
    if not np.array_equal(
        identity[["file_name", "row_index"]].to_numpy(),
        weighted_manifest[["file_name", "row_index"]].to_numpy(),
    ):
        raise RuntimeError("representation extraction changed endpoint identity or order")
    pd.DataFrame({"feature": feature_names}).to_csv(output_dir / "decision_features.csv", index=False)
    booster, xgb_device = _train_xgboost(matrix, target, weight, feature_names, config)
    booster.save_model(output_dir / "decision_xgb.ubj")
    (output_dir / "decision_xgb_config.json").write_text(
        booster.save_config(), encoding="utf-8"
    )
    _write_importance(booster, feature_names, output_dir / "decision_feature_importance.csv")
    del matrix
    gc.collect()

    prediction_dir = output_dir / "predictions"
    test_rows = _predict_validation_files(
        booster,
        model,
        variant,
        validation_files,
        data_dir,
        prediction_dir,
        config,
        raw_standardizer,
        engineered_standardizer,
        torch_device,
    )
    metrics, detail = evaluate_prediction_dir(prediction_dir, data_dir, validation_files)
    write_evaluation(output_dir / "evaluation", metrics, detail)
    summary: dict[str, object] = {
        "fold": int(fold),
        "variant": variant.name,
        "mechanism": variant.mechanism,
        "run_fingerprint": run_identity,
        "implementation_fingerprint": implementation_fingerprint(),
        "runtime": runtime_identity(),
        "config_fingerprint": config.fingerprint(),
        "endpoint_fingerprint": endpoint_fingerprint(train_manifest),
        "module_endpoint_count_fingerprint": fingerprint(
            train_manifest.groupby("file_name").size().sort_index().to_dict()
        ),
        "weight_fingerprint": weight_fingerprint(weighted_manifest),
        "split_fingerprint": split_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "train_rows": int(len(target)),
        "positive_train_rows": int(np.sum(target > 0)),
        "validation_modules": int(len(validation_files)),
        "validation_rows": int(test_rows),
        "decision_feature_count": int(len(feature_names)),
        "decision_inputs": list(variant.decision_inputs),
        "xgboost_device": xgb_device,
        "torch_device": str(torch_device) if variant.needs_encoder else "not_applicable",
        "threshold_policy": config.decision.threshold_policy,
        "threshold": float(config.decision.threshold),
        "sample_weight_used_by_xgboost": bool(config.decision.use_sample_weight),
        "representation": representation_manifest,
        "training_tensor_cache": (
            {
                **json.loads(
                    (Path(tensor_cache_dir) / "cache_manifest.json").read_text(encoding="utf-8")
                ),
                "cache_dir": str(Path(tensor_cache_dir).resolve()),
            }
            if tensor_cache_dir is not None
            else None
        ),
        "seconds": float(time.time() - started),
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate_variant(
    experiment_dir: Path,
    folds: Sequence[int],
    variant: VariantSpec,
    config: HTSFConfig,
    expected_files: Sequence[str] | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    details: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for fold in folds:
        fold_dir = Path(experiment_dir) / f"fold_{int(fold)}"
        detail = pd.read_csv(fold_dir / "evaluation" / "module_decisions.csv")
        detail["fold"] = int(fold)
        details.append(detail)
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        if summary.get("variant") != variant.name:
            raise ValueError(f"fold {fold} belongs to variant={summary.get('variant')!r}")
        if summary.get("config_fingerprint") != config.fingerprint():
            raise ValueError(f"fold {fold} config does not match this aggregation")
        if summary.get("implementation_fingerprint") != implementation_fingerprint():
            raise ValueError(f"fold {fold} was produced by a different HTSF implementation")
        if summary.get("runtime") != runtime_identity():
            raise ValueError(f"fold {fold} was produced by a different Python/ML runtime")
        summaries.append(summary)
    pooled_detail = pd.concat(details, ignore_index=True)
    if pooled_detail["file_name"].duplicated().any():
        raise ValueError("validation modules overlap across pooled folds")
    if expected_files is not None and set(pooled_detail["file_name"].astype(str)) != set(expected_files):
        raise ValueError("pooled validation files do not match the formal index")
    metrics = metrics_from_decisions(pooled_detail)
    write_evaluation(Path(experiment_dir) / "pooled", metrics, pooled_detail)
    pd.DataFrame(
        [
            {
                **{key: value for key, value in summary.items() if key not in {"metrics", "representation"}},
                **summary["metrics"],
            }
            for summary in summaries
        ]
    ).to_csv(Path(experiment_dir) / "fold_metrics.csv", index=False)
    manifest: dict[str, object] = {
        "schema_version": config.schema_version,
        "variant": variant.name,
        "mechanism": variant.mechanism,
        "config": config.to_dict(),
        "folds": [int(value) for value in folds],
        "endpoint_fingerprints": {
            str(summary["fold"]): summary["endpoint_fingerprint"] for summary in summaries
        },
        "module_endpoint_count_fingerprints": {
            str(summary["fold"]): summary["module_endpoint_count_fingerprint"]
            for summary in summaries
        },
        "weight_fingerprints": {
            str(summary["fold"]): summary["weight_fingerprint"] for summary in summaries
        },
        "split_fingerprints": {
            str(summary["fold"]): summary["split_fingerprint"] for summary in summaries
        },
        "dataset_fingerprints": {
            str(summary["fold"]): summary["dataset_fingerprint"] for summary in summaries
        },
        "xgboost_devices": {
            str(summary["fold"]): summary["xgboost_device"] for summary in summaries
        },
        "torch_devices": {
            str(summary["fold"]): summary["torch_device"] for summary in summaries
        },
        "implementation_fingerprint": implementation_fingerprint(),
        "decision_inputs": list(variant.decision_inputs),
        "train_rows_by_fold": {
            str(summary["fold"]): summary["train_rows"] for summary in summaries
        },
        "positive_train_rows_by_fold": {
            str(summary["fold"]): summary["positive_train_rows"] for summary in summaries
        },
        "decision_feature_counts": {
            str(summary["fold"]): summary["decision_feature_count"] for summary in summaries
        },
        "trainable_parameters_by_fold": {
            str(summary["fold"]): (
                summary["representation"]["trainable_parameters"]
                if summary.get("representation") is not None
                else 0
            )
            for summary in summaries
        },
        "seconds_by_fold": {
            str(summary["fold"]): summary["seconds"] for summary in summaries
        },
        "pooled_metrics": metrics,
        "versions": runtime_identity(),
    }
    (Path(experiment_dir) / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return metrics, manifest
