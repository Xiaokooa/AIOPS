"""End-to-end single-split DRFP experiment pipeline."""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ofp_unified.features import RAW_FEATURES, STATISTICAL_FEATURES

from .calibration import TemperatureCalibrator, fit_calibrator
from .config import DRFPConfig
from .data import (
    EndpointRef,
    FailurePredictionDataset,
    ModuleRecord,
    SplitManifest,
    Standardizer,
    build_endpoint_refs,
    build_fixed_split,
    load_module_records,
    read_module_record,
)
from .evaluation import (
    evaluate_ofp,
    multi_horizon_timestamp_diagnostics,
    search_validation_threshold,
)
from .features import RULE_SIGNAL_DIM, RULE_SIGNAL_NAMES
from .model import DRFPNet, parameter_statistics
from .training import batch_to_device, resolve_device, seed_everything, train_model


BRANCH_OUTPUT_KEYS = {
    "deep": "deep_hazard_logits",
    "rule": "rule_hazard_logits",
    "fusion": "fused_hazard_logits",
}
PRIMARY_BRANCH = {
    "temporal_only": "deep",
    "deep_raw_stat": "deep",
    "rule_only": "rule",
    "fixed_fusion": "fusion",
    "gated_fusion": "fusion",
}


@dataclass
class PreparedData:
    data_dir: Path
    split: SplitManifest
    train_records: list[ModuleRecord]
    validation_records: list[ModuleRecord]
    raw_standardizer: Standardizer
    stat_standardizer: Standardizer
    train_endpoints: list[EndpointRef]
    validation_training_endpoints: list[EndpointRef]
    preparation_seconds: float
    scope: str


@dataclass
class InferenceOutput:
    frame: pd.DataFrame
    hazard_logits: dict[str, np.ndarray]
    targets: np.ndarray
    valid_prefault: np.ndarray
    gate_summary: dict[str, float]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _fingerprint_records(frame: pd.DataFrame) -> str:
    records = [
        {
            "file_name": str(row.file_name),
            "folder_index": int(row.folder_index),
            "Label": int(row.Label),
            "split": str(getattr(row, "split", "unspecified")),
        }
        for row in frame.sort_values("file_name").itertuples(index=False)
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _limit_split(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.copy().reset_index(drop=True)
    if limit < 2:
        raise ValueError("module limits must be at least 2")
    positive = frame.loc[frame["Label"] > 0]
    negative = frame.loc[frame["Label"] <= 0]
    positive_count = min(len(positive), max(1, int(round(limit * len(positive) / len(frame)))))
    negative_count = min(len(negative), limit - positive_count)
    if positive_count + negative_count < limit:
        positive_count = min(len(positive), limit - negative_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("limited split must retain both faulty and normal modules")
    selected = pd.concat(
        (positive.head(positive_count), negative.head(negative_count)),
        axis=0,
    ).sort_index()
    return selected.head(limit).reset_index(drop=True)


def _dataset(
    records: Sequence[ModuleRecord],
    endpoints: Sequence[EndpointRef],
    config: DRFPConfig,
    raw_standardizer: Standardizer,
    stat_standardizer: Standardizer,
) -> FailurePredictionDataset:
    return FailurePredictionDataset(
        records,
        endpoints,
        horizons_hours=config.task.horizons_hours,
        history_hours=config.inputs.history_hours,
        grid_hours=config.inputs.grid_hours,
        rule_approach_hours=config.inputs.rule_approach_hours,
        rule_persistence_hours=config.inputs.rule_persistence_hours,
        input_variant=config.model.variant,
        raw_standardizer=raw_standardizer,
        stat_standardizer=stat_standardizer,
    )


def prepare_data(
    config: DRFPConfig,
    data_dir: Path,
    index_path: Path,
    *,
    max_train_modules: int | None = None,
    max_validation_modules: int | None = None,
    max_test_modules: int | None = None,
    progress: bool = True,
) -> PreparedData:
    """Read the fixed split once so architecture suites share identical inputs."""

    started = time.perf_counter()
    split = build_fixed_split(
        index_path,
        validation_fraction=config.split.validation_fraction,
        seed=config.split.seed,
        test_fold=config.split.test_fold,
    )
    split = SplitManifest(
        train=_limit_split(split.train, max_train_modules),
        validation=_limit_split(split.validation, max_validation_modules),
        test=_limit_split(split.test, max_test_modules),
    )
    data_dir = Path(data_dir)
    all_names = split.all_file_names()
    missing = [name for name in all_names if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset missing {len(missing)} modules; first={missing[0]}")
    print(
        "[split] "
        f"train={len(split.train)} ({int((split.train['Label'] > 0).sum())} faulty), "
        f"validation={len(split.validation)} ({int((split.validation['Label'] > 0).sum())} faulty), "
        f"test={len(split.test)} ({int((split.test['Label'] > 0).sum())} faulty)",
        flush=True,
    )
    print("[data] loading train modules", flush=True)
    train_records = load_module_records(
        data_dir, split.train["file_name"].astype(str).tolist(), progress=progress
    )
    print("[data] loading validation modules", flush=True)
    validation_records = load_module_records(
        data_dir, split.validation["file_name"].astype(str).tolist(), progress=progress
    )
    raw_standardizer = Standardizer.fit(train_records, field="raw", only_pre_failure=True)
    stat_standardizer = Standardizer.fit(
        train_records, field="statistics", only_pre_failure=True
    )
    train_refs = build_endpoint_refs(
        train_records,
        endpoints_per_module=config.sampling.endpoints_per_module,
        horizons_hours=config.task.horizons_hours,
    )
    validation_training_refs = build_endpoint_refs(
        validation_records,
        endpoints_per_module=config.sampling.endpoints_per_module,
        horizons_hours=config.task.horizons_hours,
    )
    formal_counts = (len(split.train), len(split.validation), len(split.test)) == (
        8022,
        892,
        4458,
    )
    scope = "formal_fold3" if formal_counts else "limited"
    return PreparedData(
        data_dir=data_dir,
        split=split,
        train_records=train_records,
        validation_records=validation_records,
        raw_standardizer=raw_standardizer,
        stat_standardizer=stat_standardizer,
        train_endpoints=train_refs,
        validation_training_endpoints=validation_training_refs,
        preparation_seconds=float(time.perf_counter() - started),
        scope=scope,
    )


def build_model(config: DRFPConfig) -> DRFPNet:
    return DRFPNet(
        variant=config.model.variant,
        raw_dim=len(RAW_FEATURES),
        sequence_length=config.sequence_length,
        stats_dim=len(STATISTICAL_FEATURES),
        rule_dim=RULE_SIGNAL_DIM,
        horizons=tuple(int(value) for value in config.task.horizons_hours),
        patch_length=config.model.patch_length,
        patch_stride=config.model.patch_stride,
        d_model=config.model.d_model,
        latent_dim=config.model.latent_dim,
        attention_heads=config.model.attention_heads,
        transformer_layers=config.model.transformer_layers,
        statistical_hidden=config.model.statistical_hidden,
        rule_hidden=config.model.rule_hidden,
        dropout=config.model.dropout,
    )


def infer_records(
    model: DRFPNet,
    records: Sequence[ModuleRecord],
    config: DRFPConfig,
    raw_standardizer: Standardizer,
    stat_standardizer: Standardizer,
    *,
    calibrator: TemperatureCalibrator | None = None,
    progress_label: str | None = None,
) -> InferenceOutput:
    """Infer every timestamp without materializing millions of EndpointRef objects.

    Only one module's endpoint dataset exists at a time.  Validation retains
    hazard logits for calibration; calibrated test inference retains compact
    score columns directly and discards logits batch by batch.
    """

    device = resolve_device(config.training.device)
    model = model.to(device)
    model.eval()
    module_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []
    anomaly_chunks: list[np.ndarray] = []
    collect_branches = (
        ("deep", "rule", "fusion")
        if config.model.variant in {"fixed_fusion", "gated_fusion"}
        else (PRIMARY_BRANCH[config.model.variant],)
    )
    logits: dict[str, list[np.ndarray]] = {name: [] for name in collect_branches}
    probability_chunks: dict[str, list[np.ndarray]] = {}
    targets: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    horizon_count = len(config.task.horizons_hours)
    gate_sum = np.zeros(horizon_count, dtype=np.float64)
    gate_square_sum = np.zeros(horizon_count, dtype=np.float64)
    gate_dominant_sum = np.zeros(horizon_count, dtype=np.int64)
    gate_count = 0
    primary_branch = PRIMARY_BRANCH[config.model.variant]
    primary_index = config.primary_horizon_index
    with torch.inference_mode():
        for module_id, record in enumerate(records):
            refs = [
                EndpointRef(record.file_name, 0, endpoint_index)
                for endpoint_index in range(len(record.timestamps))
            ]
            dataset = _dataset(
                [record], refs, config, raw_standardizer, stat_standardizer
            )
            loader = DataLoader(
                dataset,
                batch_size=config.training.inference_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            for batch in loader:
                inputs, batch_targets = batch_to_device(batch, device)
                output = model(**inputs)
                batch_endpoint = (
                    batch["endpoint_index"].detach().cpu().numpy().astype(np.int64)
                )
                batch_size = len(batch_endpoint)
                module_chunks.append(
                    np.full(batch_size, module_id, dtype=np.int32)
                )
                timestamp_chunks.append(record.timestamps[batch_endpoint].astype(np.int64))
                anomaly_chunks.append(record.anomaly[batch_endpoint].astype(np.int8))
                for branch in collect_branches:
                    key = BRANCH_OUTPUT_KEYS[branch]
                    value = output.get(key)
                    if not isinstance(value, torch.Tensor):
                        continue
                    values = value.detach().float().cpu().numpy()
                    if calibrator is None:
                        logits[branch].append(values)
                    else:
                        probabilities = calibrator.apply(branch, values)
                        probability_chunks.setdefault(f"{branch}_score", []).append(
                            probabilities[:, primary_index].astype(np.float32)
                        )
                        if branch == primary_branch:
                            for horizon_index, horizon in enumerate(
                                config.task.horizons_hours
                            ):
                                name = f"{branch}_p{int(horizon)}"
                                probability_chunks.setdefault(name, []).append(
                                    probabilities[:, horizon_index].astype(np.float32)
                                )
                gate = output.get("gate")
                if not isinstance(gate, torch.Tensor):
                    raise RuntimeError("model output is missing gate")
                gate_values = gate.detach().float().cpu().numpy()
                gate_sum += gate_values.sum(axis=0, dtype=np.float64)
                gate_square_sum += np.square(gate_values, dtype=np.float64).sum(axis=0)
                gate_dominant_sum += (gate_values > 0.5).sum(axis=0, dtype=np.int64)
                gate_count += len(gate_values)
                if calibrator is None:
                    targets.append(batch_targets.detach().float().cpu().numpy())
                    valid.append(
                        batch["valid_training_endpoint"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool)
                    )
            if progress_label and (
                module_id + 1 == len(records) or (module_id + 1) % 100 == 0
            ):
                print(
                    f"  [{progress_label}] inferred {module_id + 1}/{len(records)} modules",
                    flush=True,
                )
    logits_result = {
        branch: np.concatenate(chunks, axis=0)
        for branch, chunks in logits.items()
        if chunks
    }
    frame = pd.DataFrame(
        {
            "module_id": np.concatenate(module_chunks),
            "timestamp": np.concatenate(timestamp_chunks),
            "anomaly": np.concatenate(anomaly_chunks),
        }
    )
    for name, chunks in probability_chunks.items():
        frame[name] = np.concatenate(chunks)
    means = gate_sum / max(gate_count, 1)
    variances = gate_square_sum / max(gate_count, 1) - np.square(means)
    gate_summary: dict[str, float] = {"count": float(gate_count)}
    for index, horizon in enumerate(config.task.horizons_hours):
        suffix = str(int(horizon))
        gate_summary[f"gate_mean_h{suffix}"] = float(means[index])
        gate_summary[f"gate_std_h{suffix}"] = float(np.sqrt(max(variances[index], 0.0)))
        gate_summary[f"gate_rule_dominant_rate_h{suffix}"] = float(
            gate_dominant_sum[index] / max(gate_count, 1)
        )
    return InferenceOutput(
        frame=frame,
        hazard_logits=logits_result,
        targets=(
            np.concatenate(targets, axis=0)
            if targets
            else np.empty((0, horizon_count), dtype=np.float32)
        ),
        valid_prefault=(
            np.concatenate(valid, axis=0)
            if valid
            else np.empty(0, dtype=bool)
        ),
        gate_summary=gate_summary,
    )


def infer_files(
    model: DRFPNet,
    data_dir: Path,
    file_names: Sequence[str],
    config: DRFPConfig,
    raw_standardizer: Standardizer,
    stat_standardizer: Standardizer,
    calibrator: TemperatureCalibrator,
) -> InferenceOutput:
    """Read test modules incrementally, then run calibrated inference."""

    # Keeping a single module at a time avoids holding fold-3's several
    # gigabytes of 76-D statistical histories beside the training set.
    # infer_records accepts a Sequence for deterministic module ids/progress.
    # A bounded chunk keeps memory low while preserving a single compact frame.
    outputs: list[InferenceOutput] = []
    chunk_size = 32
    names = [str(name) for name in file_names]
    for start in range(0, len(names), chunk_size):
        chunk_names = names[start : start + chunk_size]
        records = [read_module_record(Path(data_dir) / name) for name in chunk_names]
        outputs.append(
            infer_records(
                model,
                records,
                config,
                raw_standardizer,
                stat_standardizer,
                calibrator=calibrator,
            )
        )
        # Rebase chunk-local ids to the fixed fold-3 module order.
        outputs[-1].frame["module_id"] += start
        if start + len(chunk_names) == len(names) or (start + len(chunk_names)) % 320 == 0:
            print(f"  [test] inferred {start + len(chunk_names)}/{len(names)} modules", flush=True)
    frame = pd.concat([output.frame for output in outputs], ignore_index=True)
    gate_count = sum(output.gate_summary["count"] for output in outputs)
    gate_summary: dict[str, float] = {"count": float(gate_count)}
    for horizon in config.task.horizons_hours:
        suffix = str(int(horizon))
        mean_key = f"gate_mean_h{suffix}"
        dominant_key = f"gate_rule_dominant_rate_h{suffix}"
        mean = sum(
            output.gate_summary[mean_key] * output.gate_summary["count"]
            for output in outputs
        ) / max(gate_count, 1.0)
        second = sum(
            (
                output.gate_summary[f"gate_std_h{suffix}"] ** 2
                + output.gate_summary[mean_key] ** 2
            )
            * output.gate_summary["count"]
            for output in outputs
        ) / max(gate_count, 1.0)
        gate_summary[mean_key] = float(mean)
        gate_summary[f"gate_std_h{suffix}"] = float(np.sqrt(max(second - mean**2, 0.0)))
        gate_summary[dominant_key] = float(
            sum(
                output.gate_summary[dominant_key] * output.gate_summary["count"]
                for output in outputs
            )
            / max(gate_count, 1.0)
        )
    return InferenceOutput(
        frame=frame,
        hazard_logits={},
        targets=np.empty((0, len(config.task.horizons_hours)), dtype=np.float32),
        valid_prefault=np.empty(0, dtype=bool),
        gate_summary=gate_summary,
    )


def _apply_calibration(
    output: InferenceOutput,
    calibrator: TemperatureCalibrator,
    config: DRFPConfig,
) -> pd.DataFrame:
    frame = output.frame.copy()
    horizon_names = [str(int(value)) for value in config.task.horizons_hours]
    primary_index = config.primary_horizon_index
    for branch, logits in output.hazard_logits.items():
        probabilities = calibrator.apply(branch, logits)
        for horizon_index, horizon in enumerate(horizon_names):
            frame[f"{branch}_p{horizon}"] = probabilities[:, horizon_index]
        frame[f"{branch}_score"] = probabilities[:, primary_index]
    return frame


def _active_branches(config: DRFPConfig, output: InferenceOutput) -> list[str]:
    if config.model.variant in {"temporal_only", "deep_raw_stat"}:
        return ["deep"]
    if config.model.variant == "rule_only":
        return ["rule"]
    return [name for name in ("deep", "rule", "fusion") if name in output.hazard_logits]


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metrics.items()
        if isinstance(value, (str, bool, int, float, np.integer, np.floating))
    }


def _gate_metrics(summary: Mapping[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in summary.items() if key != "count"}


def _fixed_policy_label(threshold: float) -> str:
    return f"fixed_{float(threshold):g}"


def _detail_with_file_names(
    detail: pd.DataFrame,
    file_names: Sequence[str],
) -> pd.DataFrame:
    result = detail.copy()
    if "module_id" in result:
        mapping = {index: str(name) for index, name in enumerate(file_names)}
        result.insert(1, "file_name", result["module_id"].map(mapping))
    return result


def _write_primary_predictions(
    frame: pd.DataFrame,
    file_names: Sequence[str],
    output_dir: Path,
    *,
    score_col: str,
    threshold: float,
    overwrite: bool,
) -> None:
    """Write the exact expected fold-3 file set and remove stale outputs."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"prediction directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    expected_ids = set(range(len(file_names)))
    observed_ids = set(pd.to_numeric(frame["module_id"], errors="raise").astype(int).unique())
    if observed_ids != expected_ids:
        raise RuntimeError("test score frame and fixed test manifest disagree")
    for module_id, group in frame.groupby("module_id", sort=False):
        module_index = int(module_id)
        scores = pd.to_numeric(group[score_col], errors="coerce")
        prediction = (scores.notna() & (scores >= float(threshold))).astype(np.int8)
        output = pd.DataFrame(
            {
                "timestamp": group["timestamp"].to_numpy(),
                "predict": prediction.to_numpy(),
                "score": scores.to_numpy(dtype=float),
            }
        ).sort_values("timestamp", kind="mergesort")
        output.to_csv(output_dir / Path(str(file_names[module_index])).name, index=False)


def run_prepared_experiment(
    prepared: PreparedData,
    config: DRFPConfig,
    output_dir: Path,
    *,
    overwrite: bool = False,
    save_long_scores: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if (output_dir / "run_manifest.json").exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains a completed run; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    started = time.perf_counter()
    train_dataset = _dataset(
        prepared.train_records,
        prepared.train_endpoints,
        config,
        prepared.raw_standardizer,
        prepared.stat_standardizer,
    )
    validation_training_dataset = _dataset(
        prepared.validation_records,
        prepared.validation_training_endpoints,
        config,
        prepared.raw_standardizer,
        prepared.stat_standardizer,
    )
    model = build_model(config)
    parameters = parameter_statistics(model)
    print(
        f"[model] variant={config.model.variant} trainable_parameters={parameters['trainable']}",
        flush=True,
    )
    training = train_model(
        model,
        train_dataset,
        validation_training_dataset,
        config,
    )
    training.history.to_csv(output_dir / "training_history.csv", index=False)
    _write_json(
        output_dir / "normalizers.json",
        {
            "fit_scope": "train modules, valid rows strictly before first failure",
            "raw": prepared.raw_standardizer.to_dict(),
            "statistical": prepared.stat_standardizer.to_dict(),
            "rule": "fixed physical scaling; no data-fitted normalization",
        },
    )
    print("[inference] validation", flush=True)
    validation_raw = infer_records(
        training.model,
        prepared.validation_records,
        config,
        prepared.raw_standardizer,
        prepared.stat_standardizer,
        progress_label="validation",
    )
    calibration_inputs = {
        branch: values[validation_raw.valid_prefault]
        for branch, values in validation_raw.hazard_logits.items()
    }
    calibration_targets = validation_raw.targets[validation_raw.valid_prefault]
    calibrator = fit_calibrator(
        calibration_inputs,
        calibration_targets,
        method=config.calibration.method,
        minimum=config.calibration.minimum_temperature,
        maximum=config.calibration.maximum_temperature,
        grid_size=config.calibration.grid_size,
    )
    _write_json(output_dir / "calibrator.json", calibrator.to_dict())
    validation_frame = _apply_calibration(validation_raw, calibrator, config)
    active_branches = _active_branches(config, validation_raw)
    branch_score_cols = {
        name: f"{name}_score"
        for name in active_branches
        if f"{name}_score" in validation_frame
    }
    selected_thresholds: dict[str, float] = {}
    for branch in active_branches:
        score_col = f"{branch}_score"
        threshold_search = search_validation_threshold(
            validation_frame,
            score_col=score_col,
            fixed_threshold=config.decision.fixed_threshold,
            grid_size=config.decision.threshold_candidates,
            quantile_count=config.decision.threshold_candidates,
            module_col="module_id",
            horizon_hours=config.task.primary_horizon_hours,
            branch_score_cols=branch_score_cols,
        )
        threshold_search.table.to_csv(
            output_dir / f"validation_threshold_search_{branch}.csv", index=False
        )
        selected_thresholds[branch] = float(threshold_search.threshold)

    # Test data are not read until calibration and every branch threshold are
    # frozen from validation.  This ordering is part of the leakage audit.
    print("[inference] fold-3 test", flush=True)
    test_output = infer_files(
        training.model,
        prepared.data_dir,
        prepared.split.test["file_name"].astype(str).tolist(),
        config,
        prepared.raw_standardizer,
        prepared.stat_standardizer,
        calibrator,
    )
    test_frame = test_output.frame
    test_branch_score_cols = {
        name: f"{name}_score"
        for name in active_branches
        if f"{name}_score" in test_frame
    }

    result_rows: list[dict[str, Any]] = []
    fixed_policy = _fixed_policy_label(config.decision.fixed_threshold)
    for branch in active_branches:
        score_col = f"{branch}_score"
        policies = [(fixed_policy, float(config.decision.fixed_threshold))]
        if config.decision.tune_on_validation:
            policies.append(
                ("validation_selected", float(selected_thresholds[branch]))
            )
        for policy, threshold in policies:
            diagnostic_thresholds = (
                {name: float(config.decision.fixed_threshold) for name in active_branches}
                if policy == fixed_policy
                else selected_thresholds
            )
            result = evaluate_ofp(
                test_frame,
                threshold=threshold,
                module_col="module_id",
                score_col=score_col,
                horizon_hours=config.task.primary_horizon_hours,
                branch_score_cols=test_branch_score_cols,
                branch_thresholds=diagnostic_thresholds,
                include_diagnostics=False,
            )
            row: dict[str, Any] = {
                "experiment_id": config.experiment_name,
                "variant": config.model.variant,
                "branch": branch,
                "threshold_policy": policy,
                "threshold": threshold,
                "changed_axis": "architecture",
                "trainable_parameters": parameters["trainable"],
                "seconds_training": training.seconds_total,
            }
            row.update(_scalar_metrics(result.metrics))
            row.update(_gate_metrics(test_output.gate_summary))
            result_rows.append(row)
            _detail_with_file_names(
                result.detail,
                prepared.split.test["file_name"].astype(str).tolist(),
            ).to_csv(
                output_dir / f"module_decisions_{branch}_{policy}.csv",
                index=False,
            )
    comparison = pd.DataFrame(result_rows)
    comparison.to_csv(output_dir / "branch_comparison.csv", index=False)

    primary_branch = PRIMARY_BRANCH[config.model.variant]
    primary_threshold = (
        selected_thresholds[primary_branch]
        if config.decision.tune_on_validation
        else config.decision.fixed_threshold
    )
    primary_policy = (
        "validation_selected" if config.decision.tune_on_validation else fixed_policy
    )
    # Compute expensive timestamp/lead/branch diagnostics exactly once, for
    # the declared primary output.  Branch diagnostics use each branch's own
    # validation-selected threshold rather than reusing the fusion threshold.
    primary_result = evaluate_ofp(
        test_frame,
        threshold=primary_threshold,
        module_col="module_id",
        score_col=f"{primary_branch}_score",
        horizon_hours=config.task.primary_horizon_hours,
        branch_score_cols=test_branch_score_cols,
        branch_thresholds=(
            selected_thresholds
            if config.decision.tune_on_validation
            else {name: float(config.decision.fixed_threshold) for name in active_branches}
        ),
        include_diagnostics=True,
    )
    primary_result.metrics.update(
        multi_horizon_timestamp_diagnostics(
            test_frame,
            {
                float(horizon): f"{primary_branch}_p{int(horizon)}"
                for horizon in config.task.horizons_hours
            },
            module_col="module_id",
        )
    )
    _detail_with_file_names(
        primary_result.detail,
        prepared.split.test["file_name"].astype(str).tolist(),
    ).to_csv(
        output_dir / f"module_decisions_{primary_branch}_{primary_policy}.csv",
        index=False,
    )
    _write_primary_predictions(
        test_frame,
        prepared.split.test["file_name"].astype(str).tolist(),
        output_dir / "ofp_predictions",
        score_col=f"{primary_branch}_score",
        threshold=primary_threshold,
        overwrite=overwrite,
    )
    if save_long_scores:
        validation_frame.to_csv(output_dir / "validation_scores.csv.gz", index=False)
        test_frame.to_csv(output_dir / "test_scores.csv.gz", index=False)

    checkpoint = {
        "state_dict": {
            name: value.detach().cpu() for name, value in training.model.state_dict().items()
        },
        "config": config.to_dict(),
        "calibrator": calibrator.to_dict(),
        "selected_thresholds": selected_thresholds,
        "raw_standardizer": prepared.raw_standardizer.to_dict(),
        "stat_standardizer": prepared.stat_standardizer.to_dict(),
    }
    torch.save(checkpoint, output_dir / "model.pt")
    split_table = pd.concat(
        (
            prepared.split.train.assign(split="train"),
            prepared.split.validation.assign(split="validation"),
            prepared.split.test.assign(split="test"),
        ),
        ignore_index=True,
    )
    split_table.to_csv(output_dir / "split_manifest.csv", index=False)
    primary_metrics = _scalar_metrics(primary_result.metrics)
    run_manifest = {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_name,
        "variant": config.model.variant,
        "scope": prepared.scope,
        "config_fingerprint": config.fingerprint(),
        "split_fingerprint": _fingerprint_records(split_table),
        "split_counts": {
            "train": len(prepared.split.train),
            "validation": len(prepared.split.validation),
            "test": len(prepared.split.test),
        },
        "training_endpoint_protocol": {
            "mode": config.sampling.mode,
            "trainable_modules": len(
                {reference.module_index for reference in train_dataset.endpoints}
            ),
            "endpoints_per_module_per_epoch": config.sampling.endpoints_per_module,
            "endpoints_per_epoch": (
                len({reference.module_index for reference in train_dataset.endpoints})
                * config.sampling.endpoints_per_module
            ),
            "sampled_endpoint_pool_size": len(train_dataset),
            "sampled_positive_pool_rows_primary_horizon": int(
                train_dataset.target_matrix[:, config.primary_horizon_index].sum()
            ),
        },
        "feature_counts": {
            "raw": len(RAW_FEATURES),
            "statistical": len(STATISTICAL_FEATURES),
            "predictive_rule_signals": RULE_SIGNAL_DIM,
        },
        "rule_signal_names": list(RULE_SIGNAL_NAMES),
        "parameter_statistics": parameters,
        "positive_weights": training.positive_weights,
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "calibration": calibrator.to_dict(),
        "primary_branch": primary_branch,
        "primary_threshold_policy": primary_policy,
        "primary_threshold": primary_threshold,
        "primary_metrics": primary_metrics,
        "seconds": {
            "data_preparation": prepared.preparation_seconds,
            "training": training.seconds_total,
            "experiment_after_preparation": float(time.perf_counter() - started),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(resolve_device(config.training.device)),
        },
    }
    _write_json(output_dir / "effective_config.json", config.to_dict())
    _write_json(output_dir / "run_manifest.json", run_manifest)
    print(
        f"[result] branch={primary_branch} threshold={primary_threshold:.6f} "
        f"final={float(primary_metrics['final_score']):.6f} "
        f"F1={float(primary_metrics['f1_score']):.6f}",
        flush=True,
    )
    return run_manifest


def release_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "PreparedData",
    "InferenceOutput",
    "prepare_data",
    "build_model",
    "infer_records",
    "infer_files",
    "run_prepared_experiment",
    "release_model_memory",
]
