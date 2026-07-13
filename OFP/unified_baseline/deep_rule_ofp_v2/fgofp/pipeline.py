"""End-to-end native sequence FGL experiment with a frozen OFP protocol."""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import FGOFPConfig
from .data import (
    LengthBucketBatchSampler,
    ModuleRecord,
    NativeSequenceDataset,
    SplitManifest,
    Standardizer,
    build_fixed_split,
    collate_native_sequences,
    load_module_records,
)
from .evaluation import (
    PROTOCOL_NAME,
    evaluate_legacy_inclusive,
    search_validation_threshold,
    write_module_predictions,
)
from .features import RAW_FEATURES
from .model import CausalSeq2SeqTCN, build_teacher_student
from .training import TrainingResult, resolve_device, seed_everything, train_models


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fingerprint_split(frame: pd.DataFrame) -> str:
    columns = ["file_name", "folder_index", "Label", "split"]
    records = frame.loc[:, columns].sort_values("file_name").to_dict("records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _limit_split(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or int(limit) >= len(frame):
        return frame.copy().reset_index(drop=True)
    if int(limit) < 2:
        raise ValueError("module limits must be at least 2")
    limit = int(limit)
    faulty = frame.loc[frame["Label"] > 0]
    normal = frame.loc[frame["Label"] <= 0]
    faulty_count = min(
        len(faulty), max(1, int(round(limit * len(faulty) / len(frame))))
    )
    normal_count = min(len(normal), limit - faulty_count)
    if faulty_count + normal_count < limit:
        faulty_count = min(len(faulty), limit - normal_count)
    if faulty_count == 0 or normal_count == 0:
        raise ValueError("limited split must retain both faulty and normal modules")
    selected = pd.concat(
        (faulty.head(faulty_count), normal.head(normal_count)), axis=0
    ).sort_index()
    return selected.head(limit).reset_index(drop=True)


def _limited_manifest(
    split: SplitManifest,
    *,
    max_train_modules: int | None,
    max_validation_modules: int | None,
    max_test_modules: int | None,
) -> SplitManifest:
    return SplitManifest(
        train=_limit_split(split.train, max_train_modules),
        validation=_limit_split(split.validation, max_validation_modules),
        test=_limit_split(split.test, max_test_modules),
    )


def _dataset(
    records: Sequence[ModuleRecord],
    standardizer: Standardizer,
    config: FGOFPConfig,
    *,
    truncate_after_first_fault: bool = False,
    include_supervision: bool = True,
) -> NativeSequenceDataset:
    return NativeSequenceDataset(
        records,
        standardizer=standardizer,
        student_horizon_hours=config.task.student_horizon_hours,
        teacher_horizon_hours=config.task.teacher_horizon_hours,
        offset_hours=config.task.future_offset_hours,
        alignment_tolerance_seconds=config.inputs.alignment_tolerance_seconds,
        expected_cadence_seconds=config.inputs.expected_cadence_seconds,
        delta_clip_steps=config.inputs.delta_clip_steps,
        truncate_after_first_fault=truncate_after_first_fault,
        include_supervision=include_supervision,
    )


def _check_files(data_dir: Path, split: SplitManifest) -> None:
    names = pd.concat(
        (split.train["file_name"], split.validation["file_name"], split.test["file_name"]),
        ignore_index=True,
    ).astype(str)
    missing = [name for name in names if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"dataset is missing {len(missing)} modules; first={missing[0]}"
        )


def _verify_index_labels(
    membership: pd.DataFrame,
    records: Sequence[ModuleRecord],
    *,
    split_name: str,
) -> None:
    expected = {
        str(row.file_name): bool(int(row.Label) > 0)
        for row in membership.itertuples(index=False)
    }
    mismatches = [
        record.file_name
        for record in records
        if expected.get(record.file_name) != (record.first_fault_index is not None)
    ]
    if mismatches:
        raise ValueError(
            f"{split_name} index/CSV fault-label mismatch for "
            f"{len(mismatches)} modules; first={mismatches[0]}"
        )


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --overwrite to reuse it"
        )
    if output_dir.exists() and overwrite:
        artifact_files = {
            "effective_config.json",
            "normalizer.json",
            "result.csv",
            "run_manifest.json",
            "split_manifest.csv",
            "test_module_decisions.csv",
            "test_scores.csv.gz",
            "training_history.csv",
            "validation_module_decisions.csv",
            "validation_scores.csv.gz",
            "validation_threshold_search.csv",
        }
        for name in artifact_files:
            target = output_dir / name
            if target.is_file():
                target.unlink()
        for name in ("checkpoints", "ofp_predictions"):
            target = output_dir / name
            if target.is_dir():
                shutil.rmtree(target)
    output_dir.mkdir(parents=True, exist_ok=True)


def _inference_loader(
    dataset: NativeSequenceDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    sampler = LengthBucketBatchSampler(
        dataset, batch_size=batch_size, shuffle=False, seed=0
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_native_sequences,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def infer_student(
    student: CausalSeq2SeqTCN,
    dataset: NativeSequenceDataset,
    config: FGOFPConfig,
    *,
    progress_label: str,
) -> pd.DataFrame:
    """Run causal student-only inference and return every original row once."""

    device = resolve_device(config.training.device)
    student = student.to(device)
    student.eval()
    loader = _inference_loader(
        dataset,
        batch_size=config.training.inference_batch_size,
        num_workers=config.training.num_workers,
    )
    pieces: list[pd.DataFrame] = []
    processed = 0
    for batch in loader:
        raw = batch["raw"].to(device=device, dtype=torch.float32, non_blocking=True)
        raw_mask = batch["raw_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
        delta = batch["delta_steps"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        lengths = batch["lengths"].to(dtype=torch.long)
        maximum = int(raw.shape[1])
        padding_mask = (
            torch.arange(maximum, device=device).unsqueeze(0)
            < lengths.to(device=device).unsqueeze(1)
        )
        logits = student(raw, raw_mask, delta, padding_mask)
        scores = torch.softmax(logits.float(), dim=-1)[..., 1].cpu().numpy()
        timestamps = batch["timestamps"].cpu().numpy()
        anomaly = batch["anomaly"].cpu().numpy()
        for index, file_name in enumerate(batch["file_names"]):
            length = int(lengths[index])
            pieces.append(
                pd.DataFrame(
                    {
                        "file_name": str(file_name),
                        "timestamp": timestamps[index, :length].astype(np.int64),
                        "anomaly": (anomaly[index, :length] > 0).astype(np.int8),
                        "score": scores[index, :length].astype(np.float32),
                    }
                )
            )
        processed += len(batch["file_names"])
        if processed == len(dataset) or processed % 100 == 0:
            print(
                f"  [{progress_label}] inferred {processed}/{len(dataset)} modules",
                flush=True,
            )
    if not pieces:
        raise ValueError("cannot infer an empty dataset")
    frame = pd.concat(pieces, ignore_index=True)
    frame["file_name"] = frame["file_name"].astype("category")
    if frame["timestamp"].dtype != np.int64:
        raise RuntimeError("inference timestamps lost int64 precision")
    return frame


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _split_table(split: SplitManifest) -> pd.DataFrame:
    return pd.concat(
        (
            split.train.assign(split="train"),
            split.validation.assign(split="validation"),
            split.test.assign(split="test"),
        ),
        ignore_index=True,
    )


def run_experiment(
    config: FGOFPConfig,
    data_dir: Path | str,
    index_path: Path | str,
    output_dir: Path | str,
    *,
    max_train_modules: int | None = None,
    max_validation_modules: int | None = None,
    max_test_modules: int | None = None,
    overwrite: bool = False,
    save_long_scores: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Train, freeze a validation threshold, then read and evaluate fold 3."""

    started = time.perf_counter()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    _prepare_output(output_dir, overwrite)
    if config.protocol.name != PROTOCOL_NAME:
        raise ValueError(f"pipeline supports only {PROTOCOL_NAME}")

    split = build_fixed_split(
        index_path,
        validation_fraction=config.split.validation_fraction,
        seed=config.split.seed,
        test_fold=config.split.test_fold,
    )
    split = _limited_manifest(
        split,
        max_train_modules=max_train_modules,
        max_validation_modules=max_validation_modules,
        max_test_modules=max_test_modules,
    )
    _check_files(data_dir, split)
    split_table = _split_table(split)
    split_table.to_csv(output_dir / "split_manifest.csv", index=False)
    _write_json(output_dir / "effective_config.json", config.to_dict())
    print(
        "[split] "
        f"train={len(split.train)} ({int((split.train['Label'] > 0).sum())} faulty), "
        f"validation={len(split.validation)} "
        f"({int((split.validation['Label'] > 0).sum())} faulty), "
        f"test={len(split.test)} ({int((split.test['Label'] > 0).sum())} faulty)",
        flush=True,
    )

    # Fold-3 CSV contents are deliberately not opened before the validation
    # threshold is frozen. Membership metadata in the public index is safe.
    data_started = time.perf_counter()
    print("[data] loading train native-row modules", flush=True)
    train_records = load_module_records(
        data_dir,
        split.train["file_name"].astype(str).tolist(),
        progress=progress,
    )
    print("[data] loading validation native-row modules", flush=True)
    validation_records = load_module_records(
        data_dir,
        split.validation["file_name"].astype(str).tolist(),
        progress=progress,
    )
    _verify_index_labels(split.train, train_records, split_name="train")
    _verify_index_labels(
        split.validation, validation_records, split_name="validation"
    )
    standardizer = Standardizer.fit(train_records)
    train_dataset = _dataset(
        train_records,
        standardizer,
        config,
        truncate_after_first_fault=True,
        include_supervision=True,
    )
    validation_training_dataset = _dataset(
        validation_records,
        standardizer,
        config,
        truncate_after_first_fault=True,
        include_supervision=True,
    )
    validation_inference_dataset = _dataset(
        validation_records,
        standardizer,
        config,
        truncate_after_first_fault=False,
        include_supervision=False,
    )
    data_preparation_seconds = float(time.perf_counter() - data_started)

    # Initialization must be seeded before constructing either model; seeding
    # only inside train_models would make isolated N0/N1 runs incomparable.
    seed_everything(config.seed)
    student, teacher = build_teacher_student(config.model)
    print(
        f"[model] student_parameters={_parameter_count(student)} "
        f"receptive_field_steps={student.receptive_field_steps}",
        flush=True,
    )
    training: TrainingResult = train_models(
        student,
        teacher,
        train_dataset,
        validation_training_dataset,
        config,
    )
    training.history.to_csv(output_dir / "training_history.csv", index=False)

    print("[inference] validation student only", flush=True)
    validation_frame = infer_student(
        training.student,
        validation_inference_dataset,
        config,
        progress_label="validation",
    )
    if config.decision.tune_on_validation:
        threshold_search = search_validation_threshold(
            validation_frame,
            score_col="score",
            fixed_threshold=config.decision.fixed_threshold,
            grid_size=config.decision.threshold_grid_size,
            quantile_count=config.decision.threshold_quantile_count,
            split_name="validation",
        )
        threshold = float(threshold_search.threshold)
        threshold_search.table.to_csv(
            output_dir / "validation_threshold_search.csv", index=False
        )
        threshold_search.detail.to_csv(
            output_dir / "validation_module_decisions.csv", index=False
        )
        threshold_policy = "validation_selected"
    else:
        threshold = float(config.decision.fixed_threshold)
        validation_result = evaluate_legacy_inclusive(
            validation_frame, threshold=threshold
        )
        validation_result.detail.to_csv(
            output_dir / "validation_module_decisions.csv", index=False
        )
        pd.DataFrame(
            [{"rank": 1, "selected": True, "threshold": threshold, **validation_result.metrics}]
        ).to_csv(output_dir / "validation_threshold_search.csv", index=False)
        threshold_policy = "fixed"
    if save_long_scores:
        validation_frame.to_csv(
            output_dir / "validation_scores.csv.gz", index=False, compression="gzip"
        )
    print(
        f"[decision] validation threshold frozen at {threshold:.9f}; "
        "fold-3 CSV loading may now begin",
        flush=True,
    )

    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    student_checkpoint = {
        "schema_version": config.schema_version,
        "config_fingerprint": config.fingerprint(),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in training.student.state_dict().items()
        },
        "model_config": config.to_dict()["model"],
        "input_config": config.to_dict()["inputs"],
        "task_config": config.to_dict()["task"],
        "input_schema": {
            "raw_features": list(RAW_FEATURES),
            "channels": ["raw", "raw_observation_mask", "delta_steps"],
            "cadence_mode": config.inputs.cadence_mode,
        },
        "standardizer": standardizer.to_dict(),
        "decision_threshold": threshold,
        "threshold_policy": threshold_policy,
        "protocol": config.to_dict()["protocol"],
        "teacher_used_at_inference": False,
    }
    if any(name.startswith("teacher") for name in student_checkpoint["state_dict"]):
        raise RuntimeError("deployable checkpoint unexpectedly contains teacher parameters")
    torch.save(student_checkpoint, checkpoints / "student_best.pt")
    if training.teacher_train_only is not None:
        torch.save(
            {
                "schema_version": config.schema_version,
                "training_only": True,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in training.teacher_train_only.state_dict().items()
                },
            },
            checkpoints / "teacher_train_only.pt",
        )

    # Release the ~15M train/validation native rows before loading fold 3.
    del (
        train_dataset,
        validation_training_dataset,
        validation_inference_dataset,
        train_records,
        validation_records,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[data] loading fold-3 test native-row modules", flush=True)
    test_load_started = time.perf_counter()
    test_records = load_module_records(
        data_dir,
        split.test["file_name"].astype(str).tolist(),
        progress=progress,
    )
    _verify_index_labels(split.test, test_records, split_name="test")
    test_dataset = _dataset(
        test_records,
        standardizer,
        config,
        truncate_after_first_fault=False,
        include_supervision=False,
    )
    print("[inference] fold-3 test student only", flush=True)
    test_frame = infer_student(
        training.student, test_dataset, config, progress_label="test"
    )
    test_result = evaluate_legacy_inclusive(test_frame, threshold=threshold)
    test_result.detail.to_csv(output_dir / "test_module_decisions.csv", index=False)
    write_module_predictions(
        test_frame,
        output_dir / "ofp_predictions",
        threshold=threshold,
        overwrite=True,
    )
    if save_long_scores:
        test_frame.to_csv(
            output_dir / "test_scores.csv.gz", index=False, compression="gzip"
        )

    variant = (
        "native_s2s_ce"
        if config.training.fgl_alpha >= 1.0
        else "native_s2s_ce_fgl"
    )
    metrics = {name: _scalar(value) for name, value in test_result.metrics.items()}
    result_row: dict[str, Any] = {
        "experiment_id": config.experiment_name,
        "variant": variant,
        "changed_axis": "training_objective",
        "protocol": PROTOCOL_NAME,
        "native_rows": True,
        "sequence_to_sequence": True,
        "teacher_used_at_inference": False,
        "decision_feature_count": 2 * len(RAW_FEATURES) + 1,
        "student_parameters": _parameter_count(training.student),
        "teacher_training_parameters": (
            _parameter_count(training.teacher_train_only)
            if training.teacher_train_only is not None
            else 0
        ),
        "student_ce_rows": int(training.coverage.get("student_ce_rows", 0)),
        "student_ce_positive_rows": int(
            training.coverage.get("student_ce_positive_rows", 0)
        ),
        "fgl_pairs": int(training.coverage.get("fgl_pairs", 0)),
        "fgl_modules": int(training.coverage.get("fgl_modules", 0)),
        "positive_fgl_pairs": int(
            training.coverage.get("positive_fgl_pairs", 0)
        ),
        "positive_fgl_modules": int(
            training.coverage.get("positive_fgl_modules", 0)
        ),
        "fgl_coverage_scale": training.fgl_coverage_scale,
        "decision_threshold": threshold,
        "threshold_policy": threshold_policy,
        "seconds_total": float(time.perf_counter() - started),
        **metrics,
    }
    pd.DataFrame([result_row]).to_csv(output_dir / "result.csv", index=False)

    coverage = {name: _scalar(value) for name, value in training.coverage.items()}
    manifest: dict[str, Any] = {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_name,
        "variant": variant,
        "config_fingerprint": config.fingerprint(),
        "split_fingerprint": _fingerprint_split(split_table),
        "split_counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "protocol": {
            "name": PROTOCOL_NAME,
            "hit_condition": "first_alarm_timestamp <= first_failure_timestamp",
            "alarm_policy": "first_threshold_crossing",
            "same_timestamp_lead_hour": 0,
            "lead_denominator": "all_faulty_modules",
            "timestamp_dtype": "int64_seconds",
            "threshold_fit_split": "validation_only",
        },
        "task": config.to_dict()["task"],
        "input": {
            "cadence_mode": "native_observed_rows",
            "raw_feature_count": len(RAW_FEATURES),
            "raw_features": list(RAW_FEATURES),
            "mask_feature_count": len(RAW_FEATURES),
            "delta_feature_count": 1,
            "resampling": False,
            "aggregation": False,
        },
        "training": {
            "objective": (
                "future_window_cross_entropy"
                if config.training.fgl_alpha >= 1.0
                else "future_window_cross_entropy_plus_fgl_kl"
            ),
            "fgl_alpha": config.training.fgl_alpha,
            "temperature": config.training.temperature,
            "fgl_coverage_scale": training.fgl_coverage_scale,
            "student_positive_weight": training.student_positive_weight,
            "teacher_positive_weight": training.teacher_positive_weight,
            "best_student_epoch": training.best_student_epoch,
            "best_student_validation_loss": training.best_student_validation_loss,
            "best_teacher_epoch": training.best_teacher_epoch,
            "best_teacher_validation_loss": training.best_teacher_validation_loss,
            "coverage": coverage,
        },
        "deployment": {
            "checkpoint": "checkpoints/student_best.pt",
            "teacher_used_at_inference": False,
            "teacher_parameters_in_student_checkpoint": False,
        },
        "data_access_order": [
            "index_membership",
            "train_csv",
            "validation_csv",
            "validation_threshold_frozen",
            "test_csv",
        ],
        "primary_metrics": metrics,
        "seconds": {
            "data_preparation_train_validation": data_preparation_seconds,
            "training": training.seconds_total,
            "test_load_and_inference": float(time.perf_counter() - test_load_started),
            "total": float(time.perf_counter() - started),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(resolve_device(config.training.device)),
        },
    }
    _write_json(output_dir / "normalizer.json", standardizer.to_dict())
    _write_json(output_dir / "run_manifest.json", manifest)
    print(
        f"[result] threshold={threshold:.9f} "
        f"final={float(metrics['final_score']):.6f} "
        f"F1={float(metrics['f1_score']):.6f}",
        flush=True,
    )
    return manifest


__all__ = ["infer_student", "run_experiment"]
