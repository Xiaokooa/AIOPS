"""Strict first-event XGBoost pipeline shared by B0, B1 and B2."""
from __future__ import annotations

import gc
import hashlib
import json
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from .evaluation import compare_with_readme, evaluate_prediction_dir, metrics_from_decisions, write_evaluation
from .features import (
    RAW_FEATURES,
    build_feature_frame,
    feature_manifest,
    feature_names as feature_names_for_set,
    groups_for_features,
    normalize_feature_set,
    validate_feature_schema_version,
)
from .labels import first_event_target, validate_time_order


@dataclass
class FoldRun:
    fold: int
    train_modules: int
    validation_modules: int
    train_rows: int
    positive_train_rows: int
    threshold: float
    horizon_hours: float
    seconds: float
    device: str
    config_fingerprint: str
    run_fingerprint: str
    metrics: dict[str, float]


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_fingerprint(
    *,
    config: dict,
    fold: int,
    data_dir: Path,
    train_files: Sequence[str],
    validation_files: Sequence[str],
    device: str,
    dataset_metadata_fingerprint: str | None = None,
) -> str:
    payload = {
        "config": config,
        "fold": int(fold),
        "data_dir": str(Path(data_dir).resolve()),
        "train_files": list(train_files),
        "validation_files": list(validation_files),
        "device": str(device),
    }
    if dataset_metadata_fingerprint is not None:
        payload["dataset_metadata_fingerprint"] = str(dataset_metadata_fingerprint)
    return config_fingerprint(payload)


def dataset_metadata_fingerprint(data_dir: Path, file_names: Sequence[str]) -> str:
    """Fingerprint file names, sizes and nanosecond mtimes without reading 1.88 GB."""

    data_dir = Path(data_dir)
    records = []
    for name in sorted(set(str(value) for value in file_names)):
        stat = (data_dir / name).stat()
        records.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
    return config_fingerprint({"files": records})


def data_split_fingerprint(
    *,
    fold: int,
    data_dir: Path,
    train_files: Sequence[str],
    validation_files: Sequence[str],
    dataset_fingerprint: str,
) -> str:
    return config_fingerprint(
        {
            "fold": int(fold),
            "data_dir": str(Path(data_dir).resolve()),
            "train_files": list(train_files),
            "validation_files": list(validation_files),
            "dataset_metadata_fingerprint": str(dataset_fingerprint),
        }
    )


def read_index(index_path: Path) -> pd.DataFrame:
    index_df = pd.read_csv(index_path)
    required = {"file_name", "folder_index", "Label"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"index file missing columns: {sorted(missing)}")
    index_df = index_df.copy()
    index_df["folder_index"] = pd.to_numeric(index_df["folder_index"], errors="raise").astype(int)
    index_df["Label"] = pd.to_numeric(index_df["Label"], errors="raise").astype(int)
    if index_df["file_name"].duplicated().any():
        raise ValueError("index contains duplicate file names")
    return index_df


def stratified_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.copy()
    if limit < 2:
        raise ValueError("a stratified module limit must be at least 2")
    positive = frame.loc[frame["Label"] > 0]
    negative = frame.loc[frame["Label"] <= 0]
    n_pos = min(len(positive), max(1, limit // 2))
    n_neg = min(len(negative), limit - n_pos)
    if n_pos + n_neg < limit:
        n_pos = min(len(positive), limit - n_neg)
    selected = pd.concat([positive.head(n_pos), negative.head(n_neg)]).sort_index()
    return selected.head(limit).copy()


def read_training_module(
    path: Path,
    horizon_hours: float,
    feature_set: str = "raw",
) -> tuple[np.ndarray, np.ndarray]:
    needed = {"timestamp", "anomaly", *RAW_FEATURES}
    frame = pd.read_csv(path, usecols=lambda c: c in needed)
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    validate_time_order(frame)
    target, valid, _ = first_event_target(frame, horizon_hours=horizon_hours)
    values = build_feature_frame(frame, feature_set).to_numpy(dtype=np.float32)
    return values[valid], target[valid]


def read_prediction_module(
    path: Path,
    feature_set: str = "raw",
) -> tuple[np.ndarray, np.ndarray]:
    needed = {"timestamp", *RAW_FEATURES}
    frame = pd.read_csv(path, usecols=lambda c: c in needed)
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    timestamps = validate_time_order(frame).astype(np.int64)
    values = build_feature_frame(frame, feature_set).to_numpy(dtype=np.float32)
    return timestamps, values


class CSVModuleIter(xgb.DataIter):
    """Stream module CSVs in bounded batches into an XGBoost matrix."""

    def __init__(
        self,
        data_dir: Path,
        file_names: Sequence[str],
        horizon_hours: float,
        feature_set: str,
        batch_files: int,
        cache_prefix: str | None = None,
        on_host: bool = True,
    ) -> None:
        super().__init__(cache_prefix=cache_prefix, release_data=True, on_host=on_host)
        self.data_dir = Path(data_dir)
        self.file_names = list(file_names)
        self.horizon_hours = float(horizon_hours)
        self.feature_set = normalize_feature_set(feature_set)
        self.feature_names = feature_names_for_set(self.feature_set)
        self.batch_files = int(batch_files)
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def next(self, input_data: Callable) -> bool:
        if self._cursor >= len(self.file_names):
            return False
        names = self.file_names[self._cursor : self._cursor + self.batch_files]
        self._cursor += len(names)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for name in names:
            x, y = read_training_module(
                self.data_dir / name,
                self.horizon_hours,
                self.feature_set,
            )
            if len(x):
                xs.append(x)
                ys.append(y)
        if not xs:
            return self.next(input_data)
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        input_data(data=X, label=y, feature_names=list(self.feature_names))
        return True


@lru_cache(maxsize=1)
def _cuda_probe() -> bool:
    """Return whether this machine, not only the wheel, can train on CUDA."""

    if not bool(xgb.build_info().get("USE_CUDA")):
        return False
    try:
        X = np.asarray([[0.0], [1.0]], dtype=np.float32)
        y = np.asarray([0.0, 1.0], dtype=np.float32)
        matrix = xgb.QuantileDMatrix(X, label=y, max_bin=2)
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 1,
                "max_bin": 2,
                "verbosity": 0,
            },
            matrix,
            num_boost_round=1,
        )
        return booster_effective_device(booster) == "cuda"
    except (xgb.core.XGBoostError, RuntimeError):
        return False


def booster_effective_device(booster: xgb.Booster) -> str:
    """Normalize the actual device recorded by a trained XGBoost booster."""

    config = json.loads(booster.save_config())
    device = str(config["learner"]["generic_param"]["device"]).lower()
    return "cuda" if device.startswith("cuda") else "cpu"


def resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not _cuda_probe():
            raise RuntimeError(
                "CUDA was explicitly requested but XGBoost could not train a "
                "CUDA booster; use --device cpu or make a GPU visible"
            )
        return "cuda"
    if requested == "auto":
        return "cuda" if _cuda_probe() else "cpu"
    raise ValueError(f"unknown device={requested!r}")


def build_training_matrix(
    iterator: CSVModuleIter,
    matrix_mode: str,
    max_bin: int,
    nthread: int,
) -> xgb.DMatrix:
    if matrix_mode == "quantile":
        return xgb.QuantileDMatrix(iterator, max_bin=max_bin, nthread=nthread)
    if matrix_mode == "external_memory":
        return xgb.ExtMemQuantileDMatrix(iterator, max_bin=max_bin, nthread=nthread)
    raise ValueError(f"unknown matrix_mode={matrix_mode!r}")


def predict_files(
    booster: xgb.Booster,
    data_dir: Path,
    file_names: Sequence[str],
    out_dir: Path,
    threshold: float,
    feature_set: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for idx, name in enumerate(file_names, 1):
        timestamps, X = read_prediction_module(data_dir / name, feature_set)
        score = booster.inplace_predict(X)
        pred = (score >= float(threshold)).astype(np.int8)
        pd.DataFrame({"timestamp": timestamps, "predict": pred, "score": score}).to_csv(
            out_dir / name, index=False
        )
        total_rows += len(X)
        if idx % 500 == 0:
            print(f"  [predict] {idx}/{len(file_names)} modules")
    return total_rows


def write_feature_importance(
    booster: xgb.Booster,
    selected_feature_names: Sequence[str],
    output_path: Path,
) -> None:
    """Write all features, including unused zero-importance features."""

    importance_types = ("weight", "gain", "cover", "total_gain", "total_cover")
    scores = {
        importance_type: booster.get_score(importance_type=importance_type)
        for importance_type in importance_types
    }
    groups = groups_for_features(selected_feature_names)
    rows = []
    for name, group in zip(selected_feature_names, groups):
        row: dict[str, object] = {"feature": name, "group": group}
        row.update(
            {
                importance_type: float(scores[importance_type].get(name, 0.0))
                for importance_type in importance_types
            }
        )
        rows.append(row)
    importance = pd.DataFrame(rows)
    total_gain = float(importance["total_gain"].sum())
    importance["total_gain_share"] = (
        importance["total_gain"] / total_gain if total_gain > 0 else 0.0
    )
    importance = importance.sort_values(
        ["total_gain", "feature"], ascending=[False, True]
    )
    importance.to_csv(output_path, index=False)


def run_fold(
    *,
    fold: int,
    index_df: pd.DataFrame,
    data_dir: Path,
    out_root: Path,
    config: dict,
    max_train_modules: int | None,
    max_validation_modules: int | None,
    overwrite: bool,
) -> FoldRun:
    feature_set = normalize_feature_set(str(config["feature_set"]))
    validate_feature_schema_version(config.get("feature_schema_version"))
    selected_feature_names = feature_names_for_set(feature_set)
    fold_dir = out_root / f"fold_{fold}"
    summary_path = fold_dir / "fold_summary.json"
    train_index = stratified_limit(index_df.loc[index_df["folder_index"] != fold], max_train_modules)
    validation_index = stratified_limit(index_df.loc[index_df["folder_index"] == fold], max_validation_modules)
    train_files = train_index["file_name"].astype(str).tolist()
    validation_files = validation_index["file_name"].astype(str).tolist()
    missing = [name for name in [*train_files, *validation_files] if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset is missing {len(missing)} indexed files; first={missing[0]}")
    dataset_fingerprint = dataset_metadata_fingerprint(
        data_dir,
        [*train_files, *validation_files],
    )
    split_fingerprint = data_split_fingerprint(
        fold=fold,
        data_dir=data_dir,
        train_files=train_files,
        validation_files=validation_files,
        dataset_fingerprint=dataset_fingerprint,
    )

    xgb_cfg = dict(config["xgboost"])
    num_boost_round = int(xgb_cfg.pop("num_boost_round"))
    device = resolve_device(str(xgb_cfg.get("device", "auto")))
    fingerprint = config_fingerprint(config)
    execution_fingerprint = run_fingerprint(
        config=config,
        fold=fold,
        data_dir=data_dir,
        train_files=train_files,
        validation_files=validation_files,
        device=device,
        dataset_metadata_fingerprint=dataset_fingerprint,
    )
    if summary_path.exists() and not overwrite:
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        if stored.get("run_fingerprint") != execution_fingerprint:
            raise ValueError(
                f"fold {fold} was produced by a different config/data/split; "
                "rerun with --overwrite"
            )
        return FoldRun(**stored)

    xgb_cfg["device"] = device
    xgb_cfg["verbosity"] = 1
    max_bin = int(xgb_cfg["max_bin"])
    nthread = int(xgb_cfg["nthread"])
    horizon_hours = float(config["horizon_hours"])
    threshold = float(config["threshold"])
    matrix_mode = str(config["matrix_mode"])
    batch_files = int(config["batch_files"])
    if matrix_mode == "external_memory" and device == "cuda":
        raise ValueError(
            "XGBoost 3.1 cannot train a CUDA booster from CPU-backed "
            "ExtMemQuantileDMatrix; use matrix_mode='quantile' for CUDA or "
            "device='cpu' for external memory"
        )

    started = time.time()
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "fold": int(fold),
                "data_dir": str(Path(data_dir).resolve()),
                "train_files": train_files,
                "validation_files": validation_files,
                "dataset_metadata_fingerprint": dataset_fingerprint,
                "data_split_fingerprint": split_fingerprint,
                "config_fingerprint": fingerprint,
                "run_fingerprint": execution_fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[fold {fold}] train_modules={len(train_files)} "
        f"validation_modules={len(validation_files)} features={len(selected_feature_names)}"
    )
    cache_label = str(config.get("experiment_name", "ofp")).replace(" ", "_")
    with tempfile.TemporaryDirectory(prefix=f"{cache_label}_fold{fold}_", dir=fold_dir) as cache_dir:
        cache_prefix = str(Path(cache_dir) / "xgb-cache") if matrix_mode == "external_memory" else None
        iterator = CSVModuleIter(
            data_dir=data_dir,
            file_names=train_files,
            horizon_hours=horizon_hours,
            feature_set=feature_set,
            batch_files=batch_files,
            cache_prefix=cache_prefix,
            on_host=matrix_mode != "external_memory",
        )
        dtrain = build_training_matrix(iterator, matrix_mode, max_bin=max_bin, nthread=nthread)
        train_rows = int(dtrain.num_row())
        train_label = dtrain.get_label()
        positive_train_rows = int(np.sum(train_label > 0))
        if train_rows <= 0 or positive_train_rows <= 0 or positive_train_rows >= train_rows:
            raise ValueError(
                f"invalid training labels: rows={train_rows}, positive={positive_train_rows}"
            )
        print(
            f"[fold {fold}] train_rows={train_rows} positive_rows={positive_train_rows} "
            f"device={device} matrix={matrix_mode}"
        )
        booster = xgb.train(xgb_cfg, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
        actual_device = booster_effective_device(booster)
        if actual_device != device:
            raise RuntimeError(
                f"XGBoost silently changed device from {device!r} to "
                f"{actual_device!r}; refusing to write a misleading run manifest"
            )
        booster.save_model(fold_dir / "model.ubj")
        (fold_dir / "booster_config.json").write_text(booster.save_config(), encoding="utf-8")
        write_feature_importance(
            booster,
            selected_feature_names,
            fold_dir / "feature_importance.csv",
        )
        del dtrain, iterator, train_label
        gc.collect()

        pred_dir = fold_dir / "predictions"
        predict_files(
            booster,
            data_dir,
            validation_files,
            pred_dir,
            threshold,
            feature_set,
        )

    metrics, detail_df = evaluate_prediction_dir(pred_dir, data_dir, validation_files)
    write_evaluation(fold_dir / "evaluation", metrics, detail_df)
    result = FoldRun(
        fold=int(fold),
        train_modules=len(train_files),
        validation_modules=len(validation_files),
        train_rows=train_rows,
        positive_train_rows=positive_train_rows,
        threshold=threshold,
        horizon_hours=horizon_hours,
        seconds=time.time() - started,
        device=device,
        config_fingerprint=fingerprint,
        run_fingerprint=execution_fingerprint,
        metrics=metrics,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def aggregate_folds(
    out_root: Path,
    folds: Sequence[int],
    config: dict,
    compare_readme: bool = True,
    expected_files: Sequence[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    details = []
    fold_rows = []
    importance_rows = []
    split_validation_files: list[str] = []
    dataset_fingerprints: list[str] = []
    split_fingerprints: dict[str, str] = {}
    expected_config_fingerprint = config_fingerprint(config)
    for fold in folds:
        fold_dir = out_root / f"fold_{fold}"
        detail = pd.read_csv(fold_dir / "evaluation" / "module_decisions.csv")
        detail["fold"] = int(fold)
        details.append(detail)
        fold_data = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        if fold_data.get("config_fingerprint") != expected_config_fingerprint:
            raise ValueError(f"fold {fold} config fingerprint does not match this aggregation")
        split_data = json.loads((fold_dir / "split_manifest.json").read_text(encoding="utf-8"))
        if split_data.get("run_fingerprint") != fold_data.get("run_fingerprint"):
            raise ValueError(f"fold {fold} split/summary fingerprints disagree")
        split_validation_files.extend(str(v) for v in split_data["validation_files"])
        dataset_fingerprints.append(str(split_data.get("dataset_metadata_fingerprint", "")))
        split_fingerprints[str(int(fold))] = str(split_data.get("data_split_fingerprint", ""))
        row = {k: v for k, v in fold_data.items() if k != "metrics"}
        row.update(fold_data["metrics"])
        fold_rows.append(row)
        importance_path = fold_dir / "feature_importance.csv"
        if importance_path.is_file():
            importance = pd.read_csv(importance_path)
            importance["fold"] = int(fold)
            importance_rows.append(importance)
    pooled_detail = pd.concat(details, ignore_index=True)
    if pooled_detail["file_name"].duplicated().any():
        raise ValueError("pooled folds contain duplicate module files")
    actual_files = set(pooled_detail["file_name"].astype(str))
    split_files = set(split_validation_files)
    if actual_files != split_files or len(split_validation_files) != len(split_files):
        raise ValueError("pooled predictions do not exactly match fold split manifests")
    if expected_files is not None:
        formal_files = set(str(v) for v in expected_files)
        if actual_files != formal_files or len(formal_files) != len(list(expected_files)):
            raise ValueError("formal pooled predictions do not exactly cover the expected index files")
    nonempty_dataset_fingerprints = {value for value in dataset_fingerprints if value}
    if len(nonempty_dataset_fingerprints) > 1:
        raise ValueError("folds were produced from different dataset metadata")
    pooled_metrics = metrics_from_decisions(pooled_detail)
    pooled_dir = out_root / "pooled"
    write_evaluation(pooled_dir, pooled_metrics, pooled_detail)
    pd.DataFrame(fold_rows).to_csv(out_root / "fold_metrics.csv", index=False)
    if importance_rows:
        fold_importance = pd.concat(importance_rows, ignore_index=True)
        fold_importance.to_csv(out_root / "feature_importance_folds.csv", index=False)
        feature_importance = (
            fold_importance.groupby(["feature", "group"], as_index=False)
            .agg(
                mean_total_gain_share=("total_gain_share", "mean"),
                std_total_gain_share=("total_gain_share", "std"),
                mean_weight=("weight", "mean"),
                folds_used=("weight", lambda values: int((values > 0).sum())),
            )
            .fillna({"std_total_gain_share": 0.0})
            .sort_values("mean_total_gain_share", ascending=False)
        )
        feature_importance.to_csv(out_root / "feature_importance_by_feature.csv", index=False)
        group_by_fold = (
            fold_importance.groupby(["fold", "group"], as_index=False)["total_gain_share"].sum()
        )
        group_importance = (
            group_by_fold.groupby("group", as_index=False)
            .agg(
                mean_total_gain_share=("total_gain_share", "mean"),
                std_total_gain_share=("total_gain_share", "std"),
            )
            .fillna({"std_total_gain_share": 0.0})
            .sort_values("mean_total_gain_share", ascending=False)
        )
        group_importance.to_csv(out_root / "feature_importance_by_group.csv", index=False)
    comparison = compare_with_readme(pooled_metrics) if compare_readme else pd.DataFrame()
    if compare_readme:
        comparison.to_csv(out_root / "readme_comparison.csv", index=False)
    manifest = {
        "config": config,
        "effective_device_by_fold": {
            str(int(row["fold"])): str(row["device"])
            for row in fold_rows
        },
        "run_fingerprint_by_fold": {
            str(int(row["fold"])): str(row["run_fingerprint"])
            for row in fold_rows
        },
        "data_split_fingerprint_by_fold": split_fingerprints,
        "dataset_metadata_fingerprint": (
            next(iter(nonempty_dataset_fingerprints))
            if nonempty_dataset_fingerprints
            else None
        ),
        "feature_manifest": feature_manifest(str(config["feature_set"])),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
        "pooled_metrics": pooled_metrics,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pooled_metrics, comparison
