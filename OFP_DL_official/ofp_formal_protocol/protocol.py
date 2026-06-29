from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEC_IN_HOUR = 3600.0
DEFAULT_HORIZON_HOURS = (1,)
PRIMARY_HORIZON_HOURS = 1
PRIMARY_HORIZON_SECONDS = int(PRIMARY_HORIZON_HOURS * SEC_IN_HOUR)
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SEED = 42

SENSORS = [
    "temperature",
    "current",
    "currentTXPower",
    "currentRXPower",
    "currentMultiRXPower1",
    "currentMultiRXPower2",
    "currentMultiRXPower3",
    "currentMultiRXPower4",
    "currentMultiTXPower1",
    "currentMultiTXPower2",
    "currentMultiTXPower3",
    "currentMultiTXPower4",
]


@dataclass(frozen=True)
class ModuleMeta:
    file_name: str
    index_label: int
    anomaly_label: int
    first_anomaly_ts: float | None
    n_rows: int
    start_ts: float | None
    end_ts: float | None
    label_mismatch: int


def read_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    required = {"file_name", "folder_index", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{index_path} missing columns: {sorted(missing)}")
    df["folder_index"] = pd.to_numeric(df["folder_index"], errors="raise").astype(int)
    df["Label"] = pd.to_numeric(df["Label"], errors="raise").astype(int)
    return df


def first_anomaly_timestamp(timestamps: np.ndarray, anomaly: np.ndarray) -> float | None:
    positive_ts = timestamps[anomaly > 0]
    positive_ts = positive_ts[np.isfinite(positive_ts)]
    if len(positive_ts) == 0:
        return None
    return float(np.min(positive_ts))


def ahead_label_first_event(
    timestamps: np.ndarray,
    anomaly: np.ndarray,
    horizon_seconds: int = PRIMARY_HORIZON_SECONDS,
) -> np.ndarray:
    """Label only the interval before the first anomaly of a module.

    This labels only timestamps before the first anomaly.  With the official
    OFP-compatible setting, the positive window is the 1-hour interval before
    that first anomaly.
    """
    tf = first_anomaly_timestamp(timestamps, anomaly)
    if tf is None:
        return np.zeros(len(timestamps), dtype=np.int8)
    timestamps = timestamps.astype(float, copy=False)
    label = (timestamps >= tf - horizon_seconds) & (timestamps < tf)
    return label.astype(np.int8)


def ahead_labels_multi_first_event(
    timestamps: np.ndarray,
    anomaly: np.ndarray,
    horizon_hours: tuple[float, ...] | list[float] = DEFAULT_HORIZON_HOURS,
) -> np.ndarray:
    """Return first-event labels for multiple ahead horizons.

    The last horizon is treated as the primary score by the deep-learning
    trainers. With the default setting this returns a single 1h-ahead column,
    constrained to timestamps before the first anomaly.
    """
    hours = tuple(float(h) for h in horizon_hours)
    if not hours:
        raise ValueError("horizon_hours must contain at least one horizon")
    columns = [
        ahead_label_first_event(timestamps, anomaly, horizon_seconds=int(float(h) * SEC_IN_HOUR))
        for h in hours
    ]
    return np.stack(columns, axis=1).astype(np.int8)


def primary_horizon_index(
    horizon_hours: tuple[float, ...] | list[float] = DEFAULT_HORIZON_HOURS,
    primary_horizon_hours: float = PRIMARY_HORIZON_HOURS,
) -> int:
    hours = np.asarray(tuple(float(h) for h in horizon_hours), dtype=np.float64)
    if len(hours) == 0:
        raise ValueError("horizon_hours must contain at least one horizon")
    return int(np.argmin(np.abs(hours - float(primary_horizon_hours))))


def valid_time_mask_first_event(timestamps: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """Return rows that are valid for first-fault prediction.

    Negative modules keep all finite timestamps. For a faulted module, only
    timestamps strictly before the first anomaly are valid; rows at or after
    the first anomaly are outside the prediction task.
    """
    timestamps = timestamps.astype(float, copy=False)
    valid = np.isfinite(timestamps)
    tf = first_anomaly_timestamp(timestamps, anomaly)
    if tf is None:
        return valid.astype(bool)
    return (valid & (timestamps < tf)).astype(bool)


def ahead_label_and_valid_mask_first_event(
    timestamps: np.ndarray,
    anomaly: np.ndarray,
    horizon_seconds: int = PRIMARY_HORIZON_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the primary first-event label and its valid pre-event mask."""
    return (
        ahead_label_first_event(timestamps, anomaly, horizon_seconds=horizon_seconds),
        valid_time_mask_first_event(timestamps, anomaly),
    )


def ahead_labels_and_valid_mask_multi_first_event(
    timestamps: np.ndarray,
    anomaly: np.ndarray,
    horizon_hours: tuple[float, ...] | list[float] = DEFAULT_HORIZON_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return multi-horizon first-event labels and the pre-event valid mask."""
    return (
        ahead_labels_multi_first_event(timestamps, anomaly, horizon_hours=horizon_hours),
        valid_time_mask_first_event(timestamps, anomaly),
    )


def read_module_meta(data_dir: Path, file_name: str, index_label: int) -> ModuleMeta:
    path = data_dir / file_name
    df = pd.read_csv(path, usecols=lambda c: c in {"timestamp", "anomaly"})
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    finite_ts = timestamps[np.isfinite(timestamps)]
    first_ts = float(np.min(finite_ts)) if len(finite_ts) else None
    last_ts = float(np.max(finite_ts)) if len(finite_ts) else None
    anomaly_label = int(np.any(anomaly > 0))
    first_fault_ts = first_anomaly_timestamp(timestamps, anomaly)
    return ModuleMeta(
        file_name=file_name,
        index_label=int(index_label),
        anomaly_label=anomaly_label,
        first_anomaly_ts=first_fault_ts,
        n_rows=int(len(df)),
        start_ts=first_ts,
        end_ts=last_ts,
        label_mismatch=int(anomaly_label != int(index_label)),
    )


def build_file_metadata(data_dir: Path, index_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    total = len(index_df)
    for i, row in enumerate(index_df.itertuples(index=False), 1):
        meta = read_module_meta(data_dir, row.file_name, int(row.Label))
        data = meta.__dict__.copy()
        data["outer_fold"] = int(row.folder_index)
        rows.append(data)
        if i % 1000 == 0:
            print(f"[meta] scanned {i}/{total} modules")
    return pd.DataFrame(rows)


def split_train_val_for_fold(
    metadata_df: pd.DataFrame,
    fold: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    train_pool = metadata_df[metadata_df["outer_fold"] != int(fold)].copy()
    test_df = metadata_df[metadata_df["outer_fold"] == int(fold)].copy()
    if train_pool.empty or test_df.empty:
        raise ValueError(f"fold {fold} has empty train or test split")

    stratify = train_pool["anomaly_label"]
    if stratify.nunique() < 2 or stratify.value_counts().min() < 2:
        train_df, val_df = train_test_split(
            train_pool,
            test_size=float(val_fraction),
            random_state=seed + int(fold),
            shuffle=True,
        )
    else:
        train_df, val_df = train_test_split(
            train_pool,
            test_size=float(val_fraction),
            random_state=seed + int(fold),
            shuffle=True,
            stratify=stratify,
        )
    return (
        train_df["file_name"].tolist(),
        val_df["file_name"].tolist(),
        test_df["file_name"].tolist(),
    )


def build_formal_manifest(
    data_dir: Path,
    index_path: Path,
    out_dir: Path,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_df = read_index(Path(index_path))
    metadata_df = build_file_metadata(data_dir, index_df)
    metadata_path = out_dir / "module_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    rows: list[dict] = []
    for fold in sorted(metadata_df["outer_fold"].dropna().astype(int).unique()):
        train_files, val_files, test_files = split_train_val_for_fold(
            metadata_df, fold, val_fraction=val_fraction, seed=seed
        )
        role_map = {
            **{name: "train" for name in train_files},
            **{name: "val" for name in val_files},
            **{name: "test" for name in test_files},
        }
        for name, role in role_map.items():
            meta = metadata_df.loc[metadata_df["file_name"] == name].iloc[0]
            rows.append(
                {
                    "fold": int(fold),
                    "role": role,
                    "file_name": name,
                    "anomaly_label": int(meta["anomaly_label"]),
                    "first_anomaly_ts": meta["first_anomaly_ts"],
                    "n_rows": int(meta["n_rows"]),
                    "source_outer_fold": int(meta["outer_fold"]),
                }
            )
    manifest_df = pd.DataFrame(rows)
    manifest_path = out_dir / "formal_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return metadata_df, manifest_df


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"fold", "role", "file_name", "anomaly_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing columns: {sorted(missing)}")
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["anomaly_label"] = pd.to_numeric(df["anomaly_label"], errors="raise").astype(int)
    return df


def files_for_fold(manifest_df: pd.DataFrame, fold: int) -> tuple[list[str], list[str], list[str]]:
    df = manifest_df[manifest_df["fold"] == int(fold)]
    train_files = df.loc[df["role"] == "train", "file_name"].tolist()
    val_files = df.loc[df["role"] == "val", "file_name"].tolist()
    test_files = df.loc[df["role"] == "test", "file_name"].tolist()
    if not train_files or not val_files or not test_files:
        raise ValueError(f"fold {fold} must have non-empty train/val/test roles")
    return train_files, val_files, test_files


def metrics_from_module_scores(module_rows: list[dict], threshold: float) -> dict[str, float]:
    true_pos, pred_pos = set(), set()
    lead_hours: list[float] = []
    all_names = set()
    for row in module_rows:
        name = str(row["file_name"])
        all_names.add(name)
        true_label = int(row["true_label"])
        true_ts = row["true_ts"]
        if true_label:
            true_pos.add(name)
        scores = np.asarray(row["scores"], dtype=float)
        timestamps = np.asarray(row["timestamps"], dtype=float)
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            valid_mask = np.isfinite(timestamps)
            if true_ts is not None:
                valid_mask = valid_mask & (timestamps < float(true_ts))
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)
        idx = np.where((scores >= float(threshold)) & valid_mask)[0]
        if len(idx) == 0:
            continue
        pred_ts = float(timestamps[int(idx[0])])
        if true_label == 0:
            pred_pos.add(name)
        elif true_ts is not None and pred_ts < float(true_ts):
            pred_pos.add(name)
            lead_hours.append((float(true_ts) - pred_ts) / SEC_IN_HOUR)

    hit_pos = true_pos & pred_pos
    tp = len(hit_pos)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = len(all_names) - tp - fp - fn

    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(all_names) if all_names else 0.0
    avg_lead_hour = float(np.mean(lead_hours)) if lead_hours else 0.0
    min_lead_hour = float(np.min(lead_hours)) if lead_hours else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    return {
        "final_score": final_score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "avg_lead_hour": avg_lead_hour,
        "min_lead_hour": min_lead_hour,
        "all_hit_cnt": tp,
        "all_predict_pos_cnt": len(pred_pos),
        "all_true_pos_cnt": len(true_pos),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_module_cnt": len(all_names),
    }


def choose_threshold_by_final_score(
    module_rows: list[dict],
    grid_size: int = 99,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in np.linspace(float(min_threshold), float(max_threshold), int(grid_size)):
        metrics = metrics_from_module_scores(module_rows, float(threshold))
        key = (
            metrics["final_score"],
            metrics["f1_score"],
            metrics["recall"],
            metrics["precision"],
        )
        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
            continue
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    assert best_metrics is not None
    best_metrics = dict(best_metrics)
    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics
