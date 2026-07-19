from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAINING_DIR = DATASET_DIR / "training"
INDEX_FILE = DATASET_DIR / "train_test_set_index(in).csv"

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


def safe_slope(values: np.ndarray | pd.Series | list[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return 0.0
    y = arr[mask]
    x = np.arange(len(arr), dtype=float)[mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def first_failure_timestamp(df: pd.DataFrame) -> int | None:
    if "anomaly" not in df.columns or "timestamp" not in df.columns:
        return None
    observed = df["observed"].astype(bool) if "observed" in df.columns else True
    anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0.0) > 0.0
    sub = df.loc[observed & anomaly, "timestamp"]
    if sub.empty:
        return None
    return int(sub.iloc[0])


# Bump this when changing resample/interpolation logic so cached payloads are invalidated.
RESAMPLE_VERSION = "v2_short_gap_only"
# Maximum gap (in 5-min slots) we are willing to linearly interpolate across.
# 6 slots = 30 min. Anything longer than this stays NaN so downstream code/models
# can see and handle the real outage rather than working on fabricated values.
INTERPOLATE_MAX_GAP_SLOTS = 6


def resample_module(raw: pd.DataFrame, freq_seconds: int = 300) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    if "timestamp" not in raw.columns:
        raise ValueError("raw module data must contain a timestamp column")

    df = raw.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError("raw module data has no valid timestamps")

    df["timestamp"] = df["timestamp"].astype("int64")
    sensors = [col for col in SENSORS if col in df.columns]
    for col in sensors:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "anomaly" in df.columns:
        df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0.0)
    else:
        df["anomaly"] = 0.0

    agg = {col: "mean" for col in sensors}
    agg["anomaly"] = "max"
    df = df[["timestamp", *sensors, "anomaly"]].groupby("timestamp", as_index=False).agg(agg)
    df = df.sort_values("timestamp").reset_index(drop=True)

    start_ts = int(df["timestamp"].iloc[0])
    end_ts = int(df["timestamp"].iloc[-1])
    grid = pd.DataFrame(
        {"timestamp": np.arange(start_ts, end_ts + freq_seconds, freq_seconds, dtype=np.int64)}
    )
    out = grid.merge(df, on="timestamp", how="left", indicator=True)
    out["observed"] = (out["_merge"] == "both").astype(int)
    out = out.drop(columns=["_merge"])
    out["anomaly"] = pd.to_numeric(out["anomaly"], errors="coerce").fillna(0.0)

    # Only interpolate INSIDE short gaps (<= INTERPOLATE_MAX_GAP_SLOTS slots = 30 min).
    # Long outages and leading/trailing NaN remain NaN so they cannot fabricate signal.
    for col in sensors:
        values = pd.to_numeric(out[col], errors="coerce")
        values = values.interpolate(
            method="linear",
            limit=INTERPOLATE_MAX_GAP_SLOTS,
            limit_area="inside",
        )
        out[col] = values.astype(float)

    nan_ratios = {col: float(pd.to_numeric(out[col], errors="coerce").isna().mean()) for col in sensors}
    meta = {
        "raw_rows": int(len(raw)),
        "resampled_rows": int(len(out)),
        "observed_ratio": float(out["observed"].mean()) if len(out) else 0.0,
        "start_ts": float(start_ts),
        "end_ts": float(end_ts),
        "sensor_count": int(len(sensors)),
        "anomaly_points": int((out["anomaly"] > 0).sum()),
        "resample_version": RESAMPLE_VERSION,
        "interpolate_max_gap_slots": INTERPOLATE_MAX_GAP_SLOTS,
        "sensor_nan_ratio_mean": float(np.mean(list(nan_ratios.values()))) if nan_ratios else 0.0,
    }
    return out, sensors, meta


def aggregate_window_features(obs: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    features: dict[str, float] = {"obs_coverage_ratio": float(obs["observed"].mean())}
    for col in sensors:
        values = pd.to_numeric(obs[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        features[f"{col}_missing_ratio"] = float(1.0 - mask.mean()) if len(values) else 1.0
        features[f"{col}_valid_points"] = float(mask.sum())
        if mask.sum() == 0:
            stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "q25": 0.0,
                "median": 0.0,
                "q75": 0.0,
                "max": 0.0,
                "range": 0.0,
                "iqr": 0.0,
                "first": 0.0,
                "last": 0.0,
                "delta": 0.0,
                "slope": 0.0,
            }
        else:
            finite = values[mask]
            q25, median, q75 = np.percentile(finite, [25, 50, 75])
            stats = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=0)),
                "min": float(finite.min()),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "max": float(finite.max()),
                "range": float(finite.max() - finite.min()),
                "iqr": float(q75 - q25),
                "first": float(finite[0]),
                "last": float(finite[-1]),
                "delta": float(finite[-1] - finite[0]),
                "slope": safe_slope(values),
            }
        for name, value in stats.items():
            features[f"{col}_{name}"] = value
    return features


def observation_array(obs: pd.DataFrame, sensors: list[str], normalize: bool = False) -> np.ndarray:
    arr = obs[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).any():
        col_medians = np.nanmedian(arr, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        inds = np.where(~np.isfinite(arr))
        arr[inds] = np.take(col_medians, inds[1])
    if normalize and arr.size:
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        arr = (arr - mean) / std
    return arr.astype(np.float32)


def split_modules_stratified(
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    index_df = pd.read_csv(INDEX_FILE)
    index_df["Label"] = pd.to_numeric(index_df["Label"], errors="coerce").fillna(0).astype(int)
    stratify = index_df["Label"] if index_df["Label"].nunique() > 1 else None
    train_val, test = train_test_split(
        index_df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify,
    )
    val_size = val_ratio / max(1.0 - test_ratio, 1e-9)
    stratify_tv = train_val["Label"] if train_val["Label"].nunique() > 1 else None
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_tv,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def evaluate_binary_scores(
    y_true: np.ndarray | pd.Series | list[int],
    y_score: np.ndarray | pd.Series | list[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=labels).ravel()
    if np.unique(y_true_arr).size > 1:
        auc = float(roc_auc_score(y_true_arr, y_score_arr))
        average_precision = float(average_precision_score(y_true_arr, y_score_arr))
    else:
        auc = 0.5
        average_precision = float(y_true_arr.mean()) if len(y_true_arr) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "auc": auc,
        "average_precision": average_precision,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "miss_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "y_pred": y_pred,
        "y_score": y_score_arr,
    }
