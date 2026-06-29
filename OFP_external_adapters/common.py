from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.ofp_protocol.evaluator import first_positive_timestamp, evaluate_prediction_folder


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


@dataclass
class AlarmCfg:
    threshold_grid: str = "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50"
    threshold_metric: str = "f1_score"
    val_fraction: float = 0.2
    smoothing: str = "ema"
    smooth_window: int = 1
    consecutive_k: int = 1
    first_alarm_only: bool = True
    seed: int = 42


def file_label_map(index_df: pd.DataFrame) -> dict[str, int]:
    return {
        str(row["file_name"]): int(row["Label"])
        for _, row in index_df[["file_name", "Label"]].drop_duplicates("file_name").iterrows()
    }


def split_train_val_files(
    train_files: list[str],
    label_by_file: dict[str, int],
    cfg: AlarmCfg,
    fold: int,
) -> tuple[list[str], list[str]]:
    frac = float(cfg.val_fraction)
    if frac <= 0.0 or len(train_files) < 4:
        return list(train_files), []
    rng = np.random.default_rng(int(cfg.seed) + 1009 * int(fold))
    train_out: list[str] = []
    val_out: list[str] = []
    for label in sorted({int(label_by_file.get(name, 0)) for name in train_files}):
        group = [name for name in train_files if int(label_by_file.get(name, 0)) == label]
        order = np.asarray(group, dtype=object)
        rng.shuffle(order)
        n_val = int(round(len(order) * frac))
        if len(order) > 1:
            n_val = min(max(1, n_val), len(order) - 1)
        else:
            n_val = 0
        val_out.extend(str(x) for x in order[:n_val])
        train_out.extend(str(x) for x in order[n_val:])
    train_out.sort()
    val_out.sort()
    if not train_out:
        return list(train_files), []
    return train_out, val_out


def parse_threshold_grid(text: str) -> list[float]:
    raw = str(text).strip()
    if not raw:
        return []
    if ":" in raw:
        parts = [float(x) for x in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("threshold_grid range must be start:end:step")
        start, end, step = parts
        if step <= 0:
            raise ValueError("threshold_grid step must be positive")
        values = []
        cur = start
        while cur <= end + 1e-12:
            values.append(round(float(cur), 10))
            cur += step
        return values
    return sorted({float(x) for x in raw.replace(";", ",").split(",") if x.strip()})


def smooth_scores(scores: np.ndarray, mode: str, window: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return values
    mode = str(mode).lower()
    window = max(1, int(window))
    if mode in {"none", "off", "0"} or window <= 1:
        return values
    series = pd.Series(values)
    if mode == "ema":
        return series.ewm(span=window, adjust=False).mean().to_numpy(dtype=np.float64)
    if mode in {"mean", "rolling"}:
        return series.rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)
    raise ValueError(f"Unknown smoothing={mode!r}")


def apply_alarm_strategy(score_frame: pd.DataFrame, threshold: float, cfg: AlarmCfg) -> pd.DataFrame:
    if score_frame.empty:
        return pd.DataFrame(columns=["timestamp", "predict", "score", "score_smooth"])
    frame = score_frame[["timestamp", "score"]].copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "score"]).sort_values("timestamp")
    values = frame["score"].to_numpy(dtype=np.float64)
    smoothed = smooth_scores(values, cfg.smoothing, cfg.smooth_window)
    raw = smoothed >= float(threshold)
    k = max(1, int(cfg.consecutive_k))
    if k > 1:
        pred = np.zeros(len(raw), dtype=bool)
        run = 0
        for idx, flag in enumerate(raw):
            run = run + 1 if bool(flag) else 0
            pred[idx] = run >= k
    else:
        pred = raw
    if bool(cfg.first_alarm_only) and pred.any():
        first = int(np.flatnonzero(pred)[0])
        only = np.zeros_like(pred, dtype=bool)
        only[first] = True
        pred = only
    frame["score_smooth"] = smoothed
    frame["predict"] = pred.astype(int)
    return frame[["timestamp", "predict", "score", "score_smooth"]]


def _first_positive_timestamp_frame(
    frame: pd.DataFrame,
    predict_column: str,
    before_ts: float | None = None,
) -> tuple[int, float | None]:
    if frame.empty or predict_column not in frame.columns or "timestamp" not in frame.columns:
        return 0, None
    label = pd.to_numeric(frame[predict_column], errors="coerce").fillna(0.0)
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
    valid = (label > 0) & timestamps.notna()
    if before_ts is not None:
        valid = valid & (timestamps.astype(float) < float(before_ts))
    positive = pd.to_numeric(frame.loc[valid, "timestamp"], errors="coerce").dropna()
    if positive.empty:
        return 0, None
    return 1, float(positive.min())


def summary_from_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    true_pos = set(detail_df.loc[detail_df["true_label"] > 0, "file_name"])
    pred_pos = set(detail_df.loc[detail_df["valid_predict_positive"] > 0, "file_name"])
    hit_pos = true_pos & pred_pos
    tp = len(hit_pos)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = len(detail_df) - tp - fp - fn
    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(detail_df) if len(detail_df) else 0.0
    lead_hours = detail_df.loc[detail_df["hit"] > 0, "lead_hour"].dropna().astype(float)
    avg_lead_hour = float(lead_hours.mean()) if not lead_hours.empty else 0.0
    min_lead_hour = float(lead_hours.min()) if not lead_hours.empty else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    return pd.DataFrame(
        {
            "Item": [
                "final_score",
                "f1_score",
                "precision",
                "recall",
                "all_hit_cnt",
                "all_predict_pos_cnt",
                "all_true_pos_cnt",
                "avg_lead_score",
                "avg_lead_hour",
                "min_lead_score",
                "min_lead_hour",
                "lead_pread_cnt",
                "accuracy",
                "tp",
                "fp",
                "fn",
                "tn",
                "evaluated_module_cnt",
            ],
            "Value": [
                final_score,
                f1,
                precision,
                recall,
                tp,
                len(pred_pos),
                len(true_pos),
                avg_lead_score,
                avg_lead_hour,
                min_lead_score,
                min_lead_hour,
                int(detail_df["hit"].sum()),
                accuracy,
                tp,
                fp,
                fn,
                tn,
                len(detail_df),
            ],
        }
    )


def evaluate_prediction_frames(
    prediction_frames: dict[str, pd.DataFrame],
    label_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name in sorted(prediction_frames):
        label_path = Path(label_dir) / name
        if not label_path.exists():
            continue
        true_label, true_ts = first_positive_timestamp(label_path, "anomaly")
        predict_label, predict_ts = _first_positive_timestamp_frame(
            prediction_frames[name],
            "predict",
            before_ts=true_ts if true_label > 0 else None,
        )
        valid_predict_positive = 0
        hit = 0
        lead_hour = None
        if predict_label > 0:
            if true_label == 0:
                valid_predict_positive = 1
            elif true_ts is not None and predict_ts is not None and true_ts > predict_ts:
                valid_predict_positive = 1
                hit = 1
                lead_hour = abs(true_ts - predict_ts) / 3600.0
        rows.append(
            {
                "file_name": name,
                "true_label": int(true_label),
                "predict_label": int(predict_label),
                "true_ts": true_ts,
                "predict_ts": predict_ts,
                "valid_predict_positive": int(valid_predict_positive),
                "hit": int(hit),
                "lead_hour": lead_hour,
            }
        )
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise ValueError("No matching score/label files to evaluate")
    return summary_from_detail(detail_df), detail_df


def select_threshold(
    score_frames: dict[str, pd.DataFrame],
    label_dir: Path,
    cfg: AlarmCfg,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    candidates = parse_threshold_grid(cfg.threshold_grid)
    if not candidates:
        candidates = [0.5]
    rows: list[dict] = []
    best_threshold = float(candidates[0])
    best_metrics: dict[str, float] = {}
    best_key: tuple[float, float, float, float, float, float] | None = None
    metric_key = str(cfg.threshold_metric)
    for threshold in candidates:
        pred_frames = {name: apply_alarm_strategy(frame, threshold, cfg) for name, frame in score_frames.items()}
        summary_df, _detail_df = evaluate_prediction_frames(pred_frames, label_dir)
        metrics = {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}
        value = float(metrics.get(metric_key, metrics.get("f1_score", 0.0)))
        item = {"threshold": float(threshold)}
        item.update(metrics)
        rows.append(item)
        key = (
            value,
            float(metrics.get("final_score", 0.0)),
            float(metrics.get("precision", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            float(metrics.get("recall", 0.0)),
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics, pd.DataFrame(rows)


def write_prediction_frames(prediction_frames: dict[str, pd.DataFrame], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, frame in prediction_frames.items():
        path = out_dir / name
        frame.to_csv(path, index=False)
        total += int(len(frame))
    return total


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def aggregate_results(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "protocol": item.get("protocol", ""),
            "threshold": item["threshold"],
            "seconds": item.get("seconds", 0.0),
        }
        row.update(item.get("metrics", {}) or {})
        rows.append(row)
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new.drop_duplicates(subset=["model", "fold", "protocol"], keep="last", inplace=True)
    new.sort_values(["model", "protocol", "fold"], inplace=True)
    new.to_csv(path, index=False)
    numeric = [c for c in new.columns if c not in {"model", "fold", "protocol"} and pd.api.types.is_numeric_dtype(new[c])]
    summary = new.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def load_score_frame(score_dir: Path, file_name: str, score_col: str = "score") -> pd.DataFrame:
    path = Path(score_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} must contain timestamp")
    if score_col not in frame.columns:
        raise ValueError(f"{path} must contain {score_col!r}")
    return frame[["timestamp", score_col]].rename(columns={score_col: "score"})


def stable_rng(name: str, seed: int) -> np.random.Generator:
    stable = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(int(seed) + int(stable))


__all__ = [
    "SENSORS",
    "AlarmCfg",
    "aggregate_results",
    "apply_alarm_strategy",
    "asdict",
    "evaluate_prediction_folder",
    "evaluate_prediction_frames",
    "file_label_map",
    "files_for_index_fold",
    "load_score_frame",
    "metrics_to_dict",
    "parse_threshold_grid",
    "read_index",
    "select_threshold",
    "split_train_val_files",
    "stable_rng",
    "write_prediction_frames",
]
