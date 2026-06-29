from __future__ import annotations

import math

import numpy as np
import pandas as pd


def binary_window_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def module_event_metrics(meta: pd.DataFrame, scores: np.ndarray, threshold: float, split_df: pd.DataFrame | None = None) -> dict[str, float | int]:
    local = meta.copy()
    local["score"] = np.asarray(scores, dtype=float)
    local["pred"] = (local["score"] >= threshold).astype(int)

    if split_df is not None:
        labels = split_df[["file_name", "Label"]].copy()
        labels["file_name"] = labels["file_name"].astype(str)
        labels["Label"] = pd.to_numeric(labels["Label"], errors="coerce").fillna(0).astype(int)
        all_modules = set(labels["file_name"].tolist())
        true_pos_modules = set(labels.loc[labels["Label"] == 1, "file_name"].tolist())
    else:
        all_modules = set(local["file_name"].astype(str).unique().tolist())
        true_pos_modules = set(local.loc[local["event_label"] == 1, "file_name"].astype(str).unique().tolist())

    predict_pos_modules: set[str] = set()
    lead_hours: list[float] = []
    first_alert_rows: list[dict[str, object]] = []
    for file_name, group in local.sort_values("pred_ts").groupby("file_name"):
        alerts = group[group["pred"] == 1]
        if alerts.empty:
            continue
        first_alert = alerts.iloc[0]
        true_ts = int(group["first_failure_ts"].max())
        event_label = int(group["event_label"].max())
        if event_label == 0:
            predict_pos_modules.add(str(file_name))
            first_alert_rows.append({"file_name": file_name, "predict_ts": int(first_alert["pred_ts"]), "true_ts": -1, "lead_hour": np.nan})
        elif true_ts > int(first_alert["pred_ts"]):
            predict_pos_modules.add(str(file_name))
            lead = (true_ts - int(first_alert["pred_ts"])) / 3600.0
            lead_hours.append(float(lead))
            first_alert_rows.append({"file_name": file_name, "predict_ts": int(first_alert["pred_ts"]), "true_ts": true_ts, "lead_hour": lead})

    hit_modules = true_pos_modules & predict_pos_modules
    tp = len(hit_modules)
    fp = len(predict_pos_modules) - tp
    fn = len(true_pos_modules) - tp
    tn = len(all_modules) - tp - fp - fn
    precision = tp / len(predict_pos_modules) if predict_pos_modules else 0.0
    recall = tp / len(true_pos_modules) if true_pos_modules else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(all_modules), 1)

    avg_lead_hour = sum(lead_hours) / max(len(true_pos_modules), 1)
    min_lead_hour = min(lead_hours) if lead_hours else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    return {
        "threshold": float(threshold),
        "final_score": float(final_score),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "all_hit_cnt": int(tp),
        "all_predict_pos_cnt": int(len(predict_pos_modules)),
        "all_true_pos_cnt": int(len(true_pos_modules)),
        "avg_lead_hour": float(avg_lead_hour),
        "min_lead_hour": float(min_lead_hour),
        "avg_lead_score": float(avg_lead_score),
        "min_lead_score": float(min_lead_score),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def choose_threshold(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    split_df: pd.DataFrame | None,
    grid_size: int = 99,
) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_metrics: dict[str, float | int] = {"final_score": -1.0}
    for threshold in np.linspace(0.01, 0.99, grid_size):
        metrics = module_event_metrics(meta, scores, float(threshold), split_df)
        if (
            metrics["final_score"] > best_metrics["final_score"]
            or (
                metrics["final_score"] == best_metrics["final_score"]
                and metrics["f1"] > best_metrics.get("f1", -1.0)
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics

