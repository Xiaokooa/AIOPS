"""Strict module-level OFP evaluator and README comparison."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .labels import first_anomaly_timestamp, validate_time_order


README_XGB_REFERENCE = {
    "final_score": 2.057151253,
    "f1_score": 0.255344418,
    "precision": 0.678947368,
    "recall": 0.157240371,
    "all_hit_cnt": 645.0,
    "all_predict_pos_cnt": 950.0,
    "all_true_pos_cnt": 4102.0,
    "avg_lead_hour": 68.44573643,
    "accuracy": 0.718665869,
}


def _module_decision(pred_path: Path, label_path: Path) -> dict[str, object]:
    label_df = pd.read_csv(label_path, usecols=lambda c: c in {"timestamp", "anomaly"})
    pred_df = pd.read_csv(pred_path, usecols=lambda c: c in {"timestamp", "predict", "score"})
    label_ts = validate_time_order(label_df)
    pred_ts = validate_time_order(pred_df)
    if len(label_ts) != len(pred_ts) or not np.array_equal(label_ts, pred_ts):
        raise ValueError(f"timestamp mismatch between {pred_path} and {label_path}")

    true_ts = first_anomaly_timestamp(label_df)
    true_label = int(true_ts is not None)
    pred = pd.to_numeric(pred_df["predict"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0
    valid_pred = pred.copy()
    if true_ts is not None:
        valid_pred &= pred_ts < true_ts
    predict_ts = float(pred_ts[valid_pred].min()) if valid_pred.any() else None
    predict_label = int(predict_ts is not None)
    hit = int(true_label > 0 and predict_label > 0)
    lead_hour = (float(true_ts) - float(predict_ts)) / 3600.0 if hit else None
    return {
        "file_name": pred_path.name,
        "true_label": true_label,
        "predict_label": predict_label,
        "true_ts": true_ts,
        "predict_ts": predict_ts,
        "valid_predict_positive": predict_label,
        "hit": hit,
        "lead_hour": lead_hour,
    }


def metrics_from_decisions(detail_df: pd.DataFrame) -> dict[str, float]:
    if detail_df.empty:
        raise ValueError("cannot evaluate an empty decision table")
    true_pos = detail_df["true_label"].to_numpy(dtype=int) > 0
    pred_pos = detail_df["valid_predict_positive"].to_numpy(dtype=int) > 0
    hit = true_pos & pred_pos
    tp = int(hit.sum())
    fp = int((~true_pos & pred_pos).sum())
    fn = int((true_pos & ~pred_pos).sum())
    tn = int((~true_pos & ~pred_pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(detail_df)
    lead = pd.to_numeric(detail_df.loc[detail_df["hit"] > 0, "lead_hour"], errors="coerce").dropna()
    avg_lead_hour = float(lead.mean()) if not lead.empty else 0.0
    min_lead_hour = float(lead.min()) if not lead.empty else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    return {
        "final_score": f1 + accuracy + avg_lead_score + min_lead_score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "all_hit_cnt": float(tp),
        "all_predict_pos_cnt": float(tp + fp),
        "all_true_pos_cnt": float(tp + fn),
        "avg_lead_score": avg_lead_score,
        "avg_lead_hour": avg_lead_hour,
        "min_lead_score": min_lead_score,
        "min_lead_hour": min_lead_hour,
        "lead_pread_cnt": float(tp),
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "evaluated_module_cnt": float(len(detail_df)),
    }


def evaluate_prediction_dir(
    prediction_dir: Path,
    label_dir: Path,
    expected_files: Iterable[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    prediction_dir = Path(prediction_dir)
    label_dir = Path(label_dir)
    names = list(dict.fromkeys(str(name) for name in expected_files))
    missing = [name for name in names if not (prediction_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} prediction files; first={missing[0]}")
    rows = [_module_decision(prediction_dir / name, label_dir / name) for name in names]
    detail_df = pd.DataFrame(rows)
    return metrics_from_decisions(detail_df), detail_df


def write_evaluation(out_dir: Path, metrics: dict[str, float], detail_df: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Item": list(metrics), "Value": list(metrics.values())}).to_csv(
        out_dir / "evaluate_result.csv", index=False
    )
    detail_df.to_csv(out_dir / "module_decisions.csv", index=False)


def compare_with_readme(metrics: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, reference in README_XGB_REFERENCE.items():
        value = float(metrics[name])
        rows.append({"metric": name, "b0": value, "readme": reference, "delta": value - reference})
    return pd.DataFrame(rows)
