# Numerical core retained for compatibility with the archived experiments.
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from OFP.ssffn._engine.data import (
    apply_threshold,
    evaluate_prediction_frames,
    parse_threshold_grid,
)


DEFAULT_LEAD_TIME_GRID = "1m,5m,15m,30m,1h,2h,5h,12h,24h"


def parse_lead_time_grid(text: str) -> list[tuple[str, float]]:
    raw = str(text).strip()
    if not raw:
        return []
    out: list[tuple[str, float]] = []
    for item in raw.replace(";", ",").split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.endswith("min"):
            value = float(token[:-3]) / 60.0
            label = f"{float(token[:-3]):g}min"
        elif token.endswith("m"):
            value = float(token[:-1]) / 60.0
            label = f"{float(token[:-1]):g}min"
        elif token.endswith("hour"):
            value = float(token[:-4])
            label = f"{value:g}h"
        elif token.endswith("hours"):
            value = float(token[:-5])
            label = f"{value:g}h"
        elif token.endswith("h"):
            value = float(token[:-1])
            label = f"{value:g}h"
        else:
            value = float(token)
            label = f"{value:g}h"
        if value < 0:
            raise ValueError("Lead-time values must be non-negative")
        out.append((label, float(value)))
    seen: set[float] = set()
    unique: list[tuple[str, float]] = []
    for label, value in out:
        key = round(float(value), 10)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, float(value)))
    return unique


def _metrics_from_summary(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def _metric_aliases(metrics: dict[str, float]) -> dict[str, float]:
    fp = float(metrics.get("fp", 0.0))
    tn = float(metrics.get("tn", 0.0))
    fn = float(metrics.get("fn", 0.0))
    tp = float(metrics.get("tp", 0.0))
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "hit_rate": float(metrics.get("recall", 0.0)),
        "false_alarm_rate": float(false_alarm_rate),
        "miss_rate": float(miss_rate),
    }


def select_threshold_for_lead_time(
    score_frames: dict[str, pd.DataFrame],
    label_dir: Path,
    thresholds: list[float],
    metric_key: str,
    min_hit_lead_hours: float,
    fixed_threshold: float,
    threshold_search: bool,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    candidates = list(thresholds) if threshold_search and thresholds else [float(fixed_threshold)]
    best_threshold = float(fixed_threshold)
    best_metrics: dict[str, float] = {}
    best_key: tuple[float, float, float, float, float, float] | None = None
    rows: list[dict[str, float]] = []

    for threshold in candidates:
        pred_frames = {name: apply_threshold(frame, float(threshold)) for name, frame in score_frames.items()}
        summary_df, _detail_df = evaluate_prediction_frames(
            pred_frames,
            label_dir,
            min_hit_lead_hours=float(min_hit_lead_hours),
        )
        metrics = _metrics_from_summary(summary_df)
        row = {"threshold": float(threshold), "min_hit_lead_hours": float(min_hit_lead_hours)}
        row.update(metrics)
        row.update(_metric_aliases(metrics))
        rows.append(row)
        value = float(metrics.get(metric_key, metrics.get("f1_score", 0.0)))
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


def run_lead_time_sweep(
    val_score_frames: dict[str, pd.DataFrame],
    test_score_frames: dict[str, pd.DataFrame],
    label_dir: Path,
    run_dir: Path,
    tag: str,
    fold: int,
    lead_time_grid: str,
    threshold_grid: str,
    threshold_metric: str,
    fixed_threshold: float,
    threshold_search: bool,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    leads = parse_lead_time_grid(lead_time_grid)
    if not leads:
        return []
    thresholds = parse_threshold_grid(threshold_grid)
    out_dir = Path(run_dir) / "lead_time_sweep" / str(tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = dict(metadata or {})
    rows: list[dict[str, Any]] = []
    for lead_label, lead_hours in leads:
        threshold, val_metrics, search_df = select_threshold_for_lead_time(
            val_score_frames,
            label_dir,
            thresholds,
            str(threshold_metric),
            float(lead_hours),
            float(fixed_threshold),
            bool(threshold_search),
        )
        test_pred_frames = {name: apply_threshold(frame, float(threshold)) for name, frame in test_score_frames.items()}
        test_summary_df, _test_detail_df = evaluate_prediction_frames(
            test_pred_frames,
            label_dir,
            min_hit_lead_hours=float(lead_hours),
        )
        test_metrics = _metrics_from_summary(test_summary_df)
        safe_label = lead_label.replace(".", "p").replace("/", "_")
        search_df.to_csv(out_dir / f"threshold_search_{safe_label}.csv", index=False)

        row: dict[str, Any] = {
            **metadata,
            "fold": int(fold),
            "tag": str(tag),
            "lead_label": str(lead_label),
            "min_hit_lead_hours": float(lead_hours),
            "threshold": float(threshold),
        }
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update({f"val_{key}": value for key, value in _metric_aliases(val_metrics).items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        row.update({f"test_{key}": value for key, value in _metric_aliases(test_metrics).items()})
        rows.append(row)
        print(
            f"[lead-sweep] tag={tag} fold={fold} lead={lead_label} "
            f"thr={threshold:.4f} test_f1={test_metrics.get('f1_score', 0.0):.5f} "
            f"test_hit_rate={test_metrics.get('recall', 0.0):.5f} "
            f"test_acc={test_metrics.get('accuracy', 0.0):.5f}",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(out_dir / "lead_time_sweep.csv", index=False)
    return rows


def append_lead_time_sweep_results(rows: list[dict[str, Any]], out_root: Path) -> None:
    if not rows:
        return
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "lead_time_sweep.csv"
    frame = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    subset = [col for col in ["run_name", "variant", "tag", "fold", "lead_label", "min_hit_lead_hours"] if col in frame.columns]
    if subset:
        frame.drop_duplicates(subset=subset, keep="last", inplace=True)
    sort_cols = [col for col in ["run_name", "variant", "tag", "min_hit_lead_hours", "fold"] if col in frame.columns]
    if sort_cols:
        frame.sort_values(sort_cols, inplace=True)
    frame.to_csv(path, index=False)

    group_cols = [col for col in ["run_name", "variant", "tag", "lead_label", "min_hit_lead_hours"] if col in frame.columns]
    numeric_cols = [
        col
        for col in frame.columns
        if col not in set(group_cols) | {"fold"} and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if group_cols and numeric_cols:
        summary = frame.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"])
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        summary.reset_index().to_csv(out_root / "lead_time_sweep_mean_std.csv", index=False)
