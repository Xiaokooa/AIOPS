from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "final_score",
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "all_hit_cnt",
    "all_predict_pos_cnt",
    "all_true_pos_cnt",
    "avg_lead_hour",
    "min_lead_hour",
]


def rows_from_fold_summaries(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(root.rglob("fold_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "model": payload.get("model", "semisup_mil"),
            "fold": payload.get("fold", path.parent.name),
            "protocol": payload.get("protocol", ""),
            "threshold": payload.get("threshold", float("nan")),
        }
        alarm = payload.get("alarm_strategy", {}) or {}
        row.update(
            {
                "alarm_smoothing": alarm.get("alarm_smoothing", ""),
                "alarm_smooth_window": alarm.get("alarm_smooth_window", ""),
                "alarm_consecutive_k": alarm.get("alarm_consecutive_k", ""),
            }
        )
        row.update(payload.get("metrics", {}) or {})
        rows.append(row)
    return pd.DataFrame(rows)


def load_rows(root: Path) -> pd.DataFrame:
    metrics_path = root / "fold_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return rows_from_fold_summaries(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact OFP result table.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    df = load_rows(args.root)
    if df.empty:
        print(f"[summary] no metrics found under {args.root}")
        return

    keep = [c for c in ["model", "fold", "threshold", "alarm_smoothing", "alarm_smooth_window", "alarm_consecutive_k", *METRIC_COLUMNS] if c in df.columns]
    table = df[keep].copy()
    for col in table.columns:
        if col not in {"model", "fold", "alarm_smoothing"}:
            converted = pd.to_numeric(table[col], errors="coerce")
            if converted.notna().any():
                table[col] = converted
    print("\n[summary] fold metrics")
    print(table.to_string(index=False, max_cols=32, float_format=lambda x: f"{x:.4f}"))

    numeric = [c for c in METRIC_COLUMNS if c in df.columns]
    if "model" in df.columns and numeric:
        grouped = df.groupby("model")[numeric].mean(numeric_only=True).reset_index()
        print("\n[summary] mean metrics")
        print(grouped.to_string(index=False, max_cols=32, float_format=lambda x: f"{x:.4f}"))

    print(f"\n[summary] root={args.root}")
    if (args.root / "fold_metrics.csv").exists():
        print(f"[summary] csv={args.root / 'fold_metrics.csv'}")
    if (args.root / "model_metrics_mean_std.csv").exists():
        print(f"[summary] mean_std={args.root / 'model_metrics_mean_std.csv'}")


if __name__ == "__main__":
    main()
