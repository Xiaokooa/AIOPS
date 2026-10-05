# Numerical core retained for compatibility with the archived experiments.
"""Unified evaluator for the OFP module-level fault-prediction protocol.

This module intentionally lives outside ``OFP/`` so the original teacher code
remains read-only.  It mirrors the scoring logic in
``OFP/model2/Tools/EvaluateResult.py`` while adding safer edge-case handling and
machine-readable outputs.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


SEC_IN_HOUR = 3600.0


@dataclass
class ModuleDecision:
    file_name: str
    true_label: int
    predict_label: int
    true_ts: float | None
    predict_ts: float | None
    valid_predict_positive: int
    hit: int
    lead_hour: float | None


def first_positive_timestamp(
    csv_path: Path,
    label_column: str,
    before_ts: float | None = None,
) -> tuple[int, float | None]:
    df = pd.read_csv(csv_path, usecols=lambda col: col in {"timestamp", label_column})
    if label_column not in df.columns:
        raise ValueError(f"{csv_path} does not contain column {label_column!r}")
    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path} does not contain column 'timestamp'")

    label = pd.to_numeric(df[label_column], errors="coerce").fillna(0.0)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    valid = (label > 0) & timestamps.notna()
    if before_ts is not None:
        valid = valid & (timestamps.astype(float) < float(before_ts))
    positive = pd.to_numeric(df.loc[valid, "timestamp"], errors="coerce").dropna()
    if positive.empty:
        return 0, None
    return 1, float(positive.min())


def evaluate_prediction_folder(
    prediction_dir: Path,
    label_dir: Path,
    predict_column: str = "predict",
    label_column: str = "anomaly",
    max_files: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_dir = Path(prediction_dir)
    label_dir = Path(label_dir)
    if not prediction_dir.exists():
        raise FileNotFoundError(prediction_dir)
    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    rows: list[ModuleDecision] = []
    prediction_files = sorted(path for path in prediction_dir.glob("*.csv") if path.is_file())
    if max_files is not None:
        prediction_files = prediction_files[: int(max_files)]
    for pred_path in prediction_files:
        label_path = label_dir / pred_path.name
        if not label_path.exists():
            continue

        true_label, true_ts = first_positive_timestamp(label_path, label_column)
        predict_label, predict_ts = first_positive_timestamp(
            pred_path,
            predict_column,
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
                lead_hour = abs(true_ts - predict_ts) / SEC_IN_HOUR

        rows.append(
            ModuleDecision(
                file_name=pred_path.name,
                true_label=true_label,
                predict_label=predict_label,
                true_ts=true_ts,
                predict_ts=predict_ts,
                valid_predict_positive=valid_predict_positive,
                hit=hit,
                lead_hour=lead_hour,
            )
        )

    detail_df = pd.DataFrame([asdict(row) for row in rows])
    if detail_df.empty:
        raise ValueError(f"No matching prediction/label CSV files under {prediction_dir} and {label_dir}")

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

    summary_df = pd.DataFrame(
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
    return summary_df, detail_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--predict_column", default="predict")
    parser.add_argument("--label_column", default="anomaly")
    parser.add_argument("--max_files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, detail_df = evaluate_prediction_folder(
        args.prediction_dir,
        args.label_dir,
        predict_column=args.predict_column,
        label_column=args.label_column,
        max_files=args.max_files,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "evaluate_result.csv"
    detail_path = args.out_dir / "module_decisions.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    print(f"[eval] {summary_path}")
    print(f"[eval] {detail_path}")
    for _, row in summary_df.iterrows():
        print(f"{row['Item']} = {row['Value']}")


if __name__ == "__main__":
    main()
