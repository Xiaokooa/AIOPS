from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from OFP.deep_learning.official.ofp_protocol.run_ofp_baselines import TEMP_OUTLIER, make_model2_features


RAW_USECOLS = {
    "timestamp",
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
    "anomaly",
}


def _read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda col: col in RAW_USECOLS)


def _first_positive_ts(df: pd.DataFrame, label_col: str, before_ts: float | None = None) -> tuple[int, float | None]:
    if label_col not in df.columns or "timestamp" not in df.columns:
        return 0, None
    label = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    valid = (label > 0) & timestamps.notna()
    if before_ts is not None:
        valid = valid & (timestamps.astype(float) < float(before_ts))
    positives = timestamps.loc[valid].astype(float)
    if positives.empty:
        return 0, None
    return 1, float(positives.min())


def audit_model2_feature_timestamp(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        raw = _read_raw(path)
        if "anomaly" not in raw.columns:
            continue
        anomaly = pd.to_numeric(raw["anomaly"], errors="coerce").fillna(0.0).to_numpy()
        if not np.any(anomaly > 0):
            continue
        timestamps = pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        first_idx = int(np.argmax(anomaly > 0))
        true_ts = float(timestamps[first_idx])
        pre_rows = int(np.sum(np.isfinite(timestamps) & (timestamps < true_ts)))

        normal, extra = make_model2_features(raw, with_label=True)
        normal_index = set(normal.index.tolist())
        extra_index = set(extra.index.tolist())

        feature_ts = np.nan
        feature_delta = np.nan
        feature_early = False
        if first_idx in normal_index:
            feature_ts = float(normal.loc[first_idx, "Ts"])
            feature_delta = float(int(feature_ts) - true_ts)
            feature_early = bool(int(feature_ts) < true_ts)

        extra_ts = np.nan
        extra_delta = np.nan
        extra_early = False
        if first_idx in extra_index:
            extra_ts = float(extra.loc[first_idx, "Ts"])
            extra_delta = float(int(extra_ts) - true_ts)
            extra_temp = float(extra.loc[first_idx, "Temp"]) if "Temp" in extra.columns else np.nan
            extra_early = bool(extra_temp == TEMP_OUTLIER and int(extra_ts) < true_ts)

        rows.append(
            {
                "file_name": path.name,
                "first_anomaly_row": first_idx,
                "true_ts": true_ts,
                "pre_fault_rows": pre_rows,
                "left_censored": int(pre_rows == 0),
                "feature_ts_float32": feature_ts,
                "feature_delta_sec": feature_delta,
                "feature_anomaly_row_looks_early": int(feature_early),
                "extra_ts_float32": extra_ts,
                "extra_delta_sec": extra_delta,
                "extra_temp_outlier_row_looks_early": int(extra_early),
                "any_anomaly_row_looks_early": int(feature_early or extra_early),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame()
        return summary, detail

    summary_items = {
        "faulty_modules": len(detail),
        "pre_fault_available_modules": int((detail["pre_fault_rows"] > 0).sum()),
        "left_censored_modules": int((detail["pre_fault_rows"] == 0).sum()),
        "feature_anomaly_row_present": int(detail["feature_ts_float32"].notna().sum()),
        "feature_anomaly_row_looks_early": int(detail["feature_anomaly_row_looks_early"].sum()),
        "extra_temp_outlier_row_looks_early": int(detail["extra_temp_outlier_row_looks_early"].sum()),
        "any_anomaly_row_looks_early": int(detail["any_anomaly_row_looks_early"].sum()),
        "left_censored_but_anomaly_row_looks_early": int(
            ((detail["pre_fault_rows"] == 0) & (detail["any_anomaly_row_looks_early"] > 0)).sum()
        ),
    }
    summary = pd.DataFrame({"Item": list(summary_items), "Value": list(summary_items.values())})
    return summary, detail


def audit_prediction_timestamp_domain(
    prediction_dir: Path,
    label_dir: Path,
    predict_col: str = "predict",
    label_col: str = "anomaly",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    prediction_files = sorted(path for path in Path(prediction_dir).glob("*.csv") if path.is_file())
    for pred_path in prediction_files:
        label_path = Path(label_dir) / pred_path.name
        if not label_path.exists():
            continue
        pred = pd.read_csv(pred_path, usecols=lambda col: col in {"timestamp", predict_col})
        label = pd.read_csv(label_path, usecols=lambda col: col in {"timestamp", label_col})
        pred_ts = pd.to_numeric(pred.get("timestamp"), errors="coerce").to_numpy(dtype=np.float64)
        label_ts = pd.to_numeric(label.get("timestamp"), errors="coerce").to_numpy(dtype=np.float64)
        pred_finite = pred_ts[np.isfinite(pred_ts)]
        label_finite = label_ts[np.isfinite(label_ts)]

        same_len = len(pred_ts) == len(label_ts)
        same_sequence = bool(same_len and np.array_equal(pred_ts, label_ts))
        subset = bool(np.isin(pred_finite, label_finite).all()) if len(pred_finite) else True

        true_label, true_ts = _first_positive_ts(label, label_col)
        before_ts = true_ts if true_label > 0 else None
        pred_label, predict_ts = _first_positive_ts(pred, predict_col, before_ts=before_ts)
        hit = int(true_label > 0 and pred_label > 0 and true_ts is not None and predict_ts is not None and predict_ts < true_ts)
        lead_sec = float(true_ts - predict_ts) if hit else np.nan
        pred_ts_in_label_set = bool(predict_ts in set(label_finite.tolist())) if predict_ts is not None else False

        rows.append(
            {
                "file_name": pred_path.name,
                "pred_rows": len(pred_ts),
                "label_rows": len(label_ts),
                "same_len": int(same_len),
                "same_timestamp_sequence": int(same_sequence),
                "pred_ts_subset_label_ts": int(subset),
                "has_label_column": int(label_col in label.columns),
                "true_label": int(true_label),
                "predict_label_before_true": int(pred_label),
                "true_ts": true_ts,
                "predict_ts": predict_ts,
                "hit": hit,
                "lead_sec": lead_sec,
                "predict_ts_in_label_timestamp_set": int(pred_ts_in_label_set),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame()
        return summary, detail
    summary_items = {
        "matched_files": len(detail),
        "same_len_files": int(detail["same_len"].sum()),
        "same_timestamp_sequence_files": int(detail["same_timestamp_sequence"].sum()),
        "pred_ts_subset_label_ts_files": int(detail["pred_ts_subset_label_ts"].sum()),
        "files_with_label_column": int(detail["has_label_column"].sum()),
        "true_positive_modules": int(detail["true_label"].sum()) if "true_label" in detail else 0,
        "hit_modules": int(detail["hit"].sum()) if "hit" in detail else 0,
        "hit_predict_ts_not_in_label_timestamp_set": int(
            ((detail["hit"] > 0) & (detail["predict_ts_in_label_timestamp_set"] == 0)).sum()
        ),
        "hit_lead_le_60s": int(((detail["hit"] > 0) & (detail["lead_sec"] <= 60)).sum()),
        "hit_lead_le_1h": int(((detail["hit"] > 0) & (detail["lead_sec"] <= 3600)).sum()),
    }
    summary = pd.DataFrame({"Item": list(summary_items), "Value": list(summary_items.values())})
    return summary, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OFP prediction timestamp domains and legacy model2 timestamp effects.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--prediction_dir", type=Path, default=None)
    parser.add_argument("--label_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("analysis_outputs/timestamp_domain_audit"))
    parser.add_argument("--predict_col", default="predict")
    parser.add_argument("--label_col", default="anomaly")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_summary, feature_detail = audit_model2_feature_timestamp(args.data_dir)
    feature_summary.to_csv(args.out_dir / "model2_feature_timestamp_summary.csv", index=False)
    feature_detail.to_csv(args.out_dir / "model2_feature_timestamp_detail.csv", index=False)
    print("[model2-feature-timestamp]")
    if feature_summary.empty:
        print("no labeled faulty modules found")
    else:
        print(feature_summary.to_string(index=False))

    if args.prediction_dir is not None and args.label_dir is not None:
        pred_summary, pred_detail = audit_prediction_timestamp_domain(
            args.prediction_dir,
            args.label_dir,
            predict_col=args.predict_col,
            label_col=args.label_col,
        )
        pred_summary.to_csv(args.out_dir / "prediction_timestamp_domain_summary.csv", index=False)
        pred_detail.to_csv(args.out_dir / "prediction_timestamp_domain_detail.csv", index=False)
        print("\n[prediction-timestamp-domain]")
        if pred_summary.empty:
            print("no matching prediction/label files found")
        else:
            print(pred_summary.to_string(index=False))

    print(f"\n[audit] out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
