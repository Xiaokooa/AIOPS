"""Reproduce OFP model1/model2-style baselines under the unified protocol.

The original ``OFP/`` directory is treated as read-only.  This script uses the
same public task ingredients:

- model1: point-level XGBoost trained on first-event 1-hour ahead labels.
- model2 RF variants: RF-style TreeModel trained on OFP default features and
  first-event 1-hour ahead labels, with threshold 0.3 by default.
- model2_rf_module: the same RF feature/model stack trained with the module-level
  fault label.  This matches the local-stat results in ``OFP/readme.md`` much
  more closely than the current ``TrainLabel=NAnomaly`` source setting.
- model2_rule: OFP expert-rule baseline over the same default feature table.

Each model exports one ``timestamp,predict`` file per module and is evaluated by
``ofp_protocol.evaluator``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder


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

OFP_NAME_MAP = {
    "timestamp": "Ts",
    "temperature": "Temp",
    "current": "Curr",
    "currentTXPower": "TxP0",
    "currentRXPower": "RxP0",
    "currentMultiRXPower1": "RxP1",
    "currentMultiRXPower2": "RxP2",
    "currentMultiRXPower3": "RxP3",
    "currentMultiRXPower4": "RxP4",
    "currentMultiTXPower1": "TxP1",
    "currentMultiTXPower2": "TxP2",
    "currentMultiTXPower3": "TxP3",
    "currentMultiTXPower4": "TxP4",
    "anomaly": "Ano",
}

OFP_BASE_COLUMNS = [
    "Ts",
    "Temp",
    "Curr",
    "TxP0",
    "RxP0",
    "RxP1",
    "RxP2",
    "RxP3",
    "RxP4",
    "TxP1",
    "TxP2",
    "TxP3",
    "TxP4",
]

MODEL2_FEATURES = [
    *OFP_BASE_COLUMNS,
    "FeCoCurrTemp",
    "FeCoCurrTxP0",
    "FeCoCurrRxP0",
    "FeCoTxP0RxP0",
    "FeTxP0-Max",
    "FeTxP0-Min",
    "FeRxP0-Max",
    "FeRxP0-Min",
    "TsDelta",
]

TEMP_OUTLIER = -255.0
NA_DEFAULT = -999.0
HORIZON_HOURS = 1
HORIZON_SECONDS = HORIZON_HOURS * 3600


@dataclass
class FoldResult:
    model: str
    fold: int
    threshold: float
    train_modules: int
    test_modules: int
    train_rows: int
    test_rows: int
    seconds: float
    metrics: dict[str, float]


def read_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    df["folder_index"] = pd.to_numeric(df["folder_index"], errors="coerce").astype(int)
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").astype(int)
    return df


def file_path(data_dir: Path, file_name: str) -> Path:
    return data_dir / file_name


def iter_file_names(values: Iterable[str], limit: int | None = None) -> list[str]:
    names = list(values)
    if limit is not None:
        names = names[: int(limit)]
    return names


def first_anomaly_ts(df: pd.DataFrame) -> float | None:
    if "anomaly" not in df.columns:
        return None
    anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0.0).to_numpy()
    if not np.any(anomaly > 0):
        return None
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    idx = int(np.argmax(anomaly > 0))
    return float(timestamps[idx])


def ahead120_label(df: pd.DataFrame) -> np.ndarray:
    """Compatibility alias for the current first-event ahead label.

    A row is positive only if it is before the first anomaly and no more than
    1 hour ahead of that first anomaly.
    """
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    first_ts = first_anomaly_ts(df)
    if first_ts is None:
        return np.zeros(len(df), dtype=np.int8)
    return ((timestamps >= first_ts - HORIZON_SECONDS) & (timestamps < first_ts)).astype(np.int8)


def first_anomaly_ahead_label(df: pd.DataFrame, horizon_seconds: float = HORIZON_SECONDS) -> np.ndarray:
    """First-warning label: only rows before the first anomaly can be positive."""
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    first_ts = first_anomaly_ts(df)
    if first_ts is None:
        return np.zeros(len(df), dtype=np.int8)
    label = (timestamps >= first_ts - float(horizon_seconds)) & (timestamps < first_ts)
    return label.astype(np.int8)


def first_event_valid_mask(df: pd.DataFrame) -> np.ndarray:
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(timestamps)
    first_ts = first_anomaly_ts(df)
    if first_ts is None:
        return valid.astype(bool)
    return (valid & (timestamps < first_ts)).astype(bool)


def load_model1_arrays(data_dir: Path, file_names: list[str], label_mode: str) -> tuple[np.ndarray, np.ndarray | None, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    total_rows = 0
    for name in file_names:
        df = pd.read_csv(file_path(data_dir, name), usecols=lambda col: col in {"timestamp", "anomaly", *SENSORS})
        x = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if label_mode == "ahead120":
            valid_mask = first_event_valid_mask(df)
            xs.append(x[valid_mask])
            ys.append(ahead120_label(df)[valid_mask])
        elif label_mode == "anomaly":
            valid_mask = first_event_valid_mask(df)
            xs.append(x[valid_mask])
            ys.append(ahead120_label(df)[valid_mask])
        elif label_mode == "none":
            xs.append(x)
            pass
        else:
            raise ValueError(label_mode)
        total_rows += len(xs[-1])
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(SENSORS)), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else None
    return X, y, total_rows


def rolling_corr(a: pd.Series, b: pd.Series, win_size: int = 5, fill: float = 99.0) -> np.ndarray:
    out = a.rolling(win_size, min_periods=win_size).corr(b).to_numpy(dtype=np.float32)
    out[: win_size - 1] = fill
    return np.nan_to_num(out, nan=NA_DEFAULT, posinf=NA_DEFAULT, neginf=NA_DEFAULT).astype(np.float32)


def make_model2_features(raw_df: pd.DataFrame, with_label: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_df.rename(columns=OFP_NAME_MAP).copy()
    df = df[~df["Temp"].isna()].copy()
    for col in OFP_BASE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if with_label and "Ano" in df.columns:
        df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").fillna(0.0)

    feature = df[OFP_BASE_COLUMNS].copy()
    feature["FeCoCurrTemp"] = rolling_corr(df["Curr"], df["Temp"], fill=99.0)
    feature["FeCoCurrTxP0"] = rolling_corr(df["Curr"], df["TxP0"], fill=-99.0)
    feature["FeCoCurrRxP0"] = rolling_corr(df["Curr"], df["RxP0"], fill=-99.0)
    feature["FeCoTxP0RxP0"] = rolling_corr(df["TxP0"], df["RxP0"], fill=99.0)
    feature["FeTxP0-Max"] = df["TxP0"] - df[["TxP1", "TxP2", "TxP3", "TxP4"]].max(axis=1)
    feature["FeTxP0-Min"] = df["TxP0"] - df[["TxP1", "TxP2", "TxP3", "TxP4"]].min(axis=1)
    feature["FeRxP0-Max"] = df["RxP0"] - df[["RxP1", "RxP2", "RxP3", "RxP4"]].max(axis=1)
    feature["FeRxP0-Min"] = df["RxP0"] - df[["RxP1", "RxP2", "RxP3", "RxP4"]].min(axis=1)
    if with_label and "Ano" in df.columns:
        feature["Ano"] = df["Ano"]

    cut_mask = (df["Temp"] == TEMP_OUTLIER) | (df["Temp"] == NA_DEFAULT) | df["Temp"].isna()
    normal = feature.loc[~cut_mask].copy()
    extra = feature.loc[cut_mask].copy()
    normal["TsDelta"] = normal["Ts"].diff().fillna(NA_DEFAULT)
    extra["TsDelta"] = extra["Ts"].diff().fillna(NA_DEFAULT)
    return normal.astype(np.float32), extra.astype(np.float32)


def load_model2_arrays(data_dir: Path, file_names: list[str], label_mode: str = "anomaly") -> tuple[np.ndarray, np.ndarray | None, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    total_rows = 0
    for name in file_names:
        raw = pd.read_csv(file_path(data_dir, name))
        feature, _extra = make_model2_features(raw, with_label="anomaly" in raw.columns)
        feature_index = feature.index.to_numpy(dtype=int)
        valid_mask = first_event_valid_mask(raw)
        feature_valid = valid_mask[feature_index] if len(feature_index) else np.zeros(0, dtype=bool)
        if label_mode == "none":
            selected = np.ones(len(feature), dtype=bool)
        else:
            selected = feature_valid
        total_rows += int(selected.sum())
        xs.append(feature.loc[selected, MODEL2_FEATURES].to_numpy(dtype=np.float32))
        if label_mode in {"anomaly", "module", "ahead120"}:
            label = ahead120_label(raw)
            ys.append(label[feature_index][selected].astype(np.int8))
        elif label_mode == "none":
            pass
        else:
            raise ValueError(label_mode)
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(MODEL2_FEATURES)), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else None
    return X, y, total_rows


def write_prediction_files(
    out_dir: Path,
    data_dir: Path,
    file_names: list[str],
    model,
    model_kind: str,
    threshold: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for name in file_names:
        raw = pd.read_csv(file_path(data_dir, name))
        timestamps = pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        if model_kind == "model1_xgb":
            X = raw[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            scores = model.predict_proba(X)[:, 1]
            pred = (scores >= threshold).astype(int)
        elif model_kind == "model2_rf":
            feature, extra = make_model2_features(raw, with_label="anomaly" in raw.columns)
            pred = np.zeros(len(raw), dtype=int)
            if len(feature):
                idx = feature.index.to_numpy(dtype=int)
                scores = positive_probability(model, feature[MODEL2_FEATURES].to_numpy(dtype=np.float32))
                pred[idx] = (scores >= threshold).astype(int)
            if len(extra):
                idx = extra.index.to_numpy(dtype=int)
                pred[idx] = (extra["Temp"].to_numpy(dtype=float) == TEMP_OUTLIER).astype(int)
        elif model_kind == "model2_rule":
            pred = rule_predict(raw)
        else:
            raise ValueError(model_kind)
        total_rows += len(pred)
        pd.DataFrame({"timestamp": timestamps, "predict": pred}).to_csv(out_dir / name, index=False)
    return total_rows


def positive_probability(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(len(X), dtype=np.float32)


def rule_predict(raw_df: pd.DataFrame) -> np.ndarray:
    feature, extra = make_model2_features(raw_df, with_label="anomaly" in raw_df.columns)
    pred = np.zeros(len(raw_df), dtype=int)
    if len(feature):
        f = feature
        rules = np.zeros(len(f), dtype=int)
        rules |= (f["Temp"] < 0).to_numpy()
        rules |= (f["Curr"] < 5000).to_numpy()
        rules |= ((f["Temp"].expanding().max() - f["Temp"].expanding().min()) > 100).to_numpy()
        rules |= ((f["Curr"].expanding().max() - f["Curr"].expanding().min()) > 6000).to_numpy()
        rules |= f["Temp"].expanding().std().fillna(0).gt(10).to_numpy()
        rules |= f["Curr"].expanding().std().fillna(0).gt(1500).to_numpy()
        for col in ["TxP0", "RxP0", "RxP1", "RxP2", "RxP3", "RxP4", "TxP1", "TxP2", "TxP3", "TxP4"]:
            rules |= (f[col] < 0).to_numpy()
        idx = feature.index.to_numpy(dtype=int)
        pred[idx] = rules.astype(int)
    if len(extra):
        idx = extra.index.to_numpy(dtype=int)
        pred[idx] = (extra["Temp"].to_numpy(dtype=float) == TEMP_OUTLIER).astype(int)
    return pred


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def run_fold(
    model_name: str,
    fold: int,
    index_df: pd.DataFrame,
    data_dir: Path,
    out_root: Path,
    max_train_files: int | None,
    max_test_files: int | None,
    xgb_estimators: int,
    rf_estimators: int,
    threshold: float,
) -> FoldResult:
    t0 = time.time()
    train_files = iter_file_names(index_df.loc[index_df["folder_index"] != fold, "file_name"], max_train_files)
    test_files = iter_file_names(index_df.loc[index_df["folder_index"] == fold, "file_name"], max_test_files)
    model_dir = out_root / model_name / f"fold_{fold}"
    pred_dir = model_dir / "predictions"
    eval_dir = model_dir / "evaluation"

    if model_name in {"model1_xgb_ahead1h", "model1_xgb_ahead120"}:
        X_train, y_train, train_rows = load_model1_arrays(data_dir, train_files, label_mode="ahead120")
        model = XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        test_rows = write_prediction_files(pred_dir, data_dir, test_files, model, "model1_xgb", threshold)
    elif model_name in {"model2_rf_anomaly", "model2_rf_module", "model2_rf_ahead1h", "model2_rf_ahead120"}:
        label_mode = {
            "model2_rf_anomaly": "ahead120",
            "model2_rf_module": "ahead120",
            "model2_rf_ahead1h": "ahead120",
            "model2_rf_ahead120": "ahead120",
        }[model_name]
        X_train, y_train, train_rows = load_model2_arrays(data_dir, train_files, label_mode=label_mode)
        model = RandomForestClassifier(
            n_estimators=rf_estimators,
            random_state=2024,
            n_jobs=1,
            verbose=0,
        )
        model.fit(X_train, y_train)
        test_rows = write_prediction_files(pred_dir, data_dir, test_files, model, "model2_rf", threshold)
    elif model_name == "model2_rule":
        train_rows = 0
        model = None
        test_rows = write_prediction_files(pred_dir, data_dir, test_files, model, "model2_rule", threshold)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)
    result = FoldResult(
        model=model_name,
        fold=fold,
        threshold=threshold,
        train_modules=len(train_files),
        test_modules=len(test_files),
        train_rows=int(train_rows),
        test_rows=int(test_rows),
        seconds=time.time() - t0,
        metrics=metrics,
    )
    (model_dir / "fold_summary.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def aggregate_results(results: list[FoldResult], out_root: Path) -> pd.DataFrame:
    rows = []
    for item in results:
        row = {
            "model": item.model,
            "fold": item.fold,
            "threshold": item.threshold,
            "train_modules": item.train_modules,
            "test_modules": item.test_modules,
            "train_rows": item.train_rows,
            "test_rows": item.test_rows,
            "seconds": item.seconds,
        }
        row.update(item.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "fold_metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        df = pd.concat([existing, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df.sort_values(["model", "fold"], inplace=True)
    df.to_csv(metrics_path, index=False)
    if not df.empty:
        metric_cols = [c for c in df.columns if c not in {"model", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
        summary = df.groupby("model")[metric_cols].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.reset_index(inplace=True)
        summary.to_csv(out_root / "model_metrics_mean_std.csv", index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_protocol/baselines"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["model1_xgb_ahead1h", "model2_rf_ahead1h"],
        choices=[
            "model1_xgb_ahead1h",
            "model1_xgb_ahead120",
            "model2_rf_anomaly",
            "model2_rf_module",
            "model2_rf_ahead1h",
            "model2_rf_ahead120",
            "model2_rule",
        ],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--xgb_estimators", type=int, default=100)
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--model1_threshold", type=float, default=0.5)
    parser.add_argument("--model2_threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    all_results: list[FoldResult] = []
    for model_name in args.models:
        threshold = args.model1_threshold if model_name in {"model1_xgb_ahead1h", "model1_xgb_ahead120"} else args.model2_threshold
        for fold in args.folds:
            print(f"[run] model={model_name} fold={fold}")
            result = run_fold(
                model_name,
                fold,
                index_df,
                args.data_dir,
                args.out_root,
                args.max_train_files,
                args.max_test_files,
                args.xgb_estimators,
                args.rf_estimators,
                threshold,
            )
            all_results.append(result)
            aggregate_results([result], args.out_root)
            print(
                f"[done] {model_name} fold={fold} "
                f"F1={result.metrics.get('f1_score', 0):.4f} "
                f"P={result.metrics.get('precision', 0):.4f} "
                f"R={result.metrics.get('recall', 0):.4f} "
                f"time={result.seconds:.1f}s"
            )
    aggregate_results(all_results, args.out_root)


if __name__ == "__main__":
    main()
