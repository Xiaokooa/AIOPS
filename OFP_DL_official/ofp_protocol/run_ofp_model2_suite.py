"""Reproduce OFP-style model2 single models and merged results.

This runner mirrors the local ingredients in ``OFP/model2`` without modifying
that directory.  It produces the table-style entries used in ``OFP/readme.md``:

- RF(100Trees)Thresh0.5
- XGBoost over the same OFP default feature table
- RF + XGBoost
- RF + XGBoost + Rule
- RF0.3
- RF0.3 + XGBoost

The RF/XGBoost training label is the current OFP-compatible first-event
1h-ahead label.  The RF path follows the current TreeModel defaults except that
``n_jobs`` is a
runtime option so we can use a modest number of CPU workers for one model at a
time.  No GPU is used.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
    first_anomaly_ahead_label,
    first_event_valid_mask,
    read_index,
)
from OFP_DL_official.ofp_protocol.run_ofp_model2_strict import (
    DEFAULT_FEATURE_LIST,
    NA_DEFAULT,
    TEMP_OUTLIER,
    extract_for_file,
    positive_probability,
)


TREE_MODEL_NAME = "ofp_model2_tree"


@dataclass
class ResultRow:
    model: str
    fold: int
    threshold: float | None
    train_modules: int
    test_modules: int
    train_rows: int
    test_rows: int
    seconds: float
    metrics: dict[str, float]


def load_train_arrays(data_dir: Path, file_names: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        raw = pd.read_csv(data_dir / name)
        labels = first_anomaly_ahead_label(raw)
        valid_mask = first_event_valid_mask(raw)
        normal, _extra = extract_for_file(data_dir, name, with_label=True)
        if len(normal):
            raw_idx = normal.index.to_numpy(dtype=int)
            selected = valid_mask[raw_idx]
            normal = normal.loc[selected]
            if len(normal) <= 0:
                continue
            xs.append(normal[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32))
            ys.append(labels[raw_idx][selected].astype(np.int8))
            n_rows += len(normal)
        if idx % 500 == 0:
            print(f"  [load] {idx}/{len(file_names)} train files, rows={n_rows}")
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(DEFAULT_FEATURE_LIST)), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int8)
    return X, y, n_rows


def write_tree_predictions(
    model,
    data_dir: Path,
    file_names: list[str],
    out_dir: Path,
    threshold: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        normal, extra = extract_for_file(data_dir, name, with_label=True)
        parts = []
        if len(normal):
            X = normal[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32)
            scores = positive_probability(model, X)
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": normal["Ts"].to_numpy(dtype=np.int64),
                        "predict": (scores >= threshold).astype(int),
                        "proba": scores,
                    }
                )
            )
        if len(extra):
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                        "predict": (pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER).astype(int),
                        "proba": np.nan,
                    }
                )
            )
        pred = pd.concat(parts, ignore_index=True).sort_values("timestamp") if parts else pd.DataFrame(columns=["timestamp", "predict", "proba"])
        pred.to_csv(out_dir / name, index=False)
        n_rows += len(pred)
        if idx % 500 == 0:
            print(f"  [predict] {idx}/{len(file_names)} files")
    return n_rows


def expanding_feature(frame: pd.DataFrame, col: str) -> dict[str, pd.Series]:
    series = pd.to_numeric(frame[col], errors="coerce")
    exp = series.expanding()
    f_min = exp.min()
    f_max = exp.max()
    f_std = exp.std()
    if len(f_std):
        f_std.iloc[0] = 0.0
    f_kurt = exp.kurt()
    return {
        "Min": f_min,
        "Diff": f_max - f_min,
        "Std": f_std,
        "Kurt": f_kurt,
    }


def rule_predict_one_file(data_dir: Path, file_name: str) -> pd.DataFrame:
    normal, extra = extract_for_file(data_dir, file_name, with_label=True)
    pred = np.zeros(len(normal), dtype=int)
    if len(normal):
        f = normal.copy()
        stats = {col: expanding_feature(f, col) for col in [
            "Temp", "Curr", "TxP0", "RxP0", "RxP1", "RxP2", "RxP3", "RxP4",
            "TxP1", "TxP2", "TxP3", "TxP4",
        ]}
        rules = np.zeros(len(f), dtype=bool)
        rules |= stats["Temp"]["Min"].lt(0).to_numpy()
        rules |= stats["Curr"]["Min"].lt(5000).to_numpy()
        rules |= stats["Temp"]["Diff"].gt(100).to_numpy()
        rules |= stats["Curr"]["Diff"].gt(6000).to_numpy()
        rules |= stats["Temp"]["Std"].gt(10).to_numpy()
        rules |= stats["Curr"]["Std"].gt(1500).to_numpy()
        rules |= stats["Temp"]["Kurt"].fillna(NA_DEFAULT).gt(500).to_numpy()

        for col in ["TxP0", "RxP0"]:
            rules |= stats[col]["Min"].lt(0).to_numpy()
            rules |= stats[col]["Diff"].gt(1000).to_numpy()
            rules |= stats[col]["Std"].gt(500).to_numpy()
        rules |= (
            stats["TxP0"]["Kurt"].fillna(NA_DEFAULT).gt(500)
            & stats["RxP0"]["Kurt"].fillna(NA_DEFAULT).gt(500)
        ).to_numpy()

        for col in ["RxP1", "RxP2", "RxP3", "RxP4"]:
            rules |= stats[col]["Min"].lt(0).to_numpy()
            rules |= stats[col]["Diff"].gt(1000).to_numpy()
            rules |= stats[col]["Std"].gt(200).to_numpy()

        for col in ["TxP1", "TxP2", "TxP3", "TxP4"]:
            rules |= stats[col]["Min"].lt(0).to_numpy()
            rules |= stats[col]["Diff"].gt(1000).to_numpy()
            rules |= stats[col]["Std"].gt(100).to_numpy()

        pred = rules.astype(int)

    parts = []
    if len(normal):
        parts.append(pd.DataFrame({"timestamp": normal["Ts"].to_numpy(dtype=np.int64), "predict": pred}))
    if len(extra):
        parts.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                    "predict": (pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER).astype(int),
                }
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values("timestamp") if parts else pd.DataFrame(columns=["timestamp", "predict"])


def write_rule_predictions(data_dir: Path, file_names: list[str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        pred = rule_predict_one_file(data_dir, name)
        pred.to_csv(out_dir / name, index=False)
        n_rows += len(pred)
        if idx % 500 == 0:
            print(f"  [rule] {idx}/{len(file_names)} files")
    return n_rows


def merge_or(pred_dirs: list[Path], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_names = sorted(p.name for p in pred_dirs[0].glob("*.csv"))
    n_rows = 0
    for name in file_names:
        merged: pd.DataFrame | None = None
        for pred_dir in pred_dirs:
            path = pred_dir / name
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=lambda c: c in {"timestamp", "predict"})
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
            df["predict"] = pd.to_numeric(df["predict"], errors="coerce").fillna(0).astype(int)
            if merged is None:
                merged = df.rename(columns={"predict": "predict_0"})
            else:
                idx = len([c for c in merged.columns if c.startswith("predict_")])
                merged = merged.merge(df.rename(columns={"predict": f"predict_{idx}"}), on="timestamp", how="outer")
        if merged is None:
            continue
        pred_cols = [c for c in merged.columns if c.startswith("predict_")]
        merged[pred_cols] = merged[pred_cols].fillna(0).astype(int)
        out = pd.DataFrame(
            {
                "timestamp": merged["timestamp"].astype("int64"),
                "predict": merged[pred_cols].max(axis=1).astype(int),
            }
        ).sort_values("timestamp")
        out.to_csv(out_dir / name, index=False)
        n_rows += len(out)
    return n_rows


def evaluate_dir(model_name: str, fold: int, pred_dir: Path, label_dir: Path, out_dir: Path) -> dict[str, float]:
    eval_dir = out_dir / model_name / f"fold_{fold}" / "evaluation"
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, label_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    n_jobs: int,
    rf_estimators: int,
    xgb_estimators: int,
    max_train_files: int | None,
    max_test_files: int | None,
    overwrite: bool,
) -> list[ResultRow]:
    train_files = index_df.loc[index_df["folder_index"] != fold, "file_name"].tolist()
    test_files = index_df.loc[index_df["folder_index"] == fold, "file_name"].tolist()
    if max_train_files is not None:
        train_files = train_files[: int(max_train_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]

    fold_started = time.time()
    X_train, y_train, train_rows = load_train_arrays(data_dir, train_files)
    print(f"[fold {fold}] train_rows={train_rows}, pos={int(y_train.sum())}, neg={int(len(y_train)-y_train.sum())}")
    results: list[ResultRow] = []

    # RF once, two thresholds.
    rf_model = RandomForestClassifier(
        n_estimators=rf_estimators,
        random_state=2024,
        verbose=0,
        n_jobs=n_jobs,
    )
    t0 = time.time()
    print(f"[fold {fold}] train RF n_estimators={rf_estimators}, n_jobs={n_jobs}")
    rf_model.fit(X_train, y_train)
    for model_name, threshold in [("rf_thresh0.5", 0.5), ("rf_thresh0.3", 0.3)]:
        pred_dir = out_root / model_name / f"fold_{fold}" / "predictions"
        if overwrite or not (out_root / model_name / f"fold_{fold}" / "fold_summary.json").exists():
            test_rows = write_tree_predictions(rf_model, data_dir, test_files, pred_dir, threshold)
            metrics = evaluate_dir(model_name, fold, pred_dir, data_dir, out_root)
            row = ResultRow(model_name, fold, threshold, len(train_files), len(test_files), train_rows, test_rows, time.time() - t0, metrics)
            (out_root / model_name / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
            (out_root / model_name / f"fold_{fold}" / "fold_summary.json").write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")
            results.append(row)
    del rf_model
    gc.collect()

    # XGBoost over model2 default feature table.
    t0 = time.time()
    print(f"[fold {fold}] train XGBoost n_estimators={xgb_estimators}, n_jobs={n_jobs}, CPU hist")
    xgb_model = XGBClassifier(
        n_estimators=xgb_estimators,
        random_state=2024,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=n_jobs,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred_dir = out_root / "xgboost" / f"fold_{fold}" / "predictions"
    test_rows = write_tree_predictions(xgb_model, data_dir, test_files, xgb_pred_dir, 0.5)
    metrics = evaluate_dir("xgboost", fold, xgb_pred_dir, data_dir, out_root)
    row = ResultRow("xgboost", fold, 0.5, len(train_files), len(test_files), train_rows, test_rows, time.time() - t0, metrics)
    (out_root / "xgboost" / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    (out_root / "xgboost" / f"fold_{fold}" / "fold_summary.json").write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")
    results.append(row)
    del xgb_model, X_train, y_train
    gc.collect()

    # Rule model.
    t0 = time.time()
    print(f"[fold {fold}] run RuleModel")
    rule_pred_dir = out_root / "rule" / f"fold_{fold}" / "predictions"
    test_rows = write_rule_predictions(data_dir, test_files, rule_pred_dir)
    metrics = evaluate_dir("rule", fold, rule_pred_dir, data_dir, out_root)
    row = ResultRow("rule", fold, None, len(train_files), len(test_files), train_rows, test_rows, time.time() - t0, metrics)
    (out_root / "rule" / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    (out_root / "rule" / f"fold_{fold}" / "fold_summary.json").write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")
    results.append(row)

    # OR merges.
    merges = {
        "rf0.5_plus_xgboost": [
            out_root / "rf_thresh0.5" / f"fold_{fold}" / "predictions",
            out_root / "xgboost" / f"fold_{fold}" / "predictions",
        ],
        "rf0.5_plus_xgboost_plus_rule": [
            out_root / "rf_thresh0.5" / f"fold_{fold}" / "predictions",
            out_root / "xgboost" / f"fold_{fold}" / "predictions",
            out_root / "rule" / f"fold_{fold}" / "predictions",
        ],
        "rf0.3_plus_xgboost": [
            out_root / "rf_thresh0.3" / f"fold_{fold}" / "predictions",
            out_root / "xgboost" / f"fold_{fold}" / "predictions",
        ],
    }
    for model_name, dirs in merges.items():
        t0 = time.time()
        pred_dir = out_root / model_name / f"fold_{fold}" / "predictions"
        test_rows = merge_or(dirs, pred_dir)
        metrics = evaluate_dir(model_name, fold, pred_dir, data_dir, out_root)
        row = ResultRow(model_name, fold, None, len(train_files), len(test_files), train_rows, test_rows, time.time() - t0, metrics)
        (out_root / model_name / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
        (out_root / model_name / f"fold_{fold}" / "fold_summary.json").write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")
        results.append(row)

    print(f"[fold {fold}] finished all entries in {(time.time() - fold_started) / 60:.1f} min")
    return results


def aggregate(results: list[ResultRow], out_root: Path) -> None:
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
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df.sort_values(["model", "fold"], inplace=True)
    df.to_csv(path, index=False)
    numeric = [c for c in df.columns if c not in {"model", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_protocol/model2_suite"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--xgb_estimators", type=int, default=10)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    args = parse_args()
    index_df = read_index(args.index_path)
    all_rows: list[ResultRow] = []
    for fold in args.folds:
        print(f"[run] OFP model2 suite fold={fold}")
        rows = run_fold(
            fold=fold,
            data_dir=args.data_dir,
            index_df=index_df,
            out_root=args.out_root,
            n_jobs=args.n_jobs,
            rf_estimators=args.rf_estimators,
            xgb_estimators=args.xgb_estimators,
            max_train_files=args.max_train_files,
            max_test_files=args.max_test_files,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        aggregate(rows, args.out_root)
        for row in rows:
            print(
                f"[done] {row.model} fold={fold} "
                f"F1={row.metrics.get('f1_score', 0):.6f} "
                f"P={row.metrics.get('precision', 0):.6f} "
                f"R={row.metrics.get('recall', 0):.6f}"
            )
    aggregate(all_rows, args.out_root)


if __name__ == "__main__":
    main()

