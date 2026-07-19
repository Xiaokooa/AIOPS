from __future__ import annotations

import argparse
import json
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

from OFP.deep_learning.official.ofp_formal_protocol.protocol import (
    PRIMARY_HORIZON_SECONDS,
    choose_threshold_by_final_score,
    files_for_fold,
    first_anomaly_timestamp,
    read_manifest,
)
from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP.deep_learning.official.ofp_protocol.run_ofp_model2_suite import merge_or, rule_predict_one_file
from OFP.deep_learning.official.ofp_protocol.run_ofp_model2_strict import DEFAULT_FEATURE_LIST, TEMP_OUTLIER, extract_for_file, positive_probability


@dataclass
class FormalTreeResult:
    model: str
    fold: int
    threshold: float
    train_modules: int
    val_modules: int
    test_modules: int
    train_rows: int
    seconds: float
    val_metrics: dict[str, float]
    metrics: dict[str, float]


def first_event_labels_from_ts(timestamps: np.ndarray, true_ts: float | None) -> np.ndarray:
    if true_ts is None:
        return np.zeros(len(timestamps), dtype=np.int8)
    labels = (timestamps.astype(float) >= true_ts - PRIMARY_HORIZON_SECONDS) & (timestamps.astype(float) < true_ts)
    return labels.astype(np.int8)


def valid_mask_from_ts(timestamps: np.ndarray, true_ts: float | None) -> np.ndarray:
    timestamps = timestamps.astype(float)
    valid = np.isfinite(timestamps)
    if true_ts is None:
        return valid
    return valid & (timestamps < float(true_ts))


def raw_event_info(data_dir: Path, file_name: str) -> tuple[int, float | None]:
    df = pd.read_csv(data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly"})
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    true_ts = first_anomaly_timestamp(timestamps, anomaly)
    return int(true_ts is not None), true_ts


def load_train_arrays(data_dir: Path, file_names: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        normal, _extra = extract_for_file(data_dir, name, with_label=True)
        if len(normal):
            _true_label, true_ts = raw_event_info(data_dir, name)
            timestamps = pd.to_numeric(normal["Ts"], errors="coerce").to_numpy(dtype=np.int64)
            labels = first_event_labels_from_ts(timestamps, true_ts)
            valid_mask = valid_mask_from_ts(timestamps, true_ts)
            normal = normal.loc[valid_mask]
            if len(normal) <= 0:
                continue
            xs.append(normal[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32))
            ys.append(labels[valid_mask])
            n_rows += len(normal)
        if idx % 500 == 0:
            print(f"  [load formal tree] {idx}/{len(file_names)} files rows={n_rows}")
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(DEFAULT_FEATURE_LIST)), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int8)
    return X, y, n_rows


def score_module(model, data_dir: Path, file_name: str) -> dict:
    normal, extra = extract_for_file(data_dir, file_name, with_label=True)
    frames = []
    if len(normal):
        X = normal[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32)
        scores = positive_probability(model, X)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_numeric(normal["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                    "score": scores,
                }
            )
        )
    if len(extra):
        temp = pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                    "score": (temp == TEMP_OUTLIER).astype(float),
                }
            )
        )
    if frames:
        scored = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    else:
        scored = pd.DataFrame({"timestamp": [], "score": []})
    true_label, true_ts = raw_event_info(data_dir, file_name)
    valid_mask = valid_mask_from_ts(scored["timestamp"].to_numpy(dtype=np.int64), true_ts)
    return {
        "file_name": file_name,
        "timestamps": scored["timestamp"].to_numpy(dtype=np.int64),
        "scores": scored["score"].to_numpy(dtype=np.float32),
        "true_label": true_label,
        "true_ts": true_ts,
        "valid_mask": valid_mask,
    }


def collect_scores(model, data_dir: Path, file_names: list[str]) -> list[dict]:
    return [score_module(model, data_dir, name) for name in file_names]


def write_predictions(module_rows: list[dict], out_dir: Path, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in module_rows:
        valid_mask = np.asarray(row.get("valid_mask", np.ones(len(row["scores"]), dtype=bool)), dtype=bool)
        pred = ((np.asarray(row["scores"], dtype=float) >= float(threshold)) & valid_mask).astype(int)
        pd.DataFrame(
            {
                "timestamp": np.asarray(row["timestamps"], dtype=np.int64),
                "predict": pred,
                "score": np.asarray(row["scores"], dtype=float),
                "valid_for_eval": valid_mask.astype(int),
            }
        ).to_csv(out_dir / row["file_name"], index=False)


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def make_model(model_name: str, n_jobs: int, rf_estimators: int, xgb_estimators: int, seed: int):
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=rf_estimators, random_state=seed, n_jobs=n_jobs, verbose=0)
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=xgb_estimators,
            random_state=seed,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
        )
    raise ValueError(model_name)


def run_fold(
    model_name: str,
    fold: int,
    data_dir: Path,
    manifest_df: pd.DataFrame,
    out_root: Path,
    n_jobs: int,
    rf_estimators: int,
    xgb_estimators: int,
    threshold_grid_size: int,
    seed: int,
    max_train_files: int | None,
    max_val_files: int | None,
    max_test_files: int | None,
) -> FormalTreeResult:
    t0 = time.time()
    train_files, val_files, test_files = files_for_fold(manifest_df, fold)
    if max_train_files is not None:
        train_files = train_files[: int(max_train_files)]
    if max_val_files is not None:
        val_files = val_files[: int(max_val_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]

    X_train, y_train, train_rows = load_train_arrays(data_dir, train_files)
    if len(y_train) == 0 or len(np.unique(y_train)) < 2:
        raise ValueError(f"{model_name} fold {fold} needs both positive and negative training rows")
    print(f"[formal tree] model={model_name} fold={fold} rows={train_rows} pos={int(y_train.sum())} neg={int(len(y_train)-y_train.sum())}")
    model = make_model(model_name, n_jobs, rf_estimators, xgb_estimators, seed + int(fold))
    model.fit(X_train, y_train)

    val_rows = collect_scores(model, data_dir, val_files)
    threshold, val_metrics = choose_threshold_by_final_score(val_rows, grid_size=threshold_grid_size)
    print(f"[formal tree] model={model_name} fold={fold} val_final={val_metrics['final_score']:.4f} thr={threshold:.3f}")

    test_rows = collect_scores(model, data_dir, test_files)
    run_dir = out_root / model_name / f"fold_{fold}"
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(test_rows, pred_dir, threshold)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)

    result = FormalTreeResult(
        model=model_name,
        fold=int(fold),
        threshold=float(threshold),
        train_modules=len(train_files),
        val_modules=len(val_files),
        test_modules=len(test_files),
        train_rows=int(train_rows),
        seconds=time.time() - t0,
        val_metrics=val_metrics,
        metrics=metrics,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result


def run_rule_fold(
    fold: int,
    data_dir: Path,
    manifest_df: pd.DataFrame,
    out_root: Path,
    max_test_files: int | None,
) -> FormalTreeResult:
    t0 = time.time()
    train_files, val_files, test_files = files_for_fold(manifest_df, fold)
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]
    run_dir = out_root / "rule" / f"fold_{fold}"
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(test_files, 1):
        pred = rule_predict_one_file(data_dir, name)
        pred.to_csv(pred_dir / name, index=False)
        if idx % 500 == 0:
            print(f"  [formal rule] fold={fold} {idx}/{len(test_files)} files")
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)
    result = FormalTreeResult(
        model="rule",
        fold=int(fold),
        threshold=float("nan"),
        train_modules=len(train_files),
        val_modules=len(val_files),
        test_modules=len(test_files),
        train_rows=0,
        seconds=time.time() - t0,
        val_metrics={},
        metrics=metrics,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result


def run_fusion_fold(
    model_name: str,
    fold: int,
    data_dir: Path,
    manifest_df: pd.DataFrame,
    out_root: Path,
    source_models: list[str],
    max_test_files: int | None,
) -> FormalTreeResult:
    t0 = time.time()
    train_files, val_files, test_files = files_for_fold(manifest_df, fold)
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]
    source_dirs = [out_root / src / f"fold_{fold}" / "predictions" for src in source_models]
    missing = [str(path) for path in source_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Fusion {model_name} fold {fold} missing source prediction dirs: {missing}")
    run_dir = out_root / model_name / f"fold_{fold}"
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    merge_or(source_dirs, pred_dir)
    if max_test_files is not None:
        allowed = set(test_files)
        for path in pred_dir.glob("*.csv"):
            if path.name not in allowed:
                path.unlink()
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)
    result = FormalTreeResult(
        model=model_name,
        fold=int(fold),
        threshold=float("nan"),
        train_modules=len(train_files),
        val_modules=len(val_files),
        test_modules=len(test_files),
        train_rows=0,
        seconds=time.time() - t0,
        val_metrics={},
        metrics=metrics,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result


def aggregate(results: list[FormalTreeResult], out_root: Path) -> None:
    rows = []
    for result in results:
        row = {
            "model": result.model,
            "fold": result.fold,
            "threshold": result.threshold,
            "train_modules": result.train_modules,
            "val_modules": result.val_modules,
            "test_modules": result.test_modules,
            "train_rows": result.train_rows,
            "seconds": result.seconds,
        }
        row.update(result.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "fold_metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df.sort_values(["model", "fold"], inplace=True)
    df.to_csv(metrics_path, index=False)
    numeric = [c for c in df.columns if c not in {"model", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF/XGBoost formal OFP baselines.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--manifest_path", type=Path, default=Path("output/ofp_formal_protocol/splits/formal_manifest.csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_formal_protocol/tree_baselines"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rule", "rf", "xgboost", "rf_xgboost", "rf_xgboost_rule"],
        choices=["rule", "rf", "xgboost", "rf_xgboost", "rf_xgboost_rule"],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--xgb_estimators", type=int, default=100)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = read_manifest(args.manifest_path)
    all_results: list[FormalTreeResult] = []
    for fold in args.folds:
        for model_name in args.models:
            if model_name in {"rf", "xgboost"}:
                result = run_fold(
                    model_name=model_name,
                    fold=fold,
                    data_dir=args.data_dir,
                    manifest_df=manifest_df,
                    out_root=args.out_root,
                    n_jobs=args.n_jobs,
                    rf_estimators=args.rf_estimators,
                    xgb_estimators=args.xgb_estimators,
                    threshold_grid_size=args.threshold_grid_size,
                    seed=args.seed,
                    max_train_files=args.max_train_files,
                    max_val_files=args.max_val_files,
                    max_test_files=args.max_test_files,
                )
            elif model_name == "rule":
                result = run_rule_fold(
                    fold=fold,
                    data_dir=args.data_dir,
                    manifest_df=manifest_df,
                    out_root=args.out_root,
                    max_test_files=args.max_test_files,
                )
            elif model_name == "rf_xgboost":
                result = run_fusion_fold(
                    model_name=model_name,
                    fold=fold,
                    data_dir=args.data_dir,
                    manifest_df=manifest_df,
                    out_root=args.out_root,
                    source_models=["rf", "xgboost"],
                    max_test_files=args.max_test_files,
                )
            elif model_name == "rf_xgboost_rule":
                result = run_fusion_fold(
                    model_name=model_name,
                    fold=fold,
                    data_dir=args.data_dir,
                    manifest_df=manifest_df,
                    out_root=args.out_root,
                    source_models=["rf", "xgboost", "rule"],
                    max_test_files=args.max_test_files,
                )
            else:
                raise ValueError(model_name)
            all_results.append(result)
            aggregate([result], args.out_root)
            print(
                f"[done formal tree] {model_name} fold={fold} "
                f"final={result.metrics.get('final_score', 0):.4f} "
                f"F1={result.metrics.get('f1_score', 0):.4f}"
            )
    aggregate(all_results, args.out_root)


if __name__ == "__main__":
    main()
