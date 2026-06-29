from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
    SENSORS,
    ahead120_label,
    first_event_valid_mask,
    metrics_to_dict,
    read_index,
)


SEC_IN_HOUR = 3600.0

KEY_ROLLING_FEATURES = [
    "temperature",
    "current",
    "currentTXPower",
    "currentRXPower",
    "rx_lane_mean",
    "tx_lane_mean",
]


@dataclass
class FoldResult:
    model: str
    fold: int
    threshold_mode: str
    threshold: float
    feature_count: int
    train_modules: int
    val_modules: int
    test_modules: int
    train_rows: int
    val_rows: int
    test_rows: int
    seconds: float
    metrics: dict[str, float]


def agg_feature_names(rolling_windows: tuple[int, ...]) -> list[str]:
    names = list(SENSORS)
    names.extend(
        [
            "rx_lane_mean",
            "rx_lane_std",
            "rx_lane_range",
            "tx_lane_mean",
            "tx_lane_std",
            "tx_lane_range",
            "rx0_lane_mean_gap",
            "tx0_lane_mean_gap",
            "tx_rx_gap",
            "abs_tx_rx_gap",
        ]
    )
    for col in KEY_ROLLING_FEATURES:
        names.append(f"{col}_delta")
    for win in rolling_windows:
        for col in KEY_ROLLING_FEATURES:
            names.append(f"{col}_rmean{win}")
            names.append(f"{col}_rstd{win}")
    return names


def build_agg_features(raw_df: pd.DataFrame, rolling_windows: tuple[int, ...]) -> pd.DataFrame:
    base = raw_df[SENSORS].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    out = base.copy()

    rx_cols = [
        "currentMultiRXPower1",
        "currentMultiRXPower2",
        "currentMultiRXPower3",
        "currentMultiRXPower4",
    ]
    tx_cols = [
        "currentMultiTXPower1",
        "currentMultiTXPower2",
        "currentMultiTXPower3",
        "currentMultiTXPower4",
    ]
    rx_lanes = base[rx_cols]
    tx_lanes = base[tx_cols]

    out["rx_lane_mean"] = rx_lanes.mean(axis=1)
    out["rx_lane_std"] = rx_lanes.std(axis=1).fillna(0.0)
    out["rx_lane_range"] = rx_lanes.max(axis=1) - rx_lanes.min(axis=1)
    out["tx_lane_mean"] = tx_lanes.mean(axis=1)
    out["tx_lane_std"] = tx_lanes.std(axis=1).fillna(0.0)
    out["tx_lane_range"] = tx_lanes.max(axis=1) - tx_lanes.min(axis=1)
    out["rx0_lane_mean_gap"] = base["currentRXPower"] - out["rx_lane_mean"]
    out["tx0_lane_mean_gap"] = base["currentTXPower"] - out["tx_lane_mean"]
    out["tx_rx_gap"] = base["currentTXPower"] - base["currentRXPower"]
    out["abs_tx_rx_gap"] = out["tx_rx_gap"].abs()

    for col in KEY_ROLLING_FEATURES:
        series = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_delta"] = series.diff().fillna(0.0)
        for win in rolling_windows:
            roll = series.rolling(int(win), min_periods=2)
            out[f"{col}_rmean{win}"] = roll.mean().fillna(series)
            out[f"{col}_rstd{win}"] = roll.std().fillna(0.0)

    return out[agg_feature_names(rolling_windows)].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def iter_file_names(values, limit: int | None = None) -> list[str]:
    names = list(values)
    if limit is not None:
        return names[: int(limit)]
    return names


def split_train_val(
    index_df: pd.DataFrame,
    train_files: list[str],
    val_fraction: float,
    seed: int,
    max_train_files: int | None,
    max_val_files: int | None,
) -> tuple[list[str], list[str]]:
    if val_fraction <= 0:
        return iter_file_names(train_files, max_train_files), []

    local = index_df[index_df["file_name"].isin(train_files)].copy()
    labels = local["Label"].astype(int).to_numpy()
    train_part, val_part = train_test_split(
        local,
        test_size=float(val_fraction),
        random_state=int(seed),
        stratify=labels,
    )
    return (
        iter_file_names(train_part["file_name"], max_train_files),
        iter_file_names(val_part["file_name"], max_val_files),
    )


def load_arrays(
    data_dir: Path,
    file_names: list[str],
    rolling_windows: tuple[int, ...],
    label_mode: str,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    total_rows = 0

    for name in file_names:
        raw = pd.read_csv(data_dir / name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        valid_mask = first_event_valid_mask(raw)
        if not np.any(valid_mask):
            continue
        features = build_agg_features(raw, rolling_windows).iloc[valid_mask]
        xs.append(features.to_numpy(dtype=np.float32))
        total_rows += int(np.sum(valid_mask))
        if label_mode in {"ahead120", "anomaly", "module"}:
            ys.append(ahead120_label(raw)[valid_mask])
        elif label_mode == "none":
            pass
        else:
            raise ValueError(label_mode)

    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(agg_feature_names(rolling_windows))), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else None
    return X, y, total_rows


def positive_probability(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(len(X), dtype=np.float32)


def score_modules(
    data_dir: Path,
    file_names: list[str],
    model: XGBClassifier,
    rolling_windows: tuple[int, ...],
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    total_rows = 0
    for name in file_names:
        raw = pd.read_csv(data_dir / name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        anomaly = pd.to_numeric(raw["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        valid_mask = first_event_valid_mask(raw)
        X = build_agg_features(raw, rolling_windows).to_numpy(dtype=np.float32)
        scores = positive_probability(model, X).astype(np.float32)
        anomaly_idx = np.where(anomaly > 0)[0]
        true_label = int(len(anomaly_idx) > 0)
        true_ts = float(timestamps[int(anomaly_idx[0])]) if true_label else None
        rows.append(
            {
                "file_name": name,
                "timestamps": timestamps,
                "scores": scores,
                "true_label": true_label,
                "true_ts": true_ts,
                "valid_mask": valid_mask,
            }
        )
        total_rows += int(np.sum(valid_mask))
    return rows, total_rows


def module_metrics(module_rows: list[dict], threshold: float) -> dict[str, float]:
    decisions = []
    for row in module_rows:
        scores = row["scores"]
        timestamps = row["timestamps"]
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            true_ts = row["true_ts"]
            valid_mask = np.ones(len(scores), dtype=bool) if true_ts is None else timestamps < float(true_ts)
        positive_idx = np.where((scores >= threshold) & valid_mask)[0]
        predict_label = int(len(positive_idx) > 0)
        predict_ts = float(timestamps[int(positive_idx[0])]) if predict_label else None
        true_label = int(row["true_label"])
        true_ts = row["true_ts"]
        valid_predict_positive = 0
        hit = 0
        lead_hour = math.nan
        if predict_label:
            if true_label == 0:
                valid_predict_positive = 1
            elif true_ts is not None and predict_ts is not None and true_ts > predict_ts:
                valid_predict_positive = 1
                hit = 1
                lead_hour = abs(true_ts - predict_ts) / SEC_IN_HOUR
        decisions.append((true_label, valid_predict_positive, hit, lead_hour))

    true_pos = sum(1 for true_label, *_ in decisions if true_label > 0)
    pred_pos = sum(1 for _true_label, valid_pred, *_ in decisions if valid_pred > 0)
    tp = sum(1 for *_rest, hit, _lead in decisions if hit > 0)
    fp = pred_pos - tp
    fn = true_pos - tp
    tn = len(decisions) - tp - fp - fn
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / true_pos if true_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(decisions) if decisions else 0.0
    leads = [lead for *_rest, lead in decisions if not math.isnan(lead)]
    avg_lead_hour = float(np.mean(leads)) if leads else 0.0
    min_lead_hour = float(np.min(leads)) if leads else 0.0
    final_score = f1 + math.tanh(avg_lead_hour) + math.tanh(min_lead_hour) + accuracy
    return {
        "final_score": final_score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "all_hit_cnt": float(tp),
        "all_predict_pos_cnt": float(pred_pos),
        "all_true_pos_cnt": float(true_pos),
        "avg_lead_hour": avg_lead_hour,
        "min_lead_hour": min_lead_hour,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "evaluated_module_cnt": float(len(decisions)),
    }


def choose_threshold(module_rows: list[dict], grid_size: int) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = module_metrics(module_rows, best_threshold)
    for threshold in np.linspace(0.01, 0.99, int(grid_size)):
        metrics = module_metrics(module_rows, float(threshold))
        if (
            metrics["final_score"],
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
        ) > (
            best_metrics["final_score"],
            best_metrics["f1_score"],
            best_metrics["precision"],
            best_metrics["recall"],
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def write_predictions(
    out_dir: Path,
    module_rows: list[dict],
    threshold: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in module_rows:
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            true_ts = row["true_ts"]
            valid_mask = np.ones(len(row["scores"]), dtype=bool) if true_ts is None else row["timestamps"] < float(true_ts)
        pred = ((row["scores"] >= threshold) & valid_mask).astype(int)
        pd.DataFrame(
            {
                "timestamp": row["timestamps"],
                "predict": pred,
                "score": row["scores"],
                "valid_for_eval": valid_mask.astype(int),
            }
        ).to_csv(out_dir / row["file_name"], index=False)


def make_xgb(args: argparse.Namespace) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=int(args.xgb_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=args.tree_method,
        n_jobs=int(args.n_jobs),
        random_state=int(args.seed),
    )


def train_model(
    args: argparse.Namespace,
    data_dir: Path,
    train_files: list[str],
    rolling_windows: tuple[int, ...],
) -> tuple[XGBClassifier, int]:
    X_train, y_train, train_rows = load_arrays(data_dir, train_files, rolling_windows, label_mode=args.label_mode)
    model = make_xgb(args)
    model.fit(X_train, y_train)
    return model, train_rows


def run_fold(args: argparse.Namespace, index_df: pd.DataFrame, fold: int) -> list[FoldResult]:
    t0 = time.time()
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    rolling_windows = tuple(int(x) for x in args.rolling_windows)
    all_train_files = iter_file_names(index_df.loc[index_df["folder_index"] != fold, "file_name"])
    test_files = iter_file_names(index_df.loc[index_df["folder_index"] == fold, "file_name"], args.max_test_files)
    train_files, val_files = split_train_val(
        index_df,
        all_train_files,
        args.val_fraction,
        args.seed + fold,
        args.max_train_files,
        args.max_val_files,
    )

    model_name = f"xgb_agg_{args.label_mode}"
    fold_dir = out_root / model_name / f"fold_{fold}"
    threshold = float(args.fixed_threshold)
    val_rows = 0
    val_metrics = {}

    if val_files:
        model_for_threshold, train_rows = train_model(args, data_dir, train_files, rolling_windows)
        val_module_rows, val_rows = score_modules(data_dir, val_files, model_for_threshold, rolling_windows)
        threshold, val_metrics = choose_threshold(val_module_rows, args.threshold_grid_size)
        (fold_dir / "validation").mkdir(parents=True, exist_ok=True)
        (fold_dir / "validation" / "threshold_metrics.json").write_text(
            json.dumps(val_metrics, indent=2),
            encoding="utf-8",
        )
        if args.retrain_full:
            model, train_rows = train_model(args, data_dir, iter_file_names(all_train_files, args.max_train_files), rolling_windows)
        else:
            model = model_for_threshold
    else:
        model, train_rows = train_model(args, data_dir, train_files, rolling_windows)

    test_module_rows, test_rows = score_modules(data_dir, test_files, model, rolling_windows)
    results: list[FoldResult] = []
    for mode, cur_threshold in [("fixed", float(args.fixed_threshold)), ("val_selected", float(threshold))]:
        pred_dir = fold_dir / mode / "predictions"
        eval_dir = fold_dir / mode / "evaluation"
        write_predictions(pred_dir, test_module_rows, cur_threshold)
        summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
        detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
        result = FoldResult(
            model=model_name,
            fold=fold,
            threshold_mode=mode,
            threshold=cur_threshold,
            feature_count=len(agg_feature_names(rolling_windows)),
            train_modules=len(train_files),
            val_modules=len(val_files),
            test_modules=len(test_files),
            train_rows=int(train_rows),
            val_rows=int(val_rows),
            test_rows=int(test_rows),
            seconds=time.time() - t0,
            metrics=metrics_to_dict(summary_df),
        )
        (fold_dir / mode / "fold_summary.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        results.append(result)
        print(
            f"[xgb-agg fold={fold} {mode}] "
            f"thr={cur_threshold:.3f} final={result.metrics['final_score']:.4f} "
            f"F1={result.metrics['f1_score']:.4f} "
            f"P={result.metrics['precision']:.4f} R={result.metrics['recall']:.4f} "
            f"lead={result.metrics['avg_lead_hour']:.2f}h"
        )
    return results


def write_summary(out_root: Path, results: list[FoldResult]) -> None:
    rows = []
    for item in results:
        row = {
            "model": item.model,
            "fold": item.fold,
            "threshold_mode": item.threshold_mode,
            "threshold": item.threshold,
            "feature_count": item.feature_count,
            "train_modules": item.train_modules,
            "val_modules": item.val_modules,
            "test_modules": item.test_modules,
            "train_rows": item.train_rows,
            "val_rows": item.val_rows,
            "test_rows": item.test_rows,
            "seconds": item.seconds,
        }
        row.update(item.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "xgb_agg_results.csv", index=False)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df.groupby(["model", "threshold_mode"])[numeric_cols].agg(["mean", "std"])
    summary.to_csv(out_root / "xgb_agg_results_mean_std.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost with aggregate OFP features.")
    parser.add_argument("--data_dir", type=Path, default=Path("../dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("../dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("../output/ofp_xgb_agg_features"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--label_mode", choices=["ahead120", "anomaly", "module"], default="ahead120")
    parser.add_argument("--rolling_windows", nargs="+", type=int, default=[12, 36])
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--retrain_full", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xgb_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--tree_method", default="hist")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(Path(args.index_path))
    results: list[FoldResult] = []
    for fold in args.folds:
        results.extend(run_fold(args, index_df, int(fold)))
    write_summary(Path(args.out_root), results)


if __name__ == "__main__":
    main()
