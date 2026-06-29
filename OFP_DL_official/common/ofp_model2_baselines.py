from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
    MODEL2_FEATURES,
    TEMP_OUTLIER,
    first_anomaly_ahead_label,
    first_event_valid_mask,
    make_model2_features,
    positive_probability,
)
from OFP_DL_official.ofp_protocol.run_ofp_model2_suite import evaluate_dir

from OFP_DL_official.common.index_split import files_for_index_fold


@dataclass
class OFPModel2Result:
    model: str
    fold: int
    threshold: float
    train_modules: int
    test_modules: int
    train_rows: int
    test_rows: int
    seconds: float
    metrics: dict[str, float]
    model_cfg: dict


def load_train_arrays_legacy_float32(data_dir: Path, file_names: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Load OFP model2 features with first-event 1-hour training labels."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        raw = pd.read_csv(data_dir / name)
        normal, _extra = make_model2_features(raw, with_label=True)
        if len(normal):
            raw_idx = normal.index.to_numpy(dtype=int)
            valid_mask = first_event_valid_mask(raw)[raw_idx]
            labels = first_anomaly_ahead_label(raw)[raw_idx]
            normal = normal.loc[valid_mask]
            if len(normal) <= 0:
                continue
            xs.append(normal[MODEL2_FEATURES].to_numpy(dtype=np.float32))
            ys.append(labels[valid_mask].astype(np.int8))
            n_rows += len(normal)
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(MODEL2_FEATURES)), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int8)
    return X, y, n_rows


def write_tree_predictions_legacy_float32(
    model,
    data_dir: Path,
    file_names: list[str],
    out_dir: Path,
    threshold: float,
) -> int:
    """Write predictions with legacy OFP feature timestamps.

    This intentionally preserves the old OFP reproduction behavior where the
    feature frame is float32, including `Ts`; converting it back to int64 can
    round large UNIX timestamps. This is required to reproduce README numbers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        raw = pd.read_csv(data_dir / name)
        normal, extra = make_model2_features(raw, with_label=True)
        parts = []
        if len(normal):
            X = normal[MODEL2_FEATURES].to_numpy(dtype=np.float32)
            scores = positive_probability(model, X)
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": normal["Ts"].to_numpy(dtype=np.int64),
                        "predict": (scores >= float(threshold)).astype(int),
                        "proba": scores,
                    }
                )
            )
        if len(extra):
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                        "predict": (
                            pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER
                        ).astype(int),
                        "proba": np.nan,
                    }
                )
            )
        pred = (
            pd.concat(parts, ignore_index=True).sort_values("timestamp")
            if parts
            else pd.DataFrame(columns=["timestamp", "predict", "proba"])
        )
        pred.to_csv(out_dir / name, index=False)
        n_rows += len(pred)
    return n_rows


def run_ofp_model2_fold(
    model_name: str,
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    n_jobs: int = 4,
    rf_estimators: int = 100,
    xgb_estimators: int = 10,
    rf_threshold: float = 0.5,
    xgb_threshold: float = 0.5,
) -> OFPModel2Result:
    """Run one OFP model2 baseline with README-style settings."""
    if model_name not in {"rf", "xgboost"}:
        raise ValueError(model_name)
    t0 = time.time()
    train_files, test_files = files_for_index_fold(index_df, int(fold))

    X_train, y_train, train_rows = load_train_arrays_legacy_float32(data_dir, train_files)
    if len(y_train) == 0 or len(np.unique(y_train)) < 2:
        raise ValueError(f"{model_name} fold {fold} needs both positive and negative training rows")
    print(
        f"[ofp-model2 {model_name} fold={fold}] train_rows={train_rows} "
        f"pos={int(y_train.sum())} neg={int(len(y_train)-y_train.sum())}"
    )

    if model_name == "rf":
        threshold = float(rf_threshold)
        model_cfg = {
            "source": "OFP model2 TreeModel",
            "n_estimators": int(rf_estimators),
            "random_state": 2024,
            "n_jobs": int(n_jobs),
            "threshold": threshold,
            "feature_count": len(MODEL2_FEATURES),
            "train_label": "first_event_1h_ahead",
            "timestamp_mode": "legacy_float32_feature_ts",
        }
        model = RandomForestClassifier(
            n_estimators=int(rf_estimators),
            random_state=2024,
            verbose=0,
            n_jobs=int(n_jobs),
        )
    else:
        threshold = float(xgb_threshold)
        model_cfg = {
            "source": "OFP model2 suite XGBoost",
            "n_estimators": int(xgb_estimators),
            "random_state": 2024,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "n_jobs": int(n_jobs),
            "threshold": threshold,
            "feature_count": len(MODEL2_FEATURES),
            "train_label": "first_event_1h_ahead",
            "timestamp_mode": "legacy_float32_feature_ts",
        }
        model = XGBClassifier(
            n_estimators=int(xgb_estimators),
            random_state=2024,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=int(n_jobs),
        )

    print(f"[ofp-model2 {model_name} fold={fold}] fit {model_cfg}")
    model.fit(X_train, y_train)

    run_dir = Path(out_root) / model_name / f"fold_{fold}"
    pred_dir = run_dir / "predictions"
    test_rows = write_tree_predictions_legacy_float32(model, data_dir, test_files, pred_dir, threshold)
    metrics = evaluate_dir(model_name, int(fold), pred_dir, data_dir, Path(out_root))

    result = OFPModel2Result(
        model=model_name,
        fold=int(fold),
        threshold=threshold,
        train_modules=len(train_files),
        test_modules=len(test_files),
        train_rows=int(train_rows),
        test_rows=int(test_rows),
        seconds=time.time() - t0,
        metrics=metrics,
        model_cfg=model_cfg,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    del model, X_train, y_train
    gc.collect()
    return result


def aggregate_model2_results(results: list[OFPModel2Result], out_root: Path) -> None:
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

