"""Strict OFP/model2 TreeModel reproduction under the local dataset layout.

The original ``OFP/model2`` package cannot be imported directly in this
workspace because ``Config.py`` hard-codes ``D:\\new_data`` and lists folders at
import time.  This runner mirrors the current default code path in
``OFP/model2/main.py`` without modifying ``OFP/``:

- model: ``MnTreeModel``
- features: ``FeatureExtractor.get_feature_by_default(win_size=5)``
- train label: first-event 1-hour ahead labels
- estimator: ``RandomForestClassifier(n_estimators=100, random_state=2024)``
- threshold: ``0.3``
- extra rows: ``Temp == -255`` are predicted positive by ``ExtraModel``
- evaluation: module-level first-warning-before-first-anomaly protocol
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP.deep_learning.official.ofp_protocol.run_ofp_baselines import (
    first_anomaly_ahead_label,
    first_event_valid_mask,
    read_index,
)


NA_DEFAULT = -999.0
TEMP_OUTLIER = -255.0
FEATURE_FILE_PREFIX = "feat_"

RAW_TO_OFP = {
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

BASE_FEATURES = [
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

DEFAULT_FEATURE_LIST = [
    *BASE_FEATURES,
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


@dataclass
class FoldSummary:
    model: str
    fold: int
    threshold: float
    n_estimators: int
    train_modules: int
    test_modules: int
    train_rows: int
    test_rows: int
    seconds: float
    metrics: dict[str, float]


def rolling_corr_like_ofp(a: pd.Series, b: pd.Series, first_fill: float) -> np.ndarray:
    """Vectorized equivalent of FeatureExtractor.get_feature_by_default.

    ``pearsonr`` returns NaN for constant windows; OFP later calls
    ``feat_df.fillna(NaDefault)``, so NaN correlation windows become -999.
    """
    out = a.rolling(5, min_periods=5).corr(b).to_numpy(dtype=np.float32)
    out[:4] = np.float32(first_fill)
    return np.nan_to_num(out, nan=NA_DEFAULT, posinf=NA_DEFAULT, neginf=NA_DEFAULT).astype(np.float32)


def read_as_ofp_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=RAW_TO_OFP)
    df = df[~df["Temp"].isna()].reset_index(drop=True)
    for col in [*BASE_FEATURES, "Ano"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def extract_default_feature_frame(df: pd.DataFrame, sn: str) -> pd.DataFrame:
    feature = pd.DataFrame(index=df.index)
    feature["FeCoCurrTemp"] = rolling_corr_like_ofp(df["Curr"], df["Temp"], 99.0)
    feature["FeCoCurrTxP0"] = rolling_corr_like_ofp(df["Curr"], df["TxP0"], -99.0)
    feature["FeCoCurrRxP0"] = rolling_corr_like_ofp(df["Curr"], df["RxP0"], -99.0)
    feature["FeCoTxP0RxP0"] = rolling_corr_like_ofp(df["TxP0"], df["RxP0"], 99.0)
    feature["FeTxP0-Max"] = df["TxP0"] - df[["TxP1", "TxP2", "TxP3", "TxP4"]].max(axis=1)
    feature["FeTxP0-Min"] = df["TxP0"] - df[["TxP1", "TxP2", "TxP3", "TxP4"]].min(axis=1)
    feature["FeRxP0-Max"] = df["RxP0"] - df[["RxP1", "RxP2", "RxP3", "RxP4"]].max(axis=1)
    feature["FeRxP0-Min"] = df["RxP0"] - df[["RxP1", "RxP2", "RxP3", "RxP4"]].min(axis=1)
    feature["sn"] = sn
    feature["step"] = np.arange(1, len(df) + 1)
    if "Ano" in df.columns:
        feature["Label"] = 1 if float(df["Ano"].sum()) > 0 else 0
    for col in df.columns:
        feature[col] = df[col]
    feature.fillna(NA_DEFAULT, inplace=True)
    return feature


def feature_data_get_df(feature_df: pd.DataFrame, with_label: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror FeatureData.get_df for MnTreeModel/default features."""
    feature_df = feature_df.copy()
    feature_df["row"] = np.arange(len(feature_df))
    cut_rows = (feature_df["Temp"] == TEMP_OUTLIER) | (feature_df["Temp"] == NA_DEFAULT)
    cut_df = feature_df.loc[cut_rows].copy()
    ret_df = feature_df.loc[~cut_rows].copy()

    ret_df["TsDelta"] = ret_df["Ts"].diff().fillna(NA_DEFAULT)
    cut_df["TsDelta"] = cut_df["Ts"].diff().fillna(NA_DEFAULT)

    cols = [*DEFAULT_FEATURE_LIST]
    if with_label and "Ano" in ret_df.columns:
        cols.append("Ano")
    ret_df = ret_df[cols].copy()
    ret_df["Ts"] = pd.to_numeric(ret_df["Ts"], errors="coerce").astype("int64")
    for col in ret_df.columns:
        if col != "Ts":
            ret_df[col] = pd.to_numeric(ret_df[col], errors="coerce").astype(np.float32)
    cut_cols = [*DEFAULT_FEATURE_LIST]
    if with_label and "Ano" in cut_df.columns:
        cut_cols.append("Ano")
    cut_df = cut_df[cut_cols].copy()
    if "Ts" in cut_df.columns:
        cut_df["Ts"] = pd.to_numeric(cut_df["Ts"], errors="coerce").astype("int64")
    for col in cut_df.columns:
        if col != "Ts":
            cut_df[col] = pd.to_numeric(cut_df[col], errors="coerce").astype(np.float32)
    return ret_df, cut_df


def extract_for_file(data_dir: Path, file_name: str, with_label: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_as_ofp_frame(data_dir / file_name)
    feature = extract_default_feature_frame(raw, Path(file_name).stem)
    return feature_data_get_df(feature, with_label=with_label)


def load_train_arrays(data_dir: Path, file_names: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    train_rows = 0
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
            train_rows += len(normal)
        if idx % 500 == 0:
            print(f"  [load] {idx}/{len(file_names)} files, rows={train_rows}")
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(DEFAULT_FEATURE_LIST)), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int8)
    return X, y, train_rows


def positive_probability(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(len(X), dtype=np.float32)


def write_fold_predictions(
    model: RandomForestClassifier,
    data_dir: Path,
    file_names: list[str],
    out_dir: Path,
    threshold: float,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for idx, name in enumerate(file_names, 1):
        normal, extra = extract_for_file(data_dir, name, with_label=True)
        frames = []
        if len(normal):
            scores = positive_probability(model, normal[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32))
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": normal["Ts"].to_numpy(dtype=np.int64),
                        "predict": (scores >= threshold).astype(int),
                        "file_name": name,
                        "proba": scores,
                    }
                )
            )
        if len(extra):
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                        "predict": (pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER).astype(int),
                        "file_name": name,
                    }
                )
            )
        if frames:
            pred = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        else:
            pred = pd.DataFrame(columns=["timestamp", "predict", "file_name", "proba"])
        pred.to_csv(out_dir / name, index=False)
        total_rows += len(pred)
        if idx % 500 == 0:
            print(f"  [predict] {idx}/{len(file_names)} files")
    return total_rows


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    n_estimators: int,
    threshold: float,
    max_train_files: int | None,
    max_test_files: int | None,
    overwrite: bool,
) -> FoldSummary:
    fold_dir = out_root / "model2_tree_strict" / f"fold_{fold}"
    summary_path = fold_dir / "fold_summary.json"
    if summary_path.exists() and not overwrite:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return FoldSummary(**data)

    t0 = time.time()
    train_files = index_df.loc[index_df["folder_index"] != fold, "file_name"].tolist()
    test_files = index_df.loc[index_df["folder_index"] == fold, "file_name"].tolist()
    if max_train_files is not None:
        train_files = train_files[: int(max_train_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]

    print(f"[fold {fold}] load train files={len(train_files)}")
    X_train, y_train, train_rows = load_train_arrays(data_dir, train_files)
    print(f"[fold {fold}] train rows={train_rows}, pos_rows={int(y_train.sum())}, neg_rows={int(len(y_train)-y_train.sum())}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=2024,
        verbose=3,
        n_jobs=1,
    )
    print(f"[fold {fold}] fitting RandomForestClassifier(n_estimators={n_estimators}, n_jobs=1)")
    model.fit(X_train, y_train)

    pred_dir = fold_dir / "predictions"
    eval_dir = fold_dir / "evaluation"
    print(f"[fold {fold}] predict files={len(test_files)}")
    test_rows = write_fold_predictions(model, data_dir, test_files, pred_dir, threshold)

    print(f"[fold {fold}] evaluate")
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)

    result = FoldSummary(
        model="model2_tree_strict",
        fold=fold,
        threshold=threshold,
        n_estimators=n_estimators,
        train_modules=len(train_files),
        test_modules=len(test_files),
        train_rows=int(train_rows),
        test_rows=int(test_rows),
        seconds=time.time() - t0,
        metrics=metrics_to_dict(summary_df),
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def aggregate(results: list[FoldSummary], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item.model,
            "fold": item.fold,
            "threshold": item.threshold,
            "n_estimators": item.n_estimators,
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
    fold_metrics = out_root / "fold_metrics.csv"
    if fold_metrics.exists():
        old = pd.read_csv(fold_metrics)
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold", "n_estimators"], keep="last", inplace=True)
        df.sort_values(["model", "n_estimators", "fold"], inplace=True)
    df.to_csv(fold_metrics, index=False)
    numeric = [c for c in df.columns if c not in {"model", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_protocol/model2_strict_tree"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Keep BLAS/OpenMP conservative. RF itself is also n_jobs=1.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args = parse_args()
    index_df = read_index(args.index_path)
    results: list[FoldSummary] = []
    for fold in args.folds:
        print(f"[run] strict OFP/model2 TreeModel fold={fold}")
        result = run_fold(
            fold=fold,
            data_dir=args.data_dir,
            index_df=index_df,
            out_root=args.out_root,
            n_estimators=args.n_estimators,
            threshold=args.threshold,
            max_train_files=args.max_train_files,
            max_test_files=args.max_test_files,
            overwrite=args.overwrite,
        )
        results.append(result)
        aggregate([result], args.out_root)
        print(
            f"[done] fold={fold} "
            f"F1={result.metrics.get('f1_score', 0):.6f} "
            f"P={result.metrics.get('precision', 0):.6f} "
            f"R={result.metrics.get('recall', 0):.6f} "
            f"time={result.seconds:.1f}s"
        )
    aggregate(results, args.out_root)


if __name__ == "__main__":
    main()
