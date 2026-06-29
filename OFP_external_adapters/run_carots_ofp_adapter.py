from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_external_adapters.common import (
    SENSORS,
    AlarmCfg,
    aggregate_results,
    apply_alarm_strategy,
    evaluate_prediction_folder,
    file_label_map,
    files_for_index_fold,
    load_score_frame,
    metrics_to_dict,
    read_index,
    select_threshold,
    split_train_val_files,
    write_prediction_frames,
)


def read_sensor_frame(path: Path) -> pd.DataFrame:
    cols = ["timestamp", *SENSORS, "anomaly"]
    frame = pd.read_csv(path, usecols=lambda col: col in cols)
    for col in ["timestamp", *SENSORS, "anomaly"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def compute_normal_stats(data_dir: Path, train_files: list[str], label_by_file: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    def accumulate(require_normal_module: bool) -> tuple[np.ndarray, np.ndarray, int]:
        sums = np.zeros(len(SENSORS), dtype=np.float64)
        sums_sq = np.zeros(len(SENSORS), dtype=np.float64)
        count = 0
        for name in train_files:
            if require_normal_module and int(label_by_file.get(name, 0)) != 0:
                continue
            frame = read_sensor_frame(data_dir / name)
            arr = frame[SENSORS].to_numpy(dtype=np.float64)
            finite = np.isfinite(arr).all(axis=1)
            if "anomaly" in frame.columns:
                non_anomaly = pd.to_numeric(frame["anomaly"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= 0
                finite = finite & non_anomaly
            arr = arr[finite]
            if len(arr) == 0:
                continue
            sums += arr.sum(axis=0)
            sums_sq += (arr**2).sum(axis=0)
            count += int(arr.shape[0])
        return sums, sums_sq, count

    sums, sums_sq, count = accumulate(require_normal_module=True)
    if count <= 0:
        sums, sums_sq, count = accumulate(require_normal_module=False)
    if count <= 0:
        raise ValueError("No normal/non-anomaly rows found for CAROTS normal_residual scoring")
    mean = sums / count
    var = np.maximum(sums_sq / count - mean**2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def score_file_normal_residual(data_dir: Path, file_name: str, mean: np.ndarray, std: np.ndarray, win_size: int) -> pd.DataFrame:
    frame = read_sensor_frame(data_dir / file_name)
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    arr = frame[SENSORS].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.abs((arr - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-6))
    point_score = np.nanmax(z, axis=1)
    if int(win_size) > 1 and len(point_score):
        score = pd.Series(point_score).rolling(int(win_size), min_periods=1).mean().to_numpy(dtype=np.float64)
    else:
        score = point_score.astype(np.float64)
    return pd.DataFrame({"timestamp": timestamps.astype(np.int64), "score": score})


def prepare_score_frames(
    score_dir: Path | None,
    data_dir: Path,
    file_names: list[str],
    score_col: str,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    win_size: int,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for idx, name in enumerate(file_names, 1):
        if score_dir is not None:
            frames[name] = load_score_frame(score_dir, name, score_col=score_col)
        else:
            if mean is None or std is None:
                raise ValueError("normal_residual scoring requires mean/std")
            frames[name] = score_file_normal_residual(data_dir, name, mean, std, win_size)
        if idx % 500 == 0:
            print(f"[carots-adapter-score] {idx}/{len(file_names)} files")
    return frames


def run_fold(args: argparse.Namespace, index_df: pd.DataFrame, fold: int, cfg: AlarmCfg) -> dict:
    started = time.time()
    label_by_file = file_label_map(index_df)
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    if int(args.max_train_files) > 0:
        train_files = train_files[: int(args.max_train_files)]
    if int(args.max_test_files) > 0:
        test_files = test_files[: int(args.max_test_files)]
    fit_files, val_files = split_train_val_files(train_files, label_by_file, cfg, int(fold))
    run_dir = Path(args.out_root) / args.model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    score_dir = Path(args.score_dir) if args.score_dir else None
    if score_dir is not None:
        protocol = "carots_external_score_adapter"
        mean = std = None
    elif args.score_mode == "normal_residual":
        protocol = "carots_normal_residual_adapter"
        mean, std = compute_normal_stats(args.data_dir, fit_files, label_by_file)
    else:
        raise ValueError("Use --score_dir for real CAROTS scores or --score_mode normal_residual for smoke scoring")

    print(
        f"[carots-adapter] fold={fold} protocol={protocol} "
        f"fit={len(fit_files)} val={len(val_files)} test={len(test_files)}"
    )
    val_scores = prepare_score_frames(score_dir, args.data_dir, val_files, args.score_col, mean, std, args.win_size)
    threshold, val_metrics, search_df = select_threshold(val_scores, args.data_dir, cfg)
    val_dir = run_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(val_dir / "threshold_search.csv", index=False)
    pd.DataFrame({"Item": list(val_metrics.keys()), "Value": list(val_metrics.values())}).to_csv(
        val_dir / "best_evaluate_result.csv",
        index=False,
    )

    test_scores = prepare_score_frames(score_dir, args.data_dir, test_files, args.score_col, mean, std, args.win_size)
    pred_frames = {name: apply_alarm_strategy(frame, threshold, cfg) for name, frame in test_scores.items()}
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    test_rows = write_prediction_frames(pred_frames, pred_dir)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, args.data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)
    result = {
        "model": args.model_name,
        "fold": int(fold),
        "protocol": protocol,
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_rows": int(test_rows),
        "seconds": time.time() - started,
        "metrics": metrics,
        "alarm_cfg": asdict(cfg),
        "score_dir": str(score_dir) if score_dir is not None else "",
        "score_mode": args.score_mode,
        "win_size": int(args.win_size),
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[carots-adapter-done] fold={fold} threshold={threshold:.4f} "
        f"f1={metrics.get('f1_score', 0.0):.5f} final={metrics.get('final_score', 0.0):.5f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAROTS-to-OFP score adapter.")
    parser.add_argument("--score_dir", type=Path, default=None, help="Directory of per-file CAROTS score CSVs.")
    parser.add_argument("--score_col", default="score")
    parser.add_argument("--score_mode", choices=["external", "normal_residual"], default="external")
    parser.add_argument("--model_name", default="carots")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_external_adapter_results/carots"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--win_size", type=int, default=10)
    parser.add_argument("--threshold_grid", default=AlarmCfg.threshold_grid)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--smoothing", choices=["none", "ema", "mean", "rolling"], default="ema")
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--consecutive_k", type=int, default=1)
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_test_files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_dir is None and args.score_mode == "external":
        raise SystemExit(
            "Real CAROTS mode needs --score_dir with per-file CSVs containing timestamp,score. "
            "For adapter smoke only, pass --score_mode normal_residual --model_name carots_normal_residual_adapter."
        )
    index_df = read_index(args.index_path)
    cfg = AlarmCfg(
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        val_fraction=args.val_fraction,
        smoothing=args.smoothing,
        smooth_window=args.smooth_window,
        consecutive_k=args.consecutive_k,
        first_alarm_only=not args.no_first_alarm_only,
        seed=args.seed,
    )
    results = []
    for fold in args.folds:
        result = run_fold(args, index_df, int(fold), cfg)
        results.append(result)
        aggregate_results([result], args.out_root)
    aggregate_results(results, args.out_root)


if __name__ == "__main__":
    main()
