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
    metrics_to_dict,
    read_index,
    select_threshold,
    split_train_val_files,
    write_prediction_frames,
)


def read_sensor_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda col: col in {"timestamp", *SENSORS})
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    for col in SENSORS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp")


def resample_frame(frame: pd.DataFrame, resample_seconds: int) -> pd.DataFrame:
    if int(resample_seconds) <= 0:
        return frame[["timestamp", *SENSORS]].copy()
    bucket = (pd.to_numeric(frame["timestamp"], errors="coerce") // int(resample_seconds)) * int(resample_seconds)
    work = frame.copy()
    work["timestamp"] = bucket.astype(np.int64)
    out = work.groupby("timestamp", as_index=False)[SENSORS].mean(numeric_only=True)
    return out.sort_values("timestamp")


def compute_normal_stats(data_dir: Path, train_files: list[str], label_by_file: dict[str, int], resample_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    def accumulate(require_normal_module: bool) -> tuple[np.ndarray, np.ndarray, int]:
        sums = np.zeros(len(SENSORS), dtype=np.float64)
        sums_sq = np.zeros(len(SENSORS), dtype=np.float64)
        count = 0
        for name in train_files:
            if require_normal_module and int(label_by_file.get(name, 0)) != 0:
                continue
            frame = resample_frame(read_sensor_frame(data_dir / name), resample_seconds)
            arr = frame[SENSORS].to_numpy(dtype=np.float64)
            finite = np.isfinite(arr).all(axis=1)
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
        raise ValueError("No normal/non-empty rows found for TimesFM normal statistics")
    mean = sums / count
    var = np.maximum(sums_sq / count - mean**2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def load_timesfm_model(args: argparse.Namespace):
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=int(args.context),
            max_horizon=int(args.horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=int(args.per_core_batch_size),
        )
    )
    return model


def _positive_max(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0.0
    return float(max(np.nanmax(arr), 0.0))


def _boundary_excess(values: np.ndarray, mean: np.ndarray, std: np.ndarray, sigma: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    scale = np.maximum(std.astype(np.float32), 1e-6)
    z = np.abs((arr - mean.reshape(1, -1)) / scale.reshape(1, -1))
    return _positive_max(z - float(sigma))


def _current_excess(history: np.ndarray, mean: np.ndarray, std: np.ndarray, sigma: float) -> float:
    return _boundary_excess(history[-1:].astype(np.float32), mean, std, sigma)


def _recent_excess(history: np.ndarray, mean: np.ndarray, std: np.ndarray, sigma: float, recent_tail: int) -> float:
    tail = max(1, min(int(recent_tail), len(history)))
    return _boundary_excess(history[-tail:].astype(np.float32), mean, std, sigma)


def _drift_excess(future: np.ndarray, history: np.ndarray, std: np.ndarray, sigma: float) -> float:
    if future.size == 0 or history.size == 0:
        return 0.0
    scale = np.maximum(std.astype(np.float32), 1e-6)
    anchor = np.nanmedian(history[-min(len(history), 6) :], axis=0).astype(np.float32)
    drift = np.abs((future.astype(np.float32) - anchor.reshape(1, -1)) / scale.reshape(1, -1))
    return _positive_max(drift - float(sigma))


def mock_forecast_future(history: np.ndarray, horizon: int) -> np.ndarray:
    last = history[-1]
    prev = history[max(0, len(history) - 4)]
    slope = last - prev
    return last.reshape(1, -1) + np.arange(1, int(horizon) + 1, dtype=np.float32).reshape(-1, 1) * slope.reshape(1, -1)


def timesfm_forecast_future(model, history: np.ndarray, horizon: int) -> np.ndarray:
    inputs = [history[:, col].astype(np.float32) for col in range(history.shape[1])]
    point, _quantiles = model.forecast(horizon=int(horizon), inputs=inputs)
    point = np.asarray(point, dtype=np.float32)
    return point.T


def timesfm_forecast_futures(model, histories: list[np.ndarray], horizon: int) -> list[np.ndarray]:
    if not histories:
        return []
    n_sensors = int(histories[0].shape[1])
    inputs = []
    for hist in histories:
        if int(hist.shape[1]) != n_sensors:
            raise ValueError("All TimesFM histories in one batch must have the same sensor count")
        inputs.extend(hist[:, col].astype(np.float32) for col in range(n_sensors))
    point, _quantiles = model.forecast(horizon=int(horizon), inputs=inputs)
    point = np.asarray(point, dtype=np.float32)
    expected = len(histories) * n_sensors
    if point.shape[0] != expected:
        raise ValueError(f"TimesFM returned {point.shape[0]} series, expected {expected}")
    future = point.reshape(len(histories), n_sensors, int(horizon)).transpose(0, 2, 1)
    return [future[idx].astype(np.float32) for idx in range(len(histories))]


def combine_risk_components(components: dict[str, float], args: argparse.Namespace) -> float:
    mode = str(args.risk_mode).lower()
    if mode == "forecast":
        return float(components["forecast"])
    if mode == "current":
        return float(max(components["current"], components["recent"]))
    weighted = {
        "forecast": float(args.forecast_weight) * float(components["forecast"]),
        "current": float(args.current_weight) * float(components["current"]),
        "recent": float(args.recent_weight) * float(components["recent"]),
        "drift": float(args.drift_weight) * float(components["drift"]),
    }
    if mode == "hybrid_sum":
        return float(sum(weighted.values()))
    if mode == "hybrid_max":
        return float(max(weighted.values()))
    raise ValueError(f"Unknown risk_mode={args.risk_mode!r}")


def risk_components_from_future(
    history: np.ndarray,
    future: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    return {
        "forecast": _boundary_excess(future, mean, std, float(args.normal_sigma)),
        "current": _current_excess(history, mean, std, float(args.normal_sigma)),
        "recent": _recent_excess(history, mean, std, float(args.normal_sigma), int(args.recent_tail)),
        "drift": _drift_excess(future, history, std, float(args.drift_sigma)),
    }


def select_score_positions(n_rows: int, min_context: int, stride: int, max_windows: int, mode: str) -> list[int]:
    if n_rows < int(min_context):
        return []
    positions = list(range(int(min_context) - 1, int(n_rows), max(1, int(stride))))
    max_windows = int(max_windows)
    if max_windows <= 0 or len(positions) <= max_windows:
        return positions
    mode = str(mode).lower()
    if mode == "head":
        return positions[:max_windows]
    if mode == "tail":
        return positions[-max_windows:]
    if mode == "even":
        idx = np.linspace(0, len(positions) - 1, max_windows)
        return [positions[int(round(x))] for x in idx]
    raise ValueError(f"Unknown position_sample={mode!r}")


def score_file(
    data_dir: Path,
    file_name: str,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    frame = resample_frame(read_sensor_frame(data_dir / file_name), int(args.resample_seconds))
    arr = frame[SENSORS].to_numpy(dtype=np.float32)
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    finite = np.isfinite(arr).all(axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    context = int(args.context)
    min_context = max(1, min(int(args.min_context), context))
    horizon = int(args.horizon)
    stride = max(1, int(args.stride))
    if len(arr) < min_context:
        return pd.DataFrame({"timestamp": timestamps.astype(np.int64), "score": np.zeros(len(arr), dtype=np.float32)})
    positions = select_score_positions(
        len(arr),
        min_context=min_context,
        stride=stride,
        max_windows=int(args.max_windows_per_file),
        mode=str(args.position_sample),
    )
    rows: list[dict] = []
    pending_histories: list[np.ndarray] = []
    pending_row_indices: list[int] = []
    for pos in positions:
        start = max(0, int(pos) - context + 1)
        hist = arr[start : int(pos) + 1]
        if not finite[pos] or not np.isfinite(hist).all():
            score = 0.0
            components = {"forecast": 0.0, "current": 0.0, "recent": 0.0, "drift": 0.0}
        elif len(hist) < int(args.forecast_min_context):
            future = np.repeat(hist[-1:].astype(np.float32), horizon, axis=0)
            components = risk_components_from_future(hist, future, mean, std, args)
            components["forecast"] = 0.0
            components["drift"] = 0.0
            score = combine_risk_components(components, args)
        elif bool(args.mock_forecast):
            future = mock_forecast_future(hist, horizon)
            components = risk_components_from_future(hist, future, mean, std, args)
            score = combine_risk_components(components, args)
        else:
            components = risk_components_from_future(
                hist,
                np.repeat(hist[-1:].astype(np.float32), horizon, axis=0),
                mean,
                std,
                args,
            )
            components["forecast"] = 0.0
            components["drift"] = 0.0
            score = combine_risk_components(components, args)
            pending_histories.append(hist)
            pending_row_indices.append(len(rows))
        rows.append(
            {
                "timestamp": int(timestamps[pos]),
                "score": float(score),
                "score_forecast": float(components["forecast"]),
                "score_current": float(components["current"]),
                "score_recent": float(components["recent"]),
                "score_drift": float(components["drift"]),
            }
        )
    if pending_histories:
        futures = timesfm_forecast_futures(model, pending_histories, horizon)
        for row_idx, hist, future in zip(pending_row_indices, pending_histories, futures):
            components = risk_components_from_future(hist, future, mean, std, args)
            score = combine_risk_components(components, args)
            rows[row_idx].update(
                {
                    "score": float(score),
                    "score_forecast": float(components["forecast"]),
                    "score_current": float(components["current"]),
                    "score_recent": float(components["recent"]),
                    "score_drift": float(components["drift"]),
                }
            )
    return pd.DataFrame(rows)


def write_score_summary(score_frames: dict[str, pd.DataFrame], path: Path) -> None:
    rows = []
    for name, frame in sorted(score_frames.items()):
        row = {"file_name": name, "n_score": int(len(frame))}
        if frame.empty or "score" not in frame.columns:
            row.update({"score_mean": 0.0, "score_p95": 0.0, "score_max": 0.0, "score_nonzero_frac": 0.0})
        else:
            values = pd.to_numeric(frame["score"], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                row.update({"score_mean": 0.0, "score_p95": 0.0, "score_max": 0.0, "score_nonzero_frac": 0.0})
            else:
                row.update(
                    {
                        "score_mean": float(np.mean(values)),
                        "score_p95": float(np.percentile(values, 95)),
                        "score_max": float(np.max(values)),
                        "score_nonzero_frac": float(np.mean(values > 0.0)),
                    }
                )
        for col in ["score_forecast", "score_current", "score_recent", "score_drift"]:
            if col in frame.columns and not frame.empty:
                vals = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
                row[f"{col}_max"] = float(np.max(vals)) if vals.size else 0.0
            else:
                row[f"{col}_max"] = 0.0
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def score_collection_summary(score_frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    values = []
    n_rows = 0
    n_files_with_score = 0
    for frame in score_frames.values():
        if frame.empty or "score" not in frame.columns:
            continue
        scores = pd.to_numeric(frame["score"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if scores.size == 0:
            continue
        values.append(scores)
        n_rows += int(scores.size)
        if float(np.nanmax(scores)) > 0.0:
            n_files_with_score += 1
    if not values:
        return {
            "files": float(len(score_frames)),
            "score_rows": 0.0,
            "files_with_score": 0.0,
            "score_mean": 0.0,
            "score_p95": 0.0,
            "score_max": 0.0,
            "score_nonzero_frac": 0.0,
        }
    all_scores = np.concatenate(values)
    return {
        "files": float(len(score_frames)),
        "score_rows": float(n_rows),
        "files_with_score": float(n_files_with_score),
        "score_mean": float(np.mean(all_scores)),
        "score_p95": float(np.percentile(all_scores, 95)),
        "score_max": float(np.max(all_scores)),
        "score_nonzero_frac": float(np.mean(all_scores > 0.0)),
    }


def print_score_collection_summary(split: str, score_frames: dict[str, pd.DataFrame]) -> None:
    stats = score_collection_summary(score_frames)
    print(
        f"[timesfm-score-summary] split={split} files={int(stats['files'])} "
        f"rows={int(stats['score_rows'])} files_with_score={int(stats['files_with_score'])} "
        f"mean={stats['score_mean']:.6g} p95={stats['score_p95']:.6g} "
        f"max={stats['score_max']:.6g} nonzero_frac={stats['score_nonzero_frac']:.4f}"
    )


def print_prediction_summary(split: str, pred_frames: dict[str, pd.DataFrame], threshold: float) -> None:
    pred_modules = 0
    pred_rows = 0
    for frame in pred_frames.values():
        if frame.empty or "predict" not in frame.columns:
            continue
        pred = pd.to_numeric(frame["predict"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        pred_rows += int(np.sum(pred > 0.0))
        pred_modules += int(np.any(pred > 0.0))
    print(
        f"[timesfm-pred-summary] split={split} threshold={float(threshold):.6g} "
        f"pred_modules={pred_modules}/{len(pred_frames)} pred_rows={pred_rows}"
    )


def prepare_score_frames(
    data_dir: Path,
    file_names: list[str],
    model,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for idx, name in enumerate(file_names, 1):
        frames[name] = score_file(data_dir, name, model, mean, std, args)
        if idx % 100 == 0:
            print(f"[timesfm-score] {idx}/{len(file_names)} files")
    return frames


def run_fold(args: argparse.Namespace, index_df: pd.DataFrame, fold: int, cfg: AlarmCfg, model) -> dict:
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
    print(
        f"[timesfm-adapter] fold={fold} fit={len(fit_files)} val={len(val_files)} test={len(test_files)} "
        f"mock={bool(args.mock_forecast)} risk={args.risk_mode} context={args.context} "
        f"min_context={args.min_context} forecast_min_context={args.forecast_min_context} horizon={args.horizon}"
    )
    mean, std = compute_normal_stats(args.data_dir, fit_files, label_by_file, int(args.resample_seconds))
    val_scores = prepare_score_frames(args.data_dir, val_files, model, mean, std, args)
    print_score_collection_summary("val", val_scores)
    threshold, val_metrics, search_df = select_threshold(val_scores, args.data_dir, cfg)
    val_dir = run_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    write_score_summary(val_scores, val_dir / "score_summary.csv")
    search_df.to_csv(val_dir / "threshold_search.csv", index=False)
    pd.DataFrame({"Item": list(val_metrics.keys()), "Value": list(val_metrics.values())}).to_csv(
        val_dir / "best_evaluate_result.csv",
        index=False,
    )
    test_scores = prepare_score_frames(args.data_dir, test_files, model, mean, std, args)
    print_score_collection_summary("test", test_scores)
    score_dir = run_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in test_scores.items():
        frame.to_csv(score_dir / name, index=False)
    write_score_summary(test_scores, score_dir / "score_summary.csv")
    pred_frames = {name: apply_alarm_strategy(frame, threshold, cfg) for name, frame in test_scores.items()}
    print_prediction_summary("test", pred_frames, threshold)
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    test_rows = write_prediction_frames(pred_frames, pred_dir)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, args.data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)
    protocol = "timesfm_mock_forecast_risk_adapter" if bool(args.mock_forecast) else "timesfm_forecast_risk_adapter"
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
        "timesfm_cfg": {
            "model_id": args.model_id,
            "context": int(args.context),
            "min_context": int(args.min_context),
            "forecast_min_context": int(args.forecast_min_context),
            "horizon": int(args.horizon),
            "resample_seconds": int(args.resample_seconds),
            "stride": int(args.stride),
            "max_windows_per_file": int(args.max_windows_per_file),
            "position_sample": str(args.position_sample),
            "mock_forecast": bool(args.mock_forecast),
            "risk_mode": str(args.risk_mode),
            "normal_sigma": float(args.normal_sigma),
            "drift_sigma": float(args.drift_sigma),
            "recent_tail": int(args.recent_tail),
            "forecast_weight": float(args.forecast_weight),
            "current_weight": float(args.current_weight),
            "recent_weight": float(args.recent_weight),
            "drift_weight": float(args.drift_weight),
        },
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[timesfm-adapter-done] fold={fold} threshold={threshold:.4f} "
        f"f1={metrics.get('f1_score', 0.0):.5f} final={metrics.get('final_score', 0.0):.5f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TimesFM forecast-risk adapter for OFP first-warning evaluation.")
    parser.add_argument("--model_name", default="timesfm")
    parser.add_argument("--model_id", default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--cache_dir", default=str(Path("model/timesfm_hf_cache")))
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_external_adapter_results/timesfm"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--min_context", type=int, default=4)
    parser.add_argument("--forecast_min_context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--resample_seconds", type=int, default=3600)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_windows_per_file", type=int, default=256)
    parser.add_argument("--position_sample", choices=["even", "head", "tail"], default="even")
    parser.add_argument("--per_core_batch_size", type=int, default=32)
    parser.add_argument("--mock_forecast", action="store_true")
    parser.add_argument("--risk_mode", choices=["forecast", "current", "hybrid_max", "hybrid_sum"], default="hybrid_max")
    parser.add_argument("--normal_sigma", type=float, default=3.0)
    parser.add_argument("--drift_sigma", type=float, default=2.0)
    parser.add_argument("--recent_tail", type=int, default=24)
    parser.add_argument("--forecast_weight", type=float, default=1.0)
    parser.add_argument("--current_weight", type=float, default=1.0)
    parser.add_argument("--recent_weight", type=float, default=0.7)
    parser.add_argument("--drift_weight", type=float, default=0.3)
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
    model = None if bool(args.mock_forecast) else load_timesfm_model(args)
    results = []
    for fold in args.folds:
        result = run_fold(args, index_df, int(fold), cfg, model)
        results.append(result)
        aggregate_results([result], args.out_root)
    aggregate_results(results, args.out_root)


if __name__ == "__main__":
    main()
