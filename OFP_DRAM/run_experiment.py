from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.common import base_utils as base

from OFP_DRAM.config import (
    INDEX_FILE,
    OFFICIAL_TEST_DIR,
    OFPDRAMFeatureConfig,
    OFPDRAMTrainConfig,
    TRAINING_DIR,
)
from OFP_DRAM.evaluation import binary_window_metrics, choose_threshold, module_event_metrics
from OFP_DRAM.features import build_feature_frame, build_module_frame, prepare_xy
from OFP_DRAM.models import available_models, predict_scores, train_model


def _limit_split(split_df: pd.DataFrame, limit: int | None, random_state: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(split_df) <= limit:
        return split_df.reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    remaining = int(limit)
    groups = list(split_df.groupby("Label", sort=True))
    for idx, (_, group) in enumerate(groups):
        if idx == len(groups) - 1:
            take = min(len(group), remaining)
        else:
            take = int(round(limit * len(group) / len(split_df)))
            take = max(1, min(len(group), take, remaining - (len(groups) - idx - 1)))
        pieces.append(group.sample(n=take, random_state=random_state + idx))
        remaining -= take
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def get_split_map(args) -> dict[str, pd.DataFrame]:
    split_map = base.split_modules_stratified(
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )
    return {
        "train": _limit_split(split_map["train"], args.max_train_modules, args.random_state),
        "val": _limit_split(split_map["val"], args.max_val_modules, args.random_state),
        "test": _limit_split(split_map["test"], args.max_test_modules, args.random_state),
    }


def save_frame_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)


def load_or_build_frames(args, feature_cfg: OFPDRAMFeatureConfig, split_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    cache_dir = Path(args.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split, split_df in split_map.items():
        cache_path = cache_dir / f"{split}_{feature_cfg.tag}.pkl"
        if cache_path.exists() and not args.force_rebuild:
            print(f"[cache] loading {split}: {cache_path}", flush=True)
            frames[split] = pd.read_pickle(cache_path)
        else:
            print(
                f"[features] building {split}: modules={len(split_df)} "
                f"step={feature_cfg.train_step_minutes if split == 'train' else feature_cfg.eval_step_minutes}m",
                flush=True,
            )
            frames[split] = build_feature_frame(
                split_df,
                feature_cfg,
                split_role=split,
                data_dir=TRAINING_DIR,
                progress_every=args.progress_every,
            )
            save_frame_cache(frames[split], cache_path)
            print(f"[cache] saved {split}: {cache_path}", flush=True)
        print(f"[{split}] {summarize_frame(frames[split])}", flush=True)
    return frames


def summarize_frame(frame: pd.DataFrame) -> dict[str, int | float]:
    if frame.empty:
        return {"windows": 0, "positive_windows": 0, "modules": 0, "event_modules": 0, "positive_ratio": 0.0}
    return {
        "windows": int(len(frame)),
        "positive_windows": int(frame["label"].sum()),
        "modules": int(frame["file_name"].nunique()),
        "event_modules": int(frame.groupby("file_name")["event_label"].max().sum()),
        "positive_ratio": float(frame["label"].mean()),
    }


def save_feature_importance(model_name: str, model, feature_cols: list[str], output_dir: Path) -> None:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return
    values = np.asarray(values, dtype=float)
    if len(values) != len(feature_cols):
        return
    total = float(values.sum())
    norm = values / total if total > 0 else values
    order = np.argsort(norm)[::-1]
    rows = [
        {"model": model_name, "rank": rank, "feature": feature_cols[idx], "importance": float(norm[idx])}
        for rank, idx in enumerate(order, start=1)
    ]
    pd.DataFrame(rows).to_csv(output_dir / "results" / f"{model_name}_feature_importance.csv", index=False)


def run_training(args) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    for sub in ["models", "results", "predictions", "cache"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    feature_cfg = OFPDRAMFeatureConfig(
        ahead_hours=args.ahead_hours,
        history_minutes=tuple(int(x) for x in args.history_minutes.split(",") if x.strip()),
        train_step_minutes=args.train_step_minutes,
        eval_step_minutes=args.eval_step_minutes,
        min_history_points=args.min_history_points,
        min_observed_ratio=args.min_observed_ratio,
        train_faulty_max_windows=args.train_faulty_max_windows,
        train_healthy_max_windows=args.train_healthy_max_windows,
    )
    train_cfg = OFPDRAMTrainConfig(
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        two_stage=not args.one_stage,
        positive_lead_alpha=args.positive_lead_alpha,
        hard_negative_alpha=args.hard_negative_alpha,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        threshold_grid_size=args.threshold_grid_size,
    )

    split_map = get_split_map(args)
    print("=" * 80, flush=True)
    print("OFP_DRAM experiment", flush=True)
    print("=" * 80, flush=True)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "models": args.models,
                "feature_config": feature_cfg.__dict__,
                "train_config": train_cfg.__dict__,
                "split_modules": {k: int(len(v)) for k, v in split_map.items()},
                "available_models": available_models(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    frames = load_or_build_frames(args, feature_cfg, split_map)
    X_train, y_train, meta_train, feature_cols = prepare_xy(frames["train"])
    X_val, y_val, meta_val, _ = prepare_xy(frames["val"])
    X_test, y_test, meta_test, _ = prepare_xy(frames["test"])

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    availability = available_models()
    summary: dict[str, object] = {
        "feature_config": feature_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "available_models": availability,
        "data": {name: summarize_frame(frame) for name, frame in frames.items()},
        "feature_count": len(feature_cols),
        "models": {},
    }

    for model_name in requested_models:
        print(f"[model] {model_name}", flush=True)
        if model_name not in availability:
            summary["models"][model_name] = {"status": "unknown_model"}
            print(f"[model] {model_name}: unknown_model", flush=True)
            continue
        if not availability[model_name]:
            summary["models"][model_name] = {"status": "missing_dependency"}
            print(f"[model] {model_name}: missing_dependency", flush=True)
            continue
        model, train_meta = train_model(model_name, X_train, y_train, meta_train, train_cfg)
        val_scores = predict_scores(model, X_val)
        threshold, threshold_metrics = choose_threshold(
            meta_val,
            y_val,
            val_scores,
            split_map["val"],
            grid_size=train_cfg.threshold_grid_size,
        )
        test_scores = predict_scores(model, X_test)
        model_summary = {
            "status": "ok",
            "training": train_meta,
            "threshold": threshold,
            "threshold_metrics_val": threshold_metrics,
            "window_metrics_test": binary_window_metrics(y_test, test_scores, threshold),
            "module_metrics_test": module_event_metrics(meta_test, test_scores, threshold, split_map["test"]),
        }
        summary["models"][model_name] = model_summary
        joblib.dump(
            {"model": model, "feature_cols": feature_cols, "threshold": threshold, "feature_config": feature_cfg},
            output_dir / "models" / f"{model_name}.joblib",
        )
        save_feature_importance(model_name, model, feature_cols, output_dir)
        print(
            f"[model] {model_name}: final={model_summary['module_metrics_test']['final_score']:.4f} "
            f"F1={model_summary['module_metrics_test']['f1']:.4f} "
            f"P={model_summary['module_metrics_test']['precision']:.4f} "
            f"R={model_summary['module_metrics_test']['recall']:.4f}",
            flush=True,
        )

    with open(output_dir / "results" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    rows = []
    for model_name, model_summary in summary["models"].items():
        row = {"model": model_name, "status": model_summary.get("status")}
        if model_summary.get("status") == "ok":
            row.update({f"test_{k}": v for k, v in model_summary["module_metrics_test"].items()})
            row["threshold"] = model_summary["threshold"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "results" / "model_metrics.csv", index=False)
    return summary


def predict_official_test(args, model_name: str) -> None:
    payload_path = Path(args.output_dir) / "models" / f"{model_name}.joblib"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = joblib.load(payload_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    threshold = float(payload["threshold"])
    feature_cfg: OFPDRAMFeatureConfig = payload["feature_config"]
    out_dir = Path(args.output_dir) / "predictions" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(args.official_test_dir)
    for path in sorted(test_dir.glob("*.csv")):
        raw = pd.read_csv(path, low_memory=False)
        raw_for_output = raw[["timestamp"]].copy()
        frame = build_module_frame(
            raw,
            file_name=path.name,
            folder_index=None,
            module_label=0,
            cfg=feature_cfg,
            split_role="predict",
        )
        raw_for_output["predict"] = 0
        if not frame.empty:
            X, _, meta, _ = prepare_xy(frame)
            X = X.reindex(columns=feature_cols, fill_value=0.0)
            scores = predict_scores(model, X)
            pred_ts = meta.loc[scores >= threshold, "pred_ts"].astype(int).tolist()
            raw_for_output.loc[raw_for_output["timestamp"].astype(int).isin(pred_ts), "predict"] = 1
        raw_for_output.to_csv(out_dir / path.name, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRAM-paper-inspired OFP experiment")
    parser.add_argument("--output_dir", default="output/OFP_DRAM/default")
    parser.add_argument("--models", default="random_forest,xgboost,lightgbm,catboost")
    parser.add_argument("--ahead_hours", type=int, default=120)
    parser.add_argument("--history_minutes", default="15,60,360,1440")
    parser.add_argument("--train_step_minutes", type=int, default=60)
    parser.add_argument("--eval_step_minutes", type=int, default=60)
    parser.add_argument("--min_history_points", type=int, default=6)
    parser.add_argument("--min_observed_ratio", type=float, default=0.50)
    parser.add_argument("--train_faulty_max_windows", type=int, default=192)
    parser.add_argument("--train_healthy_max_windows", type=int, default=32)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_estimators", type=int, default=240)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.06)
    parser.add_argument("--positive_lead_alpha", type=float, default=2.0)
    parser.add_argument("--hard_negative_alpha", type=float, default=2.0)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--one_stage", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--max_train_modules", type=int, default=None)
    parser.add_argument("--max_val_modules", type=int, default=None)
    parser.add_argument("--max_test_modules", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument("--predict_official_test", action="store_true")
    parser.add_argument("--official_model", default="lightgbm")
    parser.add_argument("--official_test_dir", default=str(OFFICIAL_TEST_DIR))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_train_modules = args.max_train_modules or 12
        args.max_val_modules = args.max_val_modules or 8
        args.max_test_modules = args.max_test_modules or 8
        args.n_estimators = min(args.n_estimators, 16)
        args.eval_step_minutes = max(args.eval_step_minutes, 60)
        args.history_minutes = "15,60"
        args.output_dir = args.output_dir if args.output_dir != "output/OFP_DRAM/default" else "output/OFP_DRAM/smoke"
    return args


def main() -> None:
    args = parse_args()
    summary = run_training(args)
    print(json.dumps({"output_dir": args.output_dir, "models": summary["models"]}, indent=2, ensure_ascii=False))
    if args.predict_official_test:
        predict_official_test(args, args.official_model)
        print(f"official predictions written to {Path(args.output_dir) / 'predictions' / args.official_model}")


if __name__ == "__main__":
    main()
