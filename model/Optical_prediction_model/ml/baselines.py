from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from model.Optical_prediction_model.common import base_utils as base
from model.Optical_prediction_model.common import cache_utils
from model.Optical_prediction_model.common import task_utils as week1
from model.Optical_prediction_model.ml.models.random_forest import train_rf
from model.Optical_prediction_model.ml.models.xgboost_baseline import train_xgb


@dataclass
class ExperimentConfig:
    task: week1.Week1TaskConfig
    output_dir: Path
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    random_state: int = 42
    rf_n_estimators: int = 300
    rf_max_depth: int = 14
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    xgb_n_estimators: int = 320
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.08
    xgb_subsample: float = 0.85
    xgb_colsample: float = 0.85
    n_jobs: int = 1
    max_train_modules: Optional[int] = None
    max_val_modules: Optional[int] = None
    max_test_modules: Optional[int] = None


META_COLS = {
    "file_name",
    "module_label",
    "window_index",
    "start_idx",
    "obs_start_idx",
    "obs_end_idx",
    "lead_start_idx",
    "lead_end_idx",
    "pred_start_idx",
    "pred_end_idx",
    "split_role",
    "scope_mode",
    "window_mode",
    "label_mode",
    "obs_start_ts",
    "obs_end_ts",
    "lead_start_ts",
    "lead_end_ts",
    "pred_start_ts",
    "pred_end_ts",
    "first_failure_ts",
    "event_label",
    "label",
    "future_has_anomaly",
    "future_anomaly_ratio",
    "first_in_future",
    "first_in_prediction",
    "obs_has_anomaly",
    "lead_has_anomaly",
    "pred_has_anomaly",
    "obs_anomaly_ratio",
    "lead_anomaly_ratio",
    "pred_anomaly_ratio",
    "time_to_first_minutes",
    "obs_coverage_ratio",
    "lead_coverage_ratio",
    "pred_coverage_ratio",
    "tps_target_minutes",
    "tps_original_label",
    "tps_is_synthetic",
}


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
    limited = pd.concat(pieces, ignore_index=True)
    return limited.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def get_split_map(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    split_map = base.split_modules_stratified(
        test_ratio=cfg.test_ratio,
        val_ratio=cfg.val_ratio,
        random_state=cfg.random_state,
    )
    for split_name, split_df in list(split_map.items()):
        limit = {
            "train": cfg.max_train_modules,
            "val": cfg.max_val_modules,
            "test": cfg.max_test_modules,
        }[split_name]
        split_map[split_name] = _limit_split(split_df, limit, cfg.random_state)
    return split_map


def load_frames(cfg: ExperimentConfig, force_rebuild: bool = False) -> dict[str, pd.DataFrame]:
    split_map = get_split_map(cfg)
    frames: dict[str, pd.DataFrame] = {}
    cache_utils.FEATURE_FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_utils.WINDOW_INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in split_map.items():
        cache_path = cache_utils.feature_frame_cache_path(split_name, split_df, cfg.task.cache_tag)
        window_index_cache_path = cache_utils.window_index_cache_path(
            split_name,
            split_df,
            cfg.task.window_index_cache_tag,
        )
        frames[split_name] = week1.build_feature_frame(
            split_df,
            cfg.task,
            split_role=split_name,
            cache_path=cache_path,
            window_index_cache_path=window_index_cache_path,
            force_rebuild=force_rebuild,
        )
    return frames


# Feature suffixes excluded from training. *_missing_ratio and *_valid_points
# encode dataset-level cohort selection bias (long vs short file cohorts) rather
# than physical pre-failure signals; including them inflates F1 by ~50% via
# leakage. See analysis/diagnose_missing_ratio_leakage.py and
# analysis/dataset_format_audit.py for evidence.
EXCLUDED_FEATURE_SUFFIXES: tuple[str, ...] = ("_missing_ratio", "_valid_points")


def prepare_xy(
    df: pd.DataFrame,
    excluded_suffixes: tuple[str, ...] = EXCLUDED_FEATURE_SUFFIXES,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
        and not any(c.endswith(s) for s in excluded_suffixes)
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["label"].to_numpy(dtype=int)
    meta_cols = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols].copy()
    return X, y, meta, feature_cols


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = base.evaluate_binary_scores(y_true, scores, threshold)
        candidate = {
            "threshold": float(threshold),
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        }
        if (
            candidate["f1"] > best["f1"]
            or (candidate["f1"] == best["f1"] and candidate["recall"] > best["recall"])
            or (
                candidate["f1"] == best["f1"]
                and candidate["recall"] == best["recall"]
                and candidate["precision"] > best["precision"]
            )
        ):
            best = candidate
    return float(best["threshold"]), best


def evaluate_model(model, X_val, y_val, X_test, y_test, test_meta, test_split_df=None) -> dict[str, object]:
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold, threshold_meta = choose_threshold(y_val, val_scores)
    test_scores = model.predict_proba(X_test)[:, 1]
    window_metrics = base.evaluate_binary_scores(y_test, test_scores, threshold)
    event_metrics = week1.evaluate_event_level(test_meta, y_test, window_metrics["y_pred"])
    if test_split_df is not None:
        event_metrics.update(
            week1.evaluate_event_level_all_modules(
                test_meta,
                y_test,
                window_metrics["y_pred"],
                test_split_df,
            )
        )
    return {
        "threshold_selection": threshold_meta,
        "window_metrics": {k: v for k, v in window_metrics.items() if k not in {"y_pred", "y_score"}},
        "event_metrics": event_metrics,
    }


def _feature_parts(feature_name: str) -> tuple[str, str]:
    for sensor in sorted(base.SENSORS, key=len, reverse=True):
        prefix = f"{sensor}_"
        if feature_name.startswith(prefix):
            return sensor, feature_name[len(prefix) :]
    return "other", feature_name


def save_feature_importances(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    feature_cols: list[str],
    output_dir: Path,
    top_k: int = 25,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for model_name, model in [("random_forest", rf_model), ("xgboost", xgb_model)]:
        values = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_cols))), dtype=float)
        total = float(values.sum())
        normalized = values / total if total > 0 else values
        order = np.argsort(normalized)[::-1]
        for rank, idx in enumerate(order, start=1):
            sensor, statistic = _feature_parts(feature_cols[idx])
            rows.append(
                {
                    "model": model_name,
                    "rank": int(rank),
                    "feature": feature_cols[idx],
                    "sensor": sensor,
                    "statistic": statistic,
                    "importance": float(normalized[idx]),
                }
            )

    imp_df = pd.DataFrame(rows)
    result_dir = output_dir / "results"
    fig_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(result_dir / "feature_importance.csv", index=False)

    group_df = (
        imp_df.groupby(["model", "sensor"], as_index=False)["importance"]
        .sum()
        .sort_values(["model", "importance"], ascending=[True, False])
    )
    group_df.to_csv(result_dir / "feature_group_importance.csv", index=False)

    palette = {"random_forest": "#4C78A8", "xgboost": "#F58518"}
    for model_name in ["random_forest", "xgboost"]:
        sub = imp_df[imp_df["model"] == model_name].head(top_k).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.32 * len(sub))))
        ax.barh(sub["feature"], sub["importance"], color=palette[model_name])
        ax.set_xlabel("Normalized importance")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        fig.savefig(fig_dir / f"{model_name}_feature_importance.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    top_features = (
        imp_df[imp_df["rank"] <= top_k]
        .sort_values(["model", "rank"])
        .to_dict("records")
    )
    top_groups = group_df.groupby("model").head(8).to_dict("records")
    return {"top_features": top_features, "top_groups": top_groups}


def plot_metrics(summary: dict[str, object], output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = ["random_forest", "xgboost"]
    metric_names = ["precision", "recall", "f1"]
    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    palette = {"random_forest": "#4C78A8", "xgboost": "#F58518"}
    labels = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
    for idx, model_name in enumerate(models):
        vals = [summary[model_name]["window_metrics"][m] for m in metric_names]
        bars = ax.bar(x + (idx - 0.5) * width, vals, width, label=labels[model_name], color=palette[model_name])
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(fig_dir / "ml_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    task_cfg: week1.Week1TaskConfig,
    output_dir: Path,
    force_rebuild: bool = False,
    use_tps: bool = False,
    tps_mode: str = "targets",
    tps_target_minutes: list[int] | None = None,
    tps_lead_minutes: int = 60,
    tps_step_minutes: int = 15,
    tps_tolerance_minutes: int = 5,
    n_jobs: int = 1,
    rf_n_estimators: int = 300,
    xgb_n_estimators: int = 320,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.08,
    xgb_subsample: float = 0.85,
    xgb_colsample: float = 0.85,
    max_train_modules: int | None = None,
    max_val_modules: int | None = None,
    max_test_modules: int | None = None,
) -> dict[str, object]:
    cfg = ExperimentConfig(
        task=task_cfg,
        output_dir=output_dir,
        n_jobs=n_jobs,
        rf_n_estimators=rf_n_estimators,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_subsample=xgb_subsample,
        xgb_colsample=xgb_colsample,
        max_train_modules=max_train_modules,
        max_val_modules=max_val_modules,
        max_test_modules=max_test_modules,
    )
    for path in [cfg.output_dir / "models", cfg.output_dir / "results", cfg.output_dir / "figures", cfg.output_dir / "cache"]:
        path.mkdir(parents=True, exist_ok=True)

    if use_tps and tps_mode == "progressive":
        tps_target_minutes = week1.progressive_tps_targets(tps_lead_minutes, tps_step_minutes)
    elif use_tps:
        tps_target_minutes = tps_target_minutes or []
    else:
        tps_target_minutes = []

    split_map = get_split_map(cfg)
    frames = load_frames(cfg, force_rebuild=force_rebuild)
    train_frame = (
        week1.apply_leadtime_tps_frame(frames["train"], tps_target_minutes, tolerance_minutes=tps_tolerance_minutes)
        if use_tps
        else frames["train"]
    )

    X_train, y_train, _, feature_cols = prepare_xy(train_frame)
    X_val, y_val, _, _ = prepare_xy(frames["val"])
    X_test, y_test, test_meta, _ = prepare_xy(frames["test"])

    rf_model, rf_time = train_rf(X_train, y_train, cfg)
    xgb_model, xgb_time = train_xgb(X_train, y_train, cfg)
    rf_eval = evaluate_model(rf_model, X_val, y_val, X_test, y_test, test_meta, split_map["test"])
    xgb_eval = evaluate_model(xgb_model, X_val, y_val, X_test, y_test, test_meta, split_map["test"])
    feature_importance = save_feature_importances(rf_model, xgb_model, feature_cols, cfg.output_dir)

    with open(cfg.output_dir / "models" / "random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(cfg.output_dir / "models" / "xgboost.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    summary = {
        "task": {
            "definition": "Week1-style observation/lead/prediction window baseline with train/eval label decoupling",
            "tag": task_cfg.tag,
            "obs_minutes": task_cfg.obs_minutes,
            "lead_minutes": task_cfg.lead_minutes,
            "pred_minutes": task_cfg.pred_minutes,
            "step_minutes": task_cfg.step_minutes,
            "feature_profile": task_cfg.feature_profile,
            "train_scope": task_cfg.train_scope,
            "eval_scope": task_cfg.eval_scope,
            "train_window_mode": task_cfg.train_window_mode,
            "eval_window_mode": task_cfg.eval_window_mode,
            "train_label_mode": task_cfg.train_label_mode,
            "eval_label_mode": task_cfg.eval_label_mode,
            "train_faulty_max_windows": task_cfg.train_faulty_max_windows,
            "train_healthy_max_windows": task_cfg.train_healthy_max_windows,
            "train_faulty_build_stride_minutes": task_cfg.train_faulty_build_stride_minutes,
            "train_healthy_build_stride_minutes": task_cfg.train_healthy_build_stride_minutes,
            "eval_fast_sampling": task_cfg.eval_fast_sampling,
            "eval_recent_minutes": task_cfg.eval_recent_minutes,
            "eval_faulty_max_windows": task_cfg.eval_faulty_max_windows,
            "eval_faulty_far_windows": task_cfg.eval_faulty_far_windows,
            "eval_healthy_max_windows": task_cfg.eval_healthy_max_windows,
            "eval_healthy_build_stride_minutes": task_cfg.eval_healthy_build_stride_minutes,
            "use_tps": use_tps,
            "tps_mode": tps_mode if use_tps else None,
            "tps_target_minutes": tps_target_minutes if use_tps else [],
            "tps_lead_minutes": tps_lead_minutes if use_tps and tps_mode == "progressive" else None,
            "tps_step_minutes": tps_step_minutes if use_tps and tps_mode == "progressive" else None,
            "tps_tolerance_minutes": tps_tolerance_minutes if use_tps else None,
            "n_jobs": cfg.n_jobs,
            "max_train_modules": cfg.max_train_modules,
            "max_val_modules": cfg.max_val_modules,
            "max_test_modules": cfg.max_test_modules,
        },
        "data": {
            split: {
                "windows": int(len(df)),
                "positive_windows": int(df["label"].sum()) if not df.empty else 0,
                "positive_ratio": float(df["label"].mean()) if not df.empty else 0.0,
                "modules": int(df["file_name"].nunique()) if not df.empty else 0,
                "event_modules": int(df.groupby("file_name")["event_label"].max().sum()) if not df.empty else 0,
                "dirty_observation_windows": int(df["obs_has_anomaly"].sum()) if not df.empty else 0,
            }
            for split, df in frames.items()
        },
        "split_index": {
            split: {
                "modules": int(len(split_df)),
                "fault_modules": int(pd.to_numeric(split_df["Label"], errors="coerce").fillna(0).astype(int).sum()),
                "healthy_modules": int(
                    (pd.to_numeric(split_df["Label"], errors="coerce").fillna(0).astype(int) == 0).sum()
                ),
            }
            for split, split_df in split_map.items()
        },
        "train_after_tps": {
            "windows": int(len(train_frame)),
            "positive_windows": int(train_frame["label"].sum()) if not train_frame.empty else 0,
            "positive_ratio": float(train_frame["label"].mean()) if not train_frame.empty else 0.0,
            "tps_synthetic_windows": int(train_frame.get("tps_is_synthetic", pd.Series(dtype=int)).sum()) if not train_frame.empty else 0,
        },
        "random_forest": {"train_time_s": round(rf_time, 2), **rf_eval},
        "xgboost": {"train_time_s": round(xgb_time, 2), **xgb_eval},
        "feature_count": len(feature_cols),
        "feature_importance": feature_importance,
    }

    with open(cfg.output_dir / "results" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_metrics(summary, cfg.output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Week1 paper-style RF/XGB baselines")
    parser.add_argument("--obs_minutes", type=int, default=30)
    parser.add_argument("--lead_minutes", type=int, default=15)
    parser.add_argument("--pred_minutes", type=int, default=60)
    parser.add_argument("--step_minutes", type=int, default=15)
    parser.add_argument("--feature_profile", type=str, default="fast", choices=["fast", "full", "raw"])
    parser.add_argument("--output_dir", type=str, default="output/Optical_prediction_model/default_run")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_n_estimators", type=int, default=320)
    parser.add_argument("--max_train_modules", type=int, default=None)
    parser.add_argument("--max_val_modules", type=int, default=None)
    parser.add_argument("--max_test_modules", type=int, default=None)
    parser.add_argument("--train_faulty_max_windows", type=int, default=96)
    parser.add_argument("--train_healthy_max_windows", type=int, default=24)
    parser.add_argument("--train_faulty_build_stride_minutes", type=int, default=30)
    parser.add_argument("--train_healthy_build_stride_minutes", type=int, default=360)
    parser.add_argument("--eval_full_windows", action="store_true")
    parser.add_argument("--eval_recent_minutes", type=int, default=120)
    parser.add_argument("--eval_faulty_max_windows", type=int, default=80)
    parser.add_argument("--eval_faulty_far_windows", type=int, default=16)
    parser.add_argument("--eval_healthy_max_windows", type=int, default=24)
    parser.add_argument("--eval_healthy_build_stride_minutes", type=int, default=360)
    parser.add_argument("--train_scope", type=str, default="prefirst", choices=["full", "prefirst"])
    parser.add_argument("--eval_scope", type=str, default="prefirst", choices=["full", "prefirst"])
    parser.add_argument("--train_window_mode", type=str, default="rolling", choices=["rolling", "fault_anchored"])
    parser.add_argument("--eval_window_mode", type=str, default="rolling", choices=["rolling", "fault_anchored"])
    parser.add_argument(
        "--train_label_mode",
        type=str,
        default="first_in_future",
        choices=["future_any", "first_in_future", "first_in_prediction"],
    )
    parser.add_argument(
        "--eval_label_mode",
        type=str,
        default="first_in_prediction",
        choices=["future_any", "first_in_future", "first_in_prediction"],
    )
    parser.add_argument("--use_tps", action="store_true")
    parser.add_argument("--tps_mode", type=str, default="targets", choices=["targets", "progressive"])
    parser.add_argument("--tps_target_minutes", type=str, default="15,30")
    parser.add_argument("--tps_lead_minutes", type=int, default=60)
    parser.add_argument("--tps_step_minutes", type=int, default=15)
    parser.add_argument("--tps_tolerance_minutes", type=int, default=5)
    args = parser.parse_args()

    task_cfg = week1.Week1TaskConfig(
        obs_minutes=args.obs_minutes,
        lead_minutes=args.lead_minutes,
        pred_minutes=args.pred_minutes,
        step_minutes=args.step_minutes,
        feature_profile=args.feature_profile,
        train_scope=args.train_scope,
        eval_scope=args.eval_scope,
        train_window_mode=args.train_window_mode,
        eval_window_mode=args.eval_window_mode,
        train_label_mode=args.train_label_mode,
        eval_label_mode=args.eval_label_mode,
        train_faulty_max_windows=args.train_faulty_max_windows,
        train_healthy_max_windows=args.train_healthy_max_windows,
        train_faulty_build_stride_minutes=args.train_faulty_build_stride_minutes,
        train_healthy_build_stride_minutes=args.train_healthy_build_stride_minutes,
        eval_fast_sampling=not args.eval_full_windows,
        eval_recent_minutes=args.eval_recent_minutes,
        eval_faulty_max_windows=args.eval_faulty_max_windows,
        eval_faulty_far_windows=args.eval_faulty_far_windows,
        eval_healthy_max_windows=args.eval_healthy_max_windows,
        eval_healthy_build_stride_minutes=args.eval_healthy_build_stride_minutes,
    )
    exp_output_dir = Path(args.output_dir)
    tps_target_minutes = [int(x) for x in args.tps_target_minutes.split(",") if x.strip()] if args.use_tps else []
    summary = run_experiment(
        task_cfg,
        exp_output_dir,
        force_rebuild=args.force_rebuild,
        use_tps=args.use_tps,
        tps_mode=args.tps_mode,
        tps_target_minutes=tps_target_minutes,
        tps_lead_minutes=args.tps_lead_minutes,
        tps_step_minutes=args.tps_step_minutes,
        tps_tolerance_minutes=args.tps_tolerance_minutes,
        n_jobs=args.n_jobs,
        rf_n_estimators=args.rf_n_estimators,
        xgb_n_estimators=args.xgb_n_estimators,
        max_train_modules=args.max_train_modules,
        max_val_modules=args.max_val_modules,
        max_test_modules=args.max_test_modules,
    )
    frames = load_frames(
        ExperimentConfig(
            task=task_cfg,
            output_dir=Path(args.output_dir),
            max_train_modules=args.max_train_modules,
            max_val_modules=args.max_val_modules,
            max_test_modules=args.max_test_modules,
        )
    )
    for split_name in ["train", "val", "test"]:
        print(week1.describe_split(split_name, frames[split_name]))
    for model_name in ["random_forest", "xgboost"]:
        wm = summary[model_name]["window_metrics"]
        print(
            f"{model_name}: P={wm['precision']:.4f} R={wm['recall']:.4f} "
            f"F1={wm['f1']:.4f} Acc={wm['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
