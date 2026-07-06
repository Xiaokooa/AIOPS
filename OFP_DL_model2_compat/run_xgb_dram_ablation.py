from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_model2_compat.run_model2_compat_deep import (
    CompatBatchedDataset,
    CompatCfg,
    Model2FeatureCache,
    cap_files_stratified,
    compat_feature_names,
    compute_feature_norm_stats,
    file_label_map,
    format_duration,
    format_rate,
    log_run_header,
    progress_bar,
    split_train_val_files,
)
from OFP_DL_model2_compat.run_model2_compat_tabular_fusion import (
    build_ml_model,
    evaluate_prediction_output,
    fit_feature_selector,
    log_stage_done,
    log_stage_start,
    positive_scores,
    save_selected_features,
    select_threshold_for_scores,
    write_threshold_predictions,
)
from OFP_DL_model2_compat.run_model2_compat_tabular_only import score_model2_files_to_memory
from OFP_DL_model2_compat.lead_time_sweep import (
    DEFAULT_LEAD_TIME_GRID,
    append_lead_time_sweep_results,
    run_lead_time_sweep,
)
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.common.trainer import format_metric_summary


RUN_NAME = "xgb_dram_ablation"
DEFAULT_THRESHOLD_GRID = "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50"


@dataclass(frozen=True)
class Variant:
    name: str
    feature_mode: str
    sample_selection: str
    temporal_positive_weight: float
    adaptive_negative_weight: float
    description: str


VARIANTS: dict[str, Variant] = {
    "plain_model2_random": Variant(
        name="plain_model2_random",
        feature_mode="model2",
        sample_selection="random",
        temporal_positive_weight=0.0,
        adaptive_negative_weight=0.0,
        description="model2 features + random selected rows; no DRAM-inspired additions",
    ),
    "plus_features": Variant(
        name="plus_features",
        feature_mode="model2_plus",
        sample_selection="random",
        temporal_positive_weight=0.0,
        adaptive_negative_weight=0.0,
        description="adds model2_plus multi-scale/statistical/rule-like features",
    ),
    "plus_hybrid_sampling": Variant(
        name="plus_hybrid_sampling",
        feature_mode="model2_plus",
        sample_selection="hybrid",
        temporal_positive_weight=0.0,
        adaptive_negative_weight=0.0,
        description="adds hybrid high-signal/random sample selection",
    ),
    "plus_hybrid_temporal": Variant(
        name="plus_hybrid_temporal",
        feature_mode="model2_plus",
        sample_selection="hybrid",
        temporal_positive_weight=2.0,
        adaptive_negative_weight=0.0,
        description="adds temporal positive sample weighting",
    ),
    "full_dram_xgb": Variant(
        name="full_dram_xgb",
        feature_mode="model2_plus",
        sample_selection="hybrid",
        temporal_positive_weight=2.0,
        adaptive_negative_weight=1.0,
        description="adds adaptive hard-negative weighting after a first XGB pass",
    ),
}


def selected_variants(raw: list[str]) -> list[Variant]:
    lowered = [str(item).lower() for item in raw]
    if "all" in lowered:
        keys = [
            "plain_model2_random",
            "plus_features",
            "plus_hybrid_sampling",
            "plus_hybrid_temporal",
            "full_dram_xgb",
        ]
    else:
        keys = lowered
    out: list[Variant] = []
    for key in keys:
        if key not in VARIANTS:
            raise ValueError(f"Unknown variant={key!r}; choose from all, {', '.join(VARIANTS)}")
        out.append(VARIANTS[key])
    return out


def make_cfg(args: argparse.Namespace, variant: Variant) -> CompatCfg:
    return CompatCfg(
        seq_len=args.seq_len,
        epochs=1,
        batch_size=args.batch_size,
        lr=3e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        fixed_threshold=args.fixed_threshold,
        target_mode=args.target_mode,
        feature_mode=variant.feature_mode,
        sampling_mode=args.sampling_mode,
        positive_windows_per_module=args.positive_windows_per_module,
        negative_windows_per_faulty_module=args.negative_windows_per_faulty_module,
        normal_windows_per_module=args.normal_windows_per_module,
        val_fraction=args.val_fraction,
        threshold_search=bool(args.threshold_search),
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        rule_mode=args.rule_mode,
        sample_selection=variant.sample_selection,
        sample_topk_fraction=args.sample_topk_fraction,
        temporal_positive_weight=variant.temporal_positive_weight,
        temporal_weight_horizon_hours=args.temporal_weight_horizon_hours,
        adaptive_negative_weight=0.0,
        adaptive_warmup_epochs=1,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        device="cpu",
        num_workers=0,
        amp=False,
        allow_tf32=False,
        log_batches=0,
        max_train_files=args.max_train_files,
        max_test_files=args.max_test_files,
        min_hit_lead_hours=args.min_hit_lead_hours,
    )


def collect_training_table_with_weights(
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    variant_name: str,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_names = compat_feature_names(cfg)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    started = log_stage_start(
        "collect_xgb_train_table",
        variant_name,
        fold,
        files=len(dataset.file_names),
        features=len(feature_names),
    )
    rows = 0
    for idx, (file_name, positions, weights) in enumerate(
        zip(dataset.file_names, dataset.selected_positions, dataset.selected_weights),
        1,
    ):
        if len(positions) == 0:
            continue
        _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = cache.get(file_name)
        selected = np.asarray(positions, dtype=np.int64)
        x_parts.append(features[selected].astype(np.float32))
        y_parts.append(labels[selected].astype(np.int8))
        w_parts.append(np.asarray(weights, dtype=np.float32))
        rows += int(len(selected))
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={variant_name} fold={fold} stage=collect_xgb_train_table "
                f"{progress_bar(idx, len(dataset.file_names), width=18)} files={idx}/{len(dataset.file_names)} "
                f"rows={rows} rate={format_rate(idx, elapsed)} elapsed={format_duration(elapsed)}",
                flush=True,
            )
    if not x_parts:
        raise ValueError(f"No rows collected for variant={variant_name}")
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int8)
    w = np.concatenate(w_parts, axis=0).astype(np.float32)
    log_stage_done(
        "collect_xgb_train_table",
        started,
        variant_name,
        fold,
        rows=len(y),
        features=x.shape[1],
        positives=int(np.sum(y > 0)),
        negatives=int(np.sum(y <= 0)),
        weight_min=f"{float(np.min(w)):.4f}",
        weight_max=f"{float(np.max(w)):.4f}",
    )
    return x, y, w, feature_names


def adaptive_negative_weights_from_xgb(
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_weight: np.ndarray,
    max_extra: float,
) -> tuple[np.ndarray, dict[str, float]]:
    max_extra = float(max_extra)
    if max_extra <= 0.0:
        return base_weight.astype(np.float32), {"updated_negative_rows": 0.0, "min_score": 0.0, "max_score": 0.0}
    scores = positive_scores(estimator, x_train)
    neg_mask = y_train <= 0
    if not np.any(neg_mask):
        return base_weight.astype(np.float32), {"updated_negative_rows": 0.0, "min_score": 0.0, "max_score": 0.0}
    neg_scores = scores[neg_mask]
    lo = float(np.nanmin(neg_scores))
    hi = float(np.nanmax(neg_scores))
    denom = max(hi - lo, 1e-8)
    scaled = np.clip((scores - lo) / denom, 0.0, 1.0).astype(np.float32)
    out = np.asarray(base_weight, dtype=np.float32).copy()
    out[neg_mask] = 1.0 + max_extra * scaled[neg_mask]
    return out, {
        "updated_negative_rows": float(np.sum(neg_mask)),
        "min_score": lo,
        "max_score": hi,
        "mean_negative_extra_weight": float(np.mean(out[neg_mask] - 1.0)),
    }


def train_xgb_variant(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    variant: Variant,
    args: argparse.Namespace,
    fold: int,
) -> tuple[Any, dict[str, Any]]:
    meta: dict[str, Any] = {
        "sample_weight_min": float(np.min(sample_weight)) if len(sample_weight) else 0.0,
        "sample_weight_max": float(np.max(sample_weight)) if len(sample_weight) else 0.0,
        "adaptive_negative_weight": float(variant.adaptive_negative_weight),
    }
    if float(variant.adaptive_negative_weight) <= 0.0:
        estimator = build_ml_model("xgb", args, y_train, int(args.seed) + int(fold))
        estimator.fit(x_train, y_train, sample_weight=sample_weight)
        meta["stage"] = "single_stage_weighted_xgb"
        return estimator, meta

    stage1 = build_ml_model("xgb", args, y_train, int(args.seed) + int(fold))
    started = time.time()
    print(f"[xgb-stage1] variant={variant.name} fold={fold} rows={len(y_train)} features={x_train.shape[1]}", flush=True)
    stage1.fit(x_train, y_train, sample_weight=sample_weight)
    final_weight, adaptive_meta = adaptive_negative_weights_from_xgb(
        stage1,
        x_train,
        y_train,
        sample_weight,
        float(variant.adaptive_negative_weight),
    )
    print(
        f"[adaptive-xgb-neg] variant={variant.name} rows={int(adaptive_meta.get('updated_negative_rows', 0))} "
        f"score_min={adaptive_meta.get('min_score', 0.0):.5f} score_max={adaptive_meta.get('max_score', 0.0):.5f} "
        f"elapsed={format_duration(time.time() - started)}",
        flush=True,
    )
    estimator = build_ml_model("xgb", args, y_train, int(args.seed) + int(fold) + 7919)
    estimator.fit(x_train, y_train, sample_weight=final_weight)
    meta["stage"] = "two_stage_adaptive_negative_xgb"
    meta["adaptive_meta"] = adaptive_meta
    meta["final_weight_min"] = float(np.min(final_weight)) if len(final_weight) else 0.0
    meta["final_weight_max"] = float(np.max(final_weight)) if len(final_weight) else 0.0
    return estimator, meta


def run_variant(
    variant: Variant,
    fold: int,
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    label_by_file: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = make_cfg(args, variant)
    run_dir = Path(args.out_root) / variant.name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_run_header(
        f"XGB DRAM-INSPIRED ABLATION: {variant.name}",
        {
            "description": variant.description,
            "fold": fold,
            "train/val/test": f"{len(train_files)}/{len(val_files)}/{len(test_files)} modules",
            "target": cfg.target_mode,
            "feature mode": cfg.feature_mode,
            "sample selection": cfg.sample_selection,
            "temporal positive weight": cfg.temporal_positive_weight,
            "adaptive negative weight": variant.adaptive_negative_weight,
            "rule mode": cfg.rule_mode,
            "out_dir": run_dir,
        },
    )
    stats_started = log_stage_start("feature_norm_stats", variant.name, fold, files=len(train_files), data_dir=args.data_dir)
    stats = compute_feature_norm_stats(args.data_dir, train_files, cfg, label_by_file)
    log_stage_done("feature_norm_stats", stats_started, variant.name, fold)
    mean, std = stats.arrays()
    cache = Model2FeatureCache(args.data_dir, mean, std, cfg, label_by_file)
    dataset_started = log_stage_start("build_xgb_dataset", variant.name, fold, files=len(train_files))
    dataset = CompatBatchedDataset(train_files, cache, cfg)
    log_stage_done(
        "build_xgb_dataset",
        dataset_started,
        variant.name,
        fold,
        rows=int(dataset.total_rows),
        source_rows=int(dataset.source_total_rows),
        pos=int(dataset.pos_rows),
        neg=int(dataset.neg_rows),
    )
    x_raw, y_train, sample_weight, raw_feature_names = collect_training_table_with_weights(cache, dataset, cfg, variant.name, fold)
    selector = fit_feature_selector(
        x_raw,
        y_train,
        raw_feature_names,
        args.selector,
        args.select_k,
        int(args.seed) + int(fold),
        args.n_jobs,
        deep_model_name=variant.name,
        fold=fold,
    )
    x_train = selector.transform(x_raw)
    save_selected_features(run_dir / "tabular" / "selected_features.csv", selector)
    xgb_started = log_stage_start("xgb_train", variant.name, fold, rows=len(y_train), features=x_train.shape[1])
    estimator, train_meta = train_xgb_variant(x_train, y_train, sample_weight, variant, args, fold)
    log_stage_done("xgb_train", xgb_started, variant.name, fold)
    with (run_dir / "xgb_model.pkl").open("wb") as fh:
        pickle.dump({"selector": selector, "model": estimator, "train_meta": train_meta}, fh)
    (run_dir / "variant.json").write_text(
        json.dumps({"variant": variant.__dict__, "cfg": asdict(cfg), "train_meta": train_meta}, indent=2),
        encoding="utf-8",
    )
    estimators = {"xgb": estimator}
    val_score_frames = score_model2_files_to_memory(estimators, selector, cache, val_files, cfg, fold=fold, stage_name="score_val_files")
    test_score_frames = score_model2_files_to_memory(estimators, selector, cache, test_files, cfg, fold=fold, stage_name="score_test_files")
    tag = f"{variant.name}_xgb"
    threshold, val_metrics = select_threshold_for_scores(val_score_frames["xgb"], args.data_dir, run_dir, tag, args)
    pred_dir = run_dir / "predictions" / tag
    eval_dir = run_dir / "evaluation" / tag
    test_rows = write_threshold_predictions(test_score_frames["xgb"], pred_dir, threshold)
    metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
    if str(args.lead_time_grid).strip():
        sweep_rows = run_lead_time_sweep(
            val_score_frames["xgb"],
            test_score_frames["xgb"],
            args.data_dir,
            run_dir,
            tag,
            int(fold),
            args.lead_time_grid,
            args.threshold_grid,
            args.threshold_metric,
            args.fixed_threshold,
            bool(args.threshold_search),
            metadata={
                "run_name": RUN_NAME,
                "variant": variant.name,
                "mode": tag,
                "feature_mode": cfg.feature_mode,
                "sample_selection": cfg.sample_selection,
                "temporal_positive_weight": float(cfg.temporal_positive_weight),
                "adaptive_negative_weight": float(variant.adaptive_negative_weight),
                "rule_mode": cfg.rule_mode,
                "selector": args.selector,
                "selected_feature_count": len(selector.indices),
            },
        )
        append_lead_time_sweep_results(sweep_rows, Path(args.out_root))
    print(f"[ablation-done] fold={fold} variant={variant.name} threshold={threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}", flush=True)
    del cache, dataset
    gc.collect()
    return {
        "variant": variant.name,
        "description": variant.description,
        "deep_model": RUN_NAME,
        "fold": int(fold),
        "mode": tag,
        "ml_model": "xgb",
        "feature_mode": cfg.feature_mode,
        "sample_selection": cfg.sample_selection,
        "temporal_positive_weight": float(cfg.temporal_positive_weight),
        "adaptive_negative_weight": float(variant.adaptive_negative_weight),
        "rule_mode": cfg.rule_mode,
        "selector": args.selector,
        "selected_feature_count": len(selector.indices),
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_rows": int(test_rows),
        "metrics": metrics,
        "train_meta": train_meta,
    }


def run_fold(fold: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    fold_started = time.time()
    index_df = read_index(args.index_path)
    base_cfg = CompatCfg(
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_train_files=args.max_train_files,
        max_test_files=args.max_test_files,
    )
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    label_by_file = file_label_map(index_df)
    if int(args.max_train_files) > 0:
        train_files = cap_files_stratified(train_files, label_by_file, int(args.max_train_files), int(args.seed) + int(fold))
    if int(args.max_test_files) > 0:
        test_files = cap_files_stratified(test_files, label_by_file, int(args.max_test_files), int(args.seed) + 7919 + int(fold))
    train_files, val_files = split_train_val_files(train_files, label_by_file, base_cfg, int(fold))
    results = []
    for variant in selected_variants(args.variants):
        results.append(run_variant(variant, int(fold), train_files, val_files, test_files, label_by_file, args))
    print(f"[fold-done] run={RUN_NAME} fold={fold} elapsed={format_duration(time.time() - fold_started)}", flush=True)
    return results


def aggregate_results(results: list[dict[str, Any]], out_root: Path, args: argparse.Namespace) -> None:
    rows = []
    for item in results:
        row = {
            "variant": item["variant"],
            "description": item["description"],
            "deep_model": item["deep_model"],
            "fold": int(item["fold"]),
            "mode": item["mode"],
            "ml_model": item["ml_model"],
            "feature_mode": item["feature_mode"],
            "sample_selection": item["sample_selection"],
            "temporal_positive_weight": float(item["temporal_positive_weight"]),
            "adaptive_negative_weight": float(item["adaptive_negative_weight"]),
            "rule_mode": item["rule_mode"],
            "selector": item["selector"],
            "selected_feature_count": int(item["selected_feature_count"]),
            "threshold": float(item["threshold"]),
            "target_mode": args.target_mode,
            "sampling_mode": args.sampling_mode,
            "min_hit_lead_hours": float(args.min_hit_lead_hours),
            "test_rows": int(item["test_rows"]),
            "train_stage": item.get("train_meta", {}).get("stage", ""),
        }
        row.update(item.get("metrics", {}))
        rows.append(row)
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        frame.drop_duplicates(subset=["variant", "fold", "rule_mode", "min_hit_lead_hours"], keep="last", inplace=True)
    frame.sort_values(["variant", "fold"], inplace=True)
    frame.to_csv(path, index=False)
    group_cols = ["variant", "feature_mode", "sample_selection", "temporal_positive_weight", "adaptive_negative_weight", "rule_mode"]
    numeric = [col for col in frame.columns if col not in set(group_cols) | {"fold"} and pd.api.types.is_numeric_dtype(frame[col])]
    summary = frame.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "variant_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost-only ablation for DRAM-inspired OFP-compatible gains.")
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--variants", nargs="+", default=["all"], help=f"all or any of: {', '.join(VARIANTS)}")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_model2_compat_results/xgb_dram_ablation"))
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.3)
    parser.add_argument("--target_mode", choices=["ahead120", "anomaly", "module_fault"], default="module_fault")
    parser.add_argument("--sampling_mode", choices=["row", "module_balanced"], default="module_balanced")
    parser.add_argument("--positive_windows_per_module", type=int, default=32)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--normal_windows_per_module", type=int, default=8)
    parser.add_argument("--rule_mode", choices=["none", "temp", "model2_simple"], default="model2_simple")
    parser.add_argument("--sample_topk_fraction", type=float, default=0.5)
    parser.add_argument("--temporal_weight_horizon_hours", type=float, default=120.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_grid", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--no_threshold_search", dest="threshold_search", action="store_false")
    parser.set_defaults(threshold_search=True)
    parser.add_argument("--min_hit_lead_hours", type=float, default=0.0)
    parser.add_argument(
        "--lead_time_grid",
        default="",
        help=f"Optional DRAM-style lead-time sweep, e.g. '{DEFAULT_LEAD_TIME_GRID}'. Values accept m/min/h suffixes.",
    )
    parser.add_argument("--max_cached_files", type=int, default=128)
    parser.add_argument("--max_train_files", type=int, default=2500)
    parser.add_argument("--max_test_files", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector", choices=["none", "variance", "f_classif", "mutual_info", "extra_trees", "model_importance"], default="extra_trees")
    parser.add_argument("--select_k", type=int, default=96)
    parser.add_argument("--ml_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--rf_max_depth", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: list[dict[str, Any]] = []
    for fold in args.folds:
        all_results.extend(run_fold(int(fold), args))
        aggregate_results(all_results, Path(args.out_root), args)
    aggregate_results(all_results, Path(args.out_root), args)


if __name__ == "__main__":
    main()
