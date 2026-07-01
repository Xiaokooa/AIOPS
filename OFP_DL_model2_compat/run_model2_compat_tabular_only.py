from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from dataclasses import asdict
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
    aggregate_results,
    evaluate_prediction_output,
    fit_feature_selector,
    log_stage_done,
    log_stage_start,
    positive_scores,
    save_selected_features,
    select_threshold_for_scores,
    train_ml_models,
    write_threshold_predictions,
)
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.common.trainer import format_metric_summary


DEFAULT_THRESHOLD_GRID = "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50"
RUN_NAME = "model2_tabular"


def collect_model2_training_table(
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    fold: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = compat_feature_names(cfg)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    started = log_stage_start(
        "collect_model2_train_table",
        RUN_NAME,
        fold,
        files=len(dataset.file_names),
        features=len(feature_names),
    )
    rows = 0
    for idx, (file_name, positions) in enumerate(zip(dataset.file_names, dataset.selected_positions), 1):
        if len(positions) == 0:
            continue
        _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = cache.get(file_name)
        selected = np.asarray(positions, dtype=np.int64)
        x_parts.append(features[selected].astype(np.float32))
        y_parts.append(labels[selected].astype(np.int8))
        rows += int(len(selected))
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold if fold is not None else '-'} "
                f"stage=collect_model2_train_table {progress_bar(idx, len(dataset.file_names), width=18)} "
                f"files={idx}/{len(dataset.file_names)} rows={rows} "
                f"rate={format_rate(idx, elapsed)} elapsed={format_duration(elapsed)}",
                flush=True,
            )
    if not x_parts:
        raise ValueError("No rows collected for model2 tabular training")
    x_out = np.concatenate(x_parts, axis=0).astype(np.float32)
    y_out = np.concatenate(y_parts, axis=0).astype(np.int8)
    log_stage_done(
        "collect_model2_train_table",
        started,
        RUN_NAME,
        fold,
        rows=len(y_out),
        features=x_out.shape[1],
        positives=int(np.sum(y_out > 0)),
        negatives=int(np.sum(y_out <= 0)),
    )
    return x_out, y_out, feature_names


def score_model2_files_to_memory(
    estimators: dict[str, Any],
    selector,
    cache: Model2FeatureCache,
    file_names: list[str],
    cfg: CompatCfg,
    fold: int | None = None,
    stage_name: str = "score_model2_files",
) -> dict[str, dict[str, pd.DataFrame]]:
    frames_by_model: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in estimators}
    started = log_stage_start(
        stage_name,
        RUN_NAME,
        fold,
        files=len(file_names),
        models=",".join(sorted(estimators)),
        features=len(selector.indices),
    )
    for idx, file_name in enumerate(file_names, 1):
        timestamps, features, _valid_mask, _labels, _anomaly_labels, rule_pred, extra = cache.get(file_name)
        x = selector.transform(features.astype(np.float32))
        for model_tag, estimator in estimators.items():
            score = positive_scores(estimator, x)
            frames: list[pd.DataFrame] = [
                pd.DataFrame(
                    {
                        "timestamp": timestamps.astype(np.int64),
                        "score": score.astype(np.float32),
                        "rule_predict": rule_pred.astype(int),
                        "source": f"{model_tag}_model2",
                    }
                )
            ]
            if len(extra):
                frames.append(
                    pd.DataFrame(
                        {
                            "timestamp": extra[:, 0].astype(np.int64),
                            "score": np.nan,
                            "rule_predict": extra[:, 1].astype(int),
                            "source": "model2_extra_rule",
                        }
                    )
                )
            frames_by_model[model_tag][file_name] = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold if fold is not None else '-'} "
                f"stage={stage_name} {progress_bar(idx, len(file_names), width=18)} "
                f"files={idx}/{len(file_names)} rate={format_rate(idx, elapsed)} "
                f"elapsed={format_duration(elapsed)}",
                flush=True,
            )
    log_stage_done(stage_name, started, RUN_NAME, fold)
    return frames_by_model


def run_fold(fold: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = CompatCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        fixed_threshold=args.fixed_threshold,
        target_mode=args.target_mode,
        feature_mode=args.feature_mode,
        sampling_mode=args.sampling_mode,
        positive_windows_per_module=args.positive_windows_per_module,
        negative_windows_per_faulty_module=args.negative_windows_per_faulty_module,
        normal_windows_per_module=args.normal_windows_per_module,
        val_fraction=args.val_fraction,
        threshold_search=bool(args.threshold_search),
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        rule_mode=args.rule_mode,
        sample_selection=args.sample_selection,
        sample_topk_fraction=args.sample_topk_fraction,
        temporal_positive_weight=args.temporal_positive_weight,
        temporal_weight_horizon_hours=args.temporal_weight_horizon_hours,
        adaptive_negative_weight=0.0,
        adaptive_warmup_epochs=args.adaptive_warmup_epochs,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        amp=bool(args.amp),
        allow_tf32=not args.no_tf32,
        log_batches=args.log_batches,
        max_train_files=args.max_train_files,
        max_test_files=args.max_test_files,
        min_hit_lead_hours=args.min_hit_lead_hours,
    )
    fold_started = time.time()
    print(f"[tabular-only-init] fold={fold} device={args.device} run={RUN_NAME}", flush=True)
    index_started = log_stage_start("read_index_split", RUN_NAME, fold, index_path=args.index_path)
    index_df = read_index(args.index_path)
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    label_by_file = file_label_map(index_df)
    if int(args.max_train_files) > 0:
        train_files = cap_files_stratified(train_files, label_by_file, int(args.max_train_files), int(args.seed) + int(fold))
    if int(args.max_test_files) > 0:
        test_files = cap_files_stratified(test_files, label_by_file, int(args.max_test_files), int(args.seed) + 7919 + int(fold))
    train_files, val_files = split_train_val_files(train_files, label_by_file, cfg, int(fold))
    log_stage_done(
        "read_index_split",
        index_started,
        RUN_NAME,
        fold,
        train=len(train_files),
        val=len(val_files),
        test=len(test_files),
    )

    run_dir = Path(args.out_root) / RUN_NAME / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_run_header(
        "OFP MODEL2-COMPAT TABULAR ONLY",
        {
            "fold": fold,
            "train/val/test": f"{len(train_files)}/{len(val_files)}/{len(test_files)} modules",
            "target": cfg.target_mode,
            "features": cfg.feature_mode,
            "ml models": " ".join(args.ml_models),
            "selector": f"{args.selector} top_k={args.select_k}",
            "sample selection": cfg.sample_selection,
            "rule mode": cfg.rule_mode,
            "min hit lead": f"{args.min_hit_lead_hours} h",
            "batch_size": cfg.batch_size,
            "out_dir": run_dir,
        },
    )

    stats_started = log_stage_start("feature_norm_stats", RUN_NAME, fold, files=len(train_files), data_dir=args.data_dir)
    stats = compute_feature_norm_stats(args.data_dir, train_files, cfg, label_by_file)
    log_stage_done("feature_norm_stats", stats_started, RUN_NAME, fold)
    mean, std = stats.arrays()
    cache_started = log_stage_start("feature_cache_init", RUN_NAME, fold, data_dir=args.data_dir)
    cache = Model2FeatureCache(args.data_dir, mean, std, cfg, label_by_file)
    log_stage_done("feature_cache_init", cache_started, RUN_NAME, fold)

    dataset_started = log_stage_start(
        "build_model2_dataset",
        RUN_NAME,
        fold,
        files=len(train_files),
        sample_selection=cfg.sample_selection,
        sampling_mode=cfg.sampling_mode,
    )
    dataset = CompatBatchedDataset(train_files, cache, cfg)
    log_stage_done(
        "build_model2_dataset",
        dataset_started,
        RUN_NAME,
        fold,
        rows=int(dataset.total_rows),
        source_rows=int(dataset.source_total_rows),
        pos=int(dataset.pos_rows),
        neg=int(dataset.neg_rows),
    )

    x_raw, y_train, raw_feature_names = collect_model2_training_table(cache, dataset, cfg, fold=fold)
    selector = fit_feature_selector(
        x_raw,
        y_train,
        raw_feature_names,
        args.selector,
        args.select_k,
        int(args.seed) + int(fold),
        args.n_jobs,
        deep_model_name=RUN_NAME,
        fold=fold,
    )
    x_train = selector.transform(x_raw)
    save_selected_features(run_dir / "tabular" / "selected_features.csv", selector)
    ml_names = [name.lower() for name in args.ml_models]
    ml_estimators = train_ml_models(
        x_train,
        y_train,
        ml_names,
        args,
        int(args.seed) + int(fold),
        deep_model_name=RUN_NAME,
        fold=fold,
    )
    trained_ml_names = list(ml_estimators.keys())
    print(
        f"[ml-ready] run={RUN_NAME} fold={fold} requested={','.join(ml_names)} "
        f"trained={','.join(trained_ml_names)}",
        flush=True,
    )
    pickle_started = log_stage_start("save_ml_models", RUN_NAME, fold, path=run_dir / "tabular" / "ml_models.pkl")
    with (run_dir / "tabular" / "ml_models.pkl").open("wb") as fh:
        pickle.dump({"selector": selector, "models": ml_estimators}, fh)
    log_stage_done("save_ml_models", pickle_started, RUN_NAME, fold)

    val_score_frames = score_model2_files_to_memory(
        ml_estimators,
        selector,
        cache,
        val_files,
        cfg,
        fold=fold,
        stage_name="score_val_files",
    )
    test_score_frames = score_model2_files_to_memory(
        ml_estimators,
        selector,
        cache,
        test_files,
        cfg,
        fold=fold,
        stage_name="score_test_files",
    )

    results: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    eval_started = log_stage_start("threshold_eval", RUN_NAME, fold, models=",".join(trained_ml_names))
    for ml_name in trained_ml_names:
        tag = f"ml_{ml_name}_model2"
        tag_started = log_stage_start("threshold_eval_one", RUN_NAME, fold, tag=tag)
        threshold, val_metrics = select_threshold_for_scores(val_score_frames[ml_name], args.data_dir, run_dir, tag, args)
        thresholds[tag] = threshold
        pred_dir = run_dir / "predictions" / tag
        eval_dir = run_dir / "evaluation" / tag
        test_rows = write_threshold_predictions(test_score_frames[ml_name], pred_dir, threshold)
        metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
        result = {
            "deep_model": RUN_NAME,
            "fold": int(fold),
            "mode": tag,
            "ml_model": ml_name,
            "ml_feature_set": "model2",
            "deep_feature_parts": [],
            "selector": args.selector,
            "selected_feature_count": len(selector.indices),
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "test_rows": int(test_rows),
            "metrics": metrics,
        }
        print(f"[tabular-only-done] fold={fold} {tag} threshold={threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}", flush=True)
        log_stage_done("threshold_eval_one", tag_started, RUN_NAME, fold, tag=tag)
        results.append(result)
    log_stage_done("threshold_eval", eval_started, RUN_NAME, fold)

    fold_summary = {
        "deep_model": RUN_NAME,
        "fold": int(fold),
        "seconds": time.time() - started,
        "cfg": asdict(cfg),
        "args": vars(args),
        "feature_norm_stats": asdict(stats),
        "raw_feature_count": len(raw_feature_names),
        "selected_features": selector.get_feature_names_out(),
        "thresholds": thresholds,
        "results": results,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2, default=str), encoding="utf-8")
    print(
        f"[fold-done] run={RUN_NAME} fold={fold} "
        f"elapsed={format_duration(time.time() - fold_started)} results={len(results)} out={run_dir}",
        flush=True,
    )

    del dataset, cache
    gc.collect()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model2-compatible standalone tabular ML experiments.")
    parser.add_argument("--models", nargs="+", default=[RUN_NAME], help=argparse.SUPPRESS)
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_model2_compat_results/model2_tabular"))
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.3)
    parser.add_argument("--target_mode", choices=["ahead120", "anomaly", "module_fault"], default="module_fault")
    parser.add_argument("--feature_mode", choices=["model2", "model2_plus"], default="model2")
    parser.add_argument("--sampling_mode", choices=["row_ratio", "module_balanced"], default="module_balanced")
    parser.add_argument("--positive_windows_per_module", type=int, default=32)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--normal_windows_per_module", type=int, default=8)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_grid", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--no_threshold_search", dest="threshold_search", action="store_false")
    parser.set_defaults(threshold_search=True)
    parser.add_argument("--rule_mode", choices=["none", "temp", "model2_simple"], default="model2_simple")
    parser.add_argument("--sample_selection", choices=["random", "signal_topk", "hybrid"], default="hybrid")
    parser.add_argument("--sample_topk_fraction", type=float, default=0.5)
    parser.add_argument("--temporal_positive_weight", type=float, default=2.0)
    parser.add_argument("--temporal_weight_horizon_hours", type=float, default=120.0)
    parser.add_argument("--adaptive_negative_weight", type=float, default=0.0)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=1)
    parser.add_argument("--min_hit_lead_hours", type=float, default=0.0)
    parser.add_argument("--ml_models", nargs="+", default=["xgb", "lgbm", "catboost"])
    parser.add_argument("--require_all_ml", action="store_true")
    parser.add_argument("--selector", choices=["none", "variance", "f_classif", "mutual_info", "extra_trees", "model_importance"], default="none")
    parser.add_argument("--select_k", type=int, default=0)
    parser.add_argument("--ml_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=0)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--max_cached_files", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--log_batches", type=int, default=0)
    parser.add_argument("--max_train_files", type=int, default=2500)
    parser.add_argument("--max_test_files", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: list[dict[str, Any]] = []
    started = time.time()
    print("=" * 72)
    print("OFP model2-compatible standalone tabular ML")
    print("=" * 72)
    print(f"folds: {' '.join(str(x) for x in args.folds)}")
    print(f"ml_models: {' '.join(args.ml_models)}")
    print(f"feature_mode: {args.feature_mode}")
    print(f"rule_mode: {args.rule_mode}")
    print(f"out_root: {args.out_root}")
    print("=" * 72)
    for fold in args.folds:
        results = run_fold(int(fold), args)
        all_results.extend(results)
        aggregate_results(all_results, Path(args.out_root), args)
    aggregate_results(all_results, Path(args.out_root), args)
    print(
        f"[suite-done] {progress_bar(len(args.folds), len(args.folds), width=18)} "
        f"elapsed={format_duration(time.time() - started)} out={args.out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
