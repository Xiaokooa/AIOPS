from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP.deep_learning.official.ofp_protocol.run_ofp_baselines import MODEL2_FEATURES

from OFP.deep_learning.official.PatchTST.segment_model import build_segment_model
from OFP.deep_learning.official.common.formal_data import (
    HORIZON_HOURS,
    PRIMARY_HORIZON_INDEX,
    assert_prediction_coverage,
    set_seed,
    write_predictions,
)
from OFP.deep_learning.official.common.index_split import files_for_index_fold, read_index
from OFP.deep_learning.official.common.ofp_feature_sequence import (
    OFPFeatureModuleCache,
    compute_ofp_feature_norm_stats,
    save_feature_norm_stats,
)
from OFP.deep_learning.official.common.segment_data import (
    SegmentForecastDataset,
    collect_segment_module_scores,
    make_segment_loader,
)
from OFP.deep_learning.official.common.trainer import metrics_to_dict
from OFP.deep_learning.official.common.trainer import format_metric_summary, runtime_device_summary
from OFP.deep_learning.official.run_patchtst_segment import (
    SegmentCfg,
    configure_torch,
    diagnostic_train_scores,
    effective_pos_weight,
    train_segment_one_epoch,
)


PROTOCOL_NAME = "ofp_index_causal_patchtst_model2_feature_segment_24h_to_1h"
MODEL_NAME = "patchtst_ofp_feature_segment"


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    cfg: SegmentCfg,
    smoke_max_batches: int = 0,
    max_train_files: int = 0,
    max_test_files: int = 0,
    skip_eval: bool = False,
) -> dict:
    if cfg.train_stride > cfg.pred_len:
        raise ValueError("train_stride must be <= pred_len for timestamp coverage")

    configure_torch(cfg)
    set_seed(cfg.seed + int(fold))
    t0 = time.time()
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    if int(max_train_files) > 0:
        train_files = train_files[: int(max_train_files)]
    if int(max_test_files) > 0:
        test_files = test_files[: int(max_test_files)]

    run_dir = Path(out_root) / MODEL_NAME / f"fold_{int(fold)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = OFPFeatureModuleCache(
        data_dir,
        max_cached_files=cfg.max_cached_files,
        timestamp_mode=cfg.timestamp_mode,
        module_cache_dir=Path(cfg.module_cache_dir) if cfg.module_cache_dir else None,
    )
    print(
        f"[patchtst-ofp-init] model={MODEL_NAME} fold={fold} protocol={PROTOCOL_NAME} "
        f"train_modules={len(train_files)} test_modules={len(test_files)} "
        f"features={len(MODEL2_FEATURES)} input_len={cfg.input_len} pred_len={cfg.pred_len} "
        f"train_stride={cfg.train_stride} test_stride={cfg.test_stride} epochs={cfg.epochs} "
        f"batch_size={cfg.batch_size} negative_ratio={cfg.negative_ratio} "
        f"pos_weight_cap={cfg.pos_weight_cap} fixed_threshold={cfg.fixed_threshold} "
        f"timestamp_mode={cfg.timestamp_mode} amp={cfg.amp} tf32={cfg.allow_tf32}",
        flush=True,
    )
    print(f"[patchtst-ofp-init] {runtime_device_summary(cfg.device)}", flush=True)

    print(
        f"[ofp-feature segment fold={fold}] computing train-only normalization "
        f"files={len(train_files)} features={len(MODEL2_FEATURES)}",
        flush=True,
    )
    norm = compute_ofp_feature_norm_stats(cache, train_files)
    mean, std = norm.arrays()
    save_feature_norm_stats(norm, run_dir / "norm_stats.json")

    dataset = SegmentForecastDataset(
        train_files,
        cache=cache,
        input_len=cfg.input_len,
        pred_len=cfg.pred_len,
        train_stride=cfg.train_stride,
        mean=mean,
        std=std,
        batch_segments=cfg.batch_size,
        negative_ratio=cfg.negative_ratio,
        seed=cfg.seed + int(fold),
        shuffle_segments=cfg.shuffle_segments,
    )
    loader = make_segment_loader(dataset, num_workers=cfg.num_workers)
    model, model_cfg = build_segment_model(cfg.input_len, cfg.pred_len, len(MODEL2_FEATURES))
    model = model.to(cfg.device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    pos_weight = effective_pos_weight(dataset.pos_rows_by_horizon, dataset.neg_rows_by_horizon, cfg)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight, device=cfg.device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(
        f"[ofp-feature segment fold={fold}] train_segments={dataset.total_segments} batches={len(loader)} "
        f"target_rows={dataset.total_target_rows} pos_rows={dataset.pos_rows} neg_rows={dataset.neg_rows} "
        f"pos_weight={pos_weight.tolist()} horizons={list(HORIZON_HOURS)} "
        f"primary_horizon={HORIZON_HOURS[PRIMARY_HORIZON_INDEX]} amp={cfg.amp} n_params={n_params}",
        flush=True,
    )

    history: list[dict] = []
    for epoch in range(1, int(cfg.epochs) + 1):
        parts = train_segment_one_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            cfg,
            epoch,
            max_batches=smoke_max_batches,
        )
        history.append({"epoch": epoch, **parts})
        cache.cache.clear()
        gc.collect()
        if int(smoke_max_batches) > 0:
            break

    train_score_diag = diagnostic_train_scores(
        model,
        loader,
        cfg,
        max_batches=max(1, int(cfg.train_score_diag_batches)),
    )
    print(
        f"[ofp-feature segment fold={fold}] train score diag "
        f"pos_mean={train_score_diag.get('pos', {}).get('mean', 'NA')} "
        f"neg_mean={train_score_diag.get('neg', {}).get('mean', 'NA')}",
        flush=True,
    )
    cache.cache.clear()
    gc.collect()

    common_result = {
        "model": MODEL_NAME,
        "fold": int(fold),
        "protocol": PROTOCOL_NAME,
        "feature_source": "OFP model2 DEFAULT_FEATURE_LIST",
        "feature_names": list(MODEL2_FEATURES),
        "label": "first_event_1h_ahead",
        "causal_boundary": "input window ends before predicted target block",
        "extra_row_handling": "Temp=-255 and other OFP extra rows are represented as features; no direct rule override",
        "threshold": float(cfg.fixed_threshold),
        "alarm_strategy": {
            "alarm_smoothing": cfg.alarm_smoothing,
            "alarm_smooth_window": int(cfg.alarm_smooth_window),
            "alarm_consecutive_k": int(cfg.alarm_consecutive_k),
        },
        "first_alarm_only": bool(cfg.first_alarm_only),
        "input_len": cfg.input_len,
        "pred_len": cfg.pred_len,
        "train_stride": cfg.train_stride,
        "test_stride": cfg.test_stride,
        "aggregation": cfg.aggregation,
        "train_modules": len(train_files),
        "test_modules": len(test_files),
        "train_segments": dataset.total_segments,
        "train_pos_segments": dataset.pos_segments,
        "train_neg_segments": dataset.neg_segments,
        "train_target_rows": dataset.total_target_rows,
        "train_pos_rows": dataset.pos_rows,
        "train_neg_rows": dataset.neg_rows,
        "train_pos_rows_by_horizon": dataset.pos_rows_by_horizon.astype(int).tolist(),
        "train_neg_rows_by_horizon": dataset.neg_rows_by_horizon.astype(int).tolist(),
        "pos_weight": pos_weight.astype(float).tolist(),
        "horizon_hours": list(HORIZON_HOURS),
        "primary_horizon_index": int(PRIMARY_HORIZON_INDEX),
        "pos_weight_cap": float(cfg.pos_weight_cap),
        "n_params": n_params,
        "shuffle_segments": cfg.shuffle_segments,
        "seconds": time.time() - t0,
        "run_cfg": asdict(cfg),
        "model_cfg": model_cfg,
        "train_score_diag": train_score_diag,
        "history": history,
    }

    if int(smoke_max_batches) > 0 or bool(skip_eval):
        result = {
            **common_result,
            "formal_result": False,
            "purpose": "smoke_train_only" if int(smoke_max_batches) > 0 else "train_only_skip_eval",
        }
        (run_dir / "feature_segment_smoke_summary.json").write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
        return result

    print(f"[ofp-feature segment fold={fold}] scoring full test modules={len(test_files)}", flush=True)
    test_rows = collect_segment_module_scores(
        model,
        cache,
        test_files,
        cfg.input_len,
        cfg.pred_len,
        cfg.test_stride,
        mean,
        std,
        cfg.batch_size,
        cfg.device,
        use_amp=cfg.amp,
        aggregation=cfg.aggregation,
        use_tqdm=cfg.use_tqdm,
        force_tqdm=cfg.force_tqdm,
        progress_desc=f"[ofp-feature segment fold={fold} test]",
    )
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(
        test_rows,
        pred_dir,
        cfg.fixed_threshold,
        alarm_smoothing=cfg.alarm_smoothing,
        alarm_smooth_window=cfg.alarm_smooth_window,
        alarm_consecutive_k=cfg.alarm_consecutive_k,
        first_alarm_only=cfg.first_alarm_only,
    )
    assert_prediction_coverage(pred_dir, test_files)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    metrics = metrics_to_dict(summary_df)
    print(
        f"[patchtst-ofp-final] fold={fold} threshold={cfg.fixed_threshold:.6g} "
        f"epochs={cfg.epochs} elapsed_min={(time.time() - t0) / 60.0:.2f} "
        f"{format_metric_summary(metrics)}",
        flush=True,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)

    result = {
        **common_result,
        "paper_ready": int(max_train_files) <= 0 and int(max_test_files) <= 0 and int(smoke_max_batches) <= 0,
        "seconds": time.time() - t0,
        "metrics": metrics,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    if cfg.save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "run_cfg": asdict(cfg),
                "threshold": cfg.fixed_threshold,
                "alarm_strategy": {
                    "alarm_smoothing": cfg.alarm_smoothing,
                    "alarm_smooth_window": int(cfg.alarm_smooth_window),
                    "alarm_consecutive_k": int(cfg.alarm_consecutive_k),
                },
                "first_alarm_only": bool(cfg.first_alarm_only),
                "feature_names": list(MODEL2_FEATURES),
                "protocol": PROTOCOL_NAME,
            },
            run_dir / "model.pt",
        )
    return result


def aggregate(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        if "metrics" not in item:
            continue
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "paper_ready": item.get("paper_ready", False),
            "protocol": item["protocol"],
            "feature_count": len(item.get("feature_names", [])),
            "train_segments": item["train_segments"],
            "train_target_rows": item["train_target_rows"],
            "seconds": item["seconds"],
        }
        row.update(item["metrics"])
        rows.append(row)
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    metrics_path = out_root / "patchtst_ofp_feature_segment_fold_metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        df_new = pd.concat([old, df_new], ignore_index=True)
        df_new.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df_new.sort_values(["model", "fold"], inplace=True)
    df_new.to_csv(metrics_path, index=False)
    numeric = [
        col
        for col in df_new.columns
        if col not in {"model", "paper_ready", "protocol"} and pd.api.types.is_numeric_dtype(df_new[col])
    ]
    summary = df_new.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    summary.reset_index().to_csv(out_root / "patchtst_ofp_feature_segment_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal PatchTST over OFP model2 feature sequences.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("OFP_DL_official_results/ofp_index_patchtst_ofp_feature_segment_e3"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--input_len", type=int, default=288)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--train_stride", type=int, default=12)
    parser.add_argument("--test_stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_cached_files", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--timestamp_mode", choices=["legacy_float32", "strict_int64"], default="legacy_float32")
    parser.add_argument("--module_cache_dir", default="")
    parser.add_argument("--amp", action="store_true", help="enable CUDA autocast mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="disable CUDA autocast mixed precision")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--log_batches", type=int, default=25)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--force_tqdm", action="store_true")
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--no_save_checkpoint", action="store_true")
    parser.add_argument("--no_shuffle_segments", action="store_true")
    parser.add_argument("--train_score_diag_batches", type=int, default=20)
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--smoke_max_batches", type=int, default=0)
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_test_files", type=int, default=0)
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> SegmentCfg:
    return SegmentCfg(
        input_len=args.input_len,
        pred_len=args.pred_len,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        fixed_threshold=args.fixed_threshold,
        seed=args.seed,
        device=args.device,
        max_cached_files=args.max_cached_files,
        num_workers=args.num_workers,
        timestamp_mode=args.timestamp_mode,
        module_cache_dir=args.module_cache_dir,
        amp=bool(args.amp) and not bool(args.no_amp),
        allow_tf32=not args.no_tf32,
        log_batches=args.log_batches,
        use_tqdm=not args.no_tqdm,
        force_tqdm=args.force_tqdm,
        aggregation=args.aggregation,
        save_checkpoint=not args.no_save_checkpoint,
        shuffle_segments=not args.no_shuffle_segments,
        train_score_diag_batches=args.train_score_diag_batches,
        alarm_smoothing=args.alarm_smoothing,
        alarm_smooth_window=args.alarm_smooth_window,
        alarm_consecutive_k=args.alarm_consecutive_k,
        first_alarm_only=not args.no_first_alarm_only,
    )


def main() -> None:
    args = parse_args()
    cfg = cfg_from_args(args)
    index_df = read_index(args.index_path)
    results = []
    for fold in args.folds:
        result = run_fold(
            int(fold),
            args.data_dir,
            index_df,
            args.out_root,
            cfg,
            smoke_max_batches=args.smoke_max_batches,
            max_train_files=args.max_train_files,
            max_test_files=args.max_test_files,
            skip_eval=args.skip_eval,
        )
        results.append(result)
        aggregate([result], args.out_root)
        if "metrics" in result:
            print(
                f"[ofp-feature segment done] fold={fold} "
                f"{format_metric_summary(result['metrics'])}",
                flush=True,
            )
        else:
            print(f"[ofp-feature segment done] fold={fold} smoke/train-only complete", flush=True)


if __name__ == "__main__":
    main()
