from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True.*",
    category=UserWarning,
)

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.common.index_split import read_index
from OFP.deep_learning.official.common.ofp_model2_baselines import aggregate_model2_results, run_ofp_model2_fold
from OFP.deep_learning.official.common.trainer import (
    OfficialDeepCfg,
    aggregate_results,
    format_metric_summary,
    run_deep_fold_index,
)
from OFP.deep_learning.official.FITS.model import build_model as build_fits
from OFP.deep_learning.official.FTEformer.model import build_model as build_fteformer
from OFP.deep_learning.official.ModernTCN.model import build_model as build_moderntcn
from OFP.deep_learning.official.iTransformer.model import build_model as build_itransformer
from OFP.deep_learning.official.PatchTST.model import build_model as build_patchtst


DEEP_BUILDERS = {
    "itransformer": build_itransformer,
    "patchtst": build_patchtst,
    "fteformer": build_fteformer,
    "moderntcn": build_moderntcn,
    "fits": build_fits,
}

TREE_MODELS = {"rf", "xgboost"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OFP-index 24h-lookback 1h-ahead legacy-timestamp benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "xgboost", "patchtst", "itransformer", "fteformer"],
        choices=sorted(set(DEEP_BUILDERS) | TREE_MODELS),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_official_results/lookback24h_ahead1h_legacy_benchmark"))
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument(
        "--timestamp_mode",
        choices=["legacy_float32", "strict_int64"],
        default="legacy_float32",
    )
    parser.add_argument("--no_save_checkpoint", action="store_true")
    parser.add_argument("--shuffle_train_rows", action="store_true")
    parser.add_argument("--log_batches", type=int, default=1000)
    parser.add_argument("--no_batched_windows", action="store_true")
    parser.add_argument("--amp", action="store_true", help="enable CUDA autocast mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="disable CUDA autocast mixed precision")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--module_cache_dir", default="")
    parser.add_argument("--input_mode", choices=["raw", "model2_features"], default="model2_features")
    parser.add_argument("--train_sampling", choices=["full", "pos_all_neg_ratio", "module_balanced"], default="module_balanced")
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--aux_bce_weight", type=float, default=0.3)
    parser.add_argument("--event_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_hit_mode", choices=["primary_window", "pre_fault"], default="primary_window")
    parser.add_argument("--event_modules_per_epoch", type=int, default=256)
    parser.add_argument("--event_max_windows_per_module", type=int, default=256)
    parser.add_argument("--event_module_batch_size", type=int, default=4)
    parser.add_argument("--event_positive_fraction", type=float, default=0.5)
    parser.add_argument("--module_positive_windows_per_module", type=int, default=32)
    parser.add_argument("--module_negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--module_normal_windows_per_module", type=int, default=8)
    parser.add_argument("--index_val_fraction", type=float, default=0.1)
    parser.add_argument("--selection_metric", choices=["final_score", "f1_score", "precision", "recall", "accuracy"], default="f1_score")
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_val_files", type=int, default=0)
    parser.add_argument("--max_test_files", type=int, default=0)
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--alarm_search_smooth_windows", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--alarm_search_consecutive_ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--force_tqdm", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--xgb_estimators", type=int, default=10)
    parser.add_argument("--rf_threshold", type=float, default=0.5)
    parser.add_argument("--xgb_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    deep_cfg = OfficialDeepCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        threshold_grid_size=args.threshold_grid_size,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        fixed_threshold=args.fixed_threshold,
        timestamp_mode=args.timestamp_mode,
        save_checkpoint=not args.no_save_checkpoint,
        shuffle_train_rows=args.shuffle_train_rows,
        log_batches=args.log_batches,
        batched_windows=not args.no_batched_windows,
        amp=bool(args.amp) and not bool(args.no_amp),
        allow_tf32=not args.no_tf32,
        module_cache_dir=args.module_cache_dir,
        input_mode=args.input_mode,
        train_sampling=args.train_sampling,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        aux_bce_weight=args.aux_bce_weight,
        event_loss_weight=args.event_loss_weight,
        event_hit_mode=args.event_hit_mode,
        event_modules_per_epoch=args.event_modules_per_epoch,
        event_max_windows_per_module=args.event_max_windows_per_module,
        event_module_batch_size=args.event_module_batch_size,
        event_positive_fraction=args.event_positive_fraction,
        module_positive_windows_per_module=args.module_positive_windows_per_module,
        module_negative_windows_per_faulty_module=args.module_negative_windows_per_faulty_module,
        module_normal_windows_per_module=args.module_normal_windows_per_module,
        index_val_fraction=args.index_val_fraction,
        selection_metric=args.selection_metric,
        max_train_files=args.max_train_files,
        max_val_files=args.max_val_files,
        max_test_files=args.max_test_files,
        alarm_smoothing=args.alarm_smoothing,
        alarm_smooth_window=args.alarm_smooth_window,
        alarm_consecutive_k=args.alarm_consecutive_k,
        alarm_search_smooth_windows=tuple(int(x) for x in args.alarm_search_smooth_windows),
        alarm_search_consecutive_ks=tuple(int(x) for x in args.alarm_search_consecutive_ks),
        first_alarm_only=not args.no_first_alarm_only,
        use_tqdm=not args.no_tqdm,
        force_tqdm=args.force_tqdm,
    )

    for model_name in args.models:
        for fold in args.folds:
            print(f"[official benchmark] model={model_name} fold={fold}")
            if model_name in TREE_MODELS:
                result = run_ofp_model2_fold(
                    model_name=model_name,
                    fold=int(fold),
                    data_dir=args.data_dir,
                    index_df=index_df,
                    out_root=args.out_root,
                    n_jobs=args.n_jobs,
                    rf_estimators=args.rf_estimators,
                    xgb_estimators=args.xgb_estimators,
                    rf_threshold=args.rf_threshold,
                    xgb_threshold=args.xgb_threshold,
                )
                aggregate_model2_results([result], args.out_root)
                print(
                    f"[official done] model={model_name} fold={fold} "
                    f"{format_metric_summary(result.metrics)}"
                )
                continue

            result = run_deep_fold_index(
                model_name=model_name,
                model_builder=DEEP_BUILDERS[model_name],
                fold=int(fold),
                data_dir=args.data_dir,
                index_df=index_df,
                out_root=args.out_root,
                cfg=deep_cfg,
            )
            aggregate_results([result], args.out_root)
            print(
                f"[official done] model={model_name} fold={fold} "
                f"{format_metric_summary(result['metrics'])}"
            )


if __name__ == "__main__":
    main()
