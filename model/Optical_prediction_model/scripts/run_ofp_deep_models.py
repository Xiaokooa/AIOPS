"""Train DL backbones under the OFP ahead-window task.

This entry point is intentionally OFP-specific:

* training labels: model1-style ahead-window labels
  (120-hour primary by default, with 12/24/72-hour auxiliary horizons)
* validation threshold: model2-style OFP module-level ``final_score``
* output compatibility: optional ``timestamp,predict`` CSV export

The script does not run unless invoked explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.deep_learning import ofp_task
from model.Optical_prediction_model.deep_learning.train import (
    TrainCfg,
    fit_fits,
    fit_itransformer,
    fit_moderntcn,
    fit_patchtst,
    set_seed,
)


EXP_ROOT = PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments"


MODEL_RUNNERS = {
    "itransformer": fit_itransformer,
    "patchtst": fit_patchtst,
    "moderntcn": fit_moderntcn,
    "fits": fit_fits,
}


PROFILE_DEFAULTS = {
    "server": {
        "epochs": 30,
        "patience": 8,
        "batch_size": 64,
        "lr": 5e-4,
        "amp": False,
        "max_train_modules": None,
        "max_val_modules": None,
        "max_test_modules": None,
    },
    "laptop": {
        "epochs": 4,
        "patience": 2,
        "batch_size": 4,
        "lr": 3e-4,
        "amp": True,
        "max_train_modules": 1000,
        "max_val_modules": 100,
        "max_test_modules": 100,
    },
    "laptop_result": {
        "epochs": 8,
        "patience": 3,
        "batch_size": 4,
        "lr": 2e-4,
        "amp": True,
        "max_train_modules": 1500,
        "max_val_modules": 200,
        "max_test_modules": 200,
    },
    "laptop_tiny": {
        "epochs": 2,
        "patience": 1,
        "batch_size": 4,
        "lr": 3e-4,
        "amp": True,
        "max_train_modules": 1000,
        "max_val_modules": 100,
        "max_test_modules": 100,
    },
}


def _arg_or_profile(value, args: argparse.Namespace, name: str):
    return value if value is not None else PROFILE_DEFAULTS[args.profile][name]


def model_cfg_for_profile(model_name: str, profile: str, obs_steps: int):
    if profile == "server":
        return None

    if model_name == "patchtst":
        from model.Optical_prediction_model.deep_learning.patchtst_wrapper import PatchTSTCfg

        if profile == "laptop_tiny":
            return PatchTSTCfg(
                seq_len=obs_steps,
                patch_len=32,
                stride=16,
                use_mask_channel=False,
                d_model=64,
                n_heads=4,
                e_layers=1,
                d_ff=128,
                dropout=0.25,
                revin=True,
            )
        return PatchTSTCfg(
            seq_len=obs_steps,
            patch_len=32,
            stride=16,
            use_mask_channel=True,
            d_model=80,
            n_heads=4,
            e_layers=2,
            d_ff=160,
            dropout=0.25 if profile == "laptop_result" else 0.2,
            revin=True,
        )

    if model_name == "itransformer":
        from model.Optical_prediction_model.deep_learning.models import ITransformerCfg

        if profile == "laptop_tiny":
            return ITransformerCfg(
                seq_len=obs_steps,
                use_mask_channel=False,
                d_model=64,
                n_heads=4,
                e_layers=1,
                d_ff=128,
                dropout=0.25,
            )
        return ITransformerCfg(
            seq_len=obs_steps,
            use_mask_channel=True,
            d_model=80,
            n_heads=4,
            e_layers=2,
            d_ff=160,
            dropout=0.25 if profile == "laptop_result" else 0.2,
        )

    if model_name == "moderntcn":
        from model.Optical_prediction_model.deep_learning.moderntcn_wrapper import ModernTCNCfg

        if profile == "laptop_tiny":
            return ModernTCNCfg(
                seq_len=obs_steps,
                patch_size=32,
                patch_stride=16,
                num_blocks=(1,),
                large_size=(9,),
                small_size=(3,),
                dims=(32,),
                dw_dims=(32,),
                use_mask_channel=False,
            )
        return ModernTCNCfg(
            seq_len=obs_steps,
            patch_size=32,
            patch_stride=16,
            num_blocks=(1,),
            large_size=(15,),
            small_size=(3,),
            dims=(48,),
            dw_dims=(48,),
            use_mask_channel=True,
        )

    if model_name == "fits":
        from model.Optical_prediction_model.deep_learning.interpretable_fits import InterpFITSCfg

        if profile == "laptop_tiny":
            return InterpFITSCfg(
                seq_len=obs_steps,
                use_mask_channel=False,
                cut_freq=12,
                d_hidden=64,
                dropout=0.25,
                individual=False,
            )
        return InterpFITSCfg(
            seq_len=obs_steps,
            use_mask_channel=True,
            cut_freq=20,
            d_hidden=96,
            dropout=0.2,
            individual=False,
        )

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OFP ahead-window DL model adapter")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_RUNNERS),
        default=list(MODEL_RUNNERS),
        help="One or more model adapters to train.",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_DEFAULTS),
        default="server",
        help="Resource profile. Use laptop/laptop_tiny for display-attached laptop GPUs.",
    )
    parser.add_argument("--obs_minutes", type=int, default=1440)
    parser.add_argument("--ahead_hours", type=int, default=120)
    parser.add_argument(
        "--aux_ahead_hours",
        nargs="*",
        type=int,
        default=[12, 24, 72],
        help="Auxiliary ahead-window horizons for multi-horizon supervision; pass none to disable.",
    )
    parser.add_argument("--train_step_minutes", type=int, default=60)
    parser.add_argument("--eval_step_minutes", type=int, default=60)
    parser.add_argument("--min_obs_coverage", type=float, default=0.80)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--threshold_metric",
        choices=["ofp_final_score", "ofp_f1_score", "window_f1"],
        default="ofp_final_score",
        help="Validation metric used to select the warning threshold.",
    )
    parser.add_argument(
        "--threshold_grid",
        choices=["coarse", "fine"],
        default="coarse",
        help="OFP validation threshold grid: coarse=0.05 step, fine=0.01 step.",
    )
    parser.add_argument(
        "--early_stop_metric",
        choices=["window_f1", "threshold_metric"],
        default="window_f1",
        help="Epoch selection metric. Keep window_f1 for faster laptop runs.",
    )
    parser.add_argument(
        "--ofp_score_postprocess",
        choices=["raw", "rolling_mean", "rolling_max", "consecutive"],
        default="raw",
        help="Module-wise score filtering before OFP warning thresholding.",
    )
    parser.add_argument(
        "--ofp_postprocess_window",
        type=int,
        default=1,
        help="Number of consecutive evaluation windows used by the OFP score postprocessor.",
    )
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Random seed for module split/subsampling. Defaults to --seed.",
    )
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--export_ofp_predictions", action="store_true")
    parser.add_argument("--max_train_modules", type=int, default=None)
    parser.add_argument("--max_val_modules", type=int, default=None)
    parser.add_argument("--max_test_modules", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=str(EXP_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    profile = PROFILE_DEFAULTS[args.profile]
    epochs = _arg_or_profile(args.epochs, args, "epochs")
    patience = _arg_or_profile(args.patience, args, "patience")
    batch_size = _arg_or_profile(args.batch_size, args, "batch_size")
    lr = _arg_or_profile(args.lr, args, "lr")
    amp = _arg_or_profile(args.amp, args, "amp")
    max_train_modules = _arg_or_profile(args.max_train_modules, args, "max_train_modules")
    max_val_modules = _arg_or_profile(args.max_val_modules, args, "max_val_modules")
    max_test_modules = _arg_or_profile(args.max_test_modules, args, "max_test_modules")

    task_cfg = ofp_task.OFPAheadTaskConfig(
        obs_minutes=args.obs_minutes,
        ahead_hours=args.ahead_hours,
        auxiliary_ahead_hours=tuple(args.aux_ahead_hours),
        train_step_minutes=args.train_step_minutes,
        eval_step_minutes=args.eval_step_minutes,
        min_obs_coverage=args.min_obs_coverage,
    )
    print("=" * 80)
    print("OFP ahead-window deep-learning adapters")
    print("=" * 80)
    print(f"profile={args.profile} train_defaults={profile}")
    print(json.dumps(ofp_task.task_summary(task_cfg), indent=2))

    frames, split_map = ofp_task.load_ofp_ahead_frames(
        cfg=task_cfg,
        force_rebuild=args.force_rebuild,
        random_state=split_seed,
        max_train_modules=max_train_modules,
        max_val_modules=max_val_modules,
        max_test_modules=max_test_modules,
    )
    for split_name, frame in frames.items():
        modules = int(frame["file_name"].nunique()) if not frame.empty else 0
        label_cols = [col for col in ofp_task.label_columns(task_cfg) if col in frame.columns]
        if not label_cols and "label" in frame.columns:
            label_cols = ["label"]
        pos_summary = ", ".join(
            f"{col}:pos={int(frame[col].sum())},ratio={float(frame[col].mean()):.4f}"
            for col in label_cols
        ) if not frame.empty else "empty"
        print(f"[{split_name}] windows={len(frame)} modules={modules} {pos_summary}")

    train_cfg = TrainCfg(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=0,
        seed=args.seed,
        threshold_metric=args.threshold_metric,
        early_stop_metric=args.early_stop_metric,
        ofp_threshold_grid=args.threshold_grid,
        ofp_score_postprocess=args.ofp_score_postprocess,
        ofp_postprocess_window=args.ofp_postprocess_window,
        export_ofp_predictions=args.export_ofp_predictions,
        use_amp=bool(amp),
        use_multi_horizon_aux=bool(args.aux_ahead_hours),
        primary_horizon_hours=args.ahead_hours,
    )

    output_root = Path(args.output_root)
    summaries = {}
    for model_name in args.models:
        suffix = model_name if args.profile == "server" else f"{args.profile}_{model_name}"
        out_dir = output_root / f"{task_cfg.tag}_{suffix}"
        model_cfg = model_cfg_for_profile(model_name, args.profile, task_cfg.obs_steps)
        print(f"\n[run] {model_name} -> {out_dir}")
        if model_cfg is not None:
            print(f"[profile] {args.profile} model_cfg={model_cfg}")
        summaries[model_name] = MODEL_RUNNERS[model_name](
            frames,
            out_dir,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            split_map=split_map,
        )
        ofp_metrics = summaries[model_name].get("ofp_metrics", {})
        print(
            f"[{model_name}] OFP final={ofp_metrics.get('final_score', 0):.4f} "
            f"F1={ofp_metrics.get('f1_score', 0):.4f} "
            f"P={ofp_metrics.get('precision', 0):.4f} "
            f"R={ofp_metrics.get('recall', 0):.4f}"
        )


if __name__ == "__main__":
    main()
