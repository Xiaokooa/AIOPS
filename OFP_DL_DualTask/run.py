"""CLI entry point for the OFP dual-task deep-learning adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OFP_DL_DualTask.data import DualTaskConfig, frame_summary, load_dual_task_frames
from OFP_DL_DualTask.train import DualTrainConfig, fit_dual_task_model, set_seed


OUTPUT_ROOT = PROJECT_ROOT / "output" / "OFP_DL_DualTask"


PROFILE_DEFAULTS = {
    "tiny": {
        "epochs": 1,
        "batch_size": 4,
        "lr": 3e-4,
        "patience": 1,
        "train_faulty_max_windows": 64,
        "train_healthy_max_windows": 8,
        "monitor_val_max_windows": 20_000,
        "final_val_max_windows": 30_000,
        "max_train_modules": 40,
        "max_val_modules": 12,
        "max_test_modules": 12,
    },
    "laptop": {
        "epochs": 6,
        "batch_size": 4,
        "lr": 3e-4,
        "patience": 2,
        "train_faulty_max_windows": 96,
        "train_healthy_max_windows": 16,
        "monitor_val_max_windows": 80_000,
        "final_val_max_windows": 120_000,
        "max_train_modules": 1200,
        "max_val_modules": 200,
        "max_test_modules": 200,
    },
    "server": {
        "epochs": 20,
        "batch_size": 32,
        "lr": 5e-4,
        "patience": 5,
        "train_faulty_max_windows": 128,
        "train_healthy_max_windows": 16,
        "monitor_val_max_windows": 200_000,
        "final_val_max_windows": 500_000,
        "max_train_modules": None,
        "max_val_modules": None,
        "max_test_modules": None,
    },
}


def _value_or_profile(value, profile: str, key: str):
    return PROFILE_DEFAULTS[profile][key] if value is None else value


def _parse_sensor_columns(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    value = str(value).strip()
    if not value or value.lower() in {"all", "none"}:
        return ()
    parts = tuple(part.strip() for part in value.replace(";", ",").split(",") if part.strip())
    if len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError(f"--sensor_columns contains duplicates: {value}")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OFP-compatible current+ahead deep model")
    parser.add_argument("--model", choices=["itransformer", "patchtst", "moderntcn", "fits"], default="fits")
    parser.add_argument("--protocol", choices=["dual_v2", "forecast_only", "dual_legacy", "hybrid_tree"], default="dual_v2")
    parser.add_argument("--profile", choices=list(PROFILE_DEFAULTS), default="tiny")
    parser.add_argument("--obs_minutes", type=int, default=1440)
    parser.add_argument("--ahead_hours", type=int, default=120)
    parser.add_argument("--train_step_minutes", type=int, default=60)
    parser.add_argument("--eval_step_minutes", type=int, default=5)
    parser.add_argument("--train_faulty_max_windows", type=int, default=None)
    parser.add_argument("--train_healthy_max_windows", type=int, default=None)
    parser.add_argument("--no_prefix_windows", action="store_true")
    parser.add_argument("--exclude_start_fault_modules", action="store_true")
    parser.add_argument(
        "--include_start_fault_modules",
        action="store_true",
        help="Opt back into the legacy all-positive OFP setting that keeps modules faulty at the first row.",
    )
    parser.add_argument(
        "--sensor_columns",
        type=_parse_sensor_columns,
        default=(),
        help="Comma-separated raw sensor columns used by the deep encoder. Default: all raw sensors.",
    )
    parser.add_argument("--current_label_mode", choices=["end", "any"], default="end")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--threshold_metric", choices=["ofp_final_score", "ofp_f1_score"], default="ofp_f1_score")
    parser.add_argument("--threshold_grid", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--prediction_mode", choices=["ahead_only", "current_only", "or"], default=None)
    parser.add_argument("--monitor_val_max_windows", type=int, default=None)
    parser.add_argument("--final_val_max_windows", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--current_loss_weight", type=float, default=None)
    parser.add_argument("--ahead_loss_weight", type=float, default=None)
    parser.add_argument("--union_loss_weight", type=float, default=None)
    parser.add_argument("--hybrid_target", choices=["ahead_label", "current_label", "label", "module_label"], default="ahead_label")
    parser.add_argument("--hybrid_tree_model", choices=["xgboost", "rf"], default="xgboost")
    parser.add_argument("--hybrid_max_train_windows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--max_train_modules", type=int, default=None)
    parser.add_argument("--max_val_modules", type=int, default=None)
    parser.add_argument("--max_test_modules", type=int, default=None)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--export_predictions", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    profile = args.profile
    protocol = str(args.protocol)
    exclude_start_fault = bool(
        args.exclude_start_fault_modules
        or protocol == "forecast_only"
        or not bool(args.include_start_fault_modules)
    )
    pre_event_only = protocol == "forecast_only"
    prediction_mode = args.prediction_mode
    if prediction_mode is None:
        prediction_mode = "or" if protocol == "dual_legacy" else "ahead_only"
    current_loss = args.current_loss_weight
    ahead_loss = args.ahead_loss_weight
    union_loss = args.union_loss_weight
    if current_loss is None:
        current_loss = 0.0 if protocol == "forecast_only" else (1.0 if protocol == "dual_legacy" else 0.2)
    if ahead_loss is None:
        ahead_loss = 1.0
    if union_loss is None:
        union_loss = 0.25 if protocol == "dual_legacy" else 0.0
    task_cfg = DualTaskConfig(
        obs_minutes=args.obs_minutes,
        ahead_hours=args.ahead_hours,
        train_step_minutes=args.train_step_minutes,
        eval_step_minutes=args.eval_step_minutes,
        train_faulty_max_windows=_value_or_profile(args.train_faulty_max_windows, profile, "train_faulty_max_windows"),
        train_healthy_max_windows=_value_or_profile(args.train_healthy_max_windows, profile, "train_healthy_max_windows"),
        include_prefix_windows=not bool(args.no_prefix_windows),
        current_label_mode=args.current_label_mode,
        exclude_start_fault_modules=exclude_start_fault,
        pre_event_only=pre_event_only,
        sensor_columns=args.sensor_columns,
        random_state=split_seed,
    )
    print("=" * 80)
    print("OFP dual-task DL adapter")
    print("=" * 80)
    print(json.dumps(task_cfg.to_dict(), indent=2))

    frames, split_map = load_dual_task_frames(
        cfg=task_cfg,
        force_rebuild=args.force_rebuild,
        random_state=split_seed,
        max_train_modules=_value_or_profile(args.max_train_modules, profile, "max_train_modules"),
        max_val_modules=_value_or_profile(args.max_val_modules, profile, "max_val_modules"),
        max_test_modules=_value_or_profile(args.max_test_modules, profile, "max_test_modules"),
    )
    for name, frame in frames.items():
        print(f"[{name}] {json.dumps(frame_summary(frame), ensure_ascii=False)}")
        if frame.empty:
            raise RuntimeError(f"{name} frame is empty; check dataset paths and config")

    train_cfg = DualTrainConfig(
        epochs=_value_or_profile(args.epochs, profile, "epochs"),
        batch_size=_value_or_profile(args.batch_size, profile, "batch_size"),
        lr=_value_or_profile(args.lr, profile, "lr"),
        patience=_value_or_profile(args.patience, profile, "patience"),
        threshold_metric=args.threshold_metric,
        threshold_grid=args.threshold_grid,
        prediction_mode=prediction_mode,
        monitor_val_max_windows=_value_or_profile(args.monitor_val_max_windows, profile, "monitor_val_max_windows"),
        final_val_max_windows=_value_or_profile(args.final_val_max_windows, profile, "final_val_max_windows"),
        eval_every=max(1, int(args.eval_every)),
        current_loss_weight=float(current_loss),
        ahead_loss_weight=float(ahead_loss),
        union_loss_weight=float(union_loss),
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        use_amp=bool(args.amp),
        export_predictions=bool(args.export_predictions),
        hybrid_tree=protocol == "hybrid_tree",
        hybrid_target=args.hybrid_target,
        hybrid_tree_model=args.hybrid_tree_model,
        hybrid_max_train_windows=int(args.hybrid_max_train_windows),
    )
    output_dir = Path(args.output_root) / f"{protocol}_{task_cfg.tag}_{profile}_{args.model}_seed{args.seed}"
    summary = fit_dual_task_model(
        frames=frames,
        split_map=split_map,
        model_name=args.model,
        output_dir=output_dir,
        task_cfg=task_cfg,
        train_cfg=train_cfg,
    )
    metrics = summary["ofp_metrics"]
    print(
        f"[done] OFP final={metrics['final_score']:.4f} "
        f"F1={metrics['f1_score']:.4f} "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f}"
    )
    if summary.get("hybrid_tree"):
        hmetrics = summary["hybrid_tree"]["test_metrics"]
        print(
            f"[hybrid] final={hmetrics['final_score']:.4f} "
            f"F1={hmetrics['f1_score']:.4f} "
            f"P={hmetrics['precision']:.4f} "
            f"R={hmetrics['recall']:.4f}"
        )
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
