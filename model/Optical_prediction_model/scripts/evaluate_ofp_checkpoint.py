"""Evaluate a saved OFP deep-learning checkpoint with threshold/postprocess sweeps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.deep_learning import ofp_task
from model.Optical_prediction_model.deep_learning.data import (
    FailureWindowDataset,
    WindowSliceCache,
    collate,
    compute_norm_stats,
)
from model.Optical_prediction_model.deep_learning.train import (
    TrainCfg,
    _primary_index,
    _resolve_target_columns,
    event_metrics_from_preds,
    run_one_epoch,
    set_seed,
)
from model.Optical_prediction_model.deep_learning.patchtst_wrapper import PatchTSTCfg, PatchTSTClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep OFP threshold/postprocess on a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--obs_minutes", type=int, default=1440)
    parser.add_argument("--ahead_hours", type=int, default=120)
    parser.add_argument("--aux_ahead_hours", nargs="*", type=int, default=[16, 24, 72])
    parser.add_argument("--train_step_minutes", type=int, default=60)
    parser.add_argument("--eval_step_minutes", type=int, default=60)
    parser.add_argument("--min_obs_coverage", type=float, default=0.80)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_modules", type=int, default=1500)
    parser.add_argument("--max_val_modules", type=int, default=200)
    parser.add_argument("--max_test_modules", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold_grid", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--postprocess_modes", nargs="+", default=["raw", "rolling_mean", "rolling_max", "consecutive"])
    parser.add_argument("--postprocess_windows", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def thresholds_for_grid(name: str) -> np.ndarray:
    if name == "coarse":
        return np.linspace(0.05, 0.95, 19)
    if name == "fine":
        return np.linspace(0.01, 0.99, 99)
    raise ValueError(f"Unknown threshold grid: {name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    task_cfg = ofp_task.OFPAheadTaskConfig(
        obs_minutes=args.obs_minutes,
        ahead_hours=args.ahead_hours,
        auxiliary_ahead_hours=tuple(args.aux_ahead_hours),
        train_step_minutes=args.train_step_minutes,
        eval_step_minutes=args.eval_step_minutes,
        min_obs_coverage=args.min_obs_coverage,
    )
    frames, split_map = ofp_task.load_ofp_ahead_frames(
        cfg=task_cfg,
        random_state=args.split_seed,
        max_train_modules=args.max_train_modules,
        max_val_modules=args.max_val_modules,
        max_test_modules=args.max_test_modules,
    )
    obs_steps = int(frames["train"].iloc[0]["obs_end_idx"] - frames["train"].iloc[0]["obs_start_idx"] + 1)
    train_cfg = TrainCfg(
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        use_amp=bool(args.amp),
        use_multi_horizon_aux=bool(args.aux_ahead_hours),
        primary_horizon_hours=args.ahead_hours,
    )
    target_columns = _resolve_target_columns(frames["train"], train_cfg)
    primary_index = _primary_index(target_columns, args.ahead_hours)

    slice_cache = WindowSliceCache()
    norm = compute_norm_stats(frames["train"], slice_cache, obs_steps, max_modules=400)
    ds_val = FailureWindowDataset(frames["val"], obs_steps, slice_cache, norm, target_columns=target_columns)
    ds_test = FailureWindowDataset(frames["test"], obs_steps, slice_cache, norm, target_columns=target_columns)
    dl_kw = dict(batch_size=args.batch_size, num_workers=0, collate_fn=collate, pin_memory=(device == "cuda"))
    val_loader = DataLoader(ds_val, shuffle=False, drop_last=False, **dl_kw)
    test_loader = DataLoader(ds_test, shuffle=False, drop_last=False, **dl_kw)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = PatchTSTCfg(**checkpoint["model_cfg"])
    model_cfg.seq_len = obs_steps
    model_cfg.n_sensors = len(norm.sensors)
    model_cfg.output_dim = len(target_columns)
    model = PatchTSTClassifier(model_cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loss_fn = nn.BCEWithLogitsLoss()

    _, val_scores, val_y = run_one_epoch(
        model,
        val_loader,
        None,
        loss_fn,
        device,
        primary_index=primary_index,
        use_amp=args.amp,
    )
    _, test_scores, test_y = run_one_epoch(
        model,
        test_loader,
        None,
        loss_fn,
        device,
        primary_index=primary_index,
        use_amp=args.amp,
    )

    thresholds = thresholds_for_grid(args.threshold_grid)
    rows = []
    for mode in args.postprocess_modes:
        for window in args.postprocess_windows:
            if mode == "raw" and window != 1:
                continue
            threshold, selection = ofp_task.choose_threshold_by_ofp_score(
                frames["val"],
                val_scores,
                split_df=split_map.get("val"),
                thresholds=thresholds,
                optimize_metric="f1_score",
                score_postprocess=mode,
                postprocess_window=window,
            )
            eval_scores = ofp_task.postprocess_scores(frames["test"], test_scores, mode=mode, window=window)
            test_pred = (eval_scores >= threshold).astype(int)
            ofp_metrics = ofp_task.evaluate_ofp_scores(
                frames["test"],
                test_scores,
                threshold,
                split_df=split_map.get("test"),
                score_postprocess=mode,
                postprocess_window=window,
            )
            event_metrics = event_metrics_from_preds(frames["test"], test_pred, split_map.get("test"))
            row = {
                "mode": mode,
                "window": int(window),
                "threshold_grid": args.threshold_grid,
                "threshold": float(threshold),
                "selection": selection,
                "ofp_metrics": ofp_metrics,
                "event_metrics": event_metrics,
            }
            rows.append(row)
            print(
                f"{mode:>12} w={window} thr={threshold:.2f} "
                f"F1={ofp_metrics['f1_score']:.4f} "
                f"P={ofp_metrics['precision']:.4f} "
                f"R={ofp_metrics['recall']:.4f} "
                f"final={ofp_metrics['final_score']:.4f}"
            )

    rows.sort(key=lambda item: item["ofp_metrics"]["f1_score"], reverse=True)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "task": ofp_task.task_summary(task_cfg),
                "target_columns": target_columns,
                "primary_index": primary_index,
                "rows": rows,
            },
            f,
            indent=2,
            default=str,
        )
    best = rows[0] if rows else {}
    if best:
        m = best["ofp_metrics"]
        print(
            f"[best] {best['mode']} w={best['window']} thr={best['threshold']:.2f} "
            f"F1={m['f1_score']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"final={m['final_score']:.4f}"
        )


if __name__ == "__main__":
    main()
