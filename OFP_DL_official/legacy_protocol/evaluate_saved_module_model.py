from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_deep_models import set_seed, split_train_val
from OFP_DL_official.ofp_protocol.run_ofp_baselines import read_index
from OFP_DL_official.legacy_protocol.run_deep_models_module_level import (
    FeatureModuleArrayCache,
    ModuleLevelCfg,
    choose_ofp_threshold,
    collect_module_scores_module_level,
    compute_feature_norm_stats,
    make_module_model,
    metrics_to_dict,
    write_predictions_with_trigger,
)


def cfg_from_dict(raw: dict, device: str) -> ModuleLevelCfg:
    valid = set(ModuleLevelCfg.__dataclass_fields__.keys())
    item = {k: v for k, v in raw.items() if k in valid}
    item["device"] = device
    item["horizon_hours"] = tuple(float(x) for x in item.get("horizon_hours", [item.get("positive_horizon_hours", 120.0)]))
    item["horizon_fusion_weights"] = tuple(float(x) for x in item.get("horizon_fusion_weights", []))
    item["lead_time_bins"] = tuple(float(x) for x in item.get("lead_time_bins", [0.0, 16.0, 24.0, 72.0, 120.0]))
    item["rolling_windows"] = tuple(int(x) for x in item.get("rolling_windows", [12, 36]))
    return ModuleLevelCfg(**item)


def run_trigger_eval(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.model_path, map_location=args.device)
    cfg = cfg_from_dict(checkpoint["run_cfg"], args.device)
    set_seed(cfg.seed + args.fold)

    index_df = read_index(args.index_path)
    train_files, val_files, test_files = split_train_val(index_df, args.fold, cfg.val_fraction, cfg.seed)

    rolling_windows = tuple(int(win) for win in cfg.rolling_windows)
    cache = FeatureModuleArrayCache(
        args.data_dir,
        max_cached_files=cfg.max_cached_files,
        feature_mode=cfg.feature_mode,
        rolling_windows=rolling_windows,
    )
    mean, std = compute_feature_norm_stats(
        args.data_dir,
        train_files,
        feature_mode=cfg.feature_mode,
        rolling_windows=rolling_windows,
        max_files=cfg.norm_max_files,
    )

    model, model_cfg = make_module_model(args.model_name, cfg.seq_len, len(cache.feature_names), cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg.device)

    print(f"[eval-saved] scoring val modules={len(val_files)}")
    val_rows = collect_module_scores_module_level(model, args.data_dir, val_files, cache, cfg, mean, std)
    print(f"[eval-saved] scoring test modules={len(test_files)}")
    test_rows = collect_module_scores_module_level(model, args.data_dir, test_files, cache, cfg, mean, std)

    variants = [
        ("point", "point", 1, 1),
        ("smooth3", "smooth", 1, 3),
        ("consecutive2", "consecutive", 2, 1),
        ("smooth3_consecutive2", "smooth_consecutive", 2, 3),
    ]
    rows = []
    for name, trigger_mode, trigger_k, smooth_window in variants:
        run_cfg = ModuleLevelCfg(**asdict(cfg))
        run_cfg.trigger_mode = trigger_mode
        run_cfg.trigger_k = trigger_k
        run_cfg.smooth_window = smooth_window
        threshold, val_metrics = choose_ofp_threshold(val_rows, run_cfg)
        pred_dir = args.out_dir / name / "predictions"
        eval_dir = args.out_dir / name / "evaluation"
        write_predictions_with_trigger(test_rows, pred_dir, threshold, run_cfg)
        summary_df, detail_df = evaluate_prediction_folder(pred_dir, args.data_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
        detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
        metrics = metrics_to_dict(summary_df)
        row = {
            "variant": name,
            "trigger_mode": trigger_mode,
            "trigger_k": trigger_k,
            "smooth_window": smooth_window,
            "threshold": threshold,
            "val_f1": val_metrics["f1_score"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"[eval-saved] {name} thr={threshold:.3f} "
            f"F1={metrics.get('f1_score', 0):.4f} "
            f"P={metrics.get('precision', 0):.4f} "
            f"R={metrics.get('recall', 0):.4f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "trigger_eval_metrics.csv", index=False)
    (args.out_dir / "eval_config.json").write_text(
        json.dumps({"model_cfg": model_cfg, "run_cfg": asdict(cfg)}, indent=2, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trigger variants for a saved module-level deep model.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--model_name", default="itransformer_ofp")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    run_trigger_eval(parse_args())


if __name__ == "__main__":
    main()

