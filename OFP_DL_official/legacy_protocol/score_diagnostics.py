from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.run_deep_models import set_seed, split_train_val
from OFP_DL_official.ofp_protocol.run_ofp_baselines import read_index
from OFP_DL_official.legacy_protocol.run_deep_models_module_level import (
    FeatureModuleArrayCache,
    ModuleLevelCfg,
    choose_ofp_threshold,
    collect_module_scores_module_level,
    compute_feature_norm_stats,
    evaluate_detail_from_scores,
    make_module_model,
)

SEC_IN_HOUR = 3600.0


def cfg_from_checkpoint(raw: dict, device: str) -> ModuleLevelCfg:
    valid = set(ModuleLevelCfg.__dataclass_fields__.keys())
    item = {k: v for k, v in raw.items() if k in valid}
    item["device"] = device
    item["horizon_hours"] = tuple(float(x) for x in item.get("horizon_hours", [item.get("positive_horizon_hours", 120.0)]))
    item["horizon_fusion_weights"] = tuple(float(x) for x in item.get("horizon_fusion_weights", []))
    item["lead_time_bins"] = tuple(float(x) for x in item.get("lead_time_bins", [0.0, 16.0, 24.0, 72.0, 120.0]))
    item["rolling_windows"] = tuple(int(x) for x in item.get("rolling_windows", [12, 36]))
    return ModuleLevelCfg(**item)


def quantile_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(values.max()),
    }


def trace_score_distribution(module_rows: list[dict], bins: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    module_records = []
    lead_records = []
    sorted_bins = sorted(set(float(x) for x in bins))
    for row in module_rows:
        scores = np.asarray(row["scores"], dtype=np.float64)
        timestamps = np.asarray(row["timestamps"], dtype=np.float64)
        true_label = int(row["true_label"])
        true_ts = row["true_ts"]

        pre_scores = np.asarray([], dtype=np.float64)
        post_scores = np.asarray([], dtype=np.float64)
        if true_label and true_ts is not None:
            pre_mask = timestamps < float(true_ts)
            post_mask = timestamps >= float(true_ts)
            pre_scores = scores[pre_mask]
            post_scores = scores[post_mask]

        module_records.append(
            {
                "file_name": row["file_name"],
                "true_label": true_label,
                "trace_max_score": float(np.nanmax(scores)) if len(scores) else float("nan"),
                "pre_event_max_score": float(np.nanmax(pre_scores)) if len(pre_scores) else float("nan"),
                "post_event_max_score": float(np.nanmax(post_scores)) if len(post_scores) else float("nan"),
                "n_timestamps": int(len(scores)),
            }
        )

        if true_label and true_ts is not None and len(sorted_bins) >= 2:
            lead_hours = (float(true_ts) - timestamps) / SEC_IN_HOUR
            for lo, hi in zip(sorted_bins[:-1], sorted_bins[1:]):
                mask = (lead_hours > lo) & (lead_hours <= hi)
                vals = scores[mask]
                lead_records.append(
                    {
                        "file_name": row["file_name"],
                        "lead_bin": f"({lo:g},{hi:g}]",
                        "lead_lo": lo,
                        "lead_hi": hi,
                        "has_samples": int(len(vals) > 0),
                        "max_score": float(np.nanmax(vals)) if len(vals) else float("nan"),
                        "n_samples": int(len(vals)),
                    }
                )
    return pd.DataFrame(module_records), pd.DataFrame(lead_records)


def lead_bin_recall(lead_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if lead_df.empty:
        return pd.DataFrame()
    rows = []
    for name, group in lead_df.groupby("lead_bin", sort=False):
        valid = group[group["has_samples"] > 0].copy()
        denom = len(valid)
        hit = int((valid["max_score"].astype(float) >= float(threshold)).sum()) if denom else 0
        rows.append(
            {
                "lead_bin": name,
                "modules_with_samples": denom,
                "bin_hit_modules": hit,
                "bin_recall": hit / denom if denom else 0.0,
                "max_score_mean": float(valid["max_score"].mean()) if denom else float("nan"),
                "max_score_p90": float(valid["max_score"].quantile(0.90)) if denom else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def diagnostic_summary(
    split_name: str,
    rows: list[dict],
    cfg: ModuleLevelCfg,
    threshold: float,
    lead_bins: tuple[float, ...],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = evaluate_detail_from_scores(rows, threshold, cfg)
    oracle_threshold, oracle_metrics = choose_ofp_threshold(rows, cfg)
    module_df, lead_df = trace_score_distribution(rows, lead_bins)

    healthy = module_df[module_df["true_label"] == 0]["trace_max_score"].to_numpy(dtype=float)
    faulty_pre = module_df[module_df["true_label"] == 1]["pre_event_max_score"].dropna().to_numpy(dtype=float)
    faulty_post = module_df[module_df["true_label"] == 1]["post_event_max_score"].dropna().to_numpy(dtype=float)
    summary = {
        "split": split_name,
        "threshold": float(threshold),
        "oracle_threshold": float(oracle_threshold),
        "threshold_final": float(metrics["final_score"]),
        "threshold_f1": float(metrics["f1_score"]),
        "threshold_precision": float(metrics["precision"]),
        "threshold_recall": float(metrics["recall"]),
        "threshold_hit": int(metrics["all_hit_cnt"]),
        "threshold_pred_pos": int(metrics["all_predict_pos_cnt"]),
        "oracle_final": float(oracle_metrics["final_score"]),
        "oracle_f1": float(oracle_metrics["f1_score"]),
        "oracle_precision": float(oracle_metrics["precision"]),
        "oracle_recall": float(oracle_metrics["recall"]),
        "oracle_hit": int(oracle_metrics["all_hit_cnt"]),
        "oracle_pred_pos": int(oracle_metrics["all_predict_pos_cnt"]),
    }
    summary.update(quantile_summary(healthy, "healthy_trace_max"))
    summary.update(quantile_summary(faulty_pre, "faulty_pre_event_max"))
    summary.update(quantile_summary(faulty_post, "faulty_post_event_max"))
    bin_df = lead_bin_recall(lead_df, threshold)
    return summary, module_df, lead_df, bin_df


def run(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.model_path, map_location=args.device)
    cfg = cfg_from_checkpoint(checkpoint["run_cfg"], args.device)
    summary_path = args.model_path.with_name("fold_summary.json")
    saved_summary = {}
    if summary_path.exists():
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(saved_summary.get("threshold", checkpoint.get("threshold", 0.5)))
    if args.trigger_mode is not None:
        cfg.trigger_mode = args.trigger_mode
    if args.trigger_k is not None:
        cfg.trigger_k = int(args.trigger_k)
    if args.smooth_window is not None:
        cfg.smooth_window = int(args.smooth_window)

    set_seed(cfg.seed + args.fold)
    index_df = read_index(args.index_path)
    train_files, val_files, test_files = split_train_val(index_df, args.fold, cfg.val_fraction, cfg.seed)

    cache = FeatureModuleArrayCache(
        args.data_dir,
        max_cached_files=cfg.max_cached_files,
        feature_mode=cfg.feature_mode,
        rolling_windows=tuple(int(x) for x in cfg.rolling_windows),
    )
    mean, std = compute_feature_norm_stats(
        args.data_dir,
        train_files,
        feature_mode=cfg.feature_mode,
        rolling_windows=tuple(int(x) for x in cfg.rolling_windows),
        max_files=cfg.norm_max_files,
    )
    model, model_cfg = make_module_model(args.model_name, cfg.seq_len, len(cache.feature_names), cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(cfg.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_files = {"val": val_files, "test": test_files}
    summaries = []
    for split_name, files in split_files.items():
        if args.max_files is not None:
            files = files[: int(args.max_files)]
        print(f"[diagnostic] scoring {split_name} modules={len(files)}")
        rows = collect_module_scores_module_level(model, args.data_dir, files, cache, cfg, mean, std)
        summary, module_df, lead_df, bin_df = diagnostic_summary(
            split_name,
            rows,
            cfg,
            threshold=threshold,
            lead_bins=tuple(float(x) for x in args.lead_bins),
        )
        summaries.append(summary)
        module_df.to_csv(args.out_dir / f"{split_name}_module_score_distribution.csv", index=False)
        lead_df.to_csv(args.out_dir / f"{split_name}_lead_bin_scores.csv", index=False)
        bin_df.to_csv(args.out_dir / f"{split_name}_lead_bin_recall.csv", index=False)
        print(
            f"[diagnostic] {split_name} F1={summary['threshold_f1']:.4f} "
            f"oracleF1={summary['oracle_f1']:.4f} "
            f"healthyP95={summary['healthy_trace_max_p95']:.4f} "
            f"faultyPreP50={summary['faulty_pre_event_max_p50']:.4f}"
        )

    pd.DataFrame(summaries).to_csv(args.out_dir / "score_diagnostic_summary.csv", index=False)
    (args.out_dir / "diagnostic_config.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "model_cfg": model_cfg,
                "run_cfg": asdict(cfg),
                "threshold": threshold,
                "lead_bins": args.lead_bins,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score diagnostics for OFP module-level deep checkpoints.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--model_name", default="itransformer_ofp")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--lead_bins", nargs="+", type=float, default=[0.0, 16.0, 24.0, 72.0, 120.0])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--trigger_mode", choices=["point", "smooth", "consecutive", "smooth_consecutive"], default=None)
    parser.add_argument("--trigger_k", type=int, default=None)
    parser.add_argument("--smooth_window", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

