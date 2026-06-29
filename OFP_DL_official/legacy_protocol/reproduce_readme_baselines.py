from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_ofp_model2_suite import merge_or


SEC_IN_HOUR = 3600.0


BASELINE_SPECS = {
    "model1_xgb_ahead120": ("existing", ["baselines/model1_xgb_ahead120"]),
    "rule": ("existing", ["model2_suite/rule"]),
    "rf_thresh0.5": ("existing", ["model2_suite/rf_thresh0.5"]),
    "rf_thresh0.3": ("existing", ["model2_suite/rf_thresh0.3"]),
    "model2_xgboost": ("existing", ["model2_suite/xgboost"]),
    "model2_rf0.5_plus_model2_xgboost": (
        "existing",
        ["model2_suite/rf0.5_plus_xgboost"],
    ),
    "model2_rf0.3_plus_model2_xgboost": (
        "existing",
        ["model2_suite/rf0.3_plus_xgboost"],
    ),
    "rf0.5_plus_model1_xgb": (
        "fusion",
        ["model2_suite/rf_thresh0.5", "baselines/model1_xgb_ahead120"],
    ),
    "rf0.3_plus_model1_xgb": (
        "fusion",
        ["model2_suite/rf_thresh0.3", "baselines/model1_xgb_ahead120"],
    ),
    "rf0.5_plus_model1_xgb_plus_rule": (
        "fusion",
        ["model2_suite/rf_thresh0.5", "baselines/model1_xgb_ahead120", "model2_suite/rule"],
    ),
    "rf0.3_plus_model1_xgb_plus_rule": (
        "fusion",
        ["model2_suite/rf_thresh0.3", "baselines/model1_xgb_ahead120", "model2_suite/rule"],
    ),
}


def source_prediction_dir(source_root: Path, spec: str, fold: int) -> Path:
    return source_root / spec / f"fold_{fold}" / "predictions"


def evaluate_detail_frame(detail_df: pd.DataFrame) -> dict[str, float]:
    true_pos = set(detail_df.loc[detail_df["true_label"] > 0, "file_name"])
    pred_pos = set(detail_df.loc[detail_df["valid_predict_positive"] > 0, "file_name"])
    hit_pos = true_pos & pred_pos
    tp = len(hit_pos)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = len(detail_df) - tp - fp - fn

    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(detail_df) if len(detail_df) else 0.0
    lead_hours = detail_df.loc[detail_df["hit"] > 0, "lead_hour"].dropna().astype(float)
    avg_lead_hour = float(lead_hours.mean()) if not lead_hours.empty else 0.0
    min_lead_hour = float(lead_hours.min()) if not lead_hours.empty else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    return {
        "final_score": final_score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "all_hit_cnt": tp,
        "all_predict_pos_cnt": len(pred_pos),
        "all_true_pos_cnt": len(true_pos),
        "avg_lead_score": avg_lead_score,
        "avg_lead_hour": avg_lead_hour,
        "min_lead_score": min_lead_score,
        "min_lead_hour": min_lead_hour,
        "lead_pread_cnt": int(detail_df["hit"].sum()),
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_module_cnt": len(detail_df),
    }


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def ensure_prediction_dir(
    model_name: str,
    mode: str,
    specs: list[str],
    fold: int,
    source_root: Path,
    out_root: Path,
) -> Path:
    if mode == "existing":
        return source_prediction_dir(source_root, specs[0], fold)
    pred_dir = out_root / model_name / f"fold_{fold}" / "predictions"
    if not pred_dir.exists() or not any(pred_dir.glob("*.csv")):
        src_dirs = [source_prediction_dir(source_root, spec, fold) for spec in specs]
        missing = [str(path) for path in src_dirs if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{model_name} fold {fold} missing source dirs: {missing}")
        merge_or(src_dirs, pred_dir)
    return pred_dir


def run_model(
    model_name: str,
    mode: str,
    specs: list[str],
    folds: list[int],
    source_root: Path,
    label_dir: Path,
    out_root: Path,
) -> tuple[list[dict], dict[str, float]]:
    fold_rows: list[dict] = []
    details: list[pd.DataFrame] = []
    for fold in folds:
        pred_dir = ensure_prediction_dir(model_name, mode, specs, fold, source_root, out_root)
        summary_df, detail_df = evaluate_prediction_folder(pred_dir, label_dir)
        eval_dir = out_root / model_name / f"fold_{fold}" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
        detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
        row = {"model": model_name, "fold": int(fold)}
        row.update(metrics_to_dict(summary_df))
        fold_rows.append(row)
        detail_df = detail_df.copy()
        detail_df["fold"] = int(fold)
        details.append(detail_df)
    pooled_detail = pd.concat(details, ignore_index=True)
    pooled_detail.to_csv(out_root / model_name / "pooled_module_decisions.csv", index=False)
    pooled = evaluate_detail_frame(pooled_detail)
    pooled["model"] = model_name
    return fold_rows, pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce README-style legacy OFP baselines.")
    parser.add_argument("--source_root", type=Path, default=Path("output/ofp_protocol"))
    parser.add_argument("--label_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_legacy_protocol/readme_repro"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--models", nargs="+", default=list(BASELINE_SPECS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_fold_rows: list[dict] = []
    pooled_rows: list[dict] = []
    for model_name in args.models:
        if model_name not in BASELINE_SPECS:
            raise ValueError(f"Unknown model {model_name!r}; choices: {sorted(BASELINE_SPECS)}")
        mode, specs = BASELINE_SPECS[model_name]
        print(f"[legacy] model={model_name} mode={mode}")
        fold_rows, pooled = run_model(
            model_name=model_name,
            mode=mode,
            specs=specs,
            folds=args.folds,
            source_root=args.source_root,
            label_dir=args.label_dir,
            out_root=args.out_root,
        )
        all_fold_rows.extend(fold_rows)
        pooled_rows.append(pooled)
        print(
            f"[legacy done] {model_name}: final={pooled['final_score']:.6f} "
            f"F1={pooled['f1_score']:.6f} P={pooled['precision']:.6f} "
            f"R={pooled['recall']:.6f} hits={pooled['all_hit_cnt']}"
        )

    fold_df = pd.DataFrame(all_fold_rows).sort_values(["model", "fold"])
    pooled_df = pd.DataFrame(pooled_rows)
    cols = ["model", *[c for c in pooled_df.columns if c != "model"]]
    pooled_df = pooled_df[cols]
    fold_df.to_csv(args.out_root / "fold_metrics.csv", index=False)
    pooled_df.to_csv(args.out_root / "readme_style_metrics.csv", index=False)
    print(f"[legacy] fold_metrics={args.out_root / 'fold_metrics.csv'}")
    print(f"[legacy] readme_style_metrics={args.out_root / 'readme_style_metrics.csv'}")


if __name__ == "__main__":
    main()


