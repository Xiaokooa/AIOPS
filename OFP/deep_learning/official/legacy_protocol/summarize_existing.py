from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


README_REFERENCE = {
    "rf_thresh0.5": {
        "final_score": 1.457455497,
        "f1_score": 0.573217391,
        "precision": 1.0,
        "recall": 0.401755241,
        "all_hit_cnt": 1648,
        "all_predict_pos_cnt": 1648,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 0.06730178,
        "accuracy": 0.816482202,
    },
    "model1_xgb_ahead120": {
        "final_score": 2.057151253,
        "f1_score": 0.255344418,
        "precision": 0.678947368,
        "recall": 0.157240371,
        "all_hit_cnt": 645,
        "all_predict_pos_cnt": 950,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 68.44573643,
        "accuracy": 0.718665869,
    },
    "teacher_model1_original": {
        "final_score": 2.057151253,
        "f1_score": 0.255344418,
        "precision": 0.678947368,
        "recall": 0.157240371,
        "all_hit_cnt": 645,
        "all_predict_pos_cnt": 950,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 68.44573643,
        "accuracy": 0.718665869,
    },
    "rf0.5_plus_model1_xgb": {
        "final_score": 2.452086322,
        "f1_score": 0.629664179,
        "precision": 0.869098712,
        "recall": 0.493661628,
        "all_hit_cnt": 2025,
        "all_predict_pos_cnt": 2330,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 21.79598217,
        "accuracy": 0.821866587,
    },
    "rf0.5_plus_model1_xgb_plus_rule": {
        "final_score": 2.451913658,
        "f1_score": 0.629566299,
        "precision": 0.868725869,
        "recall": 0.493661628,
        "all_hit_cnt": 2025,
        "all_predict_pos_cnt": 2331,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 21.79598217,
        "accuracy": 0.821791804,
    },
    "rf0.5_plus_teacher_model1_original": {
        "final_score": 2.452086322,
        "f1_score": 0.629664179,
        "precision": 0.869098712,
        "recall": 0.493661628,
        "all_hit_cnt": 2025,
        "all_predict_pos_cnt": 2330,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 21.79598217,
        "accuracy": 0.821866587,
    },
    "rf0.5_plus_teacher_model1_original_plus_rule": {
        "final_score": 2.451913658,
        "f1_score": 0.629566299,
        "precision": 0.868725869,
        "recall": 0.493661628,
        "all_hit_cnt": 2025,
        "all_predict_pos_cnt": 2331,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 21.79598217,
        "accuracy": 0.821791804,
    },
    "rf_thresh0.3": {
        "final_score": 1.474407508,
        "f1_score": 0.573465484,
        "precision": 1.0,
        "recall": 0.401999025,
        "all_hit_cnt": 1649,
        "all_predict_pos_cnt": 1649,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 0.084026683,
        "accuracy": 0.816556985,
    },
    "rf0.3_plus_model1_xgb": {
        "final_score": 2.452086322,
        "f1_score": 0.629664179,
        "precision": 0.869098712,
        "recall": 0.493661628,
        "all_hit_cnt": 2025,
        "all_predict_pos_cnt": 2330,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 21.79598217,
        "accuracy": 0.821866587,
    },
    "rf0.3_plus_teacher_model1_original": {
        "final_score": 2.452086322,
        "f1_score": 0.629664179,
        "precision": 0.869098712,
        "recall": 0.493661628,
        "all_hit_cnt": 2025,
        "all_predict_pos_cnt": 2330,
        "all_true_pos_cnt": 4102,
        "avg_lead_hour": 21.79598217,
        "accuracy": 0.821866587,
    },
}


def fold_metric_dict(summary_path: Path) -> dict[str, float]:
    summary_df = pd.read_csv(summary_path)
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


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


def collect_model(model_dir: Path, folds: list[int]) -> tuple[list[dict], dict[str, float]] | None:
    fold_rows: list[dict] = []
    details: list[pd.DataFrame] = []
    for fold in folds:
        eval_dir = model_dir / f"fold_{fold}" / "evaluation"
        summary_path = eval_dir / "evaluate_result.csv"
        detail_path = eval_dir / "module_decisions.csv"
        if not summary_path.exists() or not detail_path.exists():
            return None
        row = {"model": model_dir.name, "fold": int(fold)}
        row.update(fold_metric_dict(summary_path))
        fold_rows.append(row)

        detail_df = pd.read_csv(detail_path)
        detail_df = detail_df.copy()
        detail_df["fold"] = int(fold)
        details.append(detail_df)

    pooled_detail = pd.concat(details, ignore_index=True)
    pooled_path = model_dir / "pooled_module_decisions.csv"
    pooled_detail.to_csv(pooled_path, index=False)
    pooled = evaluate_detail_frame(pooled_detail)
    pooled["model"] = model_dir.name
    return fold_rows, pooled


def build_readme_comparison(pooled_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in pooled_df.iterrows():
        model = str(row["model"])
        reference = README_REFERENCE.get(model)
        if not reference:
            continue
        out = {"model": model}
        for metric, ref_value in reference.items():
            value = float(row[metric])
            out[f"{metric}_ours"] = value
            out[f"{metric}_readme"] = float(ref_value)
            out[f"{metric}_diff"] = value - float(ref_value)
        rows.append(out)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed legacy-compatible OFP fold evaluations."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/ofp_legacy_protocol/readme_repro"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--models", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    if args.models:
        requested = set(args.models)
        model_dirs = [path for path in model_dirs if path.name in requested]

    all_fold_rows: list[dict] = []
    pooled_rows: list[dict] = []
    skipped: list[str] = []
    for model_dir in model_dirs:
        result = collect_model(model_dir, args.folds)
        if result is None:
            skipped.append(model_dir.name)
            continue
        fold_rows, pooled = result
        all_fold_rows.extend(fold_rows)
        pooled_rows.append(pooled)

    if not pooled_rows:
        raise RuntimeError(f"No complete models found under {args.root}")

    fold_df = pd.DataFrame(all_fold_rows).sort_values(["model", "fold"])
    pooled_df = pd.DataFrame(pooled_rows).sort_values("model")
    cols = ["model", *[col for col in pooled_df.columns if col != "model"]]
    pooled_df = pooled_df[cols]

    fold_path = args.root / "fold_metrics.csv"
    pooled_path = args.root / "readme_style_metrics.csv"
    compare_path = args.root / "readme_reference_comparison.csv"
    fold_df.to_csv(fold_path, index=False)
    pooled_df.to_csv(pooled_path, index=False)
    build_readme_comparison(pooled_df).to_csv(compare_path, index=False)

    print(f"[summarize] complete_models={len(pooled_df)}")
    if skipped:
        print(f"[summarize] skipped_incomplete={','.join(skipped)}")
    print(f"[summarize] fold_metrics={fold_path}")
    print(f"[summarize] readme_style_metrics={pooled_path}")
    print(f"[summarize] readme_reference_comparison={compare_path}")


if __name__ == "__main__":
    main()
