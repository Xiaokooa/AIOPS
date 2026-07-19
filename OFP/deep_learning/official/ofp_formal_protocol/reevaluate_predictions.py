from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder


FOLD_RE = re.compile(r"fold_(\d+)$")


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def reevaluate_tree(prediction_root: Path, label_dir: Path, out_root: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(path for path in prediction_root.iterdir() if path.is_dir()):
        for fold_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            match = FOLD_RE.match(fold_dir.name)
            if not match:
                continue
            pred_dir = fold_dir / "predictions"
            if not pred_dir.exists():
                continue
            fold = int(match.group(1))
            print(f"[reeval] model={model_dir.name} fold={fold}")
            summary_df, detail_df = evaluate_prediction_folder(pred_dir, label_dir)
            out_dir = out_root / model_dir.name / fold_dir.name / "evaluation"
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(out_dir / "evaluate_result.csv", index=False)
            detail_df.to_csv(out_dir / "module_decisions.csv", index=False)
            row = {"model": model_dir.name, "fold": fold}
            row.update(metrics_to_dict(summary_df))
            rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "fold_metrics.csv", index=False)
    if not df.empty:
        numeric = [c for c in df.columns if c not in {"model", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
        summary = df.groupby("model")[numeric].agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate saved prediction folders with the current OFP evaluator.")
    parser.add_argument("--prediction_root", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--out_root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = reevaluate_tree(args.prediction_root, args.label_dir, args.out_root)
    print(f"[reeval] rows={len(df)}")
    print(f"[reeval] fold_metrics={args.out_root / 'fold_metrics.csv'}")
    print(f"[reeval] mean_std={args.out_root / 'model_metrics_mean_std.csv'}")


if __name__ == "__main__":
    main()
