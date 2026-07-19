from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP.deep_learning.official.ofp_protocol.run_ofp_baselines import SENSORS, metrics_to_dict, read_index


@dataclass
class FoldSummary:
    model: str
    fold: int
    threshold: float
    model_path: str
    test_modules: int
    test_rows: int
    seconds: float
    metrics: dict[str, float]


def predict_fold(
    booster: xgb.Booster,
    data_dir: Path,
    file_names: list[str],
    pred_dir: Path,
    threshold: float,
) -> int:
    pred_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for file_name in file_names:
        raw = pd.read_csv(data_dir / file_name)
        timestamps = pd.to_numeric(raw["timestamp"], errors="coerce").astype("int64")
        features = raw[SENSORS].apply(pd.to_numeric, errors="coerce")
        scores = booster.predict(xgb.DMatrix(features))
        pred = (scores >= threshold).astype(int)
        total_rows += len(raw)
        pd.DataFrame({"timestamp": timestamps, "predict": pred}).to_csv(
            pred_dir / file_name,
            index=False,
        )
    return total_rows


def run_model(
    model_name: str,
    model_path: Path,
    data_dir: Path,
    index_path: Path,
    out_root: Path,
    folds: list[int],
    threshold: float,
) -> None:
    index_df = read_index(index_path)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    fold_rows = []
    for fold in folds:
        t0 = time.time()
        file_names = list(index_df.loc[index_df["folder_index"] == fold, "file_name"])
        model_dir = out_root / model_name / f"fold_{fold}"
        pred_dir = model_dir / "predictions"
        eval_dir = model_dir / "evaluation"
        print(f"[teacher-model1] model={model_name} fold={fold} files={len(file_names)}")
        test_rows = predict_fold(booster, data_dir, file_names, pred_dir, threshold)
        summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
        detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
        summary = FoldSummary(
            model=model_name,
            fold=int(fold),
            threshold=float(threshold),
            model_path=str(model_path),
            test_modules=len(file_names),
            test_rows=int(test_rows),
            seconds=time.time() - t0,
            metrics=metrics_to_dict(summary_df),
        )
        (model_dir / "fold_summary.json").write_text(
            json.dumps(asdict(summary), indent=2),
            encoding="utf-8",
        )
        fold_rows.append(summary)
        print(
            f"[teacher-model1 done] fold={fold} "
            f"F1={summary.metrics['f1_score']:.6f} "
            f"P={summary.metrics['precision']:.6f} "
            f"R={summary.metrics['recall']:.6f} "
            f"hits={summary.metrics['all_hit_cnt']:.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate teacher-provided OFP model1 JSON checkpoints.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument(
        "--index_path",
        type=Path,
        default=Path("dataset/train_test_set_index(in).csv"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("output/ofp_legacy_protocol/readme_repro"),
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Model name and XGBoost JSON path. Can be repeated.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model_name, model_path in args.model:
        run_model(
            model_name=model_name,
            model_path=Path(model_path),
            data_dir=args.data_dir,
            index_path=args.index_path,
            out_root=args.out_root,
            folds=args.folds,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
