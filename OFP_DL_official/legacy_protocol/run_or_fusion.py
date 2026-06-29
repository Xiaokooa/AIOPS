from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_ofp_baselines import metrics_to_dict
from OFP_DL_official.ofp_protocol.run_ofp_model2_suite import merge_or


@dataclass
class FusionSummary:
    model: str
    fold: int
    source_dirs: list[str]
    test_rows: int
    seconds: float
    metrics: dict[str, float]


def resolve_template(template: str, fold: int) -> Path:
    return Path(template.format(fold=fold))


def run_fusion(
    model_name: str,
    source_templates: list[str],
    folds: list[int],
    out_root: Path,
    label_dir: Path,
) -> None:
    for fold in folds:
        source_dirs = [resolve_template(template, fold) for template in source_templates]
        missing = [str(path) for path in source_dirs if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{model_name} fold {fold} missing source dirs: {missing}")

        t0 = time.time()
        model_dir = out_root / model_name / f"fold_{fold}"
        pred_dir = model_dir / "predictions"
        eval_dir = model_dir / "evaluation"
        print(f"[fusion] model={model_name} fold={fold} sources={len(source_dirs)}")
        test_rows = merge_or(source_dirs, pred_dir)
        summary_df, detail_df = evaluate_prediction_folder(pred_dir, label_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
        detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
        summary = FusionSummary(
            model=model_name,
            fold=int(fold),
            source_dirs=[str(path) for path in source_dirs],
            test_rows=int(test_rows),
            seconds=time.time() - t0,
            metrics=metrics_to_dict(summary_df),
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "fold_summary.json").write_text(
            json.dumps(asdict(summary), indent=2),
            encoding="utf-8",
        )
        print(
            f"[fusion done] fold={fold} "
            f"F1={summary.metrics['f1_score']:.6f} "
            f"P={summary.metrics['precision']:.6f} "
            f"R={summary.metrics['recall']:.6f} "
            f"hits={summary.metrics['all_hit_cnt']:.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OR fusion for legacy OFP prediction folders.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Prediction directory template; use {fold} for the fold number.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("output/ofp_legacy_protocol/readme_repro"),
    )
    parser.add_argument("--label_dir", type=Path, default=Path("dataset/training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fusion(
        model_name=args.model_name,
        source_templates=args.source,
        folds=args.folds,
        out_root=args.out_root,
        label_dir=args.label_dir,
    )


if __name__ == "__main__":
    main()

