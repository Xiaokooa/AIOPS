from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_external_adapters.common import (
    AlarmCfg,
    aggregate_results,
    apply_alarm_strategy,
    evaluate_prediction_folder,
    file_label_map,
    files_for_index_fold,
    load_score_frame,
    metrics_to_dict,
    read_index,
    select_threshold,
    split_train_val_files,
    write_prediction_frames,
)


def run_fold(
    model_name: str,
    protocol: str,
    score_dir: Path,
    data_dir: Path,
    index_df: pd.DataFrame,
    fold: int,
    out_root: Path,
    cfg: AlarmCfg,
    score_col: str,
) -> dict:
    started = time.time()
    label_by_file = file_label_map(index_df)
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    _train_files, val_files = split_train_val_files(train_files, label_by_file, cfg, int(fold))
    run_dir = Path(out_root) / model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[score-adapter] model={model_name} fold={fold} val={len(val_files)} test={len(test_files)}")
    val_scores = {name: load_score_frame(score_dir, name, score_col=score_col) for name in val_files}
    threshold, val_metrics, search_df = select_threshold(val_scores, data_dir, cfg)
    val_dir = run_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(val_dir / "threshold_search.csv", index=False)
    pd.DataFrame({"Item": list(val_metrics.keys()), "Value": list(val_metrics.values())}).to_csv(
        val_dir / "best_evaluate_result.csv",
        index=False,
    )

    test_score_frames = {name: load_score_frame(score_dir, name, score_col=score_col) for name in test_files}
    pred_frames = {name: apply_alarm_strategy(frame, threshold, cfg) for name, frame in test_score_frames.items()}
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    test_rows = write_prediction_frames(pred_frames, pred_dir)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)
    result = {
        "model": model_name,
        "fold": int(fold),
        "protocol": protocol,
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_rows": int(test_rows),
        "seconds": time.time() - started,
        "metrics": metrics,
        "alarm_cfg": asdict(cfg),
        "score_dir": str(score_dir),
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[score-adapter-done] model={model_name} fold={fold} threshold={threshold:.4f} "
        f"f1={metrics.get('f1_score', 0.0):.5f} final={metrics.get('final_score', 0.0):.5f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert external score CSVs to OFP timestamp,predict outputs.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--protocol", default="external_score_adapter")
    parser.add_argument("--score_dir", type=Path, required=True)
    parser.add_argument("--score_col", default="score")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_external_adapter_results/score_adapter"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--threshold_grid", default=AlarmCfg.threshold_grid)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--smoothing", choices=["none", "ema", "mean", "rolling"], default="ema")
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--consecutive_k", type=int, default=1)
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    cfg = AlarmCfg(
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        val_fraction=args.val_fraction,
        smoothing=args.smoothing,
        smooth_window=args.smooth_window,
        consecutive_k=args.consecutive_k,
        first_alarm_only=not args.no_first_alarm_only,
        seed=args.seed,
    )
    results = []
    for fold in args.folds:
        result = run_fold(
            args.model_name,
            args.protocol,
            args.score_dir,
            args.data_dir,
            index_df,
            int(fold),
            args.out_root,
            cfg,
            args.score_col,
        )
        results.append(result)
        aggregate_results([result], args.out_root)
    aggregate_results(results, args.out_root)


if __name__ == "__main__":
    main()
