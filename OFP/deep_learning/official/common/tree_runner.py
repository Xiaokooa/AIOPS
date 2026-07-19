from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from OFP.deep_learning.official.common.index_split import read_index
from OFP.deep_learning.official.common.ofp_model2_baselines import aggregate_model2_results, run_ofp_model2_fold


def run_tree_model_cli(model_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run OFP-index model2 benchmark for {model_name}.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_official_results/ofp_index_legacy_ts_benchmark"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--xgb_estimators", type=int, default=10)
    parser.add_argument("--rf_threshold", type=float, default=0.5)
    parser.add_argument("--xgb_threshold", type=float, default=0.5)
    args = parser.parse_args()

    index_df = read_index(args.index_path)
    results = []
    for fold in args.folds:
        result = run_ofp_model2_fold(
            model_name=model_name,
            fold=int(fold),
            data_dir=args.data_dir,
            index_df=index_df,
            out_root=args.out_root,
            n_jobs=args.n_jobs,
            rf_estimators=args.rf_estimators,
            xgb_estimators=args.xgb_estimators,
            rf_threshold=args.rf_threshold,
            xgb_threshold=args.xgb_threshold,
        )
        results.append(result)
        aggregate_model2_results([result], args.out_root)
        print(
            f"[official done] model={model_name} fold={fold} "
            f"final={result.metrics.get('final_score', 0):.4f} "
            f"f1={result.metrics.get('f1_score', 0):.4f} "
            f"precision={result.metrics.get('precision', 0):.4f} "
            f"recall={result.metrics.get('recall', 0):.4f}"
        )
    aggregate_model2_results(results, args.out_root)
