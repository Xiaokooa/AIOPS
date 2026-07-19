from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from OFP.deep_learning.optical_prediction.common import base_utils as base
from OFP.deep_learning.optical_prediction.common import cache_utils
from OFP.deep_learning.optical_prediction.common import task_utils as week1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared optical-prediction caches")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--long_table", action="store_true")
    parser.add_argument("--window_index", action="store_true")
    parser.add_argument("--batch_modules", type=int, default=256)
    parser.add_argument("--obs_minutes", type=int, default=30)
    parser.add_argument("--lead_minutes", type=int, default=15)
    parser.add_argument("--pred_minutes", type=int, default=60)
    parser.add_argument("--step_minutes", type=int, default=15)
    parser.add_argument("--train_scope", type=str, default="prefirst", choices=["full", "prefirst"])
    parser.add_argument("--eval_scope", type=str, default="prefirst", choices=["full", "prefirst"])
    parser.add_argument(
        "--train_label_mode",
        type=str,
        default="first_in_future",
        choices=["future_any", "first_in_future", "first_in_prediction"],
    )
    parser.add_argument(
        "--eval_label_mode",
        type=str,
        default="first_in_prediction",
        choices=["future_any", "first_in_future", "first_in_prediction"],
    )
    args = parser.parse_args()

    index_df = pd.read_csv(base.INDEX_FILE)
    summary = cache_utils.prepare_resampled_cache(
        index_df,
        force_rebuild=args.force_rebuild,
        limit=args.limit,
    )
    print("Shared cache:", cache_utils.RESAMPLED_CACHE_DIR)
    print(summary)

    if args.long_table:
        long_meta = cache_utils.build_long_table_cache(
            index_df,
            force_rebuild=args.force_rebuild,
            limit=args.limit,
            batch_modules=args.batch_modules,
        )
        print("Long table:", cache_utils.LONG_TABLE_PATH)
        print(long_meta)

    if args.window_index:
        cfg = week1.Week1TaskConfig(
            obs_minutes=args.obs_minutes,
            lead_minutes=args.lead_minutes,
            pred_minutes=args.pred_minutes,
            step_minutes=args.step_minutes,
            train_scope=args.train_scope,
            eval_scope=args.eval_scope,
            train_label_mode=args.train_label_mode,
            eval_label_mode=args.eval_label_mode,
        )
        split_map = base.split_modules_stratified()
        index_summary = {}
        for split_name, split_df in split_map.items():
            if args.limit is not None:
                split_df = split_df.head(args.limit).reset_index(drop=True)
            cache_path = cache_utils.window_index_cache_path(
                split_name,
                split_df,
                cfg.window_index_cache_tag,
            )
            frame = week1.build_window_index_frame(
                split_df,
                cfg,
                split_role=split_name,
                cache_path=cache_path,
                force_rebuild=args.force_rebuild,
            )
            index_summary[split_name] = {
                "path": str(cache_path),
                "windows": int(len(frame)),
                "positive_windows": int(frame["label"].sum()) if not frame.empty else 0,
                "modules": int(frame["file_name"].nunique()) if not frame.empty else 0,
            }
        print("Window index:", cache_utils.WINDOW_INDEX_CACHE_DIR)
        print(index_summary)


if __name__ == "__main__":
    main()
