"""Run the protocol-matched static XGBoost control on three OFP folds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
DEFAULT_EXPERIMENT_NAME = "S0_static_xgb_current_row_3fold"
DEFAULT_OUTPUT_DIR = BASE_DIR / "artifacts" / DEFAULT_EXPERIMENT_NAME
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fgofp.static_xgb import run_static_xgb_threefold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Protocol-matched static current-row XGBoost control with exact "
            "three-fold OOF module evaluation."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training"
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--nthread", type=int, default=4)
    parser.add_argument("--num-boost-round", type=int, default=100)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--max-train-modules", type=int, default=None)
    parser.add_argument("--max-validation-modules", type=int, default=None)
    parser.add_argument("--max-test-modules", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_static_xgb_threefold(
        data_dir=args.data_dir,
        index_path=args.index_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        device=args.device,
        smoke=args.smoke,
        overwrite=args.overwrite,
        nthread=args.nthread,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        max_train_modules=args.max_train_modules,
        max_validation_modules=args.max_validation_modules,
        max_test_modules=args.max_test_modules,
        progress=True,
    )
    print(comparison.to_string(index=False), flush=True)
    print(f"[static-xgb] comparison={args.output_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
