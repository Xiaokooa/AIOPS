"""Run three outer N1 folds and pool all OFP module decisions."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fgofp.threefold import OUTER_FOLDS, aggregate_three_fold_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run N1 on OFP outer folds 1/2/3 and recompute exact pooled "
            "13,372-module metrics."
        )
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs" / "default.json")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "artifacts" / "N1_threefold_full",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument(
        "--experiment-name",
        default="N1_native_s2s_ce_fgl_3fold",
    )
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_fold_command(args: argparse.Namespace, fold: int) -> list[str]:
    fold_dir = args.output_dir / f"fold_{fold}"
    command = [
        sys.executable,
        "-B",
        str(BASE_DIR / "run_experiment.py"),
        "--config",
        str(args.config),
        "--data-dir",
        str(args.data_dir),
        "--index-path",
        str(args.index_path),
        "--output-dir",
        str(fold_dir),
        "--device",
        args.device,
        "--experiment-name",
        f"{args.experiment_name}_fold{fold}",
        "--test-fold",
        str(fold),
    ]
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    if args.inference_batch_size is not None:
        command.extend(("--inference-batch-size", str(args.inference_batch_size)))
    if args.no_mixed_precision:
        command.append("--no-mixed-precision")
    if args.smoke:
        command.append("--smoke")
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in OUTER_FOLDS:
        print(f"[threefold] running outer fold {fold}", flush=True)
        subprocess.run(
            build_fold_command(args, fold),
            cwd=str(WORKSPACE_ROOT),
            check=True,
        )
    comparison = aggregate_three_fold_results(
        args.output_dir,
        args.index_path,
        experiment_id=args.experiment_name,
        formal=not args.smoke,
    )
    print(comparison.to_string(index=False), flush=True)
    print(f"[threefold] comparison={args.output_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
