"""Run the fair CE versus CE+FGL objective ablation in isolated processes."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
EXPERIMENTS = {
    "N0_native_s2s_ce": ["--disable-fgl"],
    "N1_native_s2s_ce_fgl": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native sequence CE/FGL objective ablations."
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs" / "default.json")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=BASE_DIR / "artifacts" / "objective_suite")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for experiment in args.experiments:
        output_dir = args.artifacts_dir / experiment
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
            str(output_dir),
            "--device",
            args.device,
            "--experiment-name",
            experiment,
            *EXPERIMENTS[experiment],
        ]
        if args.batch_size is not None:
            command.extend(("--batch-size", str(args.batch_size)))
        if args.inference_batch_size is not None:
            command.extend(("--inference-batch-size", str(args.inference_batch_size)))
        if args.smoke:
            command.append("--smoke")
        if args.overwrite:
            command.append("--overwrite")
        print(f"[suite] running {experiment}", flush=True)
        subprocess.run(command, check=True, cwd=str(WORKSPACE_ROOT))
        rows.append(pd.read_csv(output_dir / "result.csv"))
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(args.artifacts_dir / "comparison.csv", index=False)
    print(args.artifacts_dir / "comparison.csv", flush=True)


if __name__ == "__main__":
    main()
