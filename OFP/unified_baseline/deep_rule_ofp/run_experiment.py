"""Train and evaluate one DRFP variant on the fixed fold-3 OFP split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = BASE_DIR.parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
for search_path in (BASE_DIR, UNIFIED_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from drfp.config import load_config
from drfp.pipeline import prepare_data, run_prepared_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Deep-Rule Optical Failure Prediction experiment."
    )
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs" / "default.json")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts" / "main")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--max-train-modules", type=int, default=None)
    parser.add_argument("--max-validation-modules", type=int, default=None)
    parser.add_argument("--max-test-modules", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-long-scores", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict:
    override: dict = {}
    if args.device is not None:
        override.setdefault("training", {})["device"] = args.device
    if args.smoke:
        override.update(
            {
                "sampling": {"endpoints_per_module": 4},
                "model": {
                    "d_model": 32,
                    "latent_dim": 32,
                    "transformer_layers": 1,
                    "attention_heads": 4,
                    "patch_length": 8,
                    "patch_stride": 8,
                    "statistical_hidden": 48,
                    "rule_hidden": 48,
                },
                "training": {
                    **override.get("training", {}),
                    "epochs": 1,
                    "batch_size": 32,
                    "inference_batch_size": 256,
                    "early_stopping_patience": 0,
                    "num_workers": 0,
                },
                "calibration": {"grid_size": 21},
                "decision": {"threshold_candidates": 21},
            }
        )
    if args.inference_batch_size is not None:
        override.setdefault("training", {})["inference_batch_size"] = int(
            args.inference_batch_size
        )
    return override


def main() -> None:
    args = parse_args()
    config = load_config(args.config, _overrides(args))
    max_train = args.max_train_modules
    max_validation = args.max_validation_modules
    max_test = args.max_test_modules
    if args.smoke:
        max_train = 24 if max_train is None else max_train
        max_validation = 8 if max_validation is None else max_validation
        max_test = 8 if max_test is None else max_test
    prepared = prepare_data(
        config,
        args.data_dir,
        args.index_path,
        max_train_modules=max_train,
        max_validation_modules=max_validation,
        max_test_modules=max_test,
        progress=True,
    )
    run_prepared_experiment(
        prepared,
        config,
        args.output_dir,
        overwrite=args.overwrite,
        save_long_scores=args.save_long_scores,
    )


if __name__ == "__main__":
    main()
