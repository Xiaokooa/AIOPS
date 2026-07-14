"""Train and evaluate one native sequence FGL-OFP experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fgofp.config import load_config
from fgofp.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Native-row sequence-to-sequence optical-module failure prediction "
            "with Future-window CE and Future-Guided KL."
        )
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--test-fold", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--fgl-alpha", type=float, default=None)
    parser.add_argument(
        "--positive-weight-mode",
        choices=("module_normalized_auto", "none"),
        default=None,
    )
    parser.add_argument(
        "--architecture",
        choices=("causal_depthwise_tcn", "rule_guided_residual_tcn"),
        default=None,
    )
    parser.add_argument("--rule-residual-scale", type=float, default=None)
    parser.add_argument("--fallback-min-gain", type=float, default=None)
    parser.add_argument("--disable-safe-rule-fallback", action="store_true")
    parser.add_argument("--future-offset-hours", type=float, default=None)
    parser.add_argument("--alignment-tolerance-seconds", type=int, default=None)
    parser.add_argument("--disable-fgl", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--max-train-modules", type=int, default=None)
    parser.add_argument("--max-validation-modules", type=int, default=None)
    parser.add_argument("--max-test-modules", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-long-scores", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.inference_batch_size is not None:
        overrides.setdefault("training", {})["inference_batch_size"] = int(
            args.inference_batch_size
        )
    if args.experiment_name:
        overrides["experiment_name"] = str(args.experiment_name)
    if args.test_fold is not None:
        overrides.setdefault("split", {})["test_fold"] = int(args.test_fold)
    if args.fgl_alpha is not None:
        overrides.setdefault("training", {})["fgl_alpha"] = float(args.fgl_alpha)
    if args.positive_weight_mode is not None:
        overrides.setdefault("training", {})["positive_weight_mode"] = str(
            args.positive_weight_mode
        )
    if args.architecture is not None:
        overrides.setdefault("model", {})["architecture"] = str(args.architecture)
    if args.rule_residual_scale is not None:
        overrides.setdefault("model", {})["rule_residual_scale"] = float(
            args.rule_residual_scale
        )
    if args.fallback_min_gain is not None:
        overrides.setdefault("decision", {})["fallback_min_gain"] = float(
            args.fallback_min_gain
        )
    if args.disable_safe_rule_fallback:
        overrides.setdefault("decision", {})["fallback_policy"] = "none"
    if args.disable_fgl:
        overrides.setdefault("training", {})["fgl_alpha"] = 1.0
    if args.no_mixed_precision:
        overrides.setdefault("training", {})["mixed_precision"] = False
    if args.future_offset_hours is not None:
        offset = float(args.future_offset_hours)
        overrides.setdefault("task", {})["future_offset_hours"] = offset
        overrides.setdefault("task", {})["teacher_horizon_hours"] = 120.0 - offset
    if args.alignment_tolerance_seconds is not None:
        overrides.setdefault("inputs", {})["alignment_tolerance_seconds"] = int(
            args.alignment_tolerance_seconds
        )
    if args.smoke:
        overrides.setdefault("model", {}).update(
            {"hidden_channels": 16, "dropout": 0.0}
        )
        overrides.setdefault("training", {}).update(
            {
                **overrides.get("training", {}),
                "teacher_epochs": 1,
                "student_epochs": 1,
                "batch_size": 8,
                "inference_batch_size": 8,
                "early_stopping_patience": 0,
                "num_workers": 0,
            }
        )
        overrides.setdefault("decision", {}).update(
            {"threshold_grid_size": 21, "threshold_quantile_count": 21}
        )
    return overrides


def main() -> None:
    args = parse_args()
    config = load_config(args.config, _overrides(args))
    max_train = args.max_train_modules
    max_validation = args.max_validation_modules
    max_test = args.max_test_modules
    if args.smoke:
        # A larger training subset than the old endpoint smoke is deliberate:
        # positive t+6h FGL pairs occur in only a minority of faulty modules.
        max_train = 256 if max_train is None else max_train
        max_validation = 64 if max_validation is None else max_validation
        max_test = 64 if max_test is None else max_test
    run_experiment(
        config,
        args.data_dir,
        args.index_path,
        args.output_dir,
        max_train_modules=max_train,
        max_validation_modules=max_validation,
        max_test_modules=max_test,
        overwrite=args.overwrite,
        save_long_scores=args.save_long_scores,
        progress=True,
    )


if __name__ == "__main__":
    main()
