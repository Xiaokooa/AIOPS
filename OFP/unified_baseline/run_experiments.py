"""Run strict B0/B1/B2 feature ablations with a shared protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ofp_unified.ablation import write_ablation_results
from ofp_unified.b0 import aggregate_folds, read_index, run_fold


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS = {variant: BASE_DIR / "configs" / f"{variant}.json" for variant in ("b0", "b1", "b2")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict OFP B0/B1/B2 XGBoost feature ablations."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(CONFIG_PATHS),
        default=["b1", "b2"],
    )
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=BASE_DIR / "artifacts")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--max-train-modules", type=int, default=None)
    parser.add_argument("--max-validation-modules", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--matrix-mode", choices=["quantile", "external_memory"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _run_variant(args: argparse.Namespace, variant: str, index_df) -> bool:
    config = json.loads(CONFIG_PATHS[variant].read_text(encoding="utf-8"))
    if config.get("variant", variant) != variant:
        raise ValueError(f"{CONFIG_PATHS[variant]} declares a different variant")
    if args.device is not None:
        config["xgboost"]["device"] = args.device
    if args.matrix_mode is not None:
        config["matrix_mode"] = args.matrix_mode

    configured_folds = [int(value) for value in config["folds"]]
    folds = configured_folds if args.folds is None else [int(value) for value in args.folds]
    if len(folds) != len(set(folds)):
        raise ValueError("--folds contains duplicates")
    invalid_folds = sorted(set(folds) - set(index_df["folder_index"].unique()))
    if invalid_folds:
        raise ValueError(f"unknown folds: {invalid_folds}")
    max_train = args.max_train_modules
    max_validation = args.max_validation_modules
    if args.smoke:
        max_train = 24 if max_train is None else max_train
        max_validation = 12 if max_validation is None else max_validation
        config["xgboost"]["num_boost_round"] = 5
        config["matrix_mode"] = "quantile"

    formal = (
        not args.smoke
        and max_train is None
        and max_validation is None
        and set(folds) == set(configured_folds)
    )
    suffix = "strict_120h" if formal else ("smoke" if args.smoke else "partial")
    out_root = args.artifacts_dir / f"{variant}_{suffix}"
    print(f"\n[variant {variant}] config={CONFIG_PATHS[variant]} out={out_root}")
    for fold in folds:
        result = run_fold(
            fold=fold,
            index_df=index_df,
            data_dir=args.data_dir,
            out_root=out_root,
            config=config,
            max_train_modules=max_train,
            max_validation_modules=max_validation,
            overwrite=args.overwrite,
        )
        print(
            f"[done] variant={variant} fold={fold} "
            f"final={result.metrics['final_score']:.6f} "
            f"F1={result.metrics['f1_score']:.6f} "
            f"P={result.metrics['precision']:.6f} "
            f"R={result.metrics['recall']:.6f} seconds={result.seconds:.1f}"
        )

    expected_files = index_df["file_name"].astype(str).tolist() if formal else None
    pooled, comparison = aggregate_folds(
        out_root,
        folds,
        config,
        compare_readme=formal,
        expected_files=expected_files,
    )
    print(
        f"[pooled] variant={variant} final={pooled['final_score']:.9f} "
        f"F1={pooled['f1_score']:.9f} P={pooled['precision']:.9f} "
        f"R={pooled['recall']:.9f}"
    )
    if formal:
        print(comparison.to_string(index=False))
    else:
        print("[comparison] skipped: partial/smoke runs are not formal results")
    return formal


def main() -> None:
    args = parse_args()
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("--variants contains duplicates")
    index_df = read_index(args.index_path)
    completed_formal = []
    for variant in args.variants:
        if _run_variant(args, variant, index_df):
            completed_formal.append(variant)
    if completed_formal:
        try:
            output = write_ablation_results(args.artifacts_dir)
        except (FileNotFoundError, ValueError) as error:
            print(f"[ablation] not written: {error}")
        else:
            print(f"[ablation] wrote {output}")


if __name__ == "__main__":
    main()
