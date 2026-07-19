from __future__ import annotations

import argparse
import json
from pathlib import Path

from ofp_unified.b0 import aggregate_folds, read_index, run_fold


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "b0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict Raw12 XGBoost OFP B0 baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "b0_strict_120h",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--max-train-modules", type=int, default=None)
    parser.add_argument("--max-validation-modules", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--matrix-mode", choices=["quantile", "external_memory"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("feature_set") != "raw":
        raise ValueError(
            "run_b0.py accepts only the raw B0 config; use run_experiments.py "
            "for B1/B2"
        )
    if args.device is not None:
        config["xgboost"]["device"] = args.device
    if args.matrix_mode is not None:
        config["matrix_mode"] = args.matrix_mode
    folds = args.folds if args.folds is not None else [int(v) for v in config["folds"]]
    max_train = args.max_train_modules
    max_validation = args.max_validation_modules
    out_root = args.out_root
    if args.smoke:
        max_train = 24 if max_train is None else max_train
        max_validation = 12 if max_validation is None else max_validation
        config["xgboost"]["num_boost_round"] = 5
        config["matrix_mode"] = "quantile"
        out_root = out_root.parent / "b0_smoke"
    index_df = read_index(args.index_path)
    invalid_folds = sorted(set(folds) - set(index_df["folder_index"].unique()))
    if invalid_folds:
        raise ValueError(f"unknown folds: {invalid_folds}")

    for fold in folds:
        result = run_fold(
            fold=int(fold),
            index_df=index_df,
            data_dir=args.data_dir,
            out_root=out_root,
            config=config,
            max_train_modules=max_train,
            max_validation_modules=max_validation,
            overwrite=args.overwrite,
        )
        print(
            f"[done] fold={fold} final={result.metrics['final_score']:.6f} "
            f"F1={result.metrics['f1_score']:.6f} P={result.metrics['precision']:.6f} "
            f"R={result.metrics['recall']:.6f} seconds={result.seconds:.1f}"
        )

    formal_comparison = (
        max_train is None
        and max_validation is None
        and set(folds) == set(int(v) for v in config["folds"])
    )
    expected_files = (
        index_df.loc[index_df["folder_index"].isin(folds), "file_name"].astype(str).tolist()
        if formal_comparison
        else None
    )
    pooled, comparison = aggregate_folds(
        out_root,
        folds,
        config,
        compare_readme=formal_comparison,
        expected_files=expected_files,
    )
    print(
        f"[pooled] final={pooled['final_score']:.9f} F1={pooled['f1_score']:.9f} "
        f"P={pooled['precision']:.9f} R={pooled['recall']:.9f}"
    )
    if formal_comparison:
        print(comparison.to_string(index=False))
    else:
        print("[comparison] skipped: partial/smoke runs are not comparable with README")


if __name__ == "__main__":
    main()
