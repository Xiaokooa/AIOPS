"""Run all architecture ablations with one shared fixed data split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = BASE_DIR.parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
for search_path in (BASE_DIR, UNIFIED_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from drfp.config import DRFPConfig, deep_merge
from drfp.pipeline import prepare_data, release_model_memory, run_prepared_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed-axis DRFP architecture ablation suite."
    )
    parser.add_argument("--protocol", type=Path, default=BASE_DIR / "configs" / "default.json")
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=BASE_DIR / "configs" / "architecture_suite.json",
    )
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=BASE_DIR / "artifacts")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--max-train-modules", type=int, default=None)
    parser.add_argument("--max-validation-modules", type=int, default=None)
    parser.add_argument("--max-test-modules", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-long-scores", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _smoke_override(device: str | None) -> dict:
    training = {
        "epochs": 1,
        "batch_size": 32,
        "inference_batch_size": 256,
        "early_stopping_patience": 0,
        "num_workers": 0,
    }
    if device is not None:
        training["device"] = device
    return {
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
        "training": training,
        "calibration": {"grid_size": 21},
        "decision": {"threshold_candidates": 21},
    }


def _audit_architecture_axis(configs: list[DRFPConfig]) -> None:
    if not configs:
        raise ValueError("suite contains no experiments")
    reference = configs[0].to_dict()
    reference.pop("experiment_name", None)
    reference["model"].pop("variant", None)
    for config in configs[1:]:
        candidate = config.to_dict()
        candidate.pop("experiment_name", None)
        candidate["model"].pop("variant", None)
        if candidate != reference:
            raise ValueError("architecture suite changes an axis other than model.variant")


def main() -> None:
    args = parse_args()
    base = json.loads(args.protocol.read_text(encoding="utf-8"))
    suite = json.loads(args.suite_config.read_text(encoding="utf-8"))
    if suite.get("changed_axis") != "architecture":
        raise ValueError("run_suite.py only accepts an architecture suite")
    experiments = list(suite.get("experiments", []))
    if args.experiments:
        requested = set(args.experiments)
        known = {str(item["id"]) for item in experiments}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown experiment ids: {unknown}")
        experiments = [item for item in experiments if str(item["id"]) in requested]
    configs: list[DRFPConfig] = []
    for experiment in experiments:
        payload = deep_merge(base, experiment.get("overrides", {}))
        payload["experiment_name"] = str(experiment["id"])
        if args.device is not None:
            payload = deep_merge(payload, {"training": {"device": args.device}})
        if args.smoke:
            payload = deep_merge(payload, _smoke_override(args.device))
        if args.inference_batch_size is not None:
            payload = deep_merge(
                payload,
                {"training": {"inference_batch_size": args.inference_batch_size}},
            )
        configs.append(DRFPConfig.from_dict(payload))
    _audit_architecture_axis(configs)
    max_train = args.max_train_modules
    max_validation = args.max_validation_modules
    max_test = args.max_test_modules
    if args.smoke:
        max_train = 24 if max_train is None else max_train
        max_validation = 8 if max_validation is None else max_validation
        max_test = 8 if max_test is None else max_test
    prepared = prepare_data(
        configs[0],
        args.data_dir,
        args.index_path,
        max_train_modules=max_train,
        max_validation_modules=max_validation,
        max_test_modules=max_test,
        progress=True,
    )
    suite_scope = "smoke" if args.smoke else prepared.scope
    output_dir = args.artifacts_dir / f"architecture_{suite_scope}"
    rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    core_metrics = (
        "final_score",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
        "avg_lead_hour",
        "min_lead_hour",
        "tp",
        "fp",
        "fn",
        "tn",
        "normal_module_false_alarm_rate",
    )
    for config in configs:
        print(f"\n[experiment] {config.experiment_name}", flush=True)
        manifest = run_prepared_experiment(
            prepared,
            config,
            output_dir / config.experiment_name,
            overwrite=args.overwrite,
            save_long_scores=args.save_long_scores,
        )
        branch_table = pd.read_csv(
            output_dir / config.experiment_name / "branch_comparison.csv"
        )
        primary_branch = str(manifest["primary_branch"])
        primary_rows = branch_table.loc[branch_table["branch"] == primary_branch]
        for branch_row in primary_rows.to_dict(orient="records"):
            row = {
                "experiment_id": config.experiment_name,
                "variant": config.model.variant,
                "changed_axis": "architecture",
                "scope": manifest["scope"],
                "primary_branch": primary_branch,
                "threshold_policy": branch_row["threshold_policy"],
                "threshold": branch_row["threshold"],
                "train_modules": manifest["split_counts"]["train"],
                "validation_modules": manifest["split_counts"]["validation"],
                "test_modules": manifest["split_counts"]["test"],
                "faulty_test_modules": branch_row.get("faulty_module_count"),
                "training_endpoints_per_epoch": manifest["training_endpoint_protocol"][
                    "endpoints_per_epoch"
                ],
                "trainable_parameters": manifest["parameter_statistics"]["trainable"],
                "seconds_data_preparation_shared": manifest["seconds"][
                    "data_preparation"
                ],
                "seconds_training": manifest["seconds"]["training"],
                "seconds_total": manifest["seconds"]["experiment_after_preparation"],
            }
            row.update({metric: branch_row.get(metric) for metric in core_metrics})
            rows.append(row)
        diagnostics = {
            "experiment_id": config.experiment_name,
            "variant": config.model.variant,
            "threshold_policy": manifest["primary_threshold_policy"],
            "primary_branch": primary_branch,
        }
        diagnostics.update(
            {
                key: value
                for key, value in manifest["primary_metrics"].items()
                if key not in core_metrics
            }
        )
        diagnostic_rows.append(diagnostics)
        release_model_memory()
    comparison = pd.DataFrame(rows)
    reference_id = (
        "D1_deep_raw_stat"
        if "D1_deep_raw_stat" in set(comparison["experiment_id"])
        else str(comparison.iloc[0]["experiment_id"])
    )
    comparison["delta_reference_id"] = reference_id
    for metric in ("final_score", "f1_score", "precision", "recall", "accuracy"):
        if metric in comparison:
            deltas = []
            for row in comparison.itertuples(index=False):
                reference = comparison.loc[
                    (comparison["experiment_id"] == reference_id)
                    & (comparison["threshold_policy"] == row.threshold_policy)
                ]
                reference_value = (
                    float(reference.iloc[0][metric]) if not reference.empty else float("nan")
                )
                deltas.append(float(getattr(row, metric)) - reference_value)
            comparison[f"delta_{metric}_vs_reference"] = deltas
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    comparison.loc[comparison["threshold_policy"] == "validation_selected"].to_csv(
        output_dir / "comparison_validation_selected.csv", index=False
    )
    comparison.loc[comparison["threshold_policy"].astype(str).str.startswith("fixed_")].to_csv(
        output_dir / "comparison_fixed_threshold.csv", index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "diagnostics.csv", index=False)
    (output_dir / "suite_manifest.json").write_text(
        json.dumps(
            {
                "suite_name": suite.get("suite_name"),
                "changed_axis": "architecture",
                "scope": suite_scope,
                "reference_id": reference_id,
                "reported_threshold_policies": sorted(
                    comparison["threshold_policy"].astype(str).unique().tolist()
                ),
                "comparison_note": (
                    "fixed threshold is the historical-policy reference; "
                    "validation_selected is comparable only to baselines tuned on the same validation split"
                ),
                "experiments": [config.experiment_name for config in configs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[suite] wrote {output_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
