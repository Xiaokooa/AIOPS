"""Run clean, one-axis-at-a-time HTSF suites."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = BASE_DIR.parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
for path in (BASE_DIR, UNIFIED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ofp_unified.b0 import (
    data_split_fingerprint,
    dataset_metadata_fingerprint,
    read_index,
    stratified_limit,
)
from ofp_htsf.config import HTSFConfig, deep_merge
from ofp_htsf.data import load_or_fit_standardizers, materialize_training_tensor_cache
from ofp_htsf.endpoints import (
    apply_static_weights,
    endpoint_source_fingerprint,
    load_or_build_endpoint_manifest,
)
from ofp_htsf.pipeline import aggregate_variant, run_fold_variant
from ofp_htsf.suite import audit_suite_configs, load_suite, write_suite_comparison
from ofp_htsf.variants import get_variant


SUITE_PATHS = {
    "architecture": BASE_DIR / "configs" / "architecture_suite.json",
    "bridge": BASE_DIR / "configs" / "bridge_suite.json",
    "sampling": BASE_DIR / "configs" / "sampling_suite.json",
    "weighting": BASE_DIR / "configs" / "weighting_suite.json",
}
CANONICAL_PROTOCOL_SHA256 = "c365f5401214ac2223e15f32478fdd831607f6c2304dc25f0f4e90544506a707"
CANONICAL_INDEX_SHA256 = "9c5c768b06390bd72c9294d60c0844556c416003c5ce3bb0c7465fbbdcd5af99"
CANONICAL_SUITE_SHA256 = {
    "architecture": "fdd568993fd2714058b4ff1a6b377ca4a9e661166ba9ba6d4ccb0f125bade513",
    "bridge": "b56ad6611184d3df456608f0d94b9322132da1e70e3bed8b892b2027653ba1da",
    "sampling": "4f649a8ce3b5f72b939e371b96859746a3b84edef401bcc1cb84188d220fa1ae",
    "weighting": "56d564cbf4ba4077a8f406ac31cbc98d94e88923daea33818bfdd4d4b502587f",
}


def _file_sha256(path: Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _index_assignment_sha256(index_df) -> str:
    ordered = index_df.loc[:, ["file_name", "folder_index", "Label"]].sort_values("file_name")
    records = [
        {
            "file_name": str(row.file_name),
            "folder_index": int(row.folder_index),
            "Label": int(row.Label),
        }
        for row in ordered.itertuples(index=False)
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run protocol-locked OFP HTSF architecture/bridge/sampling/weighting suites."
    )
    parser.add_argument("--suite", choices=sorted(SUITE_PATHS), default="architecture")
    parser.add_argument("--protocol", type=Path, default=BASE_DIR / "configs" / "protocol.json")
    parser.add_argument("--suite-config", type=Path, default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_protocol_payload(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _effective_config(
    base: dict,
    experiment: dict,
    args: argparse.Namespace,
) -> HTSFConfig:
    payload = deep_merge(base, experiment.get("overrides", {}))
    payload["experiment_name"] = str(experiment["id"])
    if args.device is not None:
        payload = deep_merge(
            payload,
            {
                "training": {"device": args.device},
                "decision": {"xgboost": {"device": args.device}},
            },
        )
    if args.smoke:
        payload = deep_merge(
            payload,
            {
                "window": {"sequence_length": 16, "patch_length": 4, "patch_stride": 2},
                "representation": {
                    "d_model": 32,
                    "latent_dim": 32,
                    "transformer_layers": 1,
                    "attention_heads": 4,
                    "engineered_hidden": 64,
                },
                "training": {"epochs": 1, "batch_size": 64},
                "decision": {"xgboost": {"num_boost_round": 5}},
            },
        )
        if payload["weighting"]["mode"] in {"anw", "tpw_anw"}:
            payload["training"]["epochs"] = 2
            payload["weighting"]["adaptive_warmup_epoch"] = 1
    return HTSFConfig.from_dict(payload)


def main() -> None:
    args = parse_args()
    suite_path = args.suite_config or SUITE_PATHS[args.suite]
    suite_source_fingerprint = _file_sha256(suite_path)
    protocol_source_fingerprint = _file_sha256(args.protocol)
    canonical_definition = (
        protocol_source_fingerprint == CANONICAL_PROTOCOL_SHA256
        and suite_source_fingerprint == CANONICAL_SUITE_SHA256[args.suite]
    )
    suite = load_suite(suite_path)
    if suite["suite_name"] != args.suite and args.suite_config is None:
        raise ValueError("suite file name and declared suite_name disagree")
    experiments = list(suite["experiments"])
    complete_experiment_ids = {str(item["id"]) for item in experiments}
    if args.experiments is not None:
        requested = set(args.experiments)
        known = {str(item["id"]) for item in experiments}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown experiment ids: {unknown}")
        experiments = [item for item in experiments if str(item["id"]) in requested]
    suite_complete = {str(item["id"]) for item in experiments} == complete_experiment_ids
    base_payload = _load_protocol_payload(args.protocol)
    configs = [_effective_config(base_payload, item, args) for item in experiments]
    variants = [get_variant(str(item["variant"])) for item in experiments]
    invariant = audit_suite_configs(
        str(suite["changed_axis"]),
        configs,
        [variant.name for variant in variants],
    )

    index_df = read_index(args.index_path)
    index_assignment_fingerprint = _index_assignment_sha256(index_df)
    configured_folds = list(configs[0].folds)
    folds = configured_folds if args.folds is None else [int(value) for value in args.folds]
    if len(folds) != len(set(folds)):
        raise ValueError("--folds contains duplicates")
    invalid = sorted(set(folds) - set(index_df["folder_index"].unique()))
    if invalid:
        raise ValueError(f"unknown folds: {invalid}")
    max_train = args.max_train_modules
    max_validation = args.max_validation_modules
    if args.smoke:
        max_train = 48 if max_train is None else max_train
        max_validation = 12 if max_validation is None else max_validation
    canonical_index = (
        len(index_df) == 13372
        and index_df["file_name"].nunique() == 13372
        and index_df.groupby("folder_index").size().to_dict() == {1: 4457, 2: 4457, 3: 4458}
        and index_df.loc[index_df["Label"] > 0].groupby("folder_index").size().to_dict()
        == {1: 1367, 2: 1367, 3: 1368}
    )
    formal = (
        not args.smoke
        and canonical_definition
        and suite_complete
        and canonical_index
        and index_assignment_fingerprint == CANONICAL_INDEX_SHA256
        and max_train is None
        and max_validation is None
        and set(folds) == set(configured_folds)
    )
    if formal:
        scope = "formal"
    elif args.smoke:
        scope = "smoke"
    elif not canonical_definition:
        scope = f"custom_{protocol_source_fingerprint[:8]}_{suite_source_fingerprint[:8]}"
    else:
        scope = "partial"
    suite_output = args.artifacts_dir / f"{suite['suite_name']}_{scope}"
    shared_cache: dict[str, tuple] = {}
    completed: list[dict] = []

    for experiment, config, variant in zip(experiments, configs, variants):
        experiment_id = str(experiment["id"])
        experiment_dir = suite_output / experiment_id
        print(
            f"\n[experiment] {experiment_id} variant={variant.name} "
            f"sampling={config.sampling.mode} weighting={config.weighting.mode}",
            flush=True,
        )
        for fold in folds:
            train_index = stratified_limit(
                index_df.loc[index_df["folder_index"] != fold], max_train
            )
            validation_index = stratified_limit(
                index_df.loc[index_df["folder_index"] == fold], max_validation
            )
            train_files = train_index["file_name"].astype(str).tolist()
            validation_files = validation_index["file_name"].astype(str).tolist()
            all_files = [*train_files, *validation_files]
            missing = [name for name in all_files if not (args.data_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"dataset missing {len(missing)} files; first={missing[0]}")
            dataset_fp = dataset_metadata_fingerprint(args.data_dir, all_files)
            split_fp = data_split_fingerprint(
                fold=fold,
                data_dir=args.data_dir,
                train_files=train_files,
                validation_files=validation_files,
                dataset_fingerprint=dataset_fp,
            )
            source_fp = endpoint_source_fingerprint(train_files, config, dataset_fp)
            cache_key = f"{fold}:{source_fp}"
            if cache_key not in shared_cache:
                shared_dir = args.artifacts_dir / "_shared" / source_fp[:16] / f"fold_{fold}"
                manifest, endpoint_meta = load_or_build_endpoint_manifest(
                    shared_dir,
                    args.data_dir,
                    train_files,
                    config,
                    dataset_fp,
                    overwrite=args.overwrite,
                )
                raw_norm, engineered_norm = load_or_fit_standardizers(
                    shared_dir / "normalizers.json",
                    args.data_dir,
                    manifest,
                    config,
                    overwrite=args.overwrite,
                )
                tensor_cache_dir = shared_dir / "training_tensors"
                materialize_training_tensor_cache(
                    tensor_cache_dir,
                    args.data_dir,
                    manifest,
                    config,
                    raw_norm,
                    engineered_norm,
                    overwrite=args.overwrite,
                )
                shared_cache[cache_key] = (
                    manifest,
                    endpoint_meta,
                    raw_norm,
                    engineered_norm,
                    tensor_cache_dir,
                )
            manifest, endpoint_meta, raw_norm, engineered_norm, tensor_cache_dir = shared_cache[cache_key]
            weighted_manifest = apply_static_weights(manifest, config)
            result = run_fold_variant(
                fold=fold,
                variant=variant,
                config=config,
                data_dir=args.data_dir,
                train_manifest=weighted_manifest,
                endpoint_meta=endpoint_meta,
                validation_files=validation_files,
                raw_standardizer=raw_norm,
                engineered_standardizer=engineered_norm,
                output_dir=experiment_dir / f"fold_{fold}",
                split_fingerprint=split_fp,
                dataset_fingerprint=dataset_fp,
                overwrite=args.overwrite,
                tensor_cache_dir=tensor_cache_dir,
            )
            print(
                f"[fold done] experiment={experiment_id} fold={fold} "
                f"final={result['metrics']['final_score']:.6f} "
                f"F1={result['metrics']['f1_score']:.6f}",
                flush=True,
            )
        expected = index_df["file_name"].astype(str).tolist() if formal else None
        metrics, run_manifest = aggregate_variant(
            experiment_dir,
            folds,
            variant,
            config,
            expected_files=expected,
        )
        completed.append(
            {
                "id": experiment_id,
                "variant": variant.name,
                "metrics": metrics,
                "manifest": run_manifest,
                "run_manifest": experiment_dir / "run_manifest.json",
            }
        )
        print(
            f"[pooled] experiment={experiment_id} final={metrics['final_score']:.6f} "
            f"F1={metrics['f1_score']:.6f} P={metrics['precision']:.6f} "
            f"R={metrics['recall']:.6f}",
            flush=True,
        )
    comparison = write_suite_comparison(
        suite_output,
        suite,
        completed,
        invariant,
        run_scope=scope,
        suite_complete=suite_complete,
        suite_config_fingerprint=suite_source_fingerprint,
        protocol_config_fingerprint=protocol_source_fingerprint,
        canonical_definition=canonical_definition,
        index_assignment_fingerprint=index_assignment_fingerprint,
    )
    print(f"[suite] wrote {comparison}", flush=True)


if __name__ == "__main__":
    main()
