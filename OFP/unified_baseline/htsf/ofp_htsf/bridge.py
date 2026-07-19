"""Compatibility gate between strict B2 and the separately sampled B2 control."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(value: Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def validate_strict_b2_bridge(
    strict_b2: Path | dict[str, Any],
    sampled_b2: Path | dict[str, Any],
    bridge_suite: Path | dict[str, Any],
) -> dict[str, Any]:
    """Fail unless the two manifests differ only in endpoint sampling/training scale."""

    strict = _load(strict_b2)
    sampled = _load(sampled_b2)
    suite = _load(bridge_suite)
    errors: list[str] = []
    strict_config = strict.get("config", {})
    sampled_config = sampled.get("config", {})
    strict_features = strict.get("feature_manifest", {})

    if suite.get("suite_name") != "bridge":
        errors.append("suite manifest is not the bridge suite")
    if suite.get("run_scope") != "formal":
        errors.append("bridge suite is not formal")
    if not suite.get("suite_complete") or not suite.get("canonical_definition"):
        errors.append("bridge suite is incomplete or non-canonical")
    suite_experiments = {
        (item.get("id"), item.get("variant")) for item in suite.get("experiments", [])
    }
    if ("H0_sampled_b2_xgb", "sampled_b2_xgb") not in suite_experiments:
        errors.append("formal bridge suite does not contain canonical H0")

    if strict_features.get("feature_set") != "raw_statistical_expert":
        errors.append("strict manifest is not B2 Raw+Statistical+Expert")
    if int(strict_features.get("feature_count", -1)) != 130:
        errors.append("strict B2 feature count is not 130")
    if strict_features.get("schema_version") != sampled_config.get("feature_schema_version"):
        errors.append("feature schema versions differ")
    if sampled.get("variant") != "sampled_b2_xgb":
        errors.append("HTSF bridge manifest is not sampled_b2_xgb")
    if sampled.get("decision_inputs") != ["b2"]:
        errors.append("sampled control does not feed only direct B2 to XGBoost")
    sampling = sampled_config.get("sampling", {})
    if sampling.get("mode") != "uniform_per_module" or int(
        sampling.get("windows_per_module", -1)
    ) != 32:
        errors.append("sampled B2 does not use canonical uniform-per-module 32 endpoints")
    weighting = sampled_config.get("weighting", {})
    if weighting.get("mode") != "none":
        errors.append("sampled B2 bridge unexpectedly uses representation weighting")
    if sorted(int(value) for value in sampled.get("folds", [])) != [1, 2, 3]:
        errors.append("sampled B2 bridge is not a complete three-fold run")
    if strict_config.get("matrix_mode") != "quantile":
        errors.append("strict B2 did not use QuantileDMatrix")
    if float(strict_config.get("horizon_hours", -1)) != float(
        sampled_config.get("task", {}).get("horizon_hours", -2)
    ):
        errors.append("first-event horizons differ")
    if float(strict_config.get("threshold", -1)) != float(
        sampled_config.get("decision", {}).get("threshold", -2)
    ):
        errors.append("fixed thresholds differ")
    strict_xgb = {
        key: value for key, value in strict_config.get("xgboost", {}).items() if key != "device"
    }
    sampled_xgb = {
        key: value
        for key, value in sampled_config.get("decision", {}).get("xgboost", {}).items()
        if key != "device"
    }
    if strict_xgb != sampled_xgb:
        errors.append("XGBoost hyperparameters differ")

    strict_splits = strict.get("data_split_fingerprint_by_fold", {})
    sampled_splits = sampled.get("split_fingerprints", {})
    if strict_splits != sampled_splits:
        errors.append("fold module split fingerprints differ")
    strict_dataset = strict.get("dataset_metadata_fingerprint")
    sampled_datasets = set(sampled.get("dataset_fingerprints", {}).values())
    if sampled_datasets != {strict_dataset}:
        errors.append("dataset metadata fingerprints differ")
    if strict.get("effective_device_by_fold") != sampled.get("xgboost_devices"):
        errors.append("effective XGBoost devices differ")
    strict_count = int(strict.get("pooled_metrics", {}).get("evaluated_module_cnt", -1))
    sampled_count = int(sampled.get("pooled_metrics", {}).get("evaluated_module_cnt", -2))
    if strict_count != 13372 or sampled_count != 13372:
        errors.append("one or both manifests do not cover all 13,372 modules")
    if errors:
        raise ValueError("incompatible strict-B2 bridge: " + "; ".join(errors))
    return {
        "compatible": True,
        "strict_variant": "b2",
        "sampled_variant": "sampled_b2_xgb",
        "controlled_change": "all valid training rows -> frozen sampled endpoint manifest",
        "feature_count": 130,
        "folds": [1, 2, 3],
        "evaluated_modules": 13372,
        "dataset_metadata_fingerprint": strict_dataset,
    }
