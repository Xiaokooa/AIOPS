"""Cross-variant result checks and B0/B1/B2 ablation table generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


VARIANT_OUTPUTS = {
    "b0": "b0_strict_120h",
    "b1": "b1_strict_120h",
    "b2": "b2_strict_120h",
}

METRIC_COLUMNS = (
    "final_score",
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "all_hit_cnt",
    "all_predict_pos_cnt",
    "avg_lead_hour",
)


def protocol_signature(config: dict) -> dict:
    """Return fields that must be identical for a fair feature ablation."""

    ignored = {
        "experiment_name",
        "variant",
        "feature_set",
        "feature_schema_version",
    }
    return {key: value for key, value in config.items() if key not in ignored}


def collect_ablation_results(
    artifacts_dir: Path,
    variants: Sequence[str] = ("b0", "b1", "b2"),
) -> pd.DataFrame:
    artifacts_dir = Path(artifacts_dir)
    rows: list[dict[str, object]] = []
    expected_protocol: dict | None = None
    expected_execution_identity: dict | None = None
    for variant in variants:
        if variant not in VARIANT_OUTPUTS:
            raise ValueError(f"unknown variant={variant!r}")
        manifest_path = artifacts_dir / VARIANT_OUTPUTS[variant] / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = protocol_signature(manifest["config"])
        if expected_protocol is None:
            expected_protocol = signature
        elif signature != expected_protocol:
            raise ValueError(
                f"{variant} protocol differs from the other variants; "
                "refusing to attribute the result to features"
            )
        execution_identity = {
            "feature_schema_version": manifest["feature_manifest"].get("schema_version"),
            "effective_device_by_fold": manifest.get("effective_device_by_fold"),
            "data_split_fingerprint_by_fold": manifest.get("data_split_fingerprint_by_fold"),
            "dataset_metadata_fingerprint": manifest.get("dataset_metadata_fingerprint"),
        }
        missing_identity = [
            key for key, value in execution_identity.items() if value in (None, {}, "")
        ]
        if missing_identity:
            raise ValueError(
                f"{variant} manifest lacks execution identity fields "
                f"{missing_identity}; rerun B0/B1/B2 with the current code"
            )
        if expected_execution_identity is None:
            expected_execution_identity = execution_identity
        elif execution_identity != expected_execution_identity:
            raise ValueError(
                f"{variant} execution identity differs (schema/device/data/split); "
                "rerun B0/B1/B2 in the same environment"
            )
        metrics = manifest["pooled_metrics"]
        row: dict[str, object] = {
            "variant": variant,
            "feature_set": manifest["feature_manifest"].get(
                "feature_set", manifest["config"]["feature_set"]
            ),
            "feature_count": int(manifest["feature_manifest"]["feature_count"]),
            "result_source": str(manifest_path.resolve()),
            "feature_schema_version": execution_identity["feature_schema_version"],
            "effective_devices": json.dumps(
                execution_identity["effective_device_by_fold"], sort_keys=True
            ),
            "dataset_metadata_fingerprint": execution_identity[
                "dataset_metadata_fingerprint"
            ],
        }
        row.update({name: float(metrics[name]) for name in METRIC_COLUMNS})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no completed B0/B1/B2 manifests under {artifacts_dir}")

    table = pd.DataFrame(rows)
    b0_rows = table.loc[table["variant"] == "b0"]
    for metric in ("final_score", "f1_score", "precision", "recall", "accuracy"):
        reference = float(b0_rows.iloc[0][metric]) if len(b0_rows) == 1 else np.nan
        table[f"delta_{metric}_vs_b0"] = table[metric] - reference
    table["delta_final_vs_previous"] = table["final_score"].diff()
    table["delta_f1_vs_previous"] = table["f1_score"].diff()
    return table


def write_ablation_results(
    artifacts_dir: Path,
    variants: Sequence[str] = ("b0", "b1", "b2"),
) -> Path:
    artifacts_dir = Path(artifacts_dir)
    table = collect_ablation_results(artifacts_dir, variants)
    output = artifacts_dir / "feature_ablation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return output
