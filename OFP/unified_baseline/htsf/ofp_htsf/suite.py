"""Suite-level one-axis audits and comparison-table generation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import HTSFConfig, fingerprint


METRICS = ("final_score", "f1_score", "precision", "recall", "accuracy", "avg_lead_hour")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _validate_identifier(value: object, label: str) -> str:
    text = str(value)
    if not SAFE_IDENTIFIER.fullmatch(text):
        raise ValueError(f"unsafe {label}={text!r}; use letters, numbers, dot, dash or underscore")
    if text.split(".", 1)[0].lower() in WINDOWS_RESERVED:
        raise ValueError(f"unsafe Windows-reserved {label}={text!r}")
    return text


def load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"suite_name", "changed_axis", "experiments"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"suite config missing fields: {sorted(missing)}")
    unknown = set(payload) - {"suite_name", "changed_axis", "description", "experiments"}
    if unknown:
        raise ValueError(f"suite config has unknown fields: {sorted(unknown)}")
    payload["suite_name"] = _validate_identifier(payload["suite_name"], "suite_name")
    if payload["changed_axis"] not in {"architecture", "sampling", "weighting"}:
        raise ValueError("suite changed_axis must be architecture, sampling, or weighting")
    identifiers = []
    for item in payload["experiments"]:
        item_missing = {"id", "variant"} - set(item)
        item_unknown = set(item) - {"id", "variant", "overrides"}
        if item_missing or item_unknown:
            raise ValueError(
                f"suite experiment fields invalid; missing={sorted(item_missing)} "
                f"unknown={sorted(item_unknown)}"
            )
        item["id"] = _validate_identifier(item["id"], "experiment id")
        identifiers.append(item["id"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("suite experiment ids must be unique")
    return payload


def audit_suite_configs(
    changed_axis: str,
    configs: Sequence[HTSFConfig],
    variants: Sequence[str],
) -> str:
    if not configs:
        raise ValueError("suite has no selected experiments")
    signatures = {config.axis_signature(changed_axis) for config in configs}
    if len(signatures) != 1:
        raise ValueError(
            f"suite changes fields outside its declared {changed_axis!r} axis"
        )
    if changed_axis != "architecture" and len(set(variants)) != 1:
        raise ValueError(f"{changed_axis} suite must fix one architecture variant")
    return next(iter(signatures))


def write_suite_comparison(
    output_dir: Path,
    suite: dict[str, Any],
    completed: Sequence[dict[str, Any]],
    invariant_signature: str,
    *,
    run_scope: str,
    suite_complete: bool,
    suite_config_fingerprint: str,
    protocol_config_fingerprint: str,
    canonical_definition: bool,
    index_assignment_fingerprint: str,
) -> Path:
    if not completed:
        raise ValueError("cannot summarize an empty suite")
    changed_axis = str(suite["changed_axis"])
    split_sets = {
        json.dumps(item["manifest"]["split_fingerprints"], sort_keys=True)
        for item in completed
    }
    dataset_sets = {
        json.dumps(item["manifest"]["dataset_fingerprints"], sort_keys=True)
        for item in completed
    }
    xgboost_device_sets = {
        json.dumps(item["manifest"]["xgboost_devices"], sort_keys=True)
        for item in completed
    }
    if len(split_sets) != 1 or len(dataset_sets) != 1:
        raise ValueError("suite experiments did not use identical module splits/data")
    if len(xgboost_device_sets) != 1:
        raise ValueError("suite experiments used different effective XGBoost devices")
    learned_torch_devices = {
        json.dumps(
            {
                fold: device
                for fold, device in item["manifest"]["torch_devices"].items()
                if device != "not_applicable"
            },
            sort_keys=True,
        )
        for item in completed
        if any(
            device != "not_applicable"
            for device in item["manifest"]["torch_devices"].values()
        )
    }
    if len(learned_torch_devices) > 1:
        raise ValueError("suite learned variants used different effective Torch devices")
    if changed_axis == "architecture":
        endpoint_sets = {
            json.dumps(item["manifest"]["endpoint_fingerprints"], sort_keys=True)
            for item in completed
        }
        weight_sets = {
            json.dumps(item["manifest"]["weight_fingerprints"], sort_keys=True)
            for item in completed
        }
        if len(endpoint_sets) != 1 or len(weight_sets) != 1:
            raise ValueError("architecture variants did not consume identical endpoints and weights")
    if str(suite["suite_name"]) == "architecture":
        widths = {
            int(value)
            for item in completed
            for value in item["manifest"]["decision_feature_counts"].values()
        }
        if len(widths) != 1:
            raise ValueError("core architecture variants must expose one equal XGBoost width")
        if any("b2" in item["manifest"]["decision_inputs"] for item in completed):
            raise ValueError("core architecture suite cannot include direct B2 decision inputs")
    if changed_axis == "weighting":
        endpoint_sets = {
            json.dumps(item["manifest"]["endpoint_fingerprints"], sort_keys=True)
            for item in completed
        }
        if len(endpoint_sets) != 1:
            raise ValueError("weighting variants did not consume identical endpoints")
    if changed_axis == "sampling":
        count_profiles = {
            json.dumps(
                item["manifest"]["module_endpoint_count_fingerprints"],
                sort_keys=True,
            )
            for item in completed
        }
        if len(count_profiles) != 1:
            raise ValueError(
                "sampling variants changed per-module endpoint budgets; "
                "the matched HSS control is invalid"
            )

    rows = []
    for item in completed:
        row = {
            "experiment_id": item["id"],
            "variant": item["variant"],
            "changed_axis": changed_axis,
            "train_rows_total": int(
                sum(item["manifest"]["train_rows_by_fold"].values())
            ),
            "positive_train_rows_total": int(
                sum(item["manifest"]["positive_train_rows_by_fold"].values())
            ),
            "decision_feature_count": int(
                next(iter(item["manifest"]["decision_feature_counts"].values()))
            ),
            "trainable_parameters": int(
                next(iter(item["manifest"]["trainable_parameters_by_fold"].values()))
            ),
            "seconds_total": float(sum(item["manifest"]["seconds_by_fold"].values())),
        }
        row.update({metric: float(item["metrics"][metric]) for metric in METRICS})
        rows.append(row)
    table = pd.DataFrame(rows)
    reference = table.iloc[0]
    for metric in ("final_score", "f1_score", "precision", "recall", "accuracy"):
        table[f"delta_{metric}_vs_{reference['experiment_id']}"] = table[metric] - float(
            reference[metric]
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.csv"
    table.to_csv(comparison_path, index=False)
    (output_dir / "suite_manifest.json").write_text(
        json.dumps(
            {
                "suite_name": suite["suite_name"],
                "changed_axis": changed_axis,
                "description": suite.get("description", ""),
                "run_scope": str(run_scope),
                "suite_complete": bool(suite_complete),
                "canonical_definition": bool(canonical_definition),
                "suite_config_fingerprint": str(suite_config_fingerprint),
                "protocol_config_fingerprint": str(protocol_config_fingerprint),
                "index_assignment_fingerprint": str(index_assignment_fingerprint),
                "invariant_signature": invariant_signature,
                "experiments": [
                    {
                        "id": item["id"],
                        "variant": item["variant"],
                        "run_manifest": str(item["run_manifest"]),
                    }
                    for item in completed
                ],
                "comparison_fingerprint": fingerprint(table.to_dict(orient="records")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return comparison_path
