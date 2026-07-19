"""Auditable training-endpoint manifests and isolated sampling/weighting rules."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import ofp_unified.labels as unified_labels
from ofp_unified.labels import first_event_target, validate_time_order

from .config import HTSFConfig, SamplingConfig, WeightingConfig, fingerprint


ENDPOINT_COLUMNS = (
    "file_name",
    "row_index",
    "timestamp",
    "target",
    "module_fault",
    "lead_hours",
)


def endpoint_implementation_fingerprint() -> str:
    paths = [Path(__file__).resolve(), Path(unified_labels.__file__).resolve()]
    return fingerprint(
        [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ]
    )


def _evenly_spaced(positions: np.ndarray, count: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    count = int(count)
    if count <= 0 or len(positions) == 0:
        return np.empty(0, dtype=np.int64)
    if count >= len(positions):
        return positions.copy()
    offsets = np.rint(np.linspace(0, len(positions) - 1, count)).astype(np.int64)
    selected = positions[offsets]
    if len(np.unique(selected)) != count:
        raise RuntimeError("even endpoint selection unexpectedly produced duplicates")
    return selected


def select_positions(
    valid_positions: np.ndarray,
    target: np.ndarray,
    module_fault: bool,
    sampling: SamplingConfig,
) -> np.ndarray:
    """Select endpoints using only the explicitly configured sampling axis."""

    valid_positions = np.asarray(valid_positions, dtype=np.int64)
    if sampling.mode == "uniform_per_module":
        return _evenly_spaced(valid_positions, sampling.windows_per_module)
    faulty_total = (
        int(sampling.positive_windows_per_faulty_module)
        + int(sampling.negative_windows_per_faulty_module)
    )
    if sampling.mode == "module_quota":
        count = faulty_total if module_fault else int(sampling.normal_windows_per_module)
        return _evenly_spaced(valid_positions, count)
    if sampling.mode != "hss":
        raise ValueError(f"unknown sampling mode={sampling.mode!r}")

    positive = valid_positions[target[valid_positions] > 0]
    negative = valid_positions[target[valid_positions] <= 0]
    if module_fault:
        selected_positive = _evenly_spaced(
            positive,
            sampling.positive_windows_per_faulty_module,
        )
        selected_negative = _evenly_spaced(
            negative,
            sampling.negative_windows_per_faulty_module,
        )
        selected = np.sort(np.concatenate([selected_positive, selected_negative])).astype(np.int64)
        target_total = min(len(valid_positions), faulty_total)
        if len(selected) < target_total:
            remaining = np.setdiff1d(valid_positions, selected, assume_unique=True)
            fill = _evenly_spaced(remaining, target_total - len(selected))
            selected = np.sort(np.concatenate([selected, fill])).astype(np.int64)
        return selected
    return _evenly_spaced(negative, sampling.normal_windows_per_module)


def build_endpoint_manifest(
    data_dir: Path,
    file_names: Sequence[str],
    config: HTSFConfig,
) -> pd.DataFrame:
    """Build the one shared endpoint list used by every architecture variant."""

    rows: list[pd.DataFrame] = []
    data_dir = Path(data_dir)
    for file_name in file_names:
        path = data_dir / str(file_name)
        frame = pd.read_csv(path, usecols=lambda name: name in {"timestamp", "anomaly"})
        if "timestamp" not in frame or "anomaly" not in frame:
            raise ValueError(f"{path} must contain timestamp and anomaly")
        timestamps = validate_time_order(frame)
        target, valid, first_ts = first_event_target(
            frame,
            horizon_hours=config.task.horizon_hours,
        )
        valid_positions = np.flatnonzero(valid).astype(np.int64)
        selected = select_positions(
            valid_positions,
            target,
            module_fault=first_ts is not None,
            sampling=config.sampling,
        )
        if len(selected) == 0:
            continue
        lead = np.full(len(selected), np.nan, dtype=np.float64)
        if first_ts is not None:
            lead = (float(first_ts) - timestamps[selected]) / 3600.0
        rows.append(
            pd.DataFrame(
                {
                    "file_name": str(file_name),
                    "row_index": selected,
                    "timestamp": timestamps[selected].astype(np.int64),
                    "target": target[selected].astype(np.int8),
                    "module_fault": int(first_ts is not None),
                    "lead_hours": lead,
                }
            )
        )
    if not rows:
        raise ValueError("endpoint sampling produced an empty manifest")
    manifest = pd.concat(rows, ignore_index=True).loc[:, ENDPOINT_COLUMNS]
    validate_endpoint_manifest(manifest, data_dir, config)
    return manifest


def validate_endpoint_manifest(
    manifest: pd.DataFrame,
    data_dir: Path | None = None,
    config: HTSFConfig | None = None,
) -> None:
    missing = set(ENDPOINT_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"endpoint manifest missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("endpoint manifest is empty")
    if manifest.duplicated(["file_name", "row_index"]).any():
        raise ValueError("endpoint manifest contains duplicate file/row pairs")
    if not manifest["target"].isin([0, 1]).all():
        raise ValueError("endpoint targets must be binary")
    faulty = manifest["module_fault"].astype(int) > 0
    if (pd.to_numeric(manifest.loc[faulty, "lead_hours"], errors="coerce") <= 0).any():
        raise ValueError("faulty training endpoints must be strictly before the first fault")
    if data_dir is not None and config is not None:
        for file_name, group in manifest.groupby("file_name", sort=False):
            frame = pd.read_csv(
                Path(data_dir) / str(file_name),
                usecols=lambda name: name in {"timestamp", "anomaly"},
            )
            target, valid, _ = first_event_target(frame, config.task.horizon_hours)
            positions = group["row_index"].to_numpy(dtype=np.int64)
            if np.any(positions < 0) or np.any(positions >= len(frame)):
                raise ValueError(f"manifest position out of range for {file_name}")
            if not valid[positions].all():
                raise ValueError(f"manifest includes post-fault endpoint for {file_name}")
            if not np.array_equal(target[positions], group["target"].to_numpy(dtype=np.int8)):
                raise ValueError(f"manifest target mismatch for {file_name}")


def endpoint_fingerprint(manifest: pd.DataFrame) -> str:
    ordered = manifest.loc[:, ENDPOINT_COLUMNS].copy()
    ordered["lead_hours"] = ordered["lead_hours"].map(
        lambda value: None if pd.isna(value) else round(float(value), 10)
    )
    return fingerprint(ordered.to_dict(orient="records"))


def apply_static_weights(manifest: pd.DataFrame, config: HTSFConfig) -> pd.DataFrame:
    """Attach TPW independently of endpoint selection and architecture."""

    result = manifest.copy()
    weights = np.ones(len(result), dtype=np.float32)
    weighting = config.weighting
    if weighting.mode in {"tpw", "tpw_anw"}:
        positive = result["target"].to_numpy(dtype=np.int8) > 0
        lead = pd.to_numeric(result["lead_hours"], errors="coerce").to_numpy(dtype=float)
        proximity = 1.0 - np.clip(lead / float(config.task.horizon_hours), 0.0, 1.0)
        proximity = np.nan_to_num(proximity, nan=0.0, posinf=0.0, neginf=0.0)
        weights[positive] += (
            float(weighting.temporal_positive_max_extra) * proximity[positive]
        ).astype(np.float32)
    result["sample_weight"] = weights
    return result


def weight_fingerprint(weighted_manifest: pd.DataFrame) -> str:
    if "sample_weight" not in weighted_manifest:
        raise ValueError("weighted manifest lacks sample_weight")
    values = [round(float(value), 8) for value in weighted_manifest["sample_weight"]]
    return fingerprint(values)


def endpoint_source_fingerprint(
    file_names: Sequence[str],
    config: HTSFConfig,
    dataset_fingerprint: str,
) -> str:
    return fingerprint(
        {
            "file_names": list(file_names),
            "dataset_fingerprint": str(dataset_fingerprint),
            "task": config.to_dict()["task"],
            "sampling": config.to_dict()["sampling"],
            "seed": int(config.seed),
            "endpoint_implementation_fingerprint": endpoint_implementation_fingerprint(),
        }
    )


def load_or_build_endpoint_manifest(
    output_dir: Path,
    data_dir: Path,
    file_names: Sequence[str],
    config: HTSFConfig,
    dataset_fingerprint: str,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_dir = Path(output_dir)
    csv_path = output_dir / "train_endpoints.csv"
    meta_path = output_dir / "train_endpoints.json"
    source = endpoint_source_fingerprint(file_names, config, dataset_fingerprint)
    if csv_path.exists() and meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_fingerprint") != source:
            raise ValueError(
                f"endpoint cache at {output_dir} belongs to another task/split/sampling config; "
                "use --overwrite or another artifacts directory"
            )
        manifest = pd.read_csv(csv_path)
        validate_endpoint_manifest(manifest, data_dir, config)
        actual = endpoint_fingerprint(manifest)
        if actual != meta.get("endpoint_fingerprint"):
            raise ValueError("endpoint manifest fingerprint mismatch")
        return manifest, meta

    manifest = build_endpoint_manifest(data_dir, file_names, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(csv_path, index=False)
    meta = {
        "source_fingerprint": source,
        "endpoint_fingerprint": endpoint_fingerprint(manifest),
        "dataset_fingerprint": str(dataset_fingerprint),
        "sampling": config.to_dict()["sampling"],
        "task": config.to_dict()["task"],
        "seed": int(config.seed),
        "endpoint_implementation_fingerprint": endpoint_implementation_fingerprint(),
        "rows": int(len(manifest)),
        "positive_rows": int((manifest["target"] > 0).sum()),
        "negative_rows": int((manifest["target"] <= 0).sum()),
        "modules": int(manifest["file_name"].nunique()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return manifest, meta
