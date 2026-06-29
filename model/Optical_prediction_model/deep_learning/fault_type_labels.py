"""Weak fault-type labels for type-aware optical-module prediction.

The dataset only provides binary failure labels.  This module derives
multi-label fault-type targets from R4 window statistics so that fault-type
expert heads can be weakly anchored to physically meaningful optical-module modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


TYPE_NAMES = [
    "thermal_anomaly",
    "current_bias_anomaly",
    "tx_power_anomaly",
    "rx_power_anomaly",
    "lane_imbalance",
]

TYPE_DISPLAY_NAMES = {
    "thermal_anomaly": "Thermal",
    "current_bias_anomaly": "Bias current",
    "tx_power_anomaly": "TX power",
    "rx_power_anomaly": "RX power",
    "lane_imbalance": "Lane imbalance",
}


@dataclass
class WeakTypeLabelMeta:
    type_names: list[str]
    score_thresholds: dict[str, float]
    fallback_thresholds: dict[str, float]
    train_positive_counts: dict[str, int]
    train_positive_coverage: float


def _robust_center_scale(values: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    center = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        q75, q25 = np.nanpercentile(arr, [75, 25])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return center, scale


def _existing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in frame.columns]


def _feature_cols(sensor: str) -> list[str]:
    suffixes = ["mean", "last", "delta", "slope", "range", "std", "missing_ratio"]
    return [f"{sensor}_{suffix}" for suffix in suffixes]


def _group_score(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    sensors: list[str],
) -> np.ndarray:
    cols: list[str] = []
    for sensor in sensors:
        cols.extend(_feature_cols(sensor))
    cols = _existing_columns(frame, cols)
    if not cols:
        return np.zeros(len(frame), dtype=np.float32)

    pieces = []
    for col in cols:
        center, scale = _robust_center_scale(reference[col])
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        pieces.append(np.abs((values - center) / scale))
    scores = np.nanmax(np.vstack(pieces), axis=0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _lane_range(frame: pd.DataFrame, prefix: str, suffix: str) -> np.ndarray:
    cols = _existing_columns(frame, [f"{prefix}{idx}_{suffix}" for idx in range(1, 5)])
    if len(cols) < 2:
        return np.zeros(len(frame), dtype=np.float32)
    values = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return (np.nanmax(values, axis=1) - np.nanmin(values, axis=1)).astype(np.float32)


def _lane_imbalance_score(frame: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    tx_score = _lane_dispersion_score(frame, reference, "currentMultiTXPower")
    rx_score = _lane_dispersion_score(frame, reference, "currentMultiRXPower")
    return np.maximum(tx_score, rx_score).astype(np.float32)


def _lane_dispersion_score(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    prefix: str,
) -> np.ndarray:
    """Return robust deviation of within-TX or within-RX lane dispersion."""
    derived = []
    ref_derived = []
    for suffix in ["mean", "last", "delta", "slope"]:
        derived.append(_lane_range(frame, prefix, suffix))
        ref_derived.append(_lane_range(reference, prefix, suffix))

    pieces = []
    for values, ref_values in zip(derived, ref_derived):
        ref_series = pd.Series(ref_values)
        center, scale = _robust_center_scale(ref_series)
        pieces.append(np.abs((values.astype(float) - center) / scale))
    scores = np.nanmax(np.vstack(pieces), axis=0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_fault_type_scores(
    frame: pd.DataFrame,
    healthy_reference: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return robust anomaly scores for each weak fault type."""
    tx_lane_score = _lane_dispersion_score(frame, healthy_reference, "currentMultiTXPower")
    rx_lane_score = _lane_dispersion_score(frame, healthy_reference, "currentMultiRXPower")
    scores = {
        "thermal_anomaly": _group_score(frame, healthy_reference, ["temperature"]),
        "current_bias_anomaly": _group_score(frame, healthy_reference, ["current"]),
        "tx_power_anomaly": _group_score(frame, healthy_reference, ["currentTXPower"]),
        "rx_power_anomaly": _group_score(frame, healthy_reference, ["currentRXPower"]),
        "lane_imbalance": np.maximum(tx_lane_score, rx_lane_score).astype(np.float32),
    }
    return scores


def annotate_fault_type_frames(
    frames: dict[str, pd.DataFrame],
    threshold_quantile: float = 0.95,
    fallback_quantile: float = 0.85,
    type_names: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], WeakTypeLabelMeta]:
    """Add weak multi-label fault-type columns to train/val/test frames.

    Positive R4 windows receive type labels when their type score exceeds the
    healthy-window threshold.  If a positive window has no threshold hit but
    has one moderately abnormal type, the strongest type is used as a fallback.
    Healthy windows receive all-zero type labels.
    """
    active_types = list(type_names or TYPE_NAMES)
    unknown = sorted(set(active_types) - set(TYPE_NAMES))
    if unknown:
        raise ValueError(f"Unknown fault types: {unknown}")

    train = frames["train"]
    healthy_ref = train[pd.to_numeric(train["label"], errors="coerce").fillna(0).astype(int) == 0]
    if healthy_ref.empty:
        healthy_ref = train

    train_scores = compute_fault_type_scores(train, healthy_ref)
    thresholds = {
        name: float(np.nanquantile(train_scores[name][train["label"].to_numpy(dtype=int) == 0], threshold_quantile))
        for name in active_types
    }
    fallback_thresholds = {
        name: float(np.nanquantile(train_scores[name][train["label"].to_numpy(dtype=int) == 0], fallback_quantile))
        for name in active_types
    }

    annotated: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        out = frame.copy()
        scores = compute_fault_type_scores(out, healthy_ref)
        y = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        label_matrix = np.zeros((len(out), len(active_types)), dtype=np.float32)
        score_matrix = np.zeros_like(label_matrix)

        for idx, name in enumerate(active_types):
            score = scores[name]
            score_matrix[:, idx] = score
            label_matrix[:, idx] = ((score >= thresholds[name]) & (y == 1)).astype(np.float32)
            out[f"weak_type_score_{name}"] = score

        positive_rows = np.where(y == 1)[0]
        for row_idx in positive_rows:
            if label_matrix[row_idx].sum() > 0:
                continue
            best_idx = int(np.argmax(score_matrix[row_idx]))
            best_name = active_types[best_idx]
            if score_matrix[row_idx, best_idx] >= fallback_thresholds[best_name]:
                label_matrix[row_idx, best_idx] = 1.0

        for idx, name in enumerate(active_types):
            out[f"weak_type_{name}"] = label_matrix[:, idx]

        type_counts = label_matrix.sum(axis=1)
        out["weak_type_mask"] = ((y == 1) & (type_counts > 0)).astype(np.float32)
        primary_idx = np.argmax(score_matrix, axis=1)
        out["weak_type_primary"] = [
            active_types[int(i)] if y[row] == 1 and type_counts[row] > 0 else "none"
            for row, i in enumerate(primary_idx)
        ]
        annotated[split] = out

    train_ann = annotated["train"]
    pos = train_ann[train_ann["label"].astype(int) == 1]
    counts = {
        name: int(pos[f"weak_type_{name}"].sum()) if not pos.empty else 0
        for name in active_types
    }
    coverage = float(train_ann.loc[train_ann["label"].astype(int) == 1, "weak_type_mask"].mean()) if not pos.empty else 0.0
    meta = WeakTypeLabelMeta(
        type_names=active_types.copy(),
        score_thresholds=thresholds,
        fallback_thresholds=fallback_thresholds,
        train_positive_counts=counts,
        train_positive_coverage=coverage,
    )
    return annotated, meta


def type_columns(type_names: list[str] | None = None) -> list[str]:
    names = type_names or TYPE_NAMES
    return [f"weak_type_{name}" for name in names]
