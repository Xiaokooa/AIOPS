"""First-event, causal target construction for OFP."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_time_order(frame: pd.DataFrame) -> np.ndarray:
    if "timestamp" not in frame.columns:
        raise ValueError("missing required column 'timestamp'")
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamp contains missing or non-numeric values")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be non-decreasing within each module")
    return timestamps


def first_anomaly_timestamp(frame: pd.DataFrame) -> float | None:
    timestamps = validate_time_order(frame)
    if "anomaly" not in frame.columns:
        return None
    anomaly = pd.to_numeric(frame["anomaly"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    positive = anomaly > 0
    if not positive.any():
        return None
    return float(timestamps[positive].min())


def first_event_target(
    frame: pd.DataFrame,
    horizon_hours: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (target, train-valid mask, first anomaly timestamp).

    Only observations strictly before the first anomaly are valid training
    rows.  For a healthy module all timestamped rows are valid negatives.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    timestamps = validate_time_order(frame)
    first_ts = first_anomaly_timestamp(frame)
    target = np.zeros(len(frame), dtype=np.int8)
    valid = np.ones(len(frame), dtype=bool)
    if first_ts is None:
        return target, valid, None
    horizon_seconds = float(horizon_hours) * 3600.0
    valid = timestamps < first_ts
    target[(timestamps >= first_ts - horizon_seconds) & valid] = 1
    return target, valid, first_ts
