from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.common import base_utils as base

from OFP_DRAM.config import OFPDRAMFeatureConfig, TRAINING_DIR


SplitRole = Literal["train", "val", "test", "predict"]

RX_LANES = [
    "currentMultiRXPower1",
    "currentMultiRXPower2",
    "currentMultiRXPower3",
    "currentMultiRXPower4",
]
TX_LANES = [
    "currentMultiTXPower1",
    "currentMultiTXPower2",
    "currentMultiTXPower3",
    "currentMultiTXPower4",
]


def _safe_slope(values: np.ndarray) -> float:
    return base.safe_slope(values)


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _series_stats(prefix: str, values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr)
    out: dict[str, float] = {
        f"{prefix}_missing_ratio": float(1.0 - mask.mean()) if len(arr) else 1.0,
        f"{prefix}_valid_points": float(mask.sum()),
    }
    if mask.sum() == 0:
        for name in (
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "range",
            "iqr",
            "first",
            "last",
            "delta",
            "slope",
            "diff_abs_mean",
            "diff_abs_max",
            "diff_large_ratio",
        ):
            out[f"{prefix}_{name}"] = 0.0
        return out

    finite = arr[mask]
    q25, median, q75 = np.percentile(finite, [25, 50, 75])
    diffs = np.diff(finite)
    abs_diffs = np.abs(diffs)
    if len(abs_diffs):
        diff_med = float(np.median(abs_diffs))
        diff_mad = float(np.median(np.abs(abs_diffs - diff_med))) + 1e-9
        large_ratio = float((abs_diffs > diff_med + 3.0 * diff_mad).mean())
        diff_abs_mean = float(abs_diffs.mean())
        diff_abs_max = float(abs_diffs.max())
    else:
        large_ratio = 0.0
        diff_abs_mean = 0.0
        diff_abs_max = 0.0

    out.update(
        {
            f"{prefix}_mean": float(finite.mean()),
            f"{prefix}_std": float(finite.std(ddof=0)),
            f"{prefix}_min": float(finite.min()),
            f"{prefix}_q25": float(q25),
            f"{prefix}_median": float(median),
            f"{prefix}_q75": float(q75),
            f"{prefix}_max": float(finite.max()),
            f"{prefix}_range": float(finite.max() - finite.min()),
            f"{prefix}_iqr": float(q75 - q25),
            f"{prefix}_first": float(finite[0]),
            f"{prefix}_last": float(finite[-1]),
            f"{prefix}_delta": float(finite[-1] - finite[0]),
            f"{prefix}_slope": _safe_slope(arr),
            f"{prefix}_diff_abs_mean": diff_abs_mean,
            f"{prefix}_diff_abs_max": diff_abs_max,
            f"{prefix}_diff_large_ratio": large_ratio,
        }
    )
    return out


def _lane_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame(index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce")


def _lane_group_features(prefix: str, df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    lanes = _lane_frame(df, cols)
    if lanes.empty:
        return _series_stats(f"{prefix}_lane_spread", np.array([], dtype=float))

    row_max = lanes.max(axis=1, skipna=True)
    row_min = lanes.min(axis=1, skipna=True)
    row_mean = lanes.mean(axis=1, skipna=True)
    row_std = lanes.std(axis=1, skipna=True).fillna(0.0)
    denom = row_mean.abs().replace(0.0, np.nan)
    row_cv = (row_std / denom).replace([np.inf, -np.inf], np.nan)
    row_spread = row_max - row_min

    out: dict[str, float] = {}
    out.update(_series_stats(f"{prefix}_lane_mean", row_mean))
    out.update(_series_stats(f"{prefix}_lane_spread", row_spread))
    out.update(_series_stats(f"{prefix}_lane_cv", row_cv))

    valid_rows = lanes.notna().any(axis=1)
    weakest = pd.Series(np.nan, index=lanes.index, dtype=float)
    strongest = pd.Series(np.nan, index=lanes.index, dtype=float)
    lane_map = {col: i + 1 for i, col in enumerate(cols)}
    if valid_rows.any():
        weakest.loc[valid_rows] = lanes.loc[valid_rows].idxmin(axis=1).map(lane_map).astype(float)
        strongest.loc[valid_rows] = lanes.loc[valid_rows].idxmax(axis=1).map(lane_map).astype(float)
    for lane_id in range(1, 5):
        out[f"{prefix}_weakest_lane{lane_id}_ratio"] = float((weakest == lane_id).mean()) if len(weakest) else 0.0
        out[f"{prefix}_strongest_lane{lane_id}_ratio"] = float((strongest == lane_id).mean()) if len(strongest) else 0.0
    return out


def _rx_tx_pair_features(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, (rx_col, tx_col) in enumerate(zip(RX_LANES, TX_LANES), start=1):
        if rx_col in df.columns and tx_col in df.columns:
            delta = pd.to_numeric(df[tx_col], errors="coerce") - pd.to_numeric(df[rx_col], errors="coerce")
            ratio = pd.to_numeric(df[tx_col], errors="coerce") / pd.to_numeric(df[rx_col], errors="coerce").replace(0, np.nan)
            out.update(_series_stats(f"lane{idx}_tx_minus_rx", delta))
            out.update(_series_stats(f"lane{idx}_tx_div_rx", ratio.replace([np.inf, -np.inf], np.nan)))
    if "currentRXPower" in df.columns:
        rx_lanes = _lane_frame(df, RX_LANES)
        if not rx_lanes.empty:
            residual = pd.to_numeric(df["currentRXPower"], errors="coerce") - rx_lanes.mean(axis=1, skipna=True)
            out.update(_series_stats("rx_total_minus_lane_mean", residual))
    if "currentTXPower" in df.columns:
        tx_lanes = _lane_frame(df, TX_LANES)
        if not tx_lanes.empty:
            residual = pd.to_numeric(df["currentTXPower"], errors="coerce") - tx_lanes.mean(axis=1, skipna=True)
            out.update(_series_stats("tx_total_minus_lane_mean", residual))
    return out


def _feature_window(df: pd.DataFrame, end_idx: int, minutes: int) -> tuple[pd.DataFrame, float]:
    steps = max(1, minutes // 5)
    start = max(0, end_idx - steps + 1)
    win = df.iloc[start : end_idx + 1]
    coverage = float(len(win) / steps)
    if "observed" in win.columns and len(win):
        coverage *= float(pd.to_numeric(win["observed"], errors="coerce").fillna(0).mean())
    return win, coverage


def extract_dram_inspired_features(
    df: pd.DataFrame,
    end_idx: int,
    sensors: list[str],
    folder_index: int | None,
    cfg: OFPDRAMFeatureConfig,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for folder_id in (1, 2, 3):
        features[f"static_folder_{folder_id}"] = float(folder_index == folder_id)

    for minutes in cfg.history_minutes:
        win, coverage = _feature_window(df, end_idx, minutes)
        tag = f"h{minutes}m"
        features[f"{tag}_coverage_ratio"] = coverage
        for col in sensors:
            if col in win.columns:
                features.update(_series_stats(f"{tag}_{col}", win[col]))
        features.update({f"{tag}_{k}": v for k, v in _lane_group_features("rx", win, RX_LANES).items()})
        features.update({f"{tag}_{k}": v for k, v in _lane_group_features("tx", win, TX_LANES).items()})
        features.update({f"{tag}_{k}": v for k, v in _rx_tx_pair_features(win).items()})
    return features


def _candidate_end_indices(n_rows: int, step_minutes: int, cfg: OFPDRAMFeatureConfig) -> list[int]:
    if n_rows <= 0:
        return []
    step = max(1, step_minutes // 5)
    start = max(0, cfg.min_history_points - 1)
    return list(range(start, n_rows, step))


def _sample_module_rows(rows: list[dict[str, object]], cfg: OFPDRAMFeatureConfig) -> list[dict[str, object]]:
    if not rows:
        return rows
    df = pd.DataFrame(rows).sort_values("pred_ts").reset_index(drop=True)
    is_faulty = int(df["event_label"].max()) == 1
    max_windows = cfg.train_faulty_max_windows if is_faulty else cfg.train_healthy_max_windows
    if len(df) <= max_windows:
        return rows

    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    if len(positives) >= max_windows:
        idx = np.linspace(0, len(positives) - 1, max_windows, dtype=int)
        return positives.iloc[idx].to_dict("records")

    neg_take = max_windows - len(positives)
    if len(negatives) > neg_take:
        idx = np.linspace(0, len(negatives) - 1, neg_take, dtype=int)
        negatives = negatives.iloc[idx]
    sampled = pd.concat([positives, negatives], ignore_index=True).sort_values("pred_ts")
    return sampled.to_dict("records")


def build_module_frame(
    raw: pd.DataFrame,
    file_name: str,
    folder_index: int | None,
    module_label: int,
    cfg: OFPDRAMFeatureConfig,
    split_role: SplitRole,
) -> pd.DataFrame:
    resampled, sensors, _ = base.resample_module(raw)
    t_first = base.first_failure_timestamp(resampled)
    step_minutes = cfg.train_step_minutes if split_role == "train" else cfg.eval_step_minutes
    rows: list[dict[str, object]] = []
    for end_idx in _candidate_end_indices(len(resampled), step_minutes, cfg):
        pred_ts = int(resampled["timestamp"].iloc[end_idx])
        if t_first is not None and pred_ts >= t_first:
            continue
        _, longest_cov = _feature_window(resampled, end_idx, max(cfg.history_minutes))
        if longest_cov < cfg.min_observed_ratio and end_idx + 1 >= max(cfg.history_minutes) // 5:
            continue
        lead_seconds = int(t_first - pred_ts) if t_first is not None else -1
        label = int(t_first is not None and 0 < lead_seconds <= cfg.ahead_seconds)
        rec = extract_dram_inspired_features(resampled, end_idx, sensors, folder_index, cfg)
        rec.update(
            {
                "file_name": file_name,
                "folder_index": int(folder_index) if folder_index is not None else -1,
                "module_label": int(module_label),
                "split_role": split_role,
                "row_index": int(end_idx),
                "pred_ts": pred_ts,
                "first_failure_ts": int(t_first) if t_first is not None else -1,
                "event_label": int(t_first is not None),
                "label": label,
                "lead_seconds": lead_seconds,
                "lead_hours": float(lead_seconds / 3600.0) if lead_seconds > 0 else np.nan,
                "ahead_hours": int(cfg.ahead_hours),
            }
        )
        rows.append(rec)
    if split_role == "train":
        rows = _sample_module_rows(rows, cfg)
    return pd.DataFrame(rows)


def build_feature_frame(
    split_df: pd.DataFrame,
    cfg: OFPDRAMFeatureConfig,
    split_role: SplitRole,
    data_dir: Path = TRAINING_DIR,
    max_modules: int | None = None,
    progress_every: int = 0,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    local = split_df.reset_index(drop=True)
    if max_modules is not None:
        local = local.head(max_modules)
    total = int(len(local))
    for idx, row in enumerate(local.itertuples(index=False), start=1):
        file_name = str(getattr(row, "file_name"))
        folder_index = int(getattr(row, "folder_index", -1))
        module_label = int(getattr(row, "Label", 0))
        path = data_dir / file_name
        if not path.exists():
            if progress_every and (idx % progress_every == 0 or idx == total):
                print(
                    f"[features:{split_role}] {idx}/{total} modules, windows={sum(len(p) for p in pieces)}",
                    flush=True,
                )
            continue
        raw = pd.read_csv(path, low_memory=False)
        frame = build_module_frame(raw, file_name, folder_index, module_label, cfg, split_role)
        if not frame.empty:
            pieces.append(frame)
        if progress_every and (idx % progress_every == 0 or idx == total):
            print(
                f"[features:{split_role}] {idx}/{total} modules, windows={sum(len(p) for p in pieces)}",
                flush=True,
            )
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


META_COLS = {
    "file_name",
    "folder_index",
    "module_label",
    "split_role",
    "row_index",
    "pred_ts",
    "first_failure_ts",
    "event_label",
    "label",
    "lead_seconds",
    "lead_hours",
    "ahead_hours",
}


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    feature_cols = [c for c in frame.columns if c not in META_COLS]
    X = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    meta = frame[[c for c in META_COLS if c in frame.columns]].copy()
    return X, y, meta, feature_cols
