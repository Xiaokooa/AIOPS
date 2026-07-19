from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from OFP.deep_learning.optical_prediction.common import base_utils as base
from OFP.deep_learning.optical_prediction.common import cache_utils


ScopeMode = Literal["full", "prefirst"]
LabelMode = Literal["future_any", "first_in_future", "first_in_prediction"]
ModeType = Literal["feature", "sequence"]
FeatureProfile = Literal["fast", "full", "raw"]
WindowMode = Literal["rolling", "fault_anchored"]
FEATURE_SCHEMA_VERSION = "v5shortgap"
WINDOW_INDEX_SCHEMA_VERSION = "windowindex_v2"


@dataclass(frozen=True)
class Week1TaskConfig:
    obs_minutes: int = 30
    lead_minutes: int = 15
    pred_minutes: int = 60
    step_minutes: int = 15
    min_obs_coverage: float = 0.80
    min_lead_coverage: float = 0.80
    min_pred_coverage: float = 0.80
    feature_profile: FeatureProfile = "fast"
    train_scope: ScopeMode = "prefirst"
    eval_scope: ScopeMode = "prefirst"
    train_window_mode: WindowMode = "rolling"
    eval_window_mode: WindowMode = "rolling"
    train_label_mode: LabelMode = "first_in_future"
    eval_label_mode: LabelMode = "first_in_prediction"
    train_faulty_max_windows: int = 96
    train_healthy_max_windows: int = 24
    train_faulty_build_stride_minutes: int = 30
    train_healthy_build_stride_minutes: int = 360
    eval_fast_sampling: bool = True
    eval_recent_minutes: int = 120
    eval_faulty_max_windows: int = 80
    eval_faulty_far_windows: int = 16
    eval_healthy_max_windows: int = 24
    eval_healthy_build_stride_minutes: int = 360

    def __post_init__(self) -> None:
        for field in (
            "obs_minutes",
            "lead_minutes",
            "pred_minutes",
            "step_minutes",
            "train_faulty_build_stride_minutes",
            "train_healthy_build_stride_minutes",
            "eval_healthy_build_stride_minutes",
        ):
            value = getattr(self, field)
            if value <= 0 or value % 5 != 0:
                raise ValueError(f"{field} must be a positive multiple of 5 minutes, got {value}")

    @property
    def obs_steps(self) -> int:
        return self.obs_minutes // 5

    @property
    def lead_steps(self) -> int:
        return self.lead_minutes // 5

    @property
    def pred_steps(self) -> int:
        return self.pred_minutes // 5

    @property
    def step_steps(self) -> int:
        return self.step_minutes // 5

    @property
    def total_steps(self) -> int:
        return self.obs_steps + self.lead_steps + self.pred_steps

    @property
    def window_mode_tag(self) -> str:
        train_mode = "fa" if self.train_window_mode == "fault_anchored" else "ro"
        eval_mode = "fa" if self.eval_window_mode == "fault_anchored" else "ro"
        return f"wm{train_mode}{eval_mode}"

    @property
    def tag(self) -> str:
        return (
            f"obs{self.obs_minutes}m_lead{self.lead_minutes}m_"
            f"pred{self.pred_minutes}m_step{self.step_minutes}m"
        )

    @property
    def cache_tag(self) -> str:
        return (
            f"{FEATURE_SCHEMA_VERSION}_{self.tag}_"
            f"feat-{self.feature_profile}_"
            f"cov{self.min_obs_coverage:.2f}-{self.min_lead_coverage:.2f}-{self.min_pred_coverage:.2f}_"
            f"{self.window_mode_tag}_"
            f"train-{self.train_scope}-{self.train_label_mode}_eval-{self.eval_scope}-{self.eval_label_mode}_"
            f"sample-f{self.train_faulty_max_windows}-h{self.train_healthy_max_windows}_"
            f"stride-f{self.train_faulty_build_stride_minutes}-h{self.train_healthy_build_stride_minutes}_"
            f"evalfast-{int(self.eval_fast_sampling)}-recent{self.eval_recent_minutes}-"
            f"ef{self.eval_faulty_max_windows}-far{self.eval_faulty_far_windows}-"
            f"eh{self.eval_healthy_max_windows}-hs{self.eval_healthy_build_stride_minutes}"
        ).replace(".", "p")

    @property
    def window_index_cache_tag(self) -> str:
        return (
            f"{WINDOW_INDEX_SCHEMA_VERSION}_{self.tag}_"
            f"cov{self.min_obs_coverage:.2f}-{self.min_lead_coverage:.2f}-{self.min_pred_coverage:.2f}_"
            f"{self.window_mode_tag}_"
            f"train-{self.train_scope}-{self.train_label_mode}_eval-{self.eval_scope}-{self.eval_label_mode}_"
            f"sample-f{self.train_faulty_max_windows}-h{self.train_healthy_max_windows}_"
            f"stride-f{self.train_faulty_build_stride_minutes}-h{self.train_healthy_build_stride_minutes}_"
            f"evalfast-{int(self.eval_fast_sampling)}-recent{self.eval_recent_minutes}-"
            f"ef{self.eval_faulty_max_windows}-far{self.eval_faulty_far_windows}-"
            f"eh{self.eval_healthy_max_windows}-hs{self.eval_healthy_build_stride_minutes}"
        ).replace(".", "p")


def _window_has_anomaly(win: pd.DataFrame) -> tuple[int, float]:
    mask = win["observed"].astype(bool) & (pd.to_numeric(win["anomaly"], errors="coerce").fillna(0.0) > 0.0)
    return int(mask.any()), float(mask.mean()) if len(win) else 0.0


def _aggregate_fast_window_features(obs_df: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    features: dict[str, float] = {"obs_coverage_ratio": float(obs_df["observed"].mean())}
    for col in sensors:
        values = pd.to_numeric(obs_df[col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        features[f"{col}_missing_ratio"] = float(1.0 - finite_mask.mean()) if len(values) else 1.0
        features[f"{col}_valid_points"] = float(finite_mask.sum())
        if finite_mask.sum() == 0:
            for name in ("mean", "std", "min", "max", "range", "first", "last", "delta", "slope"):
                features[f"{col}_{name}"] = 0.0
            continue
        finite = values[finite_mask]
        first = float(finite[0])
        last = float(finite[-1])
        features[f"{col}_mean"] = float(finite.mean())
        features[f"{col}_std"] = float(finite.std(ddof=0))
        features[f"{col}_min"] = float(finite.min())
        features[f"{col}_max"] = float(finite.max())
        features[f"{col}_range"] = float(finite.max() - finite.min())
        features[f"{col}_first"] = first
        features[f"{col}_last"] = last
        features[f"{col}_delta"] = float(last - first)
        features[f"{col}_slope"] = base.safe_slope(values)
    return features


def _aggregate_raw_window_features(obs_df: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for step_idx, (_, row) in enumerate(obs_df.reset_index(drop=True).iterrows()):
        for col in sensors:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            features[f"{col}_t{step_idx:02d}"] = float(value) if np.isfinite(value) else 0.0
    return features


def _make_label(
    t_first: int | None,
    obs_end_ts: int,
    lead_end_ts: int,
    pred_end_ts: int,
    lead_has_anomaly: int,
    pred_has_anomaly: int,
    label_mode: LabelMode,
) -> int:
    if label_mode == "future_any":
        return int(bool(lead_has_anomaly or pred_has_anomaly))
    if t_first is None:
        return 0
    if label_mode == "first_in_future":
        return int(obs_end_ts < t_first <= pred_end_ts)
    if label_mode == "first_in_prediction":
        return int(lead_end_ts < t_first <= pred_end_ts)
    raise ValueError(f"Unknown label_mode: {label_mode}")


def _uniform_take_indices(values: list[int], n: int) -> list[int]:
    if len(values) <= n:
        return list(values)
    idx = np.linspace(0, len(values) - 1, n, dtype=int)
    return [values[int(i)] for i in idx]


def _grid_starts(n_rows: int, total_steps: int, stride_steps: int) -> list[int]:
    if n_rows < total_steps:
        return []
    return list(range(0, n_rows - total_steps + 1, max(stride_steps, 1)))


def _fault_anchored_start(df: pd.DataFrame, cfg: Week1TaskConfig, t_first: int | None) -> list[int]:
    """Return one start whose prediction-window end is aligned to first failure.

    For R4 this gives:
      obs: 24h history
      lead: 1h gap
      pred: 1h prediction interval ending at the first observed failure

    This keeps the task predictive: the observation window still ends
    lead+pred minutes before the first failure timestamp.
    """
    if t_first is None or len(df) < cfg.total_steps:
        return []
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)
    pred_end_idx = int(np.searchsorted(timestamps, int(t_first), side="left"))
    if pred_end_idx >= len(timestamps) or int(timestamps[pred_end_idx]) != int(t_first):
        pred_end_idx = int(np.searchsorted(timestamps, int(t_first), side="right") - 1)
    start = pred_end_idx - cfg.total_steps + 1
    if start < 0 or start > len(df) - cfg.total_steps:
        return []
    return [int(start)]


def _candidate_starts(
    df: pd.DataFrame,
    cfg: Week1TaskConfig,
    split_role: Literal["train", "val", "test"],
    t_first: int | None,
    scope: ScopeMode,
) -> list[int]:
    window_mode = cfg.train_window_mode if split_role == "train" else cfg.eval_window_mode
    if window_mode == "fault_anchored" and t_first is not None:
        return _fault_anchored_start(df, cfg, t_first)

    if split_role == "train":
        stride_minutes = (
            cfg.train_faulty_build_stride_minutes if t_first is not None else cfg.train_healthy_build_stride_minutes
        )
        return _grid_starts(df_len := len(df), cfg.total_steps, max(cfg.step_steps, stride_minutes // 5))

    if not cfg.eval_fast_sampling:
        return _grid_starts(len(df), cfg.total_steps, cfg.step_steps)

    if t_first is None:
        stride_steps = max(cfg.step_steps, cfg.eval_healthy_build_stride_minutes // 5)
        starts = _grid_starts(len(df), cfg.total_steps, stride_steps)
        return _uniform_take_indices(starts, cfg.eval_healthy_max_windows)

    dense_starts = _grid_starts(len(df), cfg.total_steps, cfg.step_steps)
    if not dense_starts:
        return []

    near: list[int] = []
    far: list[int] = []
    recent_minutes = max(cfg.eval_recent_minutes, cfg.lead_minutes + cfg.pred_minutes)
    for start in dense_starts:
        obs_end_idx = start + cfg.obs_steps - 1
        obs_end_ts = int(df["timestamp"].iloc[obs_end_idx])
        if scope == "prefirst" and t_first <= obs_end_ts:
            continue
        time_to_first = float((t_first - obs_end_ts) / 60.0)
        if 0.0 < time_to_first <= recent_minutes:
            near.append(start)
        elif time_to_first > recent_minutes:
            far.append(start)

    near = _uniform_take_indices(near, cfg.eval_faulty_max_windows)
    far = _uniform_take_indices(far, cfg.eval_faulty_far_windows)
    return sorted(set(near + far))


def extract_week1_window_index(
    df: pd.DataFrame,
    file_name: str,
    module_label: int,
    cfg: Week1TaskConfig,
    split_role: Literal["train", "val", "test"] = "train",
) -> list[dict[str, object]]:
    if len(df) < cfg.total_steps:
        return []

    t_first = base.first_failure_timestamp(df)
    scope = cfg.train_scope if split_role == "train" else cfg.eval_scope
    label_mode = cfg.train_label_mode if split_role == "train" else cfg.eval_label_mode
    window_mode = cfg.train_window_mode if split_role == "train" else cfg.eval_window_mode

    rows: list[dict[str, object]] = []
    starts = _candidate_starts(df, cfg, split_role, t_first, scope)
    for start in starts:
        obs_start_idx = start
        obs_end_idx = start + cfg.obs_steps - 1
        lead_start_idx = start + cfg.obs_steps
        lead_end_idx = start + cfg.obs_steps + cfg.lead_steps - 1
        pred_start_idx = start + cfg.obs_steps + cfg.lead_steps
        pred_end_idx = start + cfg.total_steps - 1

        obs = df.iloc[obs_start_idx : obs_end_idx + 1]
        lead = df.iloc[lead_start_idx : lead_end_idx + 1]
        pred = df.iloc[pred_start_idx : pred_end_idx + 1]

        obs_cov = float(obs["observed"].mean())
        lead_cov = float(lead["observed"].mean()) if len(lead) else 1.0
        pred_cov = float(pred["observed"].mean()) if len(pred) else 0.0
        if (
            obs_cov < cfg.min_obs_coverage
            or lead_cov < cfg.min_lead_coverage
            or pred_cov < cfg.min_pred_coverage
        ):
            continue

        obs_end_ts = int(obs["timestamp"].iloc[-1])
        lead_start_ts = int(lead["timestamp"].iloc[0]) if len(lead) else obs_end_ts
        lead_end_ts = int(lead["timestamp"].iloc[-1]) if len(lead) else obs_end_ts
        pred_start_ts = int(pred["timestamp"].iloc[0])
        pred_end_ts = int(pred["timestamp"].iloc[-1])

        if scope == "prefirst" and t_first is not None and t_first <= obs_end_ts:
            continue

        obs_has_anomaly, obs_anomaly_ratio = _window_has_anomaly(obs)
        lead_has_anomaly, lead_anomaly_ratio = _window_has_anomaly(lead) if len(lead) else (0, 0.0)
        pred_has_anomaly, pred_anomaly_ratio = _window_has_anomaly(pred)
        future_has_anomaly = int(bool(lead_has_anomaly or pred_has_anomaly))
        future_anomaly_ratio = float(
            (lead_anomaly_ratio * len(lead) + pred_anomaly_ratio * len(pred))
            / max(len(lead) + len(pred), 1)
        )

        label = _make_label(
            t_first=t_first,
            obs_end_ts=obs_end_ts,
            lead_end_ts=lead_end_ts,
            pred_end_ts=pred_end_ts,
            lead_has_anomaly=lead_has_anomaly,
            pred_has_anomaly=pred_has_anomaly,
            label_mode=label_mode,
        )
        first_in_future = int(t_first is not None and obs_end_ts < t_first <= pred_end_ts)
        first_in_prediction = int(t_first is not None and lead_end_ts < t_first <= pred_end_ts)
        time_to_first_minutes = float((t_first - obs_end_ts) / 60.0) if t_first is not None else np.nan

        rows.append(
            {
                "file_name": file_name,
                "module_label": int(module_label),
                "window_index": int(start // max(cfg.step_steps, 1)),
                "start_idx": int(start),
                "obs_start_idx": int(obs_start_idx),
                "obs_end_idx": int(obs_end_idx),
                "lead_start_idx": int(lead_start_idx),
                "lead_end_idx": int(lead_end_idx),
                "pred_start_idx": int(pred_start_idx),
                "pred_end_idx": int(pred_end_idx),
                "split_role": split_role,
                "scope_mode": scope,
                "window_mode": window_mode,
                "label_mode": label_mode,
                "obs_start_ts": int(obs["timestamp"].iloc[0]),
                "obs_end_ts": obs_end_ts,
                "lead_start_ts": lead_start_ts,
                "lead_end_ts": lead_end_ts,
                "pred_start_ts": pred_start_ts,
                "pred_end_ts": pred_end_ts,
                "first_failure_ts": int(t_first) if t_first is not None else -1,
                "event_label": int(t_first is not None),
                "label": int(label),
                "future_has_anomaly": future_has_anomaly,
                "future_anomaly_ratio": future_anomaly_ratio,
                "first_in_future": first_in_future,
                "first_in_prediction": first_in_prediction,
                "obs_has_anomaly": obs_has_anomaly,
                "lead_has_anomaly": lead_has_anomaly,
                "pred_has_anomaly": pred_has_anomaly,
                "obs_anomaly_ratio": obs_anomaly_ratio,
                "lead_anomaly_ratio": lead_anomaly_ratio,
                "pred_anomaly_ratio": pred_anomaly_ratio,
                "time_to_first_minutes": time_to_first_minutes,
                "obs_coverage_ratio": obs_cov,
                "lead_coverage_ratio": lead_cov,
                "pred_coverage_ratio": pred_cov,
            }
        )
    return rows


def extract_week1_windows(
    df: pd.DataFrame,
    sensors: list[str],
    file_name: str,
    module_label: int,
    cfg: Week1TaskConfig,
    split_role: Literal["train", "val", "test"] = "train",
    mode: ModeType = "feature",
    normalize_sequence: bool = False,
) -> list[dict[str, object]]:
    if len(df) < cfg.total_steps:
        return []

    t_first = base.first_failure_timestamp(df)
    scope = cfg.train_scope if split_role == "train" else cfg.eval_scope
    label_mode = cfg.train_label_mode if split_role == "train" else cfg.eval_label_mode
    window_mode = cfg.train_window_mode if split_role == "train" else cfg.eval_window_mode

    rows: list[dict[str, object]] = []
    starts = _candidate_starts(df, cfg, split_role, t_first, scope)
    for start in starts:
        obs = df.iloc[start : start + cfg.obs_steps].copy()
        lead = df.iloc[start + cfg.obs_steps : start + cfg.obs_steps + cfg.lead_steps].copy()
        pred = df.iloc[start + cfg.obs_steps + cfg.lead_steps : start + cfg.total_steps].copy()

        obs_cov = float(obs["observed"].mean())
        lead_cov = float(lead["observed"].mean()) if len(lead) else 1.0
        pred_cov = float(pred["observed"].mean()) if len(pred) else 0.0
        if (
            obs_cov < cfg.min_obs_coverage
            or lead_cov < cfg.min_lead_coverage
            or pred_cov < cfg.min_pred_coverage
        ):
            continue

        obs_end_ts = int(obs["timestamp"].iloc[-1])
        lead_start_ts = int(lead["timestamp"].iloc[0]) if len(lead) else obs_end_ts
        lead_end_ts = int(lead["timestamp"].iloc[-1]) if len(lead) else obs_end_ts
        pred_start_ts = int(pred["timestamp"].iloc[0])
        pred_end_ts = int(pred["timestamp"].iloc[-1])

        if scope == "prefirst" and t_first is not None and t_first <= obs_end_ts:
            continue

        obs_has_anomaly, obs_anomaly_ratio = _window_has_anomaly(obs)
        lead_has_anomaly, lead_anomaly_ratio = _window_has_anomaly(lead) if len(lead) else (0, 0.0)
        pred_has_anomaly, pred_anomaly_ratio = _window_has_anomaly(pred)
        future_has_anomaly = int(bool(lead_has_anomaly or pred_has_anomaly))
        future_anomaly_ratio = float((lead_anomaly_ratio * len(lead) + pred_anomaly_ratio * len(pred)) / max(len(lead) + len(pred), 1))

        label = _make_label(
            t_first=t_first,
            obs_end_ts=obs_end_ts,
            lead_end_ts=lead_end_ts,
            pred_end_ts=pred_end_ts,
            lead_has_anomaly=lead_has_anomaly,
            pred_has_anomaly=pred_has_anomaly,
            label_mode=label_mode,
        )
        first_in_future = int(t_first is not None and obs_end_ts < t_first <= pred_end_ts)
        first_in_prediction = int(t_first is not None and lead_end_ts < t_first <= pred_end_ts)
        time_to_first_minutes = float((t_first - obs_end_ts) / 60.0) if t_first is not None else np.nan

        base_meta: dict[str, object] = {
            "file_name": file_name,
            "module_label": int(module_label),
            "window_index": int(start // max(cfg.step_steps, 1)),
            "split_role": split_role,
            "scope_mode": scope,
            "window_mode": window_mode,
            "label_mode": label_mode,
            "obs_start_ts": int(obs["timestamp"].iloc[0]),
            "obs_end_ts": obs_end_ts,
            "lead_start_ts": lead_start_ts,
            "lead_end_ts": lead_end_ts,
            "pred_start_ts": pred_start_ts,
            "pred_end_ts": pred_end_ts,
            "first_failure_ts": int(t_first) if t_first is not None else -1,
            "event_label": int(t_first is not None),
            "label": int(label),
            "future_has_anomaly": future_has_anomaly,
            "future_anomaly_ratio": future_anomaly_ratio,
            "first_in_future": first_in_future,
            "first_in_prediction": first_in_prediction,
            "obs_has_anomaly": obs_has_anomaly,
            "lead_has_anomaly": lead_has_anomaly,
            "pred_has_anomaly": pred_has_anomaly,
            "obs_anomaly_ratio": obs_anomaly_ratio,
            "lead_anomaly_ratio": lead_anomaly_ratio,
            "pred_anomaly_ratio": pred_anomaly_ratio,
            "time_to_first_minutes": time_to_first_minutes,
            "obs_coverage_ratio": obs_cov,
            "lead_coverage_ratio": lead_cov,
            "pred_coverage_ratio": pred_cov,
        }

        if mode == "feature":
            if cfg.feature_profile == "full":
                rec = base.aggregate_window_features(obs, sensors)
            elif cfg.feature_profile == "fast":
                rec = _aggregate_fast_window_features(obs, sensors)
            elif cfg.feature_profile == "raw":
                rec = _aggregate_raw_window_features(obs, sensors)
            else:
                raise ValueError(f"Unknown feature_profile: {cfg.feature_profile}")
            rec.update(base_meta)
            rows.append(rec)
        elif mode == "sequence":
            rec = dict(base_meta)
            rec["sequence"] = base.observation_array(obs, sensors, normalize=normalize_sequence)
            rows.append(rec)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return rows


def _build_rows(
    split_df: pd.DataFrame,
    cfg: Week1TaskConfig,
    split_role: Literal["train", "val", "test"],
    mode: ModeType,
    cache_path: Path | None,
    force_rebuild: bool,
    normalize_sequence: bool = False,
) -> list[dict[str, object]]:
    if cache_path and cache_path.exists() and not force_rebuild:
        if mode == "feature":
            return pd.read_pickle(cache_path).to_dict("records")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    rows: list[dict[str, object]] = []
    for row in split_df.itertuples(index=False):
        file_name = getattr(row, "file_name")
        module_label = int(getattr(row, "Label"))
        fpath = base.TRAINING_DIR / file_name
        if not fpath.exists():
            continue
        resampled, sensors, _ = cache_utils.load_resampled_module(file_name)
        local_rows = extract_week1_windows(
            resampled,
            sensors,
            file_name,
            module_label,
            cfg,
            split_role=split_role,
            mode=mode,
            normalize_sequence=normalize_sequence,
        )
        if split_role == "train" and mode == "feature":
            local_rows = _sample_train_rows(local_rows, cfg)
        rows.extend(local_rows)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "feature":
            pd.DataFrame(rows).to_pickle(cache_path)
        else:
            with open(cache_path, "wb") as f:
                pickle.dump(rows, f)
    return rows


def _uniform_take(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[idx].copy()


def _sample_train_rows(rows: list[dict[str, object]], cfg: Week1TaskConfig) -> list[dict[str, object]]:
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    event_label = int(df["event_label"].max()) if "event_label" in df.columns else 0
    max_windows = cfg.train_faulty_max_windows if event_label == 1 else cfg.train_healthy_max_windows
    if len(df) <= max_windows:
        return rows

    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].copy()
    near_df = pd.DataFrame()
    if event_label == 1 and "time_to_first_minutes" in df.columns:
        recent_minutes = max(cfg.eval_recent_minutes, cfg.lead_minutes + cfg.pred_minutes)
        near_mask = (
            df["time_to_first_minutes"].apply(np.isfinite)
            & (df["time_to_first_minutes"] > 0)
            & (df["time_to_first_minutes"] <= recent_minutes)
        )
        if "obs_has_anomaly" in df.columns:
            near_mask = near_mask & (df["obs_has_anomaly"].astype(int) == 0)
        near_df = df[near_mask].copy()

    priority = pd.concat([pos_df, near_df], ignore_index=True).drop_duplicates(
        subset=["file_name", "window_index"], keep="first"
    )
    if len(priority) >= max_windows:
        sampled = _uniform_take(priority.sort_values("window_index"), max_windows)
        return sampled.to_dict("records")

    keep_neg = max_windows - len(priority)
    neg_pool = neg_df.merge(
        priority[["file_name", "window_index"]],
        on=["file_name", "window_index"],
        how="left",
        indicator=True,
    )
    neg_pool = neg_pool[neg_pool["_merge"] == "left_only"].drop(columns=["_merge"])
    sampled_neg = _uniform_take(neg_pool, keep_neg)
    sampled = pd.concat([priority, sampled_neg], ignore_index=True).sort_values("window_index").reset_index(drop=True)
    return sampled.to_dict("records")


def build_window_index_frame(
    split_df: pd.DataFrame,
    cfg: Week1TaskConfig,
    split_role: Literal["train", "val", "test"],
    cache_path: Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if cache_path and cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    rows: list[dict[str, object]] = []
    for row in split_df.itertuples(index=False):
        file_name = getattr(row, "file_name")
        module_label = int(getattr(row, "Label"))
        if not (base.TRAINING_DIR / file_name).exists():
            continue
        resampled, _, _ = cache_utils.load_resampled_module(file_name)
        local_rows = extract_week1_window_index(
            resampled,
            file_name,
            module_label,
            cfg,
            split_role=split_role,
        )
        if split_role == "train":
            local_rows = _sample_train_rows(local_rows, cfg)
        rows.extend(local_rows)

    out = pd.DataFrame(rows)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)
    return out


def _feature_rows_from_window_index(
    index_df: pd.DataFrame,
    cfg: Week1TaskConfig,
    mode: ModeType = "feature",
    normalize_sequence: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if index_df.empty:
        return rows

    for file_name, group in index_df.groupby("file_name", sort=False):
        resampled, sensors, _ = cache_utils.load_resampled_module(str(file_name))
        for meta in group.to_dict("records"):
            start = int(meta["start_idx"])
            obs = resampled.iloc[start : start + cfg.obs_steps].copy()
            if mode == "feature":
                if cfg.feature_profile == "full":
                    rec = base.aggregate_window_features(obs, sensors)
                elif cfg.feature_profile == "fast":
                    rec = _aggregate_fast_window_features(obs, sensors)
                elif cfg.feature_profile == "raw":
                    rec = _aggregate_raw_window_features(obs, sensors)
                else:
                    raise ValueError(f"Unknown feature_profile: {cfg.feature_profile}")
            elif mode == "sequence":
                rec = {"sequence": base.observation_array(obs, sensors, normalize=normalize_sequence)}
            else:
                raise ValueError(f"Unknown mode: {mode}")
            rec.update(meta)
            rows.append(rec)
    return rows


def build_feature_frame(
    split_df: pd.DataFrame,
    cfg: Week1TaskConfig,
    split_role: Literal["train", "val", "test"],
    cache_path: Path | None = None,
    window_index_cache_path: Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if cache_path and cache_path.exists() and not force_rebuild:
        return pd.read_pickle(cache_path)

    window_index = build_window_index_frame(
        split_df,
        cfg,
        split_role,
        cache_path=window_index_cache_path,
        force_rebuild=force_rebuild,
    )
    rows = _feature_rows_from_window_index(window_index, cfg, mode="feature")
    out = pd.DataFrame(rows)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(cache_path)
    return out


def apply_leadtime_tps_frame(
    df: pd.DataFrame,
    target_minutes: list[int],
    tolerance_minutes: int = 5,
    require_clean_observation: bool = True,
) -> pd.DataFrame:
    """
    Week1-style lead-time sampling:
    pick windows close to selected minute offsets before the first anomaly and
    relabel them positive. This explicitly supports the case where the sampled
    observation itself has anomaly=0 but should teach the model a pre-failure
    state.
    """
    if df.empty or not target_minutes:
        return df.copy()

    base_df = df.copy()
    if "tps_is_synthetic" not in base_df.columns:
        base_df["tps_is_synthetic"] = 0
    if "tps_original_label" not in base_df.columns:
        base_df["tps_original_label"] = base_df["label"].astype(int)
    out_parts = [base_df]
    for _, group in df.groupby("file_name", sort=False):
        if int(group["event_label"].max()) != 1:
            continue
        candidates = group[group["time_to_first_minutes"].apply(np.isfinite)].copy()
        candidates = candidates[candidates["time_to_first_minutes"] > 0]
        if require_clean_observation and "obs_has_anomaly" in candidates.columns:
            candidates = candidates[candidates["obs_has_anomaly"].astype(int) == 0]
        if candidates.empty:
            continue
        for target in target_minutes:
            local = candidates[
                np.abs(candidates["time_to_first_minutes"] - float(target)) <= tolerance_minutes
            ]
            if local.empty:
                nearest_idx = (candidates["time_to_first_minutes"] - float(target)).abs().idxmin()
                local = candidates.loc[[nearest_idx]].copy()
            else:
                nearest_idx = (local["time_to_first_minutes"] - float(target)).abs().idxmin()
                local = local.loc[[nearest_idx]].copy()
            local["tps_original_label"] = local["label"].astype(int)
            local["label"] = 1
            local["tps_is_synthetic"] = 1
            local["tps_target_minutes"] = int(target)
            out_parts.append(local)
    return pd.concat(out_parts, ignore_index=True)


def progressive_tps_targets(lead_minutes: int, step_minutes: int = 15) -> list[int]:
    """NTAM-style TPS targets mapped to minute-level optical data.

    NTAM uses a leading-time hyper-parameter L and progressively collects
    failed samples at l=1..L before failure. Here each l is one
    `step_minutes` interval, so lead=60 and step=15 yields
    [15, 30, 45, 60].
    """
    if lead_minutes <= 0:
        return []
    if step_minutes <= 0:
        raise ValueError(f"step_minutes must be positive, got {step_minutes}")
    return list(range(step_minutes, lead_minutes + 1, step_minutes))


def evaluate_event_level(meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    eval_df = meta_df.copy().reset_index(drop=True)
    eval_df["y_true"] = np.asarray(y_true, dtype=int)
    eval_df["y_pred"] = np.asarray(y_pred, dtype=int)

    if "event_label" in eval_df.columns:
        event_modules = eval_df.groupby("file_name")["event_label"].max()
        event_modules = event_modules[event_modules == 1].index.tolist()
    else:
        event_modules = eval_df.groupby("file_name")["y_true"].max()
        event_modules = event_modules[event_modules == 1].index.tolist()

    evaluable_event_modules = []
    hits = 0
    hit_leads = []
    for file_name in event_modules:
        sub = eval_df[eval_df["file_name"] == file_name]
        positive_rows = sub[sub["y_true"] == 1]
        if positive_rows.empty:
            continue
        evaluable_event_modules.append(file_name)
        hit_rows = positive_rows[positive_rows["y_pred"] == 1]
        if not hit_rows.empty:
            hits += 1
            if "time_to_first_minutes" in hit_rows.columns:
                hit_leads.append(float(hit_rows["time_to_first_minutes"].max()))
            elif "time_to_failure_hours" in hit_rows.columns:
                hit_leads.append(float(hit_rows["time_to_failure_hours"].max() * 60.0))

    healthy_modules = eval_df.groupby("file_name")["event_label"].max() if "event_label" in eval_df.columns else eval_df.groupby("file_name")["y_true"].max()
    healthy_modules = healthy_modules[healthy_modules == 0].index.tolist()
    false_alarm_modules = 0
    for file_name in healthy_modules:
        if (eval_df.loc[eval_df["file_name"] == file_name, "y_pred"] == 1).any():
            false_alarm_modules += 1

    return {
        "event_modules_total": int(len(event_modules)),
        "event_modules_evaluable": int(len(evaluable_event_modules)),
        "event_modules_hit": int(hits),
        "event_hit_rate": float(hits / len(event_modules)) if event_modules else 0.0,
        "event_hit_rate_evaluable": float(hits / len(evaluable_event_modules)) if evaluable_event_modules else 0.0,
        "mean_hit_lead_minutes": float(np.mean(hit_leads)) if hit_leads else 0.0,
        "median_hit_lead_minutes": float(np.median(hit_leads)) if hit_leads else 0.0,
        "healthy_modules_total": int(len(healthy_modules)),
        "false_alarm_modules": int(false_alarm_modules),
        "healthy_module_false_alarm_rate": (
            float(false_alarm_modules / len(healthy_modules)) if healthy_modules else 0.0
        ),
    }


def evaluate_event_level_all_modules(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_df: pd.DataFrame,
    label_col: str = "Label",
) -> dict[str, object]:
    """Event metrics using every module in the original split as denominator.

    `evaluate_event_level` only sees modules that survive the window extraction
    policy. For paper reporting we also need a deployment-facing denominator:
    every faulty module in the split, including modules that fail at the first
    observed point and therefore have no clean pre-failure prediction window.

    A faulty module is counted as a hit only if one of its positive windows is
    predicted positive, matching the existing event-hit definition. Faulty
    modules without valid positive windows are explicit misses under this
    all-module view.
    """
    split_index = split_df.copy()
    split_index["file_name"] = split_index["file_name"].astype(str)
    labels = pd.to_numeric(split_index[label_col], errors="coerce").fillna(0).astype(int)
    all_fault_modules = set(split_index.loc[labels == 1, "file_name"].tolist())
    all_healthy_modules = set(split_index.loc[labels == 0, "file_name"].tolist())

    eval_df = meta_df.copy().reset_index(drop=True)
    if eval_df.empty:
        eval_df["file_name"] = pd.Series(dtype=str)
        eval_df["y_true"] = pd.Series(dtype=int)
        eval_df["y_pred"] = pd.Series(dtype=int)
    else:
        eval_df["file_name"] = eval_df["file_name"].astype(str)
        eval_df["y_true"] = np.asarray(y_true, dtype=int)
        eval_df["y_pred"] = np.asarray(y_pred, dtype=int)

    modules_with_windows = set(eval_df["file_name"].unique().tolist())
    fault_modules_with_windows = all_fault_modules & modules_with_windows
    healthy_modules_with_windows = all_healthy_modules & modules_with_windows

    positive_rows = eval_df[eval_df["y_true"] == 1]
    positive_window_modules = set(positive_rows["file_name"].unique().tolist()) & all_fault_modules
    hit_rows = positive_rows[positive_rows["y_pred"] == 1]
    hit_modules = set(hit_rows["file_name"].unique().tolist()) & all_fault_modules

    any_alert_fault_modules = set(
        eval_df.loc[(eval_df["file_name"].isin(all_fault_modules)) & (eval_df["y_pred"] == 1), "file_name"]
        .unique()
        .tolist()
    )
    false_alarm_modules = set(
        eval_df.loc[(eval_df["file_name"].isin(all_healthy_modules)) & (eval_df["y_pred"] == 1), "file_name"]
        .unique()
        .tolist()
    )

    hit_leads = []
    if not hit_rows.empty and "time_to_first_minutes" in hit_rows.columns:
        for _, group in hit_rows.groupby("file_name"):
            hit_leads.append(float(group["time_to_first_minutes"].max()))

    n_fault = len(all_fault_modules)
    n_healthy = len(all_healthy_modules)
    n_positive_window_modules = len(positive_window_modules)
    return {
        "all_fault_modules_total": int(n_fault),
        "all_fault_modules_with_windows": int(len(fault_modules_with_windows)),
        "all_fault_modules_without_windows": int(n_fault - len(fault_modules_with_windows)),
        "all_fault_modules_with_positive_windows": int(n_positive_window_modules),
        "all_fault_modules_without_positive_windows": int(n_fault - n_positive_window_modules),
        "all_fault_modules_hit": int(len(hit_modules)),
        "all_fault_module_hit_rate": float(len(hit_modules) / n_fault) if n_fault else 0.0,
        "all_fault_module_hit_rate_evaluable": (
            float(len(hit_modules) / n_positive_window_modules) if n_positive_window_modules else 0.0
        ),
        "all_fault_modules_any_alert": int(len(any_alert_fault_modules)),
        "all_fault_module_any_alert_rate": (
            float(len(any_alert_fault_modules) / n_fault) if n_fault else 0.0
        ),
        "all_fault_module_prediction_coverage": (
            float(n_positive_window_modules / n_fault) if n_fault else 0.0
        ),
        "all_fault_mean_hit_lead_minutes": float(np.mean(hit_leads)) if hit_leads else 0.0,
        "all_fault_median_hit_lead_minutes": float(np.median(hit_leads)) if hit_leads else 0.0,
        "all_healthy_modules_total": int(n_healthy),
        "all_healthy_modules_with_windows": int(len(healthy_modules_with_windows)),
        "all_healthy_modules_without_windows": int(n_healthy - len(healthy_modules_with_windows)),
        "all_healthy_false_alarm_modules": int(len(false_alarm_modules)),
        "all_healthy_module_false_alarm_rate": (
            float(len(false_alarm_modules) / n_healthy) if n_healthy else 0.0
        ),
    }


def describe_split(name: str, df: pd.DataFrame) -> str:
    if df.empty:
        return f"{name}: empty"
    total = int(len(df))
    pos = int(df["label"].sum())
    event_modules = int(df.groupby("file_name")["event_label"].max().sum())
    obs_dirty = int(df["obs_has_anomaly"].sum()) if "obs_has_anomaly" in df.columns else 0
    return (
        f"{name}: {total} windows | positive={pos} ({pos / total * 100:.2f}%) | "
        f"event_modules={event_modules} | obs_has_anomaly={obs_dirty} ({obs_dirty / total * 100:.2f}%)"
    )
