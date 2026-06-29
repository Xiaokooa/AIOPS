"""OFP-style ahead-window task helpers for deep-learning models.

This module keeps the training target close to ``OFP/model1``:
an observation window is positive when the first failure occurs within an
ahead horizon after the observation end timestamp. The 120-hour horizon remains
the primary target by default, while shorter horizons can be retained as
auxiliary multi-horizon labels.

It keeps evaluation/output close to ``OFP/model2``:
predictions are converted to per-module ``timestamp,predict`` alerts, and
module-level metrics count a hit only when the first alert is before the first
true failure timestamp.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from model.Optical_prediction_model.common import base_utils as base
from model.Optical_prediction_model.common import cache_utils


ScopeMode = Literal["full", "prefirst"]
ScorePostprocessMode = Literal["raw", "rolling_mean", "rolling_max", "consecutive"]


@dataclass(frozen=True)
class OFPAheadTaskConfig:
    obs_minutes: int = 1440
    ahead_hours: int = 120
    auxiliary_ahead_hours: tuple[int, ...] = (12, 24, 72)
    train_step_minutes: int = 60
    eval_step_minutes: int = 60
    min_obs_coverage: float = 0.80
    train_scope: ScopeMode = "prefirst"
    eval_scope: ScopeMode = "prefirst"
    train_faulty_max_windows: int = 192
    train_healthy_max_windows: int = 24
    train_faulty_stride_minutes: int = 60
    train_healthy_stride_minutes: int = 360

    def __post_init__(self) -> None:
        for field in (
            "obs_minutes",
            "train_step_minutes",
            "eval_step_minutes",
            "train_faulty_stride_minutes",
            "train_healthy_stride_minutes",
        ):
            value = int(getattr(self, field))
            if value <= 0 or value % 5 != 0:
                raise ValueError(f"{field} must be a positive multiple of 5 minutes, got {value}")
        if self.ahead_hours <= 0:
            raise ValueError(f"ahead_hours must be positive, got {self.ahead_hours}")
        for value in self.auxiliary_ahead_hours:
            if int(value) <= 0:
                raise ValueError(f"auxiliary_ahead_hours must be positive, got {value}")

    @property
    def obs_steps(self) -> int:
        return self.obs_minutes // 5

    @property
    def ahead_seconds(self) -> int:
        return self.ahead_hours * 3600

    @property
    def all_ahead_hours(self) -> tuple[int, ...]:
        return tuple(sorted({int(h) for h in (*self.auxiliary_ahead_hours, self.ahead_hours)}))

    @property
    def main_label_column(self) -> str:
        return f"label_ahead_{int(self.ahead_hours)}h"

    @property
    def tag(self) -> str:
        aux = "-".join(str(h) for h in self.auxiliary_ahead_hours) or "none"
        return (
            f"ofp_obs{self.obs_minutes}m_ahead{self.ahead_hours}h_"
            f"aux{aux}_trstep{self.train_step_minutes}m_evalstep{self.eval_step_minutes}m"
        )

    @property
    def cache_tag(self) -> str:
        return (
            f"{self.tag}_cov{self.min_obs_coverage:.2f}_"
            f"scope-{self.train_scope}-{self.eval_scope}_"
            f"sample-f{self.train_faulty_max_windows}-h{self.train_healthy_max_windows}_"
            f"stride-f{self.train_faulty_stride_minutes}-h{self.train_healthy_stride_minutes}"
        ).replace(".", "p")


def _uniform_take(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[idx].copy()


def _limit_split(split_df: pd.DataFrame, limit: int | None, random_state: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(split_df) <= limit:
        return split_df.reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    remaining = int(limit)
    groups = list(split_df.groupby("Label", sort=True))
    for idx, (_, group) in enumerate(groups):
        if idx == len(groups) - 1:
            take = min(len(group), remaining)
        else:
            take = int(round(limit * len(group) / len(split_df)))
            take = max(1, min(len(group), take, remaining - (len(groups) - idx - 1)))
        pieces.append(group.sample(n=take, random_state=random_state + idx))
        remaining -= take
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def get_split_map(
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
    max_train_modules: int | None = None,
    max_val_modules: int | None = None,
    max_test_modules: int | None = None,
) -> dict[str, pd.DataFrame]:
    split_map = base.split_modules_stratified(
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
    )
    limits = {
        "train": max_train_modules,
        "val": max_val_modules,
        "test": max_test_modules,
    }
    return {
        name: _limit_split(split_df, limits[name], random_state)
        for name, split_df in split_map.items()
    }


def _sample_train_rows(rows: list[dict[str, object]], cfg: OFPAheadTaskConfig) -> list[dict[str, object]]:
    if not rows:
        return rows
    df = pd.DataFrame(rows).sort_values("obs_end_ts").reset_index(drop=True)
    event_label = int(df["event_label"].max()) if "event_label" in df.columns else 0
    max_windows = cfg.train_faulty_max_windows if event_label == 1 else cfg.train_healthy_max_windows
    if len(df) <= max_windows:
        return rows

    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    if len(pos_df) >= max_windows:
        return _uniform_take(pos_df, max_windows).to_dict("records")

    neg_take = max_windows - len(pos_df)
    sampled = pd.concat([pos_df, _uniform_take(neg_df, neg_take)], ignore_index=True)
    sampled = sampled.sort_values("obs_end_ts").reset_index(drop=True)
    return sampled.to_dict("records")


def label_column_for_horizon(hours: int) -> str:
    return f"label_ahead_{int(hours)}h"


def label_columns(cfg: OFPAheadTaskConfig) -> list[str]:
    return [label_column_for_horizon(h) for h in cfg.all_ahead_hours]


def _extract_module_rows(
    df: pd.DataFrame,
    file_name: str,
    module_label: int,
    cfg: OFPAheadTaskConfig,
    split_role: Literal["train", "val", "test"],
) -> list[dict[str, object]]:
    if len(df) < cfg.obs_steps:
        return []

    t_first = base.first_failure_timestamp(df)
    is_train = split_role == "train"
    scope = cfg.train_scope if is_train else cfg.eval_scope
    if is_train:
        stride_minutes = cfg.train_faulty_stride_minutes if t_first is not None else cfg.train_healthy_stride_minutes
    else:
        stride_minutes = cfg.eval_step_minutes
    stride_steps = max(1, stride_minutes // 5)

    rows: list[dict[str, object]] = []
    for start in range(0, len(df) - cfg.obs_steps + 1, stride_steps):
        obs_end_idx = start + cfg.obs_steps - 1
        obs = df.iloc[start : obs_end_idx + 1]
        obs_cov = float(obs["observed"].mean()) if len(obs) else 0.0
        if obs_cov < cfg.min_obs_coverage:
            continue

        obs_start_ts = int(obs["timestamp"].iloc[0])
        obs_end_ts = int(obs["timestamp"].iloc[-1])
        if scope == "prefirst" and t_first is not None and t_first <= obs_end_ts:
            continue

        lead_seconds = int(t_first - obs_end_ts) if t_first is not None else -1
        labels_by_horizon = {
            int(hours): int(t_first is not None and 0 < lead_seconds <= int(hours) * 3600)
            for hours in cfg.all_ahead_hours
        }
        label = labels_by_horizon[int(cfg.ahead_hours)]
        time_to_first_minutes = float(lead_seconds / 60.0) if t_first is not None else np.nan
        payload = {
                "file_name": str(file_name),
                "module_label": int(module_label),
                "split_role": split_role,
                "task_family": "ofp_ahead_window",
                "label_mode": f"ahead_{cfg.ahead_hours}h_aux_{'-'.join(map(str, cfg.auxiliary_ahead_hours))}",
                "window_index": int(start // stride_steps),
                "start_idx": int(start),
                "obs_start_idx": int(start),
                "obs_end_idx": int(obs_end_idx),
                "obs_start_ts": obs_start_ts,
                "obs_end_ts": obs_end_ts,
                "pred_ts": obs_end_ts,
                "first_failure_ts": int(t_first) if t_first is not None else -1,
                "event_label": int(t_first is not None),
                "label": label,
                "time_to_first_minutes": time_to_first_minutes,
                "lead_seconds": lead_seconds,
                "obs_coverage_ratio": obs_cov,
                "ahead_hours": int(cfg.ahead_hours),
        }
        for hours, value in labels_by_horizon.items():
            payload[label_column_for_horizon(hours)] = int(value)
        rows.append(payload)
    if is_train:
        rows = _sample_train_rows(rows, cfg)
    return rows


def _frame_cache_path(split_name: str, split_df: pd.DataFrame, cfg: OFPAheadTaskConfig) -> Path:
    split_sig = cache_utils.split_fingerprint(split_df)
    return cache_utils.FEATURE_FRAME_CACHE_DIR / f"{split_name}_{split_sig}_{cfg.cache_tag}_ofp_windows.pkl"


def build_ofp_ahead_frame(
    split_df: pd.DataFrame,
    cfg: OFPAheadTaskConfig,
    split_role: Literal["train", "val", "test"],
    cache_path: Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if cache_path and cache_path.exists() and not force_rebuild:
        return pd.read_pickle(cache_path)

    rows: list[dict[str, object]] = []
    for row in split_df.itertuples(index=False):
        file_name = str(getattr(row, "file_name"))
        module_label = int(getattr(row, "Label"))
        if not (base.TRAINING_DIR / file_name).exists():
            continue
        resampled, _, _ = cache_utils.load_resampled_module(file_name)
        rows.extend(_extract_module_rows(resampled, file_name, module_label, cfg, split_role))

    out = pd.DataFrame(rows)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(cache_path)
    return out


def load_ofp_ahead_frames(
    cfg: OFPAheadTaskConfig | None = None,
    force_rebuild: bool = False,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
    max_train_modules: int | None = None,
    max_val_modules: int | None = None,
    max_test_modules: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    cfg = cfg or OFPAheadTaskConfig()
    split_map = get_split_map(
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
        max_train_modules=max_train_modules,
        max_val_modules=max_val_modules,
        max_test_modules=max_test_modules,
    )
    frames: dict[str, pd.DataFrame] = {}
    for split_name, split_df in split_map.items():
        cache_path = _frame_cache_path(split_name, split_df, cfg)
        frames[split_name] = build_ofp_ahead_frame(
            split_df,
            cfg,
            split_role=split_name,
            cache_path=cache_path,
            force_rebuild=force_rebuild,
        )
    return frames, split_map


def _split_labels(split_df: pd.DataFrame | None, frame: pd.DataFrame) -> tuple[set[str], set[str], int]:
    if split_df is None:
        labels = frame.groupby("file_name")["event_label"].max()
        all_modules = set(labels.index.astype(str).tolist())
        true_pos = set(labels[labels == 1].index.astype(str).tolist())
        return all_modules, true_pos, len(all_modules)

    local = split_df.copy()
    local["file_name"] = local["file_name"].astype(str)
    label = pd.to_numeric(local["Label"], errors="coerce").fillna(0).astype(int)
    all_modules = set(local["file_name"].tolist())
    true_pos = set(local.loc[label == 1, "file_name"].tolist())
    return all_modules, true_pos, len(local)


def postprocess_scores(
    meta_frame: pd.DataFrame,
    scores: np.ndarray | list[float],
    mode: ScorePostprocessMode | str = "raw",
    window: int = 1,
) -> np.ndarray:
    """Apply module-wise temporal filtering before warning thresholding."""
    values = np.asarray(scores, dtype=float)
    mode = str(mode or "raw")
    window = max(1, int(window))
    if mode == "raw" or window <= 1:
        return values
    if mode not in {"rolling_mean", "rolling_max", "consecutive"}:
        raise ValueError(f"Unknown OFP score postprocess mode: {mode}")
    if meta_frame.empty:
        return values
    if len(meta_frame) != len(values):
        raise ValueError(
            f"score length ({len(values)}) does not match meta_frame length ({len(meta_frame)})"
        )

    time_col = "pred_ts" if "pred_ts" in meta_frame.columns else "obs_end_ts"
    work = meta_frame[["file_name"]].copy().reset_index(drop=True)
    work["__time"] = (
        pd.to_numeric(meta_frame[time_col], errors="coerce").fillna(0).to_numpy()
        if time_col in meta_frame.columns
        else np.arange(len(values))
    )
    work["__score"] = values
    work["__order"] = np.arange(len(values))
    processed = np.zeros_like(values, dtype=float)

    sorted_work = work.sort_values(["file_name", "__time", "__order"])
    for _, group in sorted_work.groupby("file_name", sort=False):
        score_series = group["__score"].astype(float)
        if mode == "rolling_mean":
            out = score_series.rolling(window=window, min_periods=1).mean()
        elif mode == "rolling_max":
            out = score_series.rolling(window=window, min_periods=1).max()
        else:
            out = score_series.rolling(window=window, min_periods=window).min().fillna(0.0)
        processed[group["__order"].to_numpy(dtype=int)] = out.to_numpy(dtype=float)
    return processed


def evaluate_ofp_scores(
    meta_frame: pd.DataFrame,
    scores: np.ndarray | list[float],
    threshold: float,
    split_df: pd.DataFrame | None = None,
    score_postprocess: ScorePostprocessMode | str = "raw",
    postprocess_window: int = 1,
) -> dict[str, object]:
    if meta_frame.empty:
        all_modules, all_true_pos_sns, all_cnt = _split_labels(split_df, meta_frame)
        tp = 0
        fp = 0
        fn = len(all_true_pos_sns)
        tn = all_cnt - fn
        return _ofp_report(tp, fp, fn, tn, [], [], 0)

    eval_df = meta_frame.copy().reset_index(drop=True)
    eval_df["score"] = postprocess_scores(
        eval_df,
        scores,
        mode=score_postprocess,
        window=postprocess_window,
    )
    eval_df["predict"] = (eval_df["score"] >= float(threshold)).astype(int)
    all_modules, all_true_pos_sns, all_cnt = _split_labels(split_df, eval_df)

    all_predict_pos_sns: set[str] = set()
    lead_sec_list: list[float] = []
    lead_pred_sn_list: list[str] = []
    for file_name, group in eval_df.groupby("file_name", sort=False):
        file_name = str(file_name)
        if file_name not in all_modules:
            continue
        pred_rows = group[group["predict"] > 0]
        if pred_rows.empty:
            continue
        pred_ts = int(pred_rows["pred_ts"].min() if "pred_ts" in pred_rows else pred_rows["obs_end_ts"].min())
        first_failure_ts = int(group["first_failure_ts"].max()) if "first_failure_ts" in group else -1
        if first_failure_ts > 0:
            if first_failure_ts > pred_ts:
                all_predict_pos_sns.add(file_name)
                lead_sec = float(first_failure_ts - pred_ts)
                lead_sec_list.append(abs(lead_sec))
                lead_pred_sn_list.append(file_name)
        else:
            all_predict_pos_sns.add(file_name)

    all_hit_sns = all_true_pos_sns & all_predict_pos_sns
    tp = len(all_hit_sns)
    fp = len(all_predict_pos_sns) - tp
    fn = len(all_true_pos_sns) - tp
    tn = all_cnt - tp - fp - fn
    return _ofp_report(tp, fp, fn, tn, lead_sec_list, lead_pred_sn_list, len(all_true_pos_sns))


def _ofp_report(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    lead_sec_list: list[float],
    lead_pred_sn_list: list[str],
    true_pos_count: int,
) -> dict[str, object]:
    all_cnt = tp + fp + fn + tn
    pred_pos_count = tp + fp
    accuracy = (tp + tn) / all_cnt if all_cnt else 0.0
    precision = tp / pred_pos_count if pred_pos_count else 0.0
    recall = tp / true_pos_count if true_pos_count else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    avg_lead_sec = sum(lead_sec_list) / tp if tp > 0 and lead_sec_list else 0.0
    avg_lead_hour = avg_lead_sec / 3600.0
    avg_lead_score = math.tanh(avg_lead_hour)

    min_lead_sec = min(lead_sec_list) if lead_sec_list else 0.0
    min_lead_hour = min_lead_sec / 3600.0
    min_lead_score = math.tanh(min_lead_hour)

    final_score = f1_score + avg_lead_score + min_lead_score + accuracy
    return {
        "final_score": float(final_score),
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "all_hit_cnt": int(tp),
        "all_predict_pos_cnt": int(pred_pos_count),
        "all_true_pos_cnt": int(true_pos_count),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "avg_lead_score": float(avg_lead_score),
        "avg_lead_hour": float(avg_lead_hour),
        "min_lead_score": float(min_lead_score),
        "min_lead_hour": float(min_lead_hour),
        "lead_pread_cnt": int(len(lead_pred_sn_list)),
    }


def choose_threshold_by_ofp_score(
    meta_frame: pd.DataFrame,
    scores: np.ndarray | list[float],
    split_df: pd.DataFrame | None = None,
    thresholds: np.ndarray | None = None,
    optimize_metric: str = "final_score",
    score_postprocess: ScorePostprocessMode | str = "raw",
    postprocess_window: int = 1,
) -> tuple[float, dict[str, object]]:
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    allowed_metrics = {"final_score", "f1_score", "precision", "recall", "accuracy"}
    if optimize_metric not in allowed_metrics:
        raise ValueError(f"Unknown OFP threshold metric: {optimize_metric}")
    best_threshold = 0.5
    best_metrics: dict[str, object] | None = None
    for threshold in thresholds:
        metrics = evaluate_ofp_scores(
            meta_frame,
            scores,
            float(threshold),
            split_df=split_df,
            score_postprocess=score_postprocess,
            postprocess_window=postprocess_window,
        )
        candidate = {
            "threshold": float(threshold),
            "final_score": float(metrics["final_score"]),
            "f1_score": float(metrics["f1_score"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "accuracy": float(metrics["accuracy"]),
            "score_postprocess": str(score_postprocess),
            "postprocess_window": int(postprocess_window),
        }
        if best_metrics is None:
            best_threshold, best_metrics = float(threshold), candidate
            continue
        if (
            candidate[optimize_metric] > best_metrics[optimize_metric]
            or (
                candidate[optimize_metric] == best_metrics[optimize_metric]
                and candidate["final_score"] > best_metrics["final_score"]
            )
            or (
                candidate[optimize_metric] == best_metrics[optimize_metric]
                and candidate["final_score"] == best_metrics["final_score"]
                and candidate["recall"] > best_metrics["recall"]
            )
        ):
            best_threshold, best_metrics = float(threshold), candidate
    return best_threshold, best_metrics or {"threshold": 0.5}


def export_ofp_prediction_files(
    meta_frame: pd.DataFrame,
    scores: np.ndarray | list[float],
    threshold: float,
    output_dir: Path,
    source_dir: Path | None = None,
    include_score: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if meta_frame.empty:
        return

    pred_df = meta_frame[["file_name", "pred_ts"]].copy()
    pred_df["score"] = np.asarray(scores, dtype=float)
    pred_df["predict"] = (pred_df["score"] >= float(threshold)).astype(int)

    for file_name, group in pred_df.groupby("file_name", sort=False):
        file_name = str(file_name)
        group = group.sort_values("pred_ts")
        if source_dir is not None and (source_dir / file_name).exists():
            raw_ts = pd.read_csv(source_dir / file_name, usecols=["timestamp"])
            out = raw_ts.copy()
            out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("int64")
            out["predict"] = 0
            if include_score:
                out["score"] = 0.0
            agg = group.groupby("pred_ts", as_index=False).agg({"predict": "max", "score": "max"})
            pred_map = dict(zip(agg["pred_ts"].astype("int64"), agg["predict"].astype(int)))
            score_map = dict(zip(agg["pred_ts"].astype("int64"), agg["score"].astype(float)))
            out["predict"] = out["timestamp"].map(pred_map).fillna(0).astype(int)
            if include_score:
                out["score"] = out["timestamp"].map(score_map).fillna(0.0).astype(float)
        else:
            out = (
                group.groupby("pred_ts", as_index=False)
                .agg({"predict": "max", "score": "max"})
                .rename(columns={"pred_ts": "timestamp"})
            )
            out["timestamp"] = out["timestamp"].astype("int64")
            out["predict"] = out["predict"].astype(int)
            if not include_score and "score" in out.columns:
                out = out.drop(columns=["score"])
        out.to_csv(output_dir / file_name, index=False)


def task_summary(cfg: OFPAheadTaskConfig) -> dict[str, object]:
    payload = asdict(cfg)
    payload.update(
        {
            "definition": "OFP model1-style multi-horizon ahead-window training with OFP model2-style module-level evaluation",
            "tag": cfg.tag,
            "cache_tag": cfg.cache_tag,
            "ahead_seconds": cfg.ahead_seconds,
            "obs_steps": cfg.obs_steps,
            "all_ahead_hours": cfg.all_ahead_hours,
            "label_columns": label_columns(cfg),
            "main_label_column": cfg.main_label_column,
        }
    )
    return payload
