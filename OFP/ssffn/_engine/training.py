# Numerical core retained for compatibility with the archived experiments.
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OFP.ssffn._engine.timestamp_policy import install_historical_policy
install_historical_policy()
from OFP.ssffn._engine.distribution import build_sit, build_dosit, negative_topk, DistributionHSS
from OFP.ssffn._engine.module_batches import ModuleGroupedLoader, module_mean_loss, module_topk_loss
from OFP.ssffn._engine.loss_helpers import PrefixDistributionHSS, build_qrsit, module_class_weight, balanced_module_margin

from OFP.ssffn._engine.data import (
    BUILDERS,
    CompatBatchedDataset,
    CompatCfg,
    FEATURE_CACHE_VERSION,
    FeatureNormStats,
    Model2FeatureCache,
    cap_files_stratified,
    compat_alarm_logit,
    compat_feature_names,
    compute_feature_norm_stats,
    cuda_autocast,
    effective_pos_weight,
    file_label_map,
    format_duration,
    format_epoch_status,
    format_rate,
    log_run_header,
    progress_bar,
    split_train_val_files,
)
from OFP.ssffn._engine.evaluation import (
    LastLinearInputHook,
    build_ml_model,
    evaluate_prediction_output,
    log_stage_done,
    log_stage_start,
    positive_scores,
    select_threshold_for_scores,
    write_threshold_predictions,
)
from OFP.ssffn._engine.leadtime import (
    DEFAULT_LEAD_TIME_GRID,
    append_lead_time_sweep_results,
    run_lead_time_sweep,
)
from OFP.ssffn._engine.index import files_for_index_fold, read_index
from OFP.ssffn._engine.runtime import format_metric_summary, resolve_runtime_device
from OFP.ssffn._engine.feature_schema import (
    feature_group_manifest,
    select_feature_names,
)


# Neutral framework name: the temporal encoder and decision layer are recorded
# separately, so this identifier stays truthful for linear/MLP/XGB variants.
RUN_NAME = "htsf_temporal_stat_aligned"
TEMPORAL_ENCODERS = ("patchtst", "itransformer", "moderntcn", "fits", "sit", "dosit", "qrsit", "shared_sit")
BUILDERS['sit'] = build_sit
BUILDERS['dosit'] = build_dosit
BUILDERS['qrsit'] = build_qrsit
TEMPORAL_VIEW_MODES = ("level_mask", "residual_multiview")
DEFAULT_THRESHOLD_GRID = (
    "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,"
    "0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99"
)
RAW_SEQUENCE_FEATURES = [
    "Temp",
    "Curr",
    "TxP0",
    "RxP0",
    "RxP1",
    "RxP2",
    "RxP3",
    "RxP4",
    "TxP1",
    "TxP2",
    "TxP3",
    "TxP4",
    "DeltaSeconds",
]
TEMPORAL_SUMMARY_MODES = ("none", "raw_channel_stats")
TEMPORAL_SUMMARY_OPS = ("last", "mean", "std", "min", "max", "delta", "range")
FUSION_MODES = (
    "shared_sit",
    "temporal_only",
    "stat_only",
    "latent_concat",
    "attn_mean",
    "gated_attn",
    "rule_anchored",
)
DECISION_LAYERS = ("xgb", "rf", "lgbm", "catboost", "linear", "mlp")


class PositiveLinear(nn.Linear):
    """Linear head with non-negative effective weights for monotonic residual risk."""

    def reset_parameters(self) -> None:
        # softplus(-3.9) ~= 0.02, preventing the 128-dimensional additive
        # residual from producing saturated logits at initialization.
        nn.init.constant_(self.weight, -3.9)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, F.softplus(self.weight), self.bias)

    def effective_weight(self) -> torch.Tensor:
        return F.softplus(self.weight)


class FrozenLinearHead:
    """Pickle-friendly NumPy inference wrapper for the trained neural head."""

    def __init__(self, weight: np.ndarray, bias: float) -> None:
        self.coef_ = np.asarray(weight, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.asarray([float(bias)], dtype=np.float64)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "FrozenLinearHead":
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = np.asarray(x, dtype=np.float64) @ self.coef_[0] + self.intercept_[0]
        logits = np.clip(logits, -60.0, 60.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - positive, positive], axis=1)


def frozen_encoder_head(model: "TemporalStatAligner") -> FrozenLinearHead:
    head = model.classifier
    weight = head.effective_weight() if isinstance(head, PositiveLinear) else head.weight
    bias = float(head.bias.detach().cpu().reshape(-1)[0]) if head.bias is not None else 0.0
    return FrozenLinearHead(weight.detach().cpu().numpy().reshape(-1), bias)


def raw_validity_feature_name(raw_name: str) -> str:
    return "DeltaSecondsValidMask" if raw_name == "DeltaSeconds" else f"{raw_name}ValidMask"


def raw_validity_indices(cfg: CompatCfg, raw_indices: list[int] | np.ndarray) -> np.ndarray | None:
    names = compat_feature_names(cfg)
    selected_names = [names[int(idx)] for idx in raw_indices]
    mask_names = [raw_validity_feature_name(name) for name in selected_names]
    if not all(name in names for name in mask_names):
        return None
    return np.asarray([names.index(name) for name in mask_names], dtype=np.int64)


def tsf_run_name(temporal_encoder: str, decision_layer: str = "xgb") -> str:
    return f"{str(temporal_encoder).lower()}_stat_aligned_{str(decision_layer).lower()}"


def resolve_fusion_mode(fusion_mode: str, tsf_ablation: str = "full") -> str:
    legacy = {
        "full": str(fusion_mode),
        "no_stat_branch": "temporal_only",
        "no_temporal_branch": "stat_only",
        "no_cross_attention": "latent_concat",
    }
    mode = legacy.get(str(tsf_ablation), str(fusion_mode))
    if mode not in FUSION_MODES:
        raise ValueError(f"Unknown fusion mode {mode!r}; use one of {FUSION_MODES}")
    return mode


def configure_runtime(cfg: CompatCfg, seed: int, fold: int, training_seed: int = 42) -> None:
    torch.manual_seed(int(training_seed) + int(fold))
    np.random.seed(int(seed) + int(fold))
    cfg.device = resolve_runtime_device(cfg.device)
    if str(cfg.device).startswith("cuda") and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def load_or_compute_feature_norm_stats(
    data_dir: Path,
    train_files: list[str],
    cfg: CompatCfg,
    label_by_file: dict[str, int],
) -> FeatureNormStats:
    if not str(cfg.module_cache_dir).strip():
        return compute_feature_norm_stats(data_dir, train_files, cfg, label_by_file)
    fingerprint_payload = {
        "cache_version": FEATURE_CACHE_VERSION,
        "feature_mode": cfg.feature_mode,
        "target_mode": cfg.target_mode,
        "target_horizon_hours": float(cfg.target_horizon_hours),
        "preserve_timepoints": bool(cfg.preserve_timepoints),
        "rule_mode": cfg.rule_mode,
        "train_files": sorted(train_files),
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = Path(cfg.module_cache_dir) / "norm_stats" / f"{fingerprint}.json"
    if cache_path.exists():
        try:
            return FeatureNormStats(**json.loads(cache_path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            cache_path.unlink(missing_ok=True)
    stats = compute_feature_norm_stats(data_dir, train_files, cfg, label_by_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return stats


def feature_indices(
    feature_names: list[str],
    stat_feature_mode: str,
    stat_feature_groups: str = "",
    exclude_stat_feature_groups: str = "",
) -> tuple[list[int], list[str], list[int], list[str]]:
    raw_names = [name for name in RAW_SEQUENCE_FEATURES if name in feature_names]
    raw_indices = [feature_names.index(name) for name in raw_names]
    if not raw_indices:
        raise ValueError("No raw sequence features were found in compat feature names")

    raw_set = set(raw_names) | {"Ts"}
    mode = str(stat_feature_mode).lower()
    if str(stat_feature_groups).strip():
        stat_names = select_feature_names(
            feature_names,
            stat_feature_groups,
            exclude_stat_feature_groups,
        )
        stat_names = [name for name in stat_names if name not in raw_set]
    elif mode in {"ofp_expert_stat", "expert_statistical"}:
        stat_names = select_feature_names(feature_names, "expert_statistical")
    elif mode == "all_engineered":
        stat_names = [name for name in feature_names if name not in raw_set]
    elif mode == "model2_expert":
        keep = {"FeCo", "FeTxP0", "FeRxP0", "TsDelta"}
        stat_names = [name for name in feature_names if any(name.startswith(prefix) for prefix in keep)]
    elif mode == "statistics":
        keys = (
            "_mean",
            "_std",
            "_delta",
            "_exp_range",
            "Range",
            "Std",
            "Count",
            "Storm",
            "Rate",
            "Flag",
            "Elapsed",
            "DeltaSeconds",
        )
        stat_names = [name for name in feature_names if name not in raw_set and any(key in name for key in keys)]
    elif mode == "all":
        stat_names = list(feature_names)
    else:
        raise ValueError(
            "Unknown stat_feature_mode; use ofp_expert_stat, all_engineered, model2_expert, statistics, or all"
        )
    if not stat_names:
        raise ValueError(
            "The selected expert/statistical feature groups are empty. "
            "Check --feature_mode, --stat_feature_groups, and --exclude_stat_feature_groups."
        )
    stat_indices = [feature_names.index(name) for name in stat_names]
    return raw_indices, raw_names, stat_indices, stat_names


def temporal_summary_feature_names(raw_names: list[str], temporal_summary_mode: str) -> list[str]:
    mode = str(temporal_summary_mode).lower()
    if mode == "none":
        return []
    if mode != "raw_channel_stats":
        raise ValueError(f"Unknown temporal_summary_mode={temporal_summary_mode!r}; use one of {TEMPORAL_SUMMARY_MODES}")
    return [f"TemporalSummary_{name}_{op}" for name in raw_names for op in TEMPORAL_SUMMARY_OPS]


def stat_candidate_names(stat_names: list[str], raw_names: list[str], temporal_summary_mode: str) -> list[str]:
    return list(stat_names) + temporal_summary_feature_names(raw_names, temporal_summary_mode)


def compute_temporal_window_summaries(
    raw: np.ndarray,
    positions: np.ndarray,
    seq_len: int,
    temporal_summary_mode: str,
) -> np.ndarray:
    mode = str(temporal_summary_mode).lower()
    positions = np.asarray(positions, dtype=np.int64)
    if mode == "none":
        return np.zeros((len(positions), 0), dtype=np.float32)
    if mode != "raw_channel_stats":
        raise ValueError(f"Unknown temporal_summary_mode={temporal_summary_mode!r}; use one of {TEMPORAL_SUMMARY_MODES}")
    raw = np.asarray(raw, dtype=np.float32)
    n_channels = int(raw.shape[1]) if raw.ndim == 2 else 0
    out = np.zeros((len(positions), n_channels * len(TEMPORAL_SUMMARY_OPS)), dtype=np.float32)
    for row_idx, pos in enumerate(positions):
        pos = int(pos)
        if pos < 0 or pos >= len(raw):
            continue
        start = max(0, pos - int(seq_len) + 1)
        window = raw[start : pos + 1]
        if len(window) == 0:
            continue
        window = np.nan_to_num(window, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        last = window[-1]
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        min_value = window.min(axis=0)
        max_value = window.max(axis=0)
        delta = last - window[0]
        value_range = max_value - min_value
        channel_major = np.stack([last, mean, std, min_value, max_value, delta, value_range], axis=1)
        out[row_idx] = channel_major.reshape(-1)
    return np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_stat_candidate_matrix(
    features: np.ndarray,
    positions: np.ndarray,
    raw_indices: np.ndarray | list[int],
    stat_indices: np.ndarray | list[int],
    seq_len: int,
    temporal_summary_mode: str,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    stat_idx = np.asarray(stat_indices, dtype=np.int64)
    raw_idx = np.asarray(raw_indices, dtype=np.int64)
    base = (
        features[positions][:, stat_idx].astype(np.float32)
        if len(stat_idx) > 0
        else np.zeros((len(positions), 0), dtype=np.float32)
    )
    temporal = compute_temporal_window_summaries(
        features[:, raw_idx].astype(np.float32),
        positions,
        int(seq_len),
        temporal_summary_mode,
    )
    if temporal.shape[1] == 0:
        return base
    return np.concatenate([base, temporal], axis=1).astype(np.float32)


def _save_score_table(path: Path, names: list[str], scores: np.ndarray, score_name: str = "score") -> pd.DataFrame:
    frame = pd.DataFrame({"feature": list(names), score_name: np.asarray(scores, dtype=float)})
    frame.sort_values(score_name, ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _save_heatmap(path: Path, values: np.ndarray, xlabels: list[str], ylabels: list[str], title: str = "") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot-skip] path={path} reason={exc}", flush=True)
        return
    arr = np.asarray(values, dtype=float)
    width = max(4.0, min(14.0, 0.35 * max(len(xlabels), 4)))
    height = max(2.5, min(8.0, 0.35 * max(len(ylabels), 4)))
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_topk_heatmap(path: Path, frame: pd.DataFrame, value_col: str, top_k: int = 30) -> None:
    if frame.empty:
        return
    top = frame.head(int(top_k))
    _save_heatmap(
        path,
        top[[value_col]].to_numpy(dtype=float).T,
        top["feature"].astype(str).tolist(),
        [value_col],
        "",
    )


def _cap_explain_rows(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(max_rows) <= 0 or len(y) <= int(max_rows):
        return x, y
    rng = np.random.default_rng(int(seed))
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y <= 0)
    pos_keep = pos if len(pos) <= int(max_rows) // 2 else rng.choice(pos, size=int(max_rows) // 2, replace=False)
    remain = max(int(max_rows) - len(pos_keep), 0)
    neg_keep = neg if len(neg) <= remain else rng.choice(neg, size=remain, replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(keep)
    return x[keep], y[keep]


def resolve_branch_selection_budgets(
    total_select_k: int,
    temporal_select_k: int,
    stat_select_k: int,
    n_temporal_features: int,
    n_stat_features: int,
) -> tuple[int, int]:
    """Resolve an explicit branch-wise allocation for a total input-feature budget."""
    total = int(total_select_k)
    temporal = int(temporal_select_k)
    stat = int(stat_select_k)
    if min(total, temporal, stat) < 0:
        raise ValueError("Feature-selection budgets must be non-negative")
    if total > 0:
        if temporal <= 0:
            raise ValueError(
                "--total_select_k requires an explicit positive --temporal_select_k. "
                "The two branch-wise ExtraTrees importance scales are not directly comparable."
            )
        stat_from_total = total - temporal
        if stat_from_total <= 0:
            raise ValueError("--total_select_k must leave at least one feature for the expert-statistical branch")
        if stat > 0 and stat != stat_from_total:
            raise ValueError(
                f"Inconsistent Top-K allocation: total={total}, temporal={temporal}, "
                f"stat={stat}; expected stat={stat_from_total}"
            )
        stat = stat_from_total
    if temporal > int(n_temporal_features):
        raise ValueError(
            f"Requested {temporal} temporal channels, but only {n_temporal_features} are available"
        )
    if stat > int(n_stat_features):
        raise ValueError(f"Requested {stat} expert-statistical features, but only {n_stat_features} are available")
    return temporal, stat


def maybe_select_temporal_features(
    cache: Model2FeatureCache,
    base_dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    raw_names: list[str],
    args: argparse.Namespace,
    fold: int,
    run_dir: Path,
    select_k: int,
    strict: bool = False,
) -> tuple[list[int], list[str], dict[str, Any]]:
    """Rank raw DDM channels on training windows and optionally prune the temporal input."""
    selector_name = str(args.temporal_selector).lower()
    k = int(select_k)
    all_indices = list(raw_indices)
    all_names = list(raw_names)
    meta: dict[str, Any] = {
        "temporal_selector": selector_name,
        "requested_temporal_select_k": k,
        "original_temporal_feature_count": len(all_names),
    }
    if selector_name == "none":
        if strict and 0 < k < len(all_names):
            raise ValueError("A total Top-K budget requires --temporal_selector extra_trees")
        meta["selected_temporal_feature_count"] = len(all_names)
        meta["selected_features"] = [
            {"feature": name, "rank": rank, "importance": None}
            for rank, name in enumerate(all_names, 1)
        ]
        return all_indices, all_names, meta

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for name, selected in zip(base_dataset.file_names, base_dataset.selected_positions):
        if len(selected) <= 0:
            continue
        _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = cache.get(name)
        raw = features[:, np.asarray(raw_indices, dtype=np.int64)].astype(np.float32)
        x_parts.append(
            compute_temporal_window_summaries(
                raw,
                selected,
                int(cfg.seq_len),
                "raw_channel_stats",
            )
        )
        y_parts.append(labels[selected].astype(np.int8))
    if not x_parts:
        if strict:
            raise ValueError("Temporal feature selection has no training windows")
        meta.update({"selected_temporal_feature_count": len(all_names), "reason": "empty_training_selection"})
        return all_indices, all_names, meta
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    x, y = _cap_explain_rows(x, y, int(args.temporal_selector_max_rows), int(args.seed) + int(fold))
    if len(np.unique(y)) < 2:
        if strict:
            raise ValueError("Temporal feature selection requires both positive and negative training windows")
        meta.update({"selected_temporal_feature_count": len(all_names), "reason": "single_class_training_selection"})
        return all_indices, all_names, meta

    selector = ExtraTreesClassifier(
        n_estimators=int(args.temporal_selector_estimators),
        random_state=int(args.seed) + int(fold),
        n_jobs=int(args.n_jobs),
        class_weight="balanced",
    )
    started = time.time()
    selector.fit(x, y)
    descriptor_scores = np.asarray(selector.feature_importances_, dtype=float)
    operation_scores = descriptor_scores.reshape(len(all_names), len(TEMPORAL_SUMMARY_OPS))
    channel_scores = operation_scores.sum(axis=1)
    score_total = float(channel_scores.sum())
    if score_total > 0.0:
        channel_scores = channel_scores / score_total
    ranking = np.argsort(channel_scores)[::-1]
    keep_count = len(all_names) if k <= 0 else min(k, len(all_names))
    selected_local = [int(value) for value in ranking[:keep_count]]
    selected_indices = [all_indices[index] for index in selected_local]
    selected_names = [all_names[index] for index in selected_local]

    explain_dir = run_dir / "explainability"
    descriptor_names = temporal_summary_feature_names(all_names, "raw_channel_stats")
    descriptor_frame = _save_score_table(
        explain_dir / "temporal_summary_descriptor_importance.csv",
        descriptor_names,
        descriptor_scores,
        "importance",
    )
    _save_topk_heatmap(
        explain_dir / "temporal_summary_descriptor_importance_topk_heatmap.png",
        descriptor_frame,
        "importance",
    )
    selected_set = set(selected_names)
    channel_frame = pd.DataFrame(
        {
            "feature": all_names,
            "importance": channel_scores,
            "selected": [name in selected_set for name in all_names],
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    channel_frame.insert(0, "rank", np.arange(1, len(channel_frame) + 1))
    explain_dir.mkdir(parents=True, exist_ok=True)
    channel_frame.to_csv(explain_dir / "temporal_raw_channel_importance.csv", index=False)
    _save_topk_heatmap(
        explain_dir / "temporal_raw_channel_importance_topk_heatmap.png",
        channel_frame,
        "importance",
        top_k=len(all_names),
    )
    _save_heatmap(
        explain_dir / "temporal_channel_operation_importance_heatmap.png",
        operation_scores,
        list(TEMPORAL_SUMMARY_OPS),
        all_names,
    )
    selected_frame = channel_frame[channel_frame["selected"]].copy()
    selected_frame.to_csv(explain_dir / "selected_temporal_features.csv", index=False)
    meta.update(
        {
            "selected_temporal_feature_count": len(selected_names),
            "temporal_selector_rows": int(len(y)),
            "temporal_selector_positive_rows": int((y > 0).sum()),
            "temporal_selector_seconds": float(time.time() - started),
            "selected_features": selected_frame[["feature", "rank", "importance"]].to_dict("records"),
        }
    )
    return selected_indices, selected_names, meta


def maybe_select_stat_features(
    cache: Model2FeatureCache,
    base_dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    raw_names: list[str],
    stat_indices: list[int],
    stat_names: list[str],
    args: argparse.Namespace,
    fold: int,
    run_dir: Path,
    select_k: int | None = None,
    strict: bool = False,
) -> tuple[list[int], list[str], dict[str, Any], list[str]]:
    selector_name = str(args.stat_selector).lower()
    k = int(args.stat_select_k if select_k is None else select_k)
    candidate_names = stat_candidate_names(stat_names, raw_names, str(args.temporal_summary_mode))
    temporal_summary_count = len(candidate_names) - len(stat_names)
    meta: dict[str, Any] = {
        "stat_selector": selector_name,
        "requested_stat_select_k": k,
        "base_stat_feature_count": len(stat_indices),
        "temporal_summary_mode": str(args.temporal_summary_mode),
        "temporal_summary_feature_count": int(temporal_summary_count),
        "original_stat_feature_count": len(candidate_names),
    }
    all_local_indices = list(range(len(candidate_names)))
    if selector_name == "none":
        if strict and 0 < k < len(candidate_names):
            raise ValueError("A total Top-K budget requires --stat_selector extra_trees")
        meta["selected_stat_feature_count"] = len(candidate_names)
        meta["selected_features"] = [
            {"feature": name, "rank": rank, "importance": None}
            for rank, name in enumerate(candidate_names, 1)
        ]
        return all_local_indices, candidate_names, meta, candidate_names
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for name, selected in zip(base_dataset.file_names, base_dataset.selected_positions):
        if len(selected) <= 0:
            continue
        _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = cache.get(name)
        x_parts.append(
            build_stat_candidate_matrix(
                features,
                selected,
                raw_indices,
                stat_indices,
                int(cfg.seq_len),
                str(args.temporal_summary_mode),
            )
        )
        y_parts.append(labels[selected].astype(np.int8))
    if not x_parts:
        if strict:
            raise ValueError("Expert-statistical feature selection has no training windows")
        meta["selected_stat_feature_count"] = len(candidate_names)
        meta["reason"] = "empty_training_selection"
        return all_local_indices, candidate_names, meta, candidate_names
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    x, y = _cap_explain_rows(x, y, int(args.stat_selector_max_rows), int(args.seed) + int(fold))
    if len(np.unique(y)) < 2:
        if strict:
            raise ValueError("Expert-statistical feature selection requires both positive and negative training windows")
        meta["selected_stat_feature_count"] = len(candidate_names)
        meta["reason"] = "single_class_training_selection"
        return all_local_indices, candidate_names, meta, candidate_names
    selector = ExtraTreesClassifier(
        n_estimators=int(args.stat_selector_estimators),
        random_state=int(args.seed) + int(fold),
        n_jobs=int(args.n_jobs),
        class_weight="balanced",
    )
    started = time.time()
    selector.fit(x, y)
    scores = np.asarray(selector.feature_importances_, dtype=float)
    ranking = np.argsort(scores)[::-1]
    keep_count = len(ranking) if k <= 0 else min(k, len(ranking))
    local = ranking[:keep_count]
    selected_indices = [int(i) for i in local]
    selected_names = [candidate_names[int(i)] for i in local]
    importance = _save_score_table(
        run_dir / "explainability" / "stat_feature_importance.csv",
        candidate_names,
        scores,
        "importance",
    )
    selected_set = set(selected_names)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance["selected"] = importance["feature"].isin(selected_set)
    importance.to_csv(run_dir / "explainability" / "stat_feature_importance.csv", index=False)
    _save_topk_heatmap(run_dir / "explainability" / "stat_feature_importance_topk_heatmap.png", importance, "importance")
    meta.update(
        {
            "selected_stat_feature_count": len(selected_indices),
            "stat_selector_rows": int(len(y)),
            "stat_selector_positive_rows": int((y > 0).sum()),
            "stat_selector_seconds": float(time.time() - started),
            "selected_features": importance.loc[
                importance["selected"], ["feature", "rank", "importance"]
            ].to_dict("records"),
        }
    )
    (run_dir / "explainability" / "selected_stat_features.csv").write_text(
        pd.DataFrame({"feature": selected_names}).to_csv(index=False),
        encoding="utf-8",
    )
    return selected_indices, selected_names, meta, candidate_names


def deterministic_module_order(count: int, seed: int, epoch: int) -> np.ndarray:
    """Return the reproducible epoch-wise permutation used by HTSF training."""
    if int(count) <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 104729 * int(epoch))
    return rng.permutation(int(count)).astype(np.int64, copy=False)


class SelectedWindowStatDataset(IterableDataset):
    def __init__(
        self,
        base: CompatBatchedDataset,
        cache: Model2FeatureCache,
        cfg: CompatCfg,
        raw_indices: list[int],
        stat_indices: list[int],
        stat_input_indices: list[int],
        temporal_summary_mode: str,
    ) -> None:
        self.base = base
        self.cache = cache
        self.cfg = cfg
        self.raw_indices = np.asarray(raw_indices, dtype=np.int64)
        self.raw_validity_indices = raw_validity_indices(cfg, self.raw_indices)
        self.stat_indices = np.asarray(stat_indices, dtype=np.int64)
        self.stat_input_indices = np.asarray(stat_input_indices, dtype=np.int64)
        self.temporal_summary_mode = str(temporal_summary_mode)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.base)

    @property
    def file_names(self) -> list[str]:
        return self.base.file_names

    @property
    def selected_positions(self) -> list[np.ndarray]:
        return self.base.selected_positions

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible module order for one training epoch."""
        self._epoch = int(epoch)

    def __iter__(self):
        seq_len = int(self.cfg.seq_len)
        pairs = list(zip(self.base.file_names, self.base.selected_positions))
        if pairs:
            # Module filenames in the OFP index are label ordered.  Iterating
            # them verbatim makes every epoch see almost all faulty modules
            # before normal modules and causes the linear head to oscillate.
            # Shuffle modules (not windows) so module-grouped batches remain
            # intact for the first-warning-aligned bag loss.
            order = deterministic_module_order(len(pairs), int(self.cfg.seed), int(self._epoch))
            pairs = [pairs[int(index)] for index in order]
        worker = get_worker_info()
        if worker is not None:
            pairs = pairs[worker.id :: worker.num_workers]
        weight_by_name = {name: weights for name, weights in zip(self.base.file_names, self.base.selected_weights)}
        for name, selected in pairs:
            if len(selected) <= 0:
                continue
            _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = self.cache.get(name)
            raw = features[:, self.raw_indices].astype(np.float32)
            raw_validity = (
                features[:, self.raw_validity_indices].astype(np.float32)
                if self.raw_validity_indices is not None
                else np.ones_like(raw, dtype=np.float32)
            )
            stat = build_stat_candidate_matrix(
                features,
                selected,
                self.raw_indices,
                self.stat_indices,
                seq_len,
                self.temporal_summary_mode,
            )
            stat = stat[:, self.stat_input_indices].astype(np.float32)
            pad_x = np.zeros((seq_len - 1, raw.shape[1]), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, raw], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, raw_validity], axis=0))
            x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            ys = torch.from_numpy(labels[selected].astype(np.float32))
            ws = torch.from_numpy(weight_by_name.get(name, np.ones(len(selected), dtype=np.float32)).astype(np.float32))
            # Module-level flag for the first-warning bag loss: a faulty module
            # (any positive target row in the full label array) must never
            # receive the normal-module "suppress the maximum" penalty.
            module_has_positive = bool(np.any(labels > 0))
            module_flag = torch.tensor(float(module_has_positive))
            stat_tensor = torch.from_numpy(stat.astype(np.float32))
            for start in range(0, len(selected), int(self.cfg.batch_size)):
                end = start + int(self.cfg.batch_size)
                idx = torch.from_numpy(np.asarray(selected[start:end], dtype=np.int64))
                yield (
                    x_windows.index_select(0, idx).contiguous(),
                    m_windows.index_select(0, idx).contiguous(),
                    stat_tensor[start:end].contiguous(),
                    ys[start:end].contiguous(),
                    ws[start:end].contiguous(),
                    module_flag,
                )


from OFP.ssffn._engine.modules import StatisticalModuleSIT as TemporalStatAligner, RuleFreeHSS, NoHSSSampler, sensor_distribution_signal
from OFP.ssffn._engine import data as _sampling_deep
_sampling_deep.row_signal_scores = sensor_distribution_signal


def initialize_lazy_layers(model: TemporalStatAligner, dataset: SelectedWindowStatDataset, cfg: CompatCfg) -> None:
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    try:
        raw_x, raw_m, stat_x, _ys, _ws, _module_flag = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Cannot initialize fusion model because the dataset is empty") from exc
    model.eval()
    with torch.no_grad():
        model(
            raw_x.to(cfg.device),
            raw_m.to(cfg.device),
            stat_x.to(cfg.device),
        )


def write_branch_selection_manifest(
    run_dir: Path,
    temporal_meta: dict[str, Any],
    stat_meta: dict[str, Any],
    requested_total_k: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for branch, meta in (("temporal_raw", temporal_meta), ("expert_statistical", stat_meta)):
        for item in meta.get("selected_features", []):
            rows.append(
                {
                    "branch": branch,
                    "feature": item.get("feature", ""),
                    "within_branch_rank": item.get("rank"),
                    "within_branch_importance": item.get("importance"),
                }
            )
    explain_dir = run_dir / "explainability"
    explain_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(explain_dir / "selected_feature_manifest.csv", index=False)
    summary = {
        "requested_total_select_k": int(requested_total_k),
        "selected_temporal_feature_count": int(temporal_meta.get("selected_temporal_feature_count", 0)),
        "selected_stat_feature_count": int(stat_meta.get("selected_stat_feature_count", 0)),
        "selected_input_feature_count": int(
            temporal_meta.get("selected_temporal_feature_count", 0)
            + stat_meta.get("selected_stat_feature_count", 0)
        ),
        "importance_note": "Importance values are normalized within each branch and must not be compared across branches.",
    }
    (explain_dir / "branch_selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def train_fusion_encoder(
    train_files: list[str],
    cache: Model2FeatureCache,
    cfg: CompatCfg,
    raw_indices: list[int],
    raw_names: list[str],
    stat_indices: list[int],
    stat_names: list[str],
    args: argparse.Namespace,
    fold: int,
    run_dir: Path,
) -> tuple[
    TemporalStatAligner,
    CompatBatchedDataset,
    dict[str, Any],
    list[int],
    list[str],
    list[int],
    list[str],
    list[str],
]:
    started = log_stage_start(
        "build_fusion_dataset",
        RUN_NAME,
        fold,
        files=len(train_files),
        sample_selection=cfg.sample_selection,
        sampling_mode=cfg.sampling_mode,
    )
    base_dataset = CompatBatchedDataset(train_files, cache, cfg)
    if int(base_dataset.pos_rows) <= 0:
        raise ValueError(
            "The selected training subset contains no positive pre-anomaly windows. "
            "Increase --max_train_files or verify --target_mode and the trace split before using this run as evidence."
        )
    if int(args.total_select_k) > 0 and str(args.temporal_summary_mode).lower() != "none":
        raise ValueError(
            "Branch-wise total Top-K requires --temporal_summary_mode none so raw temporal summaries "
            "do not leak into the expert-statistical branch."
        )
    original_stat_candidates = stat_candidate_names(stat_names, raw_names, str(args.temporal_summary_mode))
    temporal_select_k, stat_select_k = resolve_branch_selection_budgets(
        int(args.total_select_k),
        int(args.temporal_select_k),
        int(args.stat_select_k),
        len(raw_names),
        len(original_stat_candidates),
    )
    strict_budget = int(args.total_select_k) > 0
    selected_raw_indices, selected_raw_names, temporal_selector_meta = maybe_select_temporal_features(
        cache,
        base_dataset,
        cfg,
        raw_indices,
        raw_names,
        args,
        fold,
        run_dir,
        temporal_select_k,
        strict=strict_budget,
    )
    stat_input_indices, stat_input_names, stat_selector_meta, stat_candidate_feature_names = maybe_select_stat_features(
        cache,
        base_dataset,
        cfg,
        selected_raw_indices,
        selected_raw_names,
        stat_indices,
        stat_names,
        args,
        fold,
        run_dir,
        select_k=stat_select_k,
        strict=strict_budget,
    )
    actual_total = len(selected_raw_names) + len(stat_input_names)
    if strict_budget and actual_total != int(args.total_select_k):
        raise RuntimeError(
            f"Branch selection produced {actual_total} inputs, expected total Top-K={int(args.total_select_k)}"
        )
    write_branch_selection_manifest(
        run_dir,
        temporal_selector_meta,
        stat_selector_meta,
        int(args.total_select_k),
    )
    fusion_dataset = SelectedWindowStatDataset(
        base_dataset,
        cache,
        cfg,
        selected_raw_indices,
        stat_indices,
        stat_input_indices,
        str(args.temporal_summary_mode),
    )
    log_stage_done(
        "build_fusion_dataset",
        started,
        RUN_NAME,
        fold,
        rows=int(base_dataset.total_rows),
        pos=int(base_dataset.pos_rows),
        neg=int(base_dataset.neg_rows),
    )
    model = TemporalStatAligner(
        shared_layout=args.shared_layout,
        followup=args.followup,
        module_variant=args.module_variant,
        training_seed=args.training_seed,
        temporal_encoder=args.temporal_encoder,
        seq_len=cfg.seq_len,
        n_raw_features=len(selected_raw_indices),
        n_stat_features=len(stat_input_indices),
        stat_feature_names=stat_input_names,
        embedding_dim=args.seq_embedding_dim,
        latent_dim=args.latent_dim,
        stat_hidden=args.stat_hidden,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        temporal_modality_dropout=args.temporal_modality_dropout,
        stat_modality_dropout=args.stat_modality_dropout,
        temporal_view_mode=args.temporal_view_mode,
        fusion_mode=args.fusion_mode,
        tsf_ablation=args.tsf_ablation,
    ).to(cfg.device)
    if str(args.temporal_encoder) in {'dosit','qrsit','shared_sit'}:
        endpoints = np.concatenate([cache.get(name)[1][positions][:,selected_raw_indices[:12]]
            for name,positions in zip(base_dataset.file_names,base_dataset.selected_positions) if len(positions)])
        if len(endpoints) > 100000:
            endpoints = endpoints[np.random.default_rng(int(cfg.seed)+fold).choice(len(endpoints),100000,replace=False)]
        distribution = model.sensor_input.fit_distribution(endpoints)
        (run_dir/'backbone_training_distribution.json').write_text(json.dumps(distribution,indent=2))
        del endpoints
    assert args.branch_warmup_epochs == 0 and args.temporal_aux_weight == args.stat_aux_weight == 0
    initialize_lazy_layers(model, fusion_dataset, cfg)
    loader = ModuleGroupedLoader(DataLoader(fusion_dataset, batch_size=None, shuffle=False, num_workers=int(cfg.num_workers)), 32)
    assert int(cfg.batch_size) >= max(int(cfg.positive_windows_per_module),int(cfg.normal_windows_per_module))
    assert float(args.module_bag_weight) == 0, "Grouped study uses ATAL only"
    pos_weight = effective_pos_weight(base_dataset, cfg)
    if args.transfer_loss == 'module_bce':
        pos_weight = module_class_weight(base_dataset)
    if args.transfer_loss in {'bce', 'atal'}:
        pos_weight = 1.0
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=cfg.device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=1024.0)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=1024.0)
    warmup_epochs = max(0, min(int(args.branch_warmup_epochs), max(int(cfg.epochs) - 1, 0)))
    temporal_aux_weight = max(0.0, float(args.temporal_aux_weight))
    stat_aux_weight = max(0.0, float(args.stat_aux_weight))
    module_bag_weight = max(0.0, float(args.module_bag_weight))
    use_aux = temporal_aux_weight > 0.0 or stat_aux_weight > 0.0 or warmup_epochs > 0
    if warmup_epochs > 0 and temporal_aux_weight + stat_aux_weight <= 0.0:
        raise ValueError("--branch_warmup_epochs requires a positive temporal or statistical auxiliary loss weight")
    history: list[dict[str, Any]] = []
    distribution_hss = None
    if args.transfer_hss in {'distribution_risk','prefix_risk'}:
        sampler_class = NoHSSSampler if args.module_variant == 'no_hss' else RuleFreeHSS
        distribution_hss = sampler_class(base_dataset, selected_raw_indices, int(cfg.seed) + int(fold))
        np.save(run_dir / 'hss_training_quantiles.npy', distribution_hss.knots)
    hss_history = []
    adaptive_meta: dict[str, float] = {}
    adaptive_applied = False
    train_started = time.time()
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        fusion_dataset.set_epoch(epoch)
        total_loss = 0.0
        total_fused_loss = 0.0
        total_temporal_loss = 0.0
        total_stat_loss = 0.0
        total_bag_loss = 0.0
        total_alarm_loss = 0.0
        n_seen = 0
        epoch_started = time.time()
        phase = "branch-warmup" if epoch <= warmup_epochs else "fusion"
        for batch_idx, (raw_x, raw_m, stat_x, ys, ws, module_flag) in enumerate(loader, 1):
            raw_x = raw_x.to(cfg.device)
            raw_m = raw_m.to(cfg.device)
            stat_x = stat_x.to(cfg.device)
            ys = ys.to(cfg.device).float()
            ws = ws.to(cfg.device).float()
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(amp_enabled):
                if use_aux:
                    logits, temporal_logits, stat_logits = model.forward_with_aux(raw_x, raw_m, stat_x)
                    fused_loss = module_mean_loss(loss_fn(logits, ys), ws, module_flag)
                    temporal_loss = module_mean_loss(loss_fn(temporal_logits, ys), ws, module_flag)
                    stat_loss = module_mean_loss(loss_fn(stat_logits, ys), ws, module_flag)
                    loss = temporal_aux_weight * temporal_loss + stat_aux_weight * stat_loss
                    if epoch > warmup_epochs:
                        loss = loss + fused_loss
                else:
                    logits = model(raw_x, raw_m, stat_x)
                    fused_loss = module_mean_loss(loss_fn(logits, ys), ws, module_flag)
                    temporal_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
                    stat_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
                    loss = fused_loss
                bag_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
                if module_bag_weight > 0.0 and epoch > warmup_epochs:
                    # Dataset batches are emitted module by module.  MIL
                    # semantics aligned with the first-warning evaluator: a
                    # faulty module must keep at least one of its sampled
                    # pre-event windows above the decision surface; a normal
                    # module must keep its maximum window below it.  Faulty
                    # modules whose sampled windows contain no positive row
                    # (e.g. every sampled row is post-event) receive no bag
                    # penalty: suppressing their maximum would fight the
                    # detector instead of shaping where it fires.
                    if bool(torch.any(ys > 0.5)):
                        bag_loss = F.softplus(-torch.max(logits[ys > 0.5]))
                    elif float(module_flag.item()) <= 0.0:
                        bag_loss = F.softplus(torch.max(logits))
                    loss = loss + module_bag_weight * bag_loss
                alarm_loss = torch.zeros((), dtype=fused_loss.dtype, device=fused_loss.device)
                if args.transfer_loss == 'atal' and epoch > max(1, warmup_epochs):
                    alarm_loss = module_topk_loss(logits, module_flag, k=4)
                    loss = loss + 0.3 * alarm_loss
                if args.transfer_loss == 'module_margin' and epoch > max(1,warmup_epochs):
                    alarm_loss = balanced_module_margin(logits,module_flag,margin=.5)
                    loss = loss + .2*alarm_loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"Non-finite HTSF loss at fold={fold}, epoch={epoch}, batch={batch_idx}; "
                    "the run was stopped before corrupting the saved model."
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # An overflowing update is an expected signal to GradScaler:
                # it skips that step and lowers the scale.  Raising here would
                # prevent the scaler from doing exactly that recovery.
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip), error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip), error_if_nonfinite=True)
                optimizer.step()
            batch_n = int(raw_x.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_n
            total_fused_loss += float(fused_loss.detach().cpu()) * batch_n
            total_temporal_loss += float(temporal_loss.detach().cpu()) * batch_n
            total_stat_loss += float(stat_loss.detach().cpu()) * batch_n
            total_bag_loss += float(bag_loss.detach().cpu()) * batch_n
            total_alarm_loss += float(alarm_loss.detach().cpu()) * batch_n
            n_seen += batch_n
            if int(cfg.log_batches) > 0 and batch_idx % int(cfg.log_batches) == 0:
                elapsed = time.time() - epoch_started
                print(
                    f"[batch] ep={epoch:02d} batch={batch_idx:>5}/{len(loader):<5} "
                    f"{progress_bar(batch_idx, len(loader), width=16)} "
                    f"phase={phase} loss={total_loss / max(n_seen, 1):.5f} "
                    f"fused={total_fused_loss / max(n_seen, 1):.5f} "
                    f"temporal={total_temporal_loss / max(n_seen, 1):.5f} "
                    f"stat={total_stat_loss / max(n_seen, 1):.5f} rows={n_seen} "
                    f"bag={total_bag_loss / max(n_seen, 1):.5f} "
                    f"rate={format_rate(n_seen, elapsed)} elapsed={format_duration(elapsed)}",
                    flush=True,
                )
        epoch_seconds = time.time() - epoch_started
        elapsed_total = time.time() - train_started
        eta = (elapsed_total / max(epoch, 1)) * max(int(cfg.epochs) - epoch, 0)
        avg_loss = total_loss / max(n_seen, 1)
        history.append(
            {
                "epoch": float(epoch),
                "phase": phase,
                "loss": float(avg_loss),
                "fused_loss": float(total_fused_loss / max(n_seen, 1)),
                "temporal_aux_loss": float(total_temporal_loss / max(n_seen, 1)),
                "stat_aux_loss": float(total_stat_loss / max(n_seen, 1)),
                "module_bag_loss": float(total_bag_loss / max(n_seen, 1)),
                "normal_top4_loss": float(total_alarm_loss / max(n_seen, 1)),
                "rows_seen": float(n_seen),
                "elapsed_seconds": epoch_seconds,
            }
        )
        print(
            format_epoch_status(
                "align-train",
                RUN_NAME,
                fold,
                epoch,
                int(cfg.epochs),
                float(avg_loss),
                int(n_seen),
                float(epoch_seconds),
                elapsed_total,
                eta,
            ),
            flush=True,
        )
        # Retain a recoverable training checkpoint for long formal runs.  This
        # is deliberately separate from the final inference artifact so an
        # interrupted run never masquerades as a completed experiment.
        checkpoint_path = run_dir / "training_checkpoint.pt"
        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "amp_enabled": bool(amp_enabled),
            },
            checkpoint_path,
        )
        if distribution_hss is not None and epoch % 2 == 0 and epoch < int(cfg.epochs):
            print(f'[distribution-hss] refreshing after epoch {epoch}', flush=True)
            hss_history.append(distribution_hss.refresh(model, cfg, selected_raw_indices,
                stat_indices, stat_input_indices, str(args.temporal_summary_mode),
                score_selected_positions, epoch))
            (run_dir / 'hss_history.json').write_text(json.dumps(hss_history, indent=2))
        if (
            float(cfg.adaptive_negative_weight) > 0.0
            and not adaptive_applied
            and epoch >= int(cfg.adaptive_warmup_epochs)
            and epoch < int(cfg.epochs)
        ):
            score_started = log_stage_start("adaptive_negative_weight", RUN_NAME, fold, after_epoch=epoch)
            scores = score_selected_positions(
                model,
                cache,
                base_dataset,
                cfg,
                selected_raw_indices,
                stat_indices,
                stat_input_indices,
                str(args.temporal_summary_mode),
            )
            adaptive_meta = base_dataset.apply_adaptive_negative_weights(scores, float(cfg.adaptive_negative_weight))
            adaptive_meta["applied_after_epoch"] = float(epoch)
            adaptive_applied = True
            print(
                f"[adaptive-neg] rows={int(adaptive_meta.get('updated_negative_rows', 0))} "
                f"score_min={adaptive_meta.get('min_score', 0.0):.5f} "
                f"score_max={adaptive_meta.get('max_score', 0.0):.5f} "
                f"max_extra={cfg.adaptive_negative_weight}",
                flush=True,
            )
            log_stage_done("adaptive_negative_weight", score_started, RUN_NAME, fold)
    meta = {
        "architecture": model.temporal_encoder_cfg,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "modules_per_optimizer_step": 32,
        "transfer_loss": args.transfer_loss,
        "module_margin_alpha": .2 if args.transfer_loss == "module_margin" else 0.,
        "transfer_hss": args.transfer_hss,
        "distribution_hss_history": hss_history,
        "sampling_policy": distribution_hss.policy_metadata,
        "train_rows": int(base_dataset.total_rows),
        "train_pos_rows": int(base_dataset.pos_rows),
        "train_neg_rows": int(base_dataset.neg_rows),
        "pos_weight": float(pos_weight),
        "branch_warmup_epochs": int(warmup_epochs),
        "temporal_aux_weight": float(temporal_aux_weight),
        "stat_aux_weight": float(stat_aux_weight),
        "module_bag_weight": float(module_bag_weight),
        "temporal_view_mode": str(args.temporal_view_mode),
        "temporal_modality_dropout": float(args.temporal_modality_dropout),
        "stat_modality_dropout": float(args.stat_modality_dropout),
        "training_seed": int(args.training_seed),
        "data_seed": int(args.seed),
        "history": history,
        "adaptive_negative_weight_meta": adaptive_meta,
        "temporal_feature_selection": temporal_selector_meta,
        "stat_feature_selection": stat_selector_meta,
        "branch_selection": {
            "requested_total_select_k": int(args.total_select_k),
            "effective_temporal_select_k": int(temporal_select_k),
            "effective_stat_select_k": int(stat_select_k),
            "selected_input_feature_count": int(actual_total),
        },
    }
    return (
        model,
        base_dataset,
        meta,
        selected_raw_indices,
        selected_raw_names,
        stat_input_indices,
        stat_input_names,
        stat_candidate_feature_names,
    )


@torch.no_grad()
def encode_file_positions(
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    file_name: str,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    temporal_summary_mode: str,
    positions: np.ndarray | None = None,
    return_branch_latents: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    timestamps, features, _valid_mask, labels, _anomaly_labels, rule_pred, extra = cache.get(file_name)
    n_rows = int(len(features))
    if positions is None:
        positions = np.arange(n_rows, dtype=np.int64)
    else:
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[(positions >= 0) & (positions < n_rows)]
    if n_rows == 0 or len(positions) == 0:
        return {
            "timestamps": np.zeros(0, dtype=np.int64),
            "fused": np.zeros((0, 0), dtype=np.float32),
            "temporal_latent": np.zeros((0, 0), dtype=np.float32),
            "statistical_latent": np.zeros((0, 0), dtype=np.float32),
            "logits": np.zeros(0, dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int8),
            "weights": np.ones(0, dtype=np.float32),
            "rule_pred": np.zeros(0, dtype=np.int8),
            "extra": extra,
        }
    raw = features[:, np.asarray(raw_indices, dtype=np.int64)].astype(np.float32)
    validity_indices = raw_validity_indices(cfg, raw_indices)
    raw_validity = (
        features[:, validity_indices].astype(np.float32)
        if validity_indices is not None
        else np.ones_like(raw, dtype=np.float32)
    )
    seq_len = int(cfg.seq_len)
    stat = build_stat_candidate_matrix(
        features,
        positions,
        raw_indices,
        stat_indices,
        seq_len,
        temporal_summary_mode,
    )
    stat = stat[:, np.asarray(stat_input_indices, dtype=np.int64)].astype(np.float32)
    pad_x = np.zeros((seq_len - 1, raw.shape[1]), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = torch.from_numpy(np.concatenate([pad_x, raw], axis=0))
    m_pad = torch.from_numpy(np.concatenate([pad_m, raw_validity], axis=0))
    x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
    m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
    fused_parts: list[np.ndarray] = []
    temporal_parts: list[np.ndarray] = []
    statistical_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    # Only inference is batched more widely; all windows and training updates stay.
    eval_batch_size = 2048
    for start in range(0, len(positions), eval_batch_size):
        batch_pos = positions[start : start + eval_batch_size]
        idx = torch.from_numpy(batch_pos.astype(np.int64))
        raw_x = x_windows.index_select(0, idx).contiguous().to(cfg.device)
        raw_m = m_windows.index_select(0, idx).contiguous().to(cfg.device)
        stat_x = torch.from_numpy(stat[start : start + len(batch_pos)].astype(np.float32)).to(cfg.device)
        with cuda_autocast(bool(cfg.amp) and str(cfg.device).startswith("cuda")):
            if return_branch_latents:
                fused, explain = model.encode(raw_x, raw_m, stat_x, return_explain=True)
            else:
                fused = model.encode(raw_x, raw_m, stat_x)
                explain = None
            logits = model.classifier(fused).squeeze(-1)
        fused_parts.append(fused.float().detach().cpu().numpy().astype(np.float32))
        if explain is not None:
            temporal_parts.append(explain["seq_latent"].float().cpu().numpy().astype(np.float32))
            statistical_parts.append(explain["stat_latent"].float().cpu().numpy().astype(np.float32))
        logit_parts.append(logits.float().detach().cpu().numpy().astype(np.float32))
    return {
        "timestamps": timestamps[positions].astype(np.int64),
        "fused": np.concatenate(fused_parts, axis=0).astype(np.float32),
        "temporal_latent": (
            np.concatenate(temporal_parts, axis=0).astype(np.float32)
            if temporal_parts
            else np.zeros((len(positions), 0), dtype=np.float32)
        ),
        "statistical_latent": (
            np.concatenate(statistical_parts, axis=0).astype(np.float32)
            if statistical_parts
            else np.zeros((len(positions), 0), dtype=np.float32)
        ),
        "logits": np.concatenate(logit_parts, axis=0).astype(np.float32),
        "labels": labels[positions].astype(np.int8),
        "weights": np.ones(len(positions), dtype=np.float32),
        "rule_pred": rule_pred[positions].astype(np.int8),
        "extra": extra,
    }


def score_selected_positions(
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    temporal_summary_mode: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for file_name, positions in zip(dataset.file_names, dataset.selected_positions):
        block = encode_file_positions(
            model,
            cache,
            file_name,
            cfg,
            raw_indices,
            stat_indices,
            stat_input_indices,
            temporal_summary_mode,
            positions,
        )
        out[file_name] = (1.0 / (1.0 + np.exp(-block["logits"]))).astype(np.float32)
    return out


def collect_fused_training_table(
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    temporal_summary_mode: str,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    started = log_stage_start("collect_fused_train_table", RUN_NAME, fold, files=len(dataset.file_names))
    weight_by_name = {name: weights for name, weights in zip(dataset.file_names, dataset.selected_weights)}
    for idx, (file_name, positions) in enumerate(zip(dataset.file_names, dataset.selected_positions), 1):
        if len(positions) == 0:
            continue
        block = encode_file_positions(
            model,
            cache,
            file_name,
            cfg,
            raw_indices,
            stat_indices,
            stat_input_indices,
            temporal_summary_mode,
            positions,
        )
        x_parts.append(block["fused"])
        y_parts.append(block["labels"])
        w_parts.append(weight_by_name.get(file_name, np.ones(len(positions), dtype=np.float32)).astype(np.float32))
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold} stage=collect_fused_train_table "
                f"{progress_bar(idx, len(dataset.file_names), width=18)} files={idx}/{len(dataset.file_names)} "
                f"rows={sum(len(part) for part in y_parts)} rate={format_rate(idx, elapsed)} "
                f"elapsed={format_duration(elapsed)}",
                flush=True,
            )
    if not x_parts:
        raise ValueError("No fused rows collected for XGBoost")
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int8)
    w = np.concatenate(w_parts, axis=0).astype(np.float32)
    log_stage_done(
        "collect_fused_train_table",
        started,
        RUN_NAME,
        fold,
        rows=len(y),
        features=x.shape[1],
        positives=int(np.sum(y > 0)),
        negatives=int(np.sum(y <= 0)),
    )
    return x, y, w, [f"aligned_latent_{idx:03d}" for idx in range(x.shape[1])]


def save_decision_latent_importance(estimator: Any, latent_names: list[str], run_dir: Path) -> None:
    scores = getattr(estimator, "feature_importances_", None)
    if scores is None:
        return
    frame = _save_score_table(
        run_dir / "explainability" / "decision_latent_importance.csv",
        latent_names,
        np.asarray(scores, dtype=float),
        "importance",
    )
    _save_topk_heatmap(run_dir / "explainability" / "decision_latent_importance_topk_heatmap.png", frame, "importance")


def pre_fusion_latent_names(latent_dim: int) -> list[str]:
    temporal = [f"temporal_latent_{idx:03d}" for idx in range(int(latent_dim))]
    statistical = [f"expert_statistical_latent_{idx:03d}" for idx in range(int(latent_dim))]
    return temporal + statistical


def collect_pre_fusion_training_table(
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    temporal_summary_mode: str,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    started = log_stage_start("collect_pre_fusion_latents", RUN_NAME, fold, files=len(dataset.file_names))
    weight_by_name = {name: weights for name, weights in zip(dataset.file_names, dataset.selected_weights)}
    for idx, (file_name, positions) in enumerate(zip(dataset.file_names, dataset.selected_positions), 1):
        if len(positions) == 0:
            continue
        block = encode_file_positions(
            model,
            cache,
            file_name,
            cfg,
            raw_indices,
            stat_indices,
            stat_input_indices,
            temporal_summary_mode,
            positions,
            return_branch_latents=True,
        )
        temporal = block["temporal_latent"]
        statistical = block["statistical_latent"]
        if temporal.shape[1] == 0 or statistical.shape[1] == 0:
            raise RuntimeError("The encoder did not expose both pre-fusion latent branches")
        x_parts.append(np.concatenate([temporal, statistical], axis=1).astype(np.float32))
        y_parts.append(block["labels"])
        w_parts.append(weight_by_name.get(file_name, np.ones(len(positions), dtype=np.float32)).astype(np.float32))
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold} stage=collect_pre_fusion_latents "
                f"{progress_bar(idx, len(dataset.file_names), width=18)} files={idx}/{len(dataset.file_names)} "
                f"rows={sum(len(part) for part in y_parts)} rate={format_rate(idx, elapsed)} "
                f"elapsed={format_duration(elapsed)}",
                flush=True,
            )
    if not x_parts:
        raise ValueError("No pre-fusion latent rows were collected")
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int8)
    weights = np.concatenate(w_parts, axis=0).astype(np.float32)
    if x.shape[1] % 2 != 0:
        raise RuntimeError(f"Expected equal temporal/statistical latent dimensions, got {x.shape[1]}")
    names = pre_fusion_latent_names(x.shape[1] // 2)
    log_stage_done(
        "collect_pre_fusion_latents",
        started,
        RUN_NAME,
        fold,
        rows=len(y),
        features=x.shape[1],
        positives=int(np.sum(y > 0)),
        negatives=int(np.sum(y <= 0)),
    )
    return x, y, weights, names


def _save_latent_branch_composition(path: Path, temporal_count: int, statistical_count: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot-skip] path={path} reason={exc}", flush=True)
        return
    labels = ["Temporal latent", "Expert-statistical latent"]
    values = [int(temporal_count), int(statistical_count)]
    fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=180)
    bars = ax.bar(labels, values, color=["#2F6B9A", "#D58A3A"], width=0.62)
    ax.set_ylabel("Selected dimensions")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def select_pre_fusion_latent_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    top_k: int,
    args: argparse.Namespace,
    fold: int,
    run_dir: Path,
) -> tuple[list[int], dict[str, Any]]:
    if x_train.shape[1] != len(feature_names):
        raise ValueError("Pre-fusion latent names do not match the latent matrix")
    k = int(top_k)
    if k <= 0 or k > x_train.shape[1]:
        raise ValueError(f"--latent_probe_topk must be in [1, {x_train.shape[1]}], got {k}")
    selector_x, selector_y = _cap_explain_rows(
        x_train,
        y_train,
        int(args.latent_probe_max_rows),
        int(args.seed) + int(fold) + 2003,
    )
    if len(np.unique(selector_y)) < 2:
        raise ValueError("Pre-fusion latent selection requires both positive and negative training windows")
    selector = ExtraTreesClassifier(
        n_estimators=int(args.latent_probe_estimators),
        random_state=int(args.seed) + int(fold) + 2003,
        n_jobs=int(args.n_jobs),
        class_weight="balanced",
    )
    started = time.time()
    selector.fit(selector_x, selector_y)
    importance = np.asarray(selector.feature_importances_, dtype=float)
    ranking = np.argsort(importance)[::-1]
    selected_indices = [int(value) for value in ranking[:k]]
    selected_set = set(selected_indices)
    latent_dim = x_train.shape[1] // 2
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "branch": ["temporal" if idx < latent_dim else "expert_statistical" for idx in range(len(feature_names))],
            "latent_dimension": [idx if idx < latent_dim else idx - latent_dim for idx in range(len(feature_names))],
            "importance": importance,
            "selected": [idx in selected_set for idx in range(len(feature_names))],
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    explain_dir = run_dir / "explainability" / "pre_fusion_latent_probe"
    explain_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(explain_dir / "latent_feature_importance.csv", index=False)
    selected_frame = frame[frame["selected"]].copy()
    selected_frame.to_csv(explain_dir / f"latent_top{k}_features.csv", index=False)
    _save_topk_heatmap(
        explain_dir / f"latent_top{k}_importance_heatmap.png",
        selected_frame,
        "importance",
        top_k=k,
    )
    matrix = np.stack([importance[:latent_dim], importance[latent_dim:]], axis=0)
    _save_heatmap(
        explain_dir / "latent_importance_by_branch_heatmap.png",
        matrix,
        [str(idx) for idx in range(latent_dim)],
        ["temporal", "expert-statistical"],
    )
    temporal_selected = int((selected_frame["branch"] == "temporal").sum())
    statistical_selected = int((selected_frame["branch"] == "expert_statistical").sum())
    temporal_mass = float(importance[:latent_dim].sum())
    statistical_mass = float(importance[latent_dim:].sum())
    summary = pd.DataFrame(
        [
            {
                "top_k": k,
                "temporal_selected": temporal_selected,
                "expert_statistical_selected": statistical_selected,
                "temporal_importance_sum": temporal_mass,
                "expert_statistical_importance_sum": statistical_mass,
                "selector_rows": int(len(selector_y)),
                "selector_positive_rows": int((selector_y > 0).sum()),
            }
        ]
    )
    summary.to_csv(explain_dir / "latent_branch_summary.csv", index=False)
    _save_latent_branch_composition(
        explain_dir / f"latent_top{k}_branch_composition.png",
        temporal_selected,
        statistical_selected,
    )
    meta = {
        **summary.iloc[0].to_dict(),
        "selector": "extra_trees",
        "selector_estimators": int(args.latent_probe_estimators),
        "selector_seconds": float(time.time() - started),
        "selected_indices": selected_indices,
        "selected_features": [feature_names[index] for index in selected_indices],
    }
    (explain_dir / "latent_probe_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return selected_indices, meta


def save_pre_fusion_probe_importance(
    estimator: Any,
    selected_names: list[str],
    run_dir: Path,
    top_k: int,
) -> None:
    scores = getattr(estimator, "feature_importances_", None)
    if scores is None:
        return
    frame = _save_score_table(
        run_dir / "explainability" / "pre_fusion_latent_probe" / f"top{int(top_k)}_xgb_importance.csv",
        selected_names,
        np.asarray(scores, dtype=float),
        "importance",
    )
    _save_topk_heatmap(
        run_dir / "explainability" / "pre_fusion_latent_probe" / f"top{int(top_k)}_xgb_importance_heatmap.png",
        frame,
        "importance",
        top_k=int(top_k),
    )


def summarize_tsf_explainability(
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    raw_names: list[str],
    stat_names: list[str],
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    max_rows = int(args.explain_max_rows)
    if bool(args.no_explainability) or max_rows == 0:
        return
    started = time.time()
    explain_dir = run_dir / "explainability"
    explain_dir.mkdir(parents=True, exist_ok=True)
    explain_dataset = SelectedWindowStatDataset(
        dataset,
        cache,
        cfg,
        raw_indices,
        stat_indices,
        stat_input_indices,
        str(args.temporal_summary_mode),
    )
    loader = DataLoader(explain_dataset, batch_size=None, shuffle=False, num_workers=0)
    model.eval()
    rows_seen = 0
    attn_sum: np.ndarray | None = None
    gate_sum = 0.0
    gate_sq_sum = 0.0
    gate_count = 0
    raw_attr = np.zeros(len(raw_names), dtype=np.float64)
    stat_attr = np.zeros(len(stat_names), dtype=np.float64)
    time_attr = np.zeros(int(cfg.seq_len), dtype=np.float64)
    raw_denom = 0
    stat_denom = 0
    time_denom = 0
    for raw_x, raw_m, stat_x, _ys, _ws, _module_flag in loader:
        if rows_seen >= max_rows:
            break
        take = min(int(raw_x.shape[0]), max_rows - rows_seen)
        raw_x = raw_x[:take].to(cfg.device).detach().requires_grad_(True)
        raw_m = raw_m[:take].to(cfg.device)
        stat_x = stat_x[:take].to(cfg.device).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        fused, info = model.encode(raw_x, raw_m, stat_x, return_explain=True)
        logits = model.classifier(fused).squeeze(-1)
        score = torch.sigmoid(logits).sum()
        grad_raw, grad_stat = torch.autograd.grad(score, (raw_x, stat_x), allow_unused=True)
        attention = info["attention"].detach().cpu().numpy()
        attn_batch = attention.sum(axis=0)
        attn_sum = attn_batch if attn_sum is None else attn_sum + attn_batch
        gate = info["gate"].detach().cpu().numpy()
        gate_sum += float(gate.sum())
        gate_sq_sum += float(np.square(gate).sum())
        gate_count += int(gate.size)
        if grad_raw is not None:
            raw_scores = (grad_raw.detach() * raw_x.detach()).abs().cpu().numpy()
            raw_attr += raw_scores.sum(axis=(0, 1))
            time_attr += raw_scores.sum(axis=(0, 2))
            raw_denom += int(raw_scores.shape[0] * raw_scores.shape[1])
            time_denom += int(raw_scores.shape[0] * raw_scores.shape[2])
        if grad_stat is not None:
            stat_scores = (grad_stat.detach() * stat_x.detach()).abs().cpu().numpy()
            stat_attr += stat_scores.sum(axis=0)
            stat_denom += int(stat_scores.shape[0])
        rows_seen += take
    if rows_seen <= 0:
        return
    if attn_sum is not None:
        attn_by_head = attn_sum / float(rows_seen)
        attn_matrix = attn_by_head.mean(axis=0)
        if attn_matrix.shape == (2, 2):
            query_labels = ["temporal", "statistical"]
            key_labels = ["temporal", "statistical"]
        else:
            query_labels = list(model.rule_feature_names[: attn_matrix.shape[0]])
            if len(query_labels) < attn_matrix.shape[0]:
                query_labels.extend(
                    f"rule_{idx}" for idx in range(len(query_labels), attn_matrix.shape[0])
                )
            key_labels = [f"patch_{idx}" for idx in range(attn_matrix.shape[1])]
        pd.DataFrame(
            attn_matrix,
            index=[f"query_{x}" for x in query_labels],
            columns=[f"key_{x}" for x in key_labels],
        ).to_csv(
            explain_dir / "cross_attention_matrix.csv"
        )
        rows = []
        for head in range(attn_by_head.shape[0]):
            for qi, q in enumerate(query_labels):
                for ki, key in enumerate(key_labels):
                    rows.append({"head": head, "query": q, "key": key, "weight": float(attn_by_head[head, qi, ki])})
        pd.DataFrame(rows).to_csv(explain_dir / "cross_attention_by_head.csv", index=False)
        _save_heatmap(explain_dir / "cross_attention_heatmap.png", attn_matrix, key_labels, query_labels)
    if gate_count > 0:
        gate_mean = gate_sum / float(gate_count)
        gate_var = max(gate_sq_sum / float(gate_count) - gate_mean * gate_mean, 0.0)
        gate_row = {"rows": rows_seen, "gate_std": float(np.sqrt(gate_var))}
        if model.fusion_mode == "rule_anchored":
            gate_row["gate_mean_temporal_residual_admission"] = gate_mean
        else:
            gate_row["gate_mean_temporal_share"] = gate_mean
            gate_row["gate_mean_statistical_share"] = 1.0 - gate_mean
        pd.DataFrame([gate_row]).to_csv(explain_dir / "modality_gate_summary.csv", index=False)
    if raw_denom > 0:
        raw_frame = _save_score_table(
            explain_dir / "raw_feature_attribution.csv",
            raw_names,
            raw_attr / float(raw_denom),
            "gradient_x_input",
        )
        _save_topk_heatmap(explain_dir / "raw_feature_attribution_topk_heatmap.png", raw_frame, "gradient_x_input")
    if stat_denom > 0:
        stat_frame = _save_score_table(
            explain_dir / "stat_feature_attribution.csv",
            stat_names,
            stat_attr / float(stat_denom),
            "gradient_x_input",
        )
        _save_topk_heatmap(explain_dir / "stat_feature_attribution_topk_heatmap.png", stat_frame, "gradient_x_input")
    if time_denom > 0:
        time_names = [f"t-{int(cfg.seq_len) - 1 - i}" for i in range(int(cfg.seq_len))]
        _save_score_table(
            explain_dir / "temporal_position_attribution.csv",
            time_names,
            time_attr / float(time_denom),
            "gradient_x_input",
        )
    print(
        f"[explainability] rows={rows_seen} out={explain_dir} elapsed={format_duration(time.time() - started)}",
        flush=True,
    )


def build_aligned_xgb_model(args: argparse.Namespace, y_train: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    classes = np.unique(y_train)
    if len(classes) < 2:
        from sklearn.dummy import DummyClassifier

        constant = int(classes[0]) if len(classes) else 0
        return DummyClassifier(strategy="constant", constant=constant), {
            "xgb_balance_mode": "dummy",
            "xgb_scale_pos_weight": 1.0,
        }

    from xgboost import XGBClassifier

    pos = float(np.sum(y_train > 0))
    neg = float(np.sum(y_train <= 0))
    raw_scale = neg / max(pos, 1.0)
    balance_mode = str(args.xgb_balance_mode).lower()
    if balance_mode == "auto":
        scale_pos_weight = raw_scale
    elif balance_mode == "sqrt":
        scale_pos_weight = float(np.sqrt(raw_scale))
    elif balance_mode == "none":
        scale_pos_weight = 1.0
    else:
        raise ValueError("Unknown xgb_balance_mode; use auto, sqrt, or none")

    estimator = XGBClassifier(
        n_estimators=int(args.ml_n_estimators),
        max_depth=int(args.xgb_max_depth),
        learning_rate=float(args.xgb_lr),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample_bytree),
        eval_metric="logloss",
        tree_method=str(args.xgb_tree_method),
        device=str(args.xgb_device),
        random_state=int(seed),
        n_jobs=int(args.n_jobs),
        scale_pos_weight=float(scale_pos_weight),
    )
    meta: dict[str, Any] = {
        "xgb_balance_mode": balance_mode,
        "xgb_scale_pos_weight": float(scale_pos_weight),
        "xgb_raw_scale_pos_weight": float(raw_scale),
    }
    return estimator, meta


def build_decision_model(args: argparse.Namespace, y_train: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    decision_layer = str(args.decision_layer).lower()
    if decision_layer == "xgb":
        estimator, meta = build_aligned_xgb_model(args, y_train, seed)
    elif decision_layer == "linear":
        linear_class_weight = None if str(args.linear_class_weight).lower() == "none" else "balanced"
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight=linear_class_weight,
            random_state=int(seed),
            n_jobs=int(args.n_jobs),
        )
        meta = {
            "xgb_balance_mode": "not_applicable",
            "xgb_scale_pos_weight": 1.0,
            "xgb_raw_scale_pos_weight": 1.0,
            "linear_class_weight": str(args.linear_class_weight).lower(),
        }
    elif decision_layer == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=int(seed),
        )
        meta = {
            "xgb_balance_mode": "not_applicable",
            "xgb_scale_pos_weight": 1.0,
            "xgb_raw_scale_pos_weight": 1.0,
        }
    else:
        estimator = build_ml_model(decision_layer, args, y_train, seed)
        meta = {
            "xgb_balance_mode": "not_applicable",
            "xgb_scale_pos_weight": 1.0,
            "xgb_raw_scale_pos_weight": 1.0,
        }
    meta["decision_layer"] = decision_layer
    return estimator, meta


def fit_decision_model(
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    use_sample_weight: bool,
) -> bool:
    if not use_sample_weight:
        estimator.fit(x_train, y_train)
        return False
    try:
        estimator.fit(x_train, y_train, sample_weight=sample_weight)
        return True
    except TypeError:
        estimator.fit(x_train, y_train)
        return False


def score_fused_files_to_memory(
    estimator: Any,
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    file_names: list[str],
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    temporal_summary_mode: str,
    fold: int,
    stage_name: str,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    started = log_stage_start(stage_name, RUN_NAME, fold, files=len(file_names))
    for idx, file_name in enumerate(file_names, 1):
        block = encode_file_positions(
            model,
            cache,
            file_name,
            cfg,
            raw_indices,
            stat_indices,
            stat_input_indices,
            temporal_summary_mode,
        )
        score = positive_scores(estimator, block["fused"])
        parts = [
            pd.DataFrame(
                {
                    "timestamp": block["timestamps"].astype(np.int64),
                    "score": score.astype(np.float32),
                    "rule_predict": block["rule_pred"].astype(int),
                    "source": RUN_NAME,
                }
            )
        ]
        extra = block["extra"]
        if len(extra):
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": extra[:, 0].astype(np.int64),
                        "score": np.nan,
                        "rule_predict": extra[:, 1].astype(int),
                        "source": "model2_extra_rule",
                    }
                )
            )
        frames[file_name] = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold} stage={stage_name} "
                f"{progress_bar(idx, len(file_names), width=18)} files={idx}/{len(file_names)} "
                f"rate={format_rate(idx, elapsed)} elapsed={format_duration(elapsed)}",
                flush=True,
            )
    log_stage_done(stage_name, started, RUN_NAME, fold)
    return frames


def score_pre_fusion_latent_files_to_memory(
    estimator: Any,
    selected_indices: list[int],
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    file_names: list[str],
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    temporal_summary_mode: str,
    fold: int,
    stage_name: str,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    selected = np.asarray(selected_indices, dtype=np.int64)
    started = log_stage_start(stage_name, RUN_NAME, fold, files=len(file_names), top_k=int(top_k))
    for idx, file_name in enumerate(file_names, 1):
        block = encode_file_positions(
            model,
            cache,
            file_name,
            cfg,
            raw_indices,
            stat_indices,
            stat_input_indices,
            temporal_summary_mode,
            return_branch_latents=True,
        )
        pre_fusion = np.concatenate(
            [block["temporal_latent"], block["statistical_latent"]],
            axis=1,
        ).astype(np.float32)
        score = (
            positive_scores(estimator, pre_fusion[:, selected])
            if len(pre_fusion) > 0
            else np.zeros(0, dtype=np.float32)
        )
        parts = [
            pd.DataFrame(
                {
                    "timestamp": block["timestamps"].astype(np.int64),
                    "score": score.astype(np.float32),
                    "rule_predict": block["rule_pred"].astype(int),
                    "source": f"pre_fusion_latent_top{int(top_k)}",
                }
            )
        ]
        extra = block["extra"]
        if len(extra):
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": extra[:, 0].astype(np.int64),
                        "score": np.nan,
                        "rule_predict": extra[:, 1].astype(int),
                        "source": "model2_extra_rule",
                    }
                )
            )
        frames[file_name] = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold} stage={stage_name} "
                f"{progress_bar(idx, len(file_names), width=18)} files={idx}/{len(file_names)} "
                f"rate={format_rate(idx, elapsed)} elapsed={format_duration(elapsed)}",
                flush=True,
            )
    log_stage_done(stage_name, started, RUN_NAME, fold)
    return frames


def run_pre_fusion_latent_probe(
    model: TemporalStatAligner,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    train_meta: dict[str, Any],
    val_files: list[str],
    test_files: list[str],
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    stat_input_indices: list[int],
    temporal_summary_mode: str,
    fold: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    top_k = int(args.latent_probe_topk)
    x_train, y_train, sample_weight, latent_names = collect_pre_fusion_training_table(
        model,
        cache,
        dataset,
        cfg,
        raw_indices,
        stat_indices,
        stat_input_indices,
        temporal_summary_mode,
        int(fold),
    )
    selected_indices, selector_meta = select_pre_fusion_latent_features(
        x_train,
        y_train,
        latent_names,
        top_k,
        args,
        int(fold),
        run_dir,
    )
    selected_names = [latent_names[index] for index in selected_indices]
    x_selected = np.ascontiguousarray(x_train[:, np.asarray(selected_indices, dtype=np.int64)])
    estimator, decision_meta = build_decision_model(args, y_train, int(args.seed) + int(fold) + 4001)
    train_started = log_stage_start(
        "latent_probe_train",
        RUN_NAME,
        fold,
        rows=len(y_train),
        features=x_selected.shape[1],
        decision_layer=args.decision_layer,
    )
    sample_weight_used = fit_decision_model(
        estimator,
        x_selected,
        y_train,
        sample_weight,
        bool(args.xgb_use_sample_weight),
    )
    decision_meta["sample_weight_used"] = bool(sample_weight_used)
    log_stage_done("latent_probe_train", train_started, RUN_NAME, fold)
    probe_dir = run_dir / "latent_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    model_path = probe_dir / f"pre_fusion_top{top_k}_{str(args.decision_layer).lower()}.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(
            {
                "model": estimator,
                "feature_names": selected_names,
                "selected_indices": selected_indices,
                "selector_meta": selector_meta,
                "decision_meta": decision_meta,
            },
            fh,
        )
    save_pre_fusion_probe_importance(estimator, selected_names, run_dir, top_k)
    del x_train, x_selected
    gc.collect()

    val_scores = score_pre_fusion_latent_files_to_memory(
        estimator,
        selected_indices,
        model,
        cache,
        val_files,
        cfg,
        raw_indices,
        stat_indices,
        stat_input_indices,
        temporal_summary_mode,
        int(fold),
        "latent_probe_score_val",
        top_k,
    )
    test_started = time.time()
    test_scores = score_pre_fusion_latent_files_to_memory(
        estimator,
        selected_indices,
        model,
        cache,
        test_files,
        cfg,
        raw_indices,
        stat_indices,
        stat_input_indices,
        temporal_summary_mode,
        int(fold),
        "latent_probe_score_test",
        top_k,
    )
    test_score_seconds = float(time.time() - test_started)
    scored_test_rows = int(sum(len(frame) for frame in test_scores.values()))
    tag = f"pre_fusion_latent_top{top_k}_{str(args.decision_layer).lower()}"
    threshold, val_metrics = select_threshold_for_scores(val_scores, args.data_dir, run_dir, tag, args)
    pred_dir = run_dir / "predictions" / tag
    eval_dir = run_dir / "evaluation" / tag
    test_rows = write_threshold_predictions(
        test_scores,
        pred_dir,
        threshold,
        confirm_k=int(args.confirm_k),
        confirm_m=int(args.confirm_m),
    )
    if test_files:
        metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
    else:
        # Inner CV never reads or evaluates the fixed test partition.
        metrics = {}

    print(
        f"[latent-probe-done] fold={fold} top_k={top_k} threshold={threshold:.4f} "
        f"temporal_selected={int(selector_meta['temporal_selected'])} "
        f"expert_stat_selected={int(selector_meta['expert_statistical_selected'])} "
        f"{format_metric_summary(metrics)}",
        flush=True,
    )
    del val_scores, test_scores
    gc.collect()
    return {
        "deep_model": RUN_NAME,
        "method_family": "HTSF latent probe",
        "method_label": f"{args.method_label or 'HTSF'} pre-fusion Top-{top_k}",
        "experiment_id": f"{args.experiment_id}_latent_top{top_k}".strip("_"),
        "temporal_encoder": args.temporal_encoder,
        "fold": int(fold),
        "mode": tag,
        "ml_model": args.decision_layer,
        "decision_layer": args.decision_layer,
        "ml_feature_set": "pre_fusion_latent_topk",
        "selector": "extra_trees",
        "selected_feature_count": top_k,
        "fusion_mode": "pre_fusion_latent_topk",
        "tsf_ablation": "latent_probe",
        "xgb_balance_mode": args.xgb_balance_mode,
        "xgb_use_sample_weight": bool(args.xgb_use_sample_weight),
        "xgb_scale_pos_weight": float(decision_meta.get("xgb_scale_pos_weight", 1.0)),
        "linear_class_weight": str(decision_meta.get("linear_class_weight", "not_applicable")),
        "threshold": float(threshold),
        "latent_probe_topk": top_k,
        "latent_probe_temporal_selected": int(selector_meta["temporal_selected"]),
        "latent_probe_statistical_selected": int(selector_meta["expert_statistical_selected"]),
        "latent_probe_temporal_importance": float(selector_meta["temporal_importance_sum"]),
        "latent_probe_statistical_importance": float(selector_meta["expert_statistical_importance_sum"]),
        "stat_selector": args.stat_selector,
        "stat_select_k": int(args.stat_select_k),
        "temporal_selector": args.temporal_selector,
        "temporal_select_k": int(args.temporal_select_k),
        "total_select_k": int(args.total_select_k),
        "temporal_summary_mode": args.temporal_summary_mode,
        "stat_feature_count": int(train_meta.get("stat_feature_selection", {}).get("selected_stat_feature_count", 0)),
        "sample_selection": args.sample_selection,
        "sample_topk_fraction": float(args.sample_topk_fraction),
        "temporal_positive_weight": float(args.temporal_positive_weight),
        "adaptive_negative_weight": float(args.adaptive_negative_weight),
        "train_rows": int(train_meta.get("train_rows", 0)),
        "train_pos_rows": int(train_meta.get("train_pos_rows", 0)),
        "train_neg_rows": int(train_meta.get("train_neg_rows", 0)),
        "test_score_seconds": test_score_seconds,
        "scored_test_rows": scored_test_rows,
        "scoring_rows_per_second": float(scored_test_rows / max(test_score_seconds, 1e-9)),
        "scoring_ms_per_window": float(1000.0 * test_score_seconds / max(scored_test_rows, 1)),
        "encoder_model_mb": float((run_dir / "aligned_encoder.pt").stat().st_size / (1024.0 * 1024.0)),
        "decision_model_mb": float(model_path.stat().st_size / (1024.0 * 1024.0)),
        "val_metrics": val_metrics,
        "test_rows": int(test_rows),
        "metrics": metrics,
    }


def run_fold(fold: int, args: argparse.Namespace, partitions=None) -> list[dict[str, Any]]:
    cfg = CompatCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        fixed_threshold=args.fixed_threshold,
        target_mode=args.target_mode,
        target_horizon_hours=args.target_horizon_hours,
        feature_mode=args.feature_mode,
        preserve_timepoints=bool(args.preserve_timepoints),
        sampling_mode="row_ratio" if str(args.sampling_mode).lower() == "row" else args.sampling_mode,
        positive_windows_per_module=args.positive_windows_per_module,
        negative_windows_per_faulty_module=args.negative_windows_per_faulty_module,
        normal_windows_per_module=args.normal_windows_per_module,
        val_fraction=args.val_fraction,
        threshold_search=bool(args.threshold_search),
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        rule_mode=args.rule_mode,
        sample_selection=args.sample_selection,
        sample_topk_fraction=args.sample_topk_fraction,
        sample_signal_mode=args.sample_signal_mode,
        sample_signal_temporal_fraction=args.sample_signal_temporal_fraction,
        temporal_positive_weight=args.temporal_positive_weight,
        temporal_weight_horizon_hours=args.temporal_weight_horizon_hours,
        adaptive_negative_weight=args.adaptive_negative_weight,
        adaptive_warmup_epochs=args.adaptive_warmup_epochs,
        max_cached_files=args.max_cached_files,
        module_cache_dir=str(args.module_cache_dir) if args.module_cache_dir else "",
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        amp=bool(args.amp),
        allow_tf32=not args.no_tf32,
        log_batches=args.log_batches,
        max_train_files=args.max_train_files,
        max_test_files=args.max_test_files,
        min_hit_lead_hours=args.min_hit_lead_hours,
    )
    configure_runtime(cfg, int(args.seed), int(fold), int(args.training_seed))
    fold_started = time.time()
    print(f"[aligned-init] fold={fold} device={cfg.device} torch={torch.__version__}", flush=True)
    index_df = read_index(args.index_path)
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    label_by_file = file_label_map(index_df)
    if int(args.max_train_files) > 0:
        train_files = cap_files_stratified(train_files, label_by_file, int(args.max_train_files), int(args.seed) + int(fold))
    if int(args.max_test_files) > 0:
        test_files = cap_files_stratified(test_files, label_by_file, int(args.max_test_files), int(args.seed) + 7919 + int(fold))
    train_files, val_files = split_train_val_files(train_files, label_by_file, cfg, int(fold))
    if partitions is not None:
        train_files, val_files, test_files = (list(partitions[key]) for key in ('train', 'validation', 'test'))
    sets = [set(train_files), set(val_files), set(test_files)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('A module occurs in more than one split')

    run_dir = Path(args.out_root) / RUN_NAME / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_mode = resolve_fusion_mode(args.fusion_mode, args.tsf_ablation)
    log_run_header(
        "SSFFN: SENSOR AND STATISTICAL FEATURE FUSION NETWORK",
        {
            "experiment id": args.experiment_id or "default",
            "temporal encoder": args.temporal_encoder,
            "temporal view": args.temporal_view_mode,
            "decision layer": args.decision_layer,
            "decision source": args.decision_source,
            "fold": fold,
            "train/val/test": f"{len(train_files)}/{len(val_files)}/{len(test_files)} modules",
            "target": cfg.target_mode,
            "feature mode": cfg.feature_mode,
            "stat features": args.stat_feature_mode,
            "stat feature groups": args.stat_feature_groups or "legacy-mode selection",
            "excluded stat groups": args.exclude_stat_feature_groups or "none",
            "temporal summaries": args.temporal_summary_mode,
            "fusion mode": resolved_mode,
            "tsf ablation": args.tsf_ablation,
            "latent probe top-k": int(args.latent_probe_topk) if int(args.latent_probe_topk) > 0 else "disabled",
            "rule mode": cfg.rule_mode,
            "alarm confirmation": f"{int(args.confirm_k)}-of-{int(args.confirm_m)} learned path only",
            "threshold constraint": args.threshold_constraint,
            "sample selection": cfg.sample_selection,
            "sample signal": cfg.sample_signal_mode,
            "signal temporal mix": cfg.sample_signal_temporal_fraction,
            "branch warmup": int(args.branch_warmup_epochs),
            "aux loss weights": f"temporal={args.temporal_aux_weight} stat={args.stat_aux_weight}",
            "module bag weight": float(args.module_bag_weight),
            "modality dropout": (
                f"temporal={args.temporal_modality_dropout} stat={args.stat_modality_dropout}"
            ),
            "xgb balance": args.xgb_balance_mode,
            "xgb sample weight": bool(args.xgb_use_sample_weight),
            "seq_len": cfg.seq_len,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "device": cfg.device,
            "out_dir": run_dir,
        },
    )
    stats_started = log_stage_start("feature_norm_stats", RUN_NAME, fold, files=len(train_files), data_dir=args.data_dir)
    stats = load_or_compute_feature_norm_stats(args.data_dir, train_files, cfg, label_by_file)
    log_stage_done("feature_norm_stats", stats_started, RUN_NAME, fold)
    (run_dir / 'normalization.json').write_text(json.dumps(asdict(stats), indent=2), encoding='utf-8')
    (run_dir / 'split.json').write_text(json.dumps(dict(train=train_files, validation=val_files, test=test_files), indent=2), encoding='utf-8')
    mean, std = stats.arrays()
    cache = Model2FeatureCache(args.data_dir, mean, std, cfg, label_by_file)
    names = compat_feature_names(cfg)
    all_raw_indices, all_raw_names, stat_indices, stat_names = feature_indices(
        names,
        args.stat_feature_mode,
        args.stat_feature_groups,
        args.exclude_stat_feature_groups,
    )
    stat_candidate_feature_names = stat_candidate_names(
        stat_names,
        all_raw_names,
        str(args.temporal_summary_mode),
    )
    original_stat_feature_count = len(stat_candidate_feature_names)
    (
        model,
        dataset,
        train_meta,
        raw_indices,
        raw_names,
        stat_input_indices,
        stat_input_names,
        stat_candidate_feature_names,
    ) = train_fusion_encoder(
        train_files,
        cache,
        cfg,
        all_raw_indices,
        all_raw_names,
        stat_indices,
        stat_names,
        args,
        int(fold),
        run_dir,
    )
    (run_dir / "feature_groups.json").write_text(
        json.dumps(
            {
                "temporal_encoder": args.temporal_encoder,
                "temporal_view_mode": args.temporal_view_mode,
                "decision_layer": args.decision_layer,
                "experiment_id": args.experiment_id,
                "raw_sequence_feature_candidates": all_raw_names,
                "raw_sequence_features": raw_names,
                "base_statistic_features": stat_names,
                "statistic_feature_candidates": stat_candidate_feature_names,
                "statistic_features": stat_input_names,
                "temporal_summary_mode": args.temporal_summary_mode,
                "temporal_summary_features": temporal_summary_feature_names(raw_names, str(args.temporal_summary_mode)),
                "feature_group_counts": {
                    "raw_sequence_features_original": len(all_raw_names),
                    "raw_sequence_features": len(raw_names),
                    "raw_sequence_window_values": len(raw_names) * int(cfg.seq_len),
                    "base_statistic_features": len(stat_names),
                    "temporal_summary_features": len(stat_candidate_feature_names) - len(stat_names),
                    "statistic_features_original": original_stat_feature_count,
                    "statistic_features_used": len(stat_input_names),
                    "sensor_token_dim": int(args.latent_dim),
                    "expert_token_dim": int(args.latent_dim),
                    "shared_output_dim": int(args.latent_dim),
                },
                "all_compat_features": names,
                "all_feature_groups": feature_group_manifest(names),
                "selected_stat_feature_groups": feature_group_manifest(stat_input_names),
                "train_meta": train_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "release_model": "SSFFN-split3",
            "module_variant": args.module_variant,
            "training_seed": args.training_seed,
            "normalization": asdict(stats),
            "cfg": asdict(cfg),
            "temporal_encoder": args.temporal_encoder,
            "temporal_view_mode": args.temporal_view_mode,
            "temporal_encoder_cfg": model.temporal_encoder_cfg,
            "patchtst_cfg": model.temporal_encoder_cfg if str(args.temporal_encoder) == "patchtst" else None,
            "raw_feature_candidates": all_raw_names,
            "raw_features": raw_names,
            "base_stat_features": stat_names,
            "stat_feature_candidates": stat_candidate_feature_names,
            "stat_features": stat_input_names,
            "stat_input_indices": stat_input_indices,
            "temporal_summary_mode": args.temporal_summary_mode,
            "fusion_mode": resolved_mode,
            "tsf_ablation": args.tsf_ablation,
            "train_meta": train_meta,
        },
        run_dir / "aligned_encoder.pt",
    )
    # Release the large training cache before full inference when sharing the GPU.
    cache.cache.clear()
    cfg.max_cached_files = 512
    assert str(args.decision_source) == 'encoder_head'
    x_train = np.zeros((0, model.latent_dim),dtype=np.float32)
    y_train = np.concatenate(dataset.selected_labels)
    sample_weight = np.concatenate(dataset.selected_weights)
    latent_names = [f'fused_{i:03d}' for i in range(model.latent_dim)]
    summarize_tsf_explainability(
        model,
        cache,
        dataset,
        cfg,
        raw_indices,
        stat_indices,
        stat_input_indices,
        raw_names,
        stat_input_names,
        run_dir,
        args,
    )
    if str(args.decision_source).lower() == "encoder_head":
        estimator = frozen_encoder_head(model)
        decision_meta = {
            "decision_layer": "linear",
            "decision_source": "encoder_head",
            "xgb_balance_mode": "not_applicable",
            "xgb_scale_pos_weight": 1.0,
            "xgb_raw_scale_pos_weight": 1.0,
            "linear_class_weight": "end_to_end_bce",
        }
    else:
        estimator, decision_meta = build_decision_model(args, y_train, int(args.seed) + int(fold))
        decision_meta["decision_source"] = "refit_latent"
    ml_started = log_stage_start(
        "decision_train_on_fused_latent",
        RUN_NAME,
        fold,
        rows=len(y_train),
        features=x_train.shape[1],
        decision_layer=args.decision_layer,
        balance=decision_meta.get("xgb_balance_mode", "not_applicable"),
        scale_pos_weight=f"{float(decision_meta.get('xgb_scale_pos_weight', 1.0)):.4f}",
        sample_weight=bool(args.xgb_use_sample_weight),
    )
    sample_weight_used = False
    if str(args.decision_source).lower() != "encoder_head":
        sample_weight_used = fit_decision_model(
            estimator,
            x_train,
            y_train,
            sample_weight,
            bool(args.xgb_use_sample_weight),
        )
    decision_meta["sample_weight_used"] = bool(sample_weight_used)
    log_stage_done("decision_train_on_fused_latent", ml_started, RUN_NAME, fold)
    save_decision_latent_importance(estimator, latent_names, run_dir)
    with (run_dir / "decision_layer.pkl").open("wb") as fh:
        pickle.dump({"model": estimator, "feature_names": latent_names, "decision_meta": decision_meta}, fh)
    if str(args.decision_layer).lower() == "xgb":
        with (run_dir / "xgb_fused_latent.pkl").open("wb") as fh:
            pickle.dump({"model": estimator, "feature_names": latent_names, "xgb_meta": decision_meta}, fh)
    val_scores = score_fused_files_to_memory(
        estimator,
        model,
        cache,
        val_files,
        cfg,
        raw_indices,
        stat_indices,
        stat_input_indices,
        str(args.temporal_summary_mode),
        int(fold),
        "score_val_files",
    )
    test_score_started = time.time()
    test_scores = score_fused_files_to_memory(
        estimator,
        model,
        cache,
        test_files,
        cfg,
        raw_indices,
        stat_indices,
        stat_input_indices,
        str(args.temporal_summary_mode),
        int(fold),
        "score_test_files",
    )
    test_score_seconds = float(time.time() - test_score_started)
    scored_test_rows = int(sum(len(frame) for frame in test_scores.values()))
    scoring_rows_per_second = scored_test_rows / max(test_score_seconds, 1e-9)
    scoring_ms_per_window = 1000.0 * test_score_seconds / max(scored_test_rows, 1)
    decision_tag = (
        "encoder_head_linear"
        if str(args.decision_source).lower() == "encoder_head"
        else f"aligned_latent_{str(args.decision_layer).lower()}"
    )
    threshold, val_metrics = select_threshold_for_scores(val_scores, args.data_dir, run_dir, decision_tag, args)
    pred_dir = run_dir / "predictions" / decision_tag
    eval_dir = run_dir / "evaluation" / decision_tag
    test_rows = write_threshold_predictions(
        test_scores,
        pred_dir,
        threshold,
        confirm_k=int(args.confirm_k),
        confirm_m=int(args.confirm_m),
    )
    if test_files:
        metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
    else:
        # Inner CV never reads or evaluates the fixed test partition.
        metrics = {}

    if str(args.lead_time_grid).strip():
        sweep_rows = run_lead_time_sweep(
            val_scores,
            test_scores,
            args.data_dir,
            run_dir,
            decision_tag,
            int(fold),
            args.lead_time_grid,
            args.threshold_grid,
            args.threshold_metric,
            args.fixed_threshold,
            bool(args.threshold_search),
            metadata={
                "run_name": RUN_NAME,
                "experiment_id": args.experiment_id,
                "temporal_encoder": args.temporal_encoder,
                "decision_layer": args.decision_layer,
                "mode": decision_tag,
                "feature_mode": args.feature_mode,
                "stat_feature_mode": args.stat_feature_mode,
                "temporal_summary_mode": args.temporal_summary_mode,
                "fusion_mode": resolved_mode,
                "tsf_ablation": args.tsf_ablation,
                "rule_mode": cfg.rule_mode,
                "xgb_balance_mode": args.xgb_balance_mode,
                "xgb_use_sample_weight": bool(args.xgb_use_sample_weight),
                "sample_signal_mode": args.sample_signal_mode,
                "sample_signal_temporal_fraction": float(args.sample_signal_temporal_fraction),
                "branch_warmup_epochs": int(args.branch_warmup_epochs),
                "temporal_aux_weight": float(args.temporal_aux_weight),
                "stat_aux_weight": float(args.stat_aux_weight),
                "temporal_modality_dropout": float(args.temporal_modality_dropout),
                "stat_modality_dropout": float(args.stat_modality_dropout),
            },
        )
        append_lead_time_sweep_results(sweep_rows, Path(args.out_root))
    print(f"[aligned-done] fold={fold} threshold={threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}", flush=True)
    decision_feature_count = int(x_train.shape[1])
    result = {
        "deep_model": RUN_NAME,
        "method_family": "HTSF",
        "method_label": args.method_label or "HTSF",
        "experiment_id": args.experiment_id,
        "temporal_encoder": args.temporal_encoder,
        "temporal_view_mode": args.temporal_view_mode,
        "fold": int(fold),
        "mode": decision_tag,
        "ml_model": args.decision_layer,
        "decision_layer": args.decision_layer,
        "decision_source": args.decision_source,
        "ml_feature_set": "aligned_latent",
        "selector": "none",
        "selected_feature_count": decision_feature_count,
        "fusion_mode": resolved_mode,
        "tsf_ablation": args.tsf_ablation,
        "rule_mode": cfg.rule_mode,
        "xgb_balance_mode": args.xgb_balance_mode,
        "xgb_use_sample_weight": bool(args.xgb_use_sample_weight),
        "xgb_scale_pos_weight": float(decision_meta.get("xgb_scale_pos_weight", 1.0)),
        "threshold": float(threshold),
        "threshold_constraint": args.threshold_constraint,
        "confirm_k": int(args.confirm_k),
        "confirm_m": int(args.confirm_m),
        "temporal_selector": args.temporal_selector,
        "temporal_select_k": int(args.temporal_select_k),
        "stat_selector": args.stat_selector,
        "stat_select_k": int(args.stat_select_k),
        "total_select_k": int(args.total_select_k),
        "temporal_summary_mode": args.temporal_summary_mode,
        "temporal_feature_count": len(raw_names),
        "stat_feature_count": len(stat_input_names),
        "selected_input_feature_count": len(raw_names) + len(stat_input_names),
        "stat_feature_groups": args.stat_feature_groups,
        "exclude_stat_feature_groups": args.exclude_stat_feature_groups,
        "sample_selection": args.sample_selection,
        "sample_topk_fraction": float(args.sample_topk_fraction),
        "sample_signal_mode": args.sample_signal_mode,
        "sample_signal_temporal_fraction": float(args.sample_signal_temporal_fraction),
        "branch_warmup_epochs": int(args.branch_warmup_epochs),
        "temporal_aux_weight": float(args.temporal_aux_weight),
        "stat_aux_weight": float(args.stat_aux_weight),
        "module_bag_weight": float(args.module_bag_weight),
        "temporal_modality_dropout": float(args.temporal_modality_dropout),
        "stat_modality_dropout": float(args.stat_modality_dropout),
        "temporal_positive_weight": float(args.temporal_positive_weight),
        "adaptive_negative_weight": float(args.adaptive_negative_weight),
        "train_rows": int(train_meta.get("train_rows", 0)),
        "train_pos_rows": int(train_meta.get("train_pos_rows", 0)),
        "train_neg_rows": int(train_meta.get("train_neg_rows", 0)),
        "test_score_seconds": test_score_seconds,
        "scored_test_rows": scored_test_rows,
        "scoring_rows_per_second": float(scoring_rows_per_second),
        "scoring_ms_per_window": float(scoring_ms_per_window),
        "encoder_model_mb": float((run_dir / "aligned_encoder.pt").stat().st_size / (1024.0 * 1024.0)),
        "decision_model_mb": float((run_dir / "decision_layer.pkl").stat().st_size / (1024.0 * 1024.0)),
        "val_metrics": val_metrics,
        "test_rows": int(test_rows),
        "metrics": metrics,
    }
    results = [result]
    del x_train, val_scores, test_scores
    gc.collect()
    if int(args.latent_probe_topk) > 0:
        results.append(
            run_pre_fusion_latent_probe(
                model,
                cache,
                dataset,
                train_meta,
                val_files,
                test_files,
                cfg,
                raw_indices,
                stat_indices,
                stat_input_indices,
                str(args.temporal_summary_mode),
                int(fold),
                run_dir,
                args,
            )
        )
    print(f"[fold-done] run={RUN_NAME} fold={fold} elapsed={format_duration(time.time() - fold_started)}", flush=True)
    model.close()
    del model, cache, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def aggregate_results(results: list[dict[str, Any]], out_root: Path, args: argparse.Namespace) -> None:
    rows = []
    for item in results:
        row = {
            "deep_model": item["deep_model"],
            "method_family": item.get("method_family", "HTSF"),
            "method_label": item.get("method_label", getattr(args, "method_label", "HTSF")),
            "experiment_id": item.get("experiment_id", getattr(args, "experiment_id", "")),
            "temporal_encoder": item.get("temporal_encoder", getattr(args, "temporal_encoder", "")),
            "temporal_view_mode": item.get(
                "temporal_view_mode", getattr(args, "temporal_view_mode", "level_mask")
            ),
            "fold": int(item["fold"]),
            "mode": item["mode"],
            "ml_model": item.get("ml_model", ""),
            "decision_layer": item.get("decision_layer", getattr(args, "decision_layer", "xgb")),
            "decision_source": item.get("decision_source", getattr(args, "decision_source", "refit_latent")),
            "ml_feature_set": item.get("ml_feature_set", ""),
            "selector": item.get("selector", ""),
            "selected_feature_count": int(item.get("selected_feature_count", 0)),
            "fusion_mode": item.get("fusion_mode", args.fusion_mode),
            "tsf_ablation": item.get("tsf_ablation", args.tsf_ablation),
            "xgb_balance_mode": item.get("xgb_balance_mode", args.xgb_balance_mode),
            "xgb_use_sample_weight": bool(item.get("xgb_use_sample_weight", args.xgb_use_sample_weight)),
            "xgb_scale_pos_weight": float(item.get("xgb_scale_pos_weight", 1.0)),
            "linear_class_weight": item.get("linear_class_weight", getattr(args, "linear_class_weight", "not_applicable")),
            "threshold": float(item.get("threshold", 0.0)),
            "threshold_constraint": item.get("threshold_constraint", getattr(args, "threshold_constraint", "none")),
            "confirm_k": int(item.get("confirm_k", getattr(args, "confirm_k", 1))),
            "confirm_m": int(item.get("confirm_m", getattr(args, "confirm_m", 1))),
            "target_mode": args.target_mode,
            "feature_mode": args.feature_mode,
            "stat_feature_mode": args.stat_feature_mode,
            "temporal_summary_mode": item.get("temporal_summary_mode", args.temporal_summary_mode),
            "temporal_selector": item.get("temporal_selector", args.temporal_selector),
            "temporal_select_k": int(item.get("temporal_select_k", args.temporal_select_k)),
            "stat_selector": item.get("stat_selector", args.stat_selector),
            "stat_select_k": int(item.get("stat_select_k", args.stat_select_k)),
            "total_select_k": int(item.get("total_select_k", args.total_select_k)),
            "latent_probe_topk": int(item.get("latent_probe_topk", 0)),
            "latent_probe_temporal_selected": int(item.get("latent_probe_temporal_selected", 0)),
            "latent_probe_statistical_selected": int(item.get("latent_probe_statistical_selected", 0)),
            "latent_probe_temporal_importance": float(item.get("latent_probe_temporal_importance", 0.0)),
            "latent_probe_statistical_importance": float(item.get("latent_probe_statistical_importance", 0.0)),
            "temporal_feature_count": int(item.get("temporal_feature_count", 0)),
            "stat_feature_count": int(item.get("stat_feature_count", 0)),
            "selected_input_feature_count": int(item.get("selected_input_feature_count", 0)),
            "stat_feature_groups": item.get("stat_feature_groups", getattr(args, "stat_feature_groups", "")),
            "exclude_stat_feature_groups": item.get(
                "exclude_stat_feature_groups",
                getattr(args, "exclude_stat_feature_groups", ""),
            ),
            "sampling_mode": args.sampling_mode,
            "sample_selection": item.get("sample_selection", args.sample_selection),
            "sample_topk_fraction": float(item.get("sample_topk_fraction", args.sample_topk_fraction)),
            "sample_signal_mode": item.get("sample_signal_mode", args.sample_signal_mode),
            "sample_signal_temporal_fraction": float(
                item.get("sample_signal_temporal_fraction", args.sample_signal_temporal_fraction)
            ),
            "branch_warmup_epochs": int(item.get("branch_warmup_epochs", args.branch_warmup_epochs)),
            "temporal_aux_weight": float(item.get("temporal_aux_weight", args.temporal_aux_weight)),
            "stat_aux_weight": float(item.get("stat_aux_weight", args.stat_aux_weight)),
            "module_bag_weight": float(item.get("module_bag_weight", getattr(args, "module_bag_weight", 0.0))),
            "temporal_modality_dropout": float(
                item.get("temporal_modality_dropout", args.temporal_modality_dropout)
            ),
            "stat_modality_dropout": float(item.get("stat_modality_dropout", args.stat_modality_dropout)),
            "temporal_positive_weight": float(
                item.get("temporal_positive_weight", args.temporal_positive_weight)
            ),
            "adaptive_negative_weight": float(
                item.get("adaptive_negative_weight", args.adaptive_negative_weight)
            ),
            "rule_mode": item.get("rule_mode", args.rule_mode),
            "min_hit_lead_hours": float(args.min_hit_lead_hours),
            "test_rows": int(item.get("test_rows", 0)),
            "train_rows": int(item.get("train_rows", 0)),
            "train_pos_rows": int(item.get("train_pos_rows", 0)),
            "train_neg_rows": int(item.get("train_neg_rows", 0)),
            "test_score_seconds": float(item.get("test_score_seconds", 0.0)),
            "scored_test_rows": int(item.get("scored_test_rows", 0)),
            "scoring_rows_per_second": float(item.get("scoring_rows_per_second", 0.0)),
            "scoring_ms_per_window": float(item.get("scoring_ms_per_window", 0.0)),
            "encoder_model_mb": float(item.get("encoder_model_mb", 0.0)),
            "decision_model_mb": float(item.get("decision_model_mb", 0.0)),
        }
        row.update(item.get("metrics", {}))
        rows.append(row)
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        dedupe_cols = [
            "deep_model",
            "experiment_id",
            "temporal_encoder",
            "temporal_view_mode",
            "fold",
            "mode",
            "decision_layer",
            "decision_source",
            "stat_feature_mode",
            "stat_feature_groups",
            "exclude_stat_feature_groups",
            "temporal_summary_mode",
            "temporal_selector",
            "temporal_select_k",
            "stat_selector",
            "stat_select_k",
            "total_select_k",
            "latent_probe_topk",
            "fusion_mode",
            "tsf_ablation",
            "xgb_balance_mode",
            "xgb_use_sample_weight",
            "linear_class_weight",
            "sampling_mode",
            "sample_selection",
            "sample_topk_fraction",
            "sample_signal_mode",
            "sample_signal_temporal_fraction",
            "branch_warmup_epochs",
            "temporal_aux_weight",
            "stat_aux_weight",
            "module_bag_weight",
            "temporal_modality_dropout",
            "stat_modality_dropout",
            "temporal_positive_weight",
            "adaptive_negative_weight",
            "rule_mode",
            "threshold_constraint",
            "confirm_k",
            "confirm_m",
        ]
        for col in dedupe_cols:
            if col in frame.columns and (pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col])):
                frame[col] = frame[col].fillna("")
        frame.drop_duplicates(
            subset=dedupe_cols,
            keep="last",
            inplace=True,
        )
    frame.sort_values(["deep_model", "mode", "fold"], inplace=True)
    frame.to_csv(path, index=False)
    group_cols = [
        "deep_model",
        "method_family",
        "method_label",
        "experiment_id",
        "temporal_encoder",
        "temporal_view_mode",
        "mode",
        "decision_layer",
        "decision_source",
        "target_mode",
        "feature_mode",
        "stat_feature_mode",
        "stat_feature_groups",
        "exclude_stat_feature_groups",
        "temporal_summary_mode",
        "temporal_selector",
        "temporal_select_k",
        "stat_selector",
        "stat_select_k",
        "total_select_k",
        "latent_probe_topk",
        "fusion_mode",
        "tsf_ablation",
        "sampling_mode",
        "sample_selection",
        "sample_topk_fraction",
        "sample_signal_mode",
        "sample_signal_temporal_fraction",
        "branch_warmup_epochs",
        "temporal_aux_weight",
        "stat_aux_weight",
        "module_bag_weight",
        "temporal_modality_dropout",
        "stat_modality_dropout",
        "temporal_positive_weight",
        "adaptive_negative_weight",
        "rule_mode",
        "threshold_constraint",
        "confirm_k",
        "confirm_m",
        "xgb_balance_mode",
        "xgb_use_sample_weight",
        "linear_class_weight",
        "min_hit_lead_hours",
    ]
    non_metric_cols = set(group_cols) | {"fold", "ml_model", "ml_feature_set", "selector"}
    numeric = [col for col in frame.columns if col not in non_metric_cols and pd.api.types.is_numeric_dtype(frame[col])]
    summary = frame.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTSF dual-view representation learning with a replaceable temporal encoder and decision layer."
    )
    parser.add_argument("--experiment_id", default="")
    parser.add_argument('--module_variant', choices=['full','no_sensor','no_statistics','no_sit','no_hss'], default='full')
    parser.add_argument('--followup',choices=['a4_typed','a4_no_expert','a4_no_dropout','a4_semantic','a4_random80'],default='a4_typed')
    parser.add_argument('--training_seed',type=int,default=42)
    parser.add_argument('--shared_layout', choices=['typed','sensor_grouped'], default='typed')
    parser.add_argument('--transfer_loss', choices=['weighted_bce', 'bce', 'atal', 'module_bce', 'module_margin'], default='weighted_bce')
    parser.add_argument('--transfer_hss', choices=['original', 'distribution_risk', 'prefix_risk'], default='original')
    parser.add_argument("--method_label", default="HTSF")
    parser.add_argument(
        "--temporal_encoder",
        choices=list(TEMPORAL_ENCODERS),
        default="patchtst",
        help="Temporal branch backbone used inside the TSF-XGBoost framework.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_model2_compat_results/patchtst_stat_aligned_xgb"))
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.3)
    parser.add_argument(
        "--target_mode",
        choices=["pre_event", "ahead_horizon", "ahead120", "anomaly", "module_fault"],
        default="pre_event",
    )
    parser.add_argument(
        "--target_horizon_hours",
        type=float,
        default=120.0,
        help="Explicit horizon used only when --target_mode ahead_horizon; lead-time evaluation remains independent.",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["model2", "model2_plus", "ofp", "ofp_plus"],
        default="ofp",
    )
    parser.add_argument("--preserve_timepoints", dest="preserve_timepoints", action="store_true")
    parser.add_argument("--legacy_drop_timepoints", dest="preserve_timepoints", action="store_false")
    parser.set_defaults(preserve_timepoints=True)
    parser.add_argument(
        "--stat_feature_mode",
        choices=["ofp_expert_stat", "all_engineered", "model2_expert", "statistics", "all"],
        default="ofp_expert_stat",
    )
    parser.add_argument(
        "--stat_feature_groups",
        default="",
        help="Comma-separated semantic feature groups or aliases from OFP/model2/ExperimentFeatureSchema.py.",
    )
    parser.add_argument(
        "--exclude_stat_feature_groups",
        default="",
        help="Comma-separated semantic groups removed after --stat_feature_groups is expanded.",
    )
    parser.add_argument(
        "--temporal_summary_mode",
        choices=list(TEMPORAL_SUMMARY_MODES),
        default="none",
        help="Optional temporal raw-channel summary features added to the statistic/expert branch candidate pool.",
    )
    parser.add_argument("--temporal_selector", choices=["none", "extra_trees"], default="none")
    parser.add_argument("--temporal_select_k", type=int, default=0)
    parser.add_argument("--temporal_selector_estimators", type=int, default=200)
    parser.add_argument("--temporal_selector_max_rows", type=int, default=200000)
    parser.add_argument("--stat_selector", choices=["none", "extra_trees"], default="none")
    parser.add_argument("--stat_select_k", type=int, default=0)
    parser.add_argument("--stat_selector_estimators", type=int, default=200)
    parser.add_argument("--stat_selector_max_rows", type=int, default=200000)
    parser.add_argument(
        "--total_select_k",
        type=int,
        default=0,
        help=(
            "Total number of selected raw temporal channels plus expert-statistical features. "
            "When positive, --temporal_select_k is required and the remaining budget is assigned "
            "to the expert-statistical branch."
        ),
    )
    parser.add_argument(
        "--sampling_mode",
        choices=["row", "row_ratio", "module_balanced"],
        default="module_balanced",
    )
    parser.add_argument("--positive_windows_per_module", type=int, default=32)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--normal_windows_per_module", type=int, default=8)
    parser.add_argument(
        "--rule_mode",
        choices=["none", "temp", "model2_simple", "ofp_rules"],
        default="none",
    )
    parser.add_argument("--sample_selection", choices=["random", "signal_topk", "hybrid"], default="hybrid")
    parser.add_argument("--sample_topk_fraction", type=float, default=0.5)
    parser.add_argument(
        "--sample_signal_mode",
        choices=["sensor_distribution"],
        default="sensor_distribution",
        help="Rule-free deviation of the 12 training-standardized sensor measurements.",
    )
    parser.add_argument("--sample_signal_temporal_fraction", type=float, default=0.5)
    parser.add_argument("--temporal_positive_weight", type=float, default=0.0)
    parser.add_argument("--temporal_weight_horizon_hours", type=float, default=120.0)
    parser.add_argument("--adaptive_negative_weight", type=float, default=0.0)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_grid", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument(
        "--threshold_constraint",
        choices=["none", "rule_nondegrade"],
        default="none",
        help="Optionally require validation F1/recall to be no worse than the hard Rule fallback.",
    )
    parser.add_argument("--threshold_constraint_tolerance", type=float, default=0.0)
    parser.add_argument("--confirm_k", type=int, default=1)
    parser.add_argument("--confirm_m", type=int, default=1)
    parser.add_argument("--no_threshold_search", dest="threshold_search", action="store_false")
    parser.set_defaults(threshold_search=True)
    parser.add_argument("--min_hit_lead_hours", type=float, default=0.0)
    parser.add_argument(
        "--lead_time_grid",
        default="",
        help=f"Optional DRAM-style lead-time sweep, e.g. '{DEFAULT_LEAD_TIME_GRID}'. Values accept m/min/h suffixes.",
    )
    parser.add_argument("--max_cached_files", type=int, default=128)
    parser.add_argument(
        "--module_cache_dir",
        type=Path,
        default=None,
        help="Optional shared disk cache for unnormalized OFP feature arrays.",
    )
    parser.add_argument("--max_train_files", type=int, default=2500)
    parser.add_argument("--max_test_files", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_batches", type=int, default=100)
    parser.add_argument("--explain_max_rows", type=int, default=2048)
    parser.add_argument("--no_explainability", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--seq_embedding_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument(
        "--latent_probe_topk",
        type=int,
        default=0,
        help=(
            "When positive, fit one ExtraTrees selector on the concatenated pre-fusion temporal and "
            "expert-statistical latents, train an additional decision model on the global Top-K, and "
            "report it alongside the fused HTSF result."
        ),
    )
    parser.add_argument("--latent_probe_estimators", type=int, default=300)
    parser.add_argument("--latent_probe_max_rows", type=int, default=300000)
    parser.add_argument("--stat_hidden", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--branch_warmup_epochs",
        type=int,
        default=0,
        help="Initial epochs trained only with branch auxiliary heads before enabling fused supervision.",
    )
    parser.add_argument("--temporal_aux_weight", type=float, default=0.0)
    parser.add_argument("--stat_aux_weight", type=float, default=0.0)
    parser.add_argument(
        "--module_bag_weight",
        type=float,
        default=0.0,
        help="Weight for module-level max-risk loss aligned with first-warning false alarms.",
    )
    parser.add_argument("--temporal_modality_dropout", type=float, default=0.0)
    parser.add_argument("--stat_modality_dropout", type=float, default=0.0)
    parser.add_argument(
        "--temporal_view_mode",
        choices=list(TEMPORAL_VIEW_MODES),
        default="level_mask",
        help="Local temporal-token inputs: raw level/mask or level/mask plus causal differences and EMA residuals.",
    )
    parser.add_argument("--fusion_mode", choices=list(FUSION_MODES), default="gated_attn")
    parser.add_argument(
        "--decision_source",
        choices=["encoder_head", "refit_latent"],
        default="refit_latent",
        help="Use the end-to-end neural linear head or refit a separate estimator on frozen latents.",
    )
    parser.add_argument(
        "--tsf_ablation",
        choices=["full", "no_stat_branch", "no_temporal_branch", "no_cross_attention"],
        default="full",
        help="TSF module ablation used for Table 5.",
    )
    parser.add_argument(
        "--decision_layer",
        choices=list(DECISION_LAYERS),
        default="xgb",
        help="Classifier applied to the learned HTSF representation.",
    )
    parser.add_argument("--xgb_balance_mode", choices=["auto", "sqrt", "none"], default="auto")
    parser.add_argument("--xgb_use_sample_weight", dest="xgb_use_sample_weight", action="store_true")
    parser.add_argument("--no_xgb_sample_weight", dest="xgb_use_sample_weight", action="store_false")
    parser.set_defaults(xgb_use_sample_weight=True)
    parser.add_argument(
        "--linear_class_weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Class weighting for the linear decision layer; use none when sampling/loss balancing is already applied.",
    )
    parser.add_argument("--ml_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--xgb_device", default="cuda")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--rf_max_depth", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    global RUN_NAME
    torch.set_num_threads(2)
    args = parse_args()
    args.temporal_encoder = str(args.temporal_encoder).lower()
    args.decision_layer = str(args.decision_layer).lower()
    RUN_NAME = "shared_sit"
    if str(args.gpu_id).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip()
    results: list[dict[str, Any]] = []
    for fold in args.folds:
        fold_results = run_fold(int(fold), args)
        results.extend(fold_results)
        marker = Path(args.out_root) / RUN_NAME / f'fold_{fold}' / 'transfer_result.json'
        marker.write_text(json.dumps(fold_results, indent=2), encoding='utf-8')
    # Parallel folds only write their own directory. Pooled aggregation is done
    # once by the study analyzer, never by competing per-fold processes.


if __name__ == "__main__":
    main()
