"""Training and OFP evaluation for dual-head deep models."""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.common import base_utils as base
from OFP_DL_DualTask.data import (
    DualTaskConfig,
    DualTaskWindowDataset,
    WindowSliceCache,
    collate_dual,
    compute_norm_stats,
    subset_norm_stats,
)


@dataclass
class DualTrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-2
    patience: int = 5
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    threshold_metric: str = "ofp_f1_score"
    threshold_grid: str = "coarse"
    prediction_mode: str = "ahead_only"
    monitor_val_max_windows: int = 200_000
    final_val_max_windows: int = 0
    eval_every: int = 1
    current_loss_weight: float = 1.0
    ahead_loss_weight: float = 1.0
    union_loss_weight: float = 0.25
    use_amp: bool = False
    export_predictions: bool = False
    hybrid_tree: bool = False
    hybrid_target: str = "ahead_label"
    hybrid_tree_model: str = "xgboost"
    hybrid_max_train_windows: int = 300_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _primary_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def threshold_values(kind: str) -> np.ndarray:
    if kind == "coarse":
        return np.linspace(0.05, 0.95, 19)
    if kind == "fine":
        return np.linspace(0.01, 0.99, 99)
    raise ValueError(f"unknown threshold grid: {kind}")


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
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    avg_lead_sec = sum(lead_sec_list) / tp if tp > 0 and lead_sec_list else 0.0
    avg_lead_hour = avg_lead_sec / 3600.0
    min_lead_sec = min(lead_sec_list) if lead_sec_list else 0.0
    min_lead_hour = min_lead_sec / 3600.0
    avg_lead_score = math.tanh(avg_lead_hour)
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


def _split_labels(split_df: pd.DataFrame | None, frame: pd.DataFrame) -> tuple[set[str], set[str], int]:
    if split_df is None:
        labels = frame.groupby("file_name")["event_label"].max()
        modules = set(labels.index.astype(str).tolist())
        true_pos = set(labels[labels == 1].index.astype(str).tolist())
        return modules, true_pos, len(modules)
    local = split_df.copy()
    local["file_name"] = local["file_name"].astype(str)
    label = pd.to_numeric(local["Label"], errors="coerce").fillna(0).astype(int)
    modules = set(local["file_name"].tolist())
    true_pos = set(local.loc[label == 1, "file_name"].tolist())
    return modules, true_pos, len(local)


def evaluate_dual_ofp_scores(
    meta_frame: pd.DataFrame,
    current_scores: np.ndarray,
    ahead_scores: np.ndarray,
    current_threshold: float,
    ahead_threshold: float,
    split_df: pd.DataFrame | None = None,
    prediction_mode: str = "ahead_only",
) -> dict[str, object]:
    if meta_frame.empty:
        modules, true_pos, all_cnt = _split_labels(split_df, meta_frame)
        fn = len(true_pos)
        return _ofp_report(0, 0, fn, all_cnt - fn, [], [], len(true_pos))

    eval_df = meta_frame.copy().reset_index(drop=True)
    eval_df["current_score"] = np.asarray(current_scores, dtype=float)
    eval_df["ahead_score"] = np.asarray(ahead_scores, dtype=float)
    mode = str(prediction_mode).lower()
    if mode == "ahead_only":
        pred_mask = eval_df["ahead_score"] >= float(ahead_threshold)
    elif mode == "current_only":
        pred_mask = eval_df["current_score"] >= float(current_threshold)
    elif mode in {"or", "dual_or", "legacy_or"}:
        pred_mask = (
            (eval_df["current_score"] >= float(current_threshold))
            | (eval_df["ahead_score"] >= float(ahead_threshold))
        )
    else:
        raise ValueError(f"unknown prediction_mode={prediction_mode!r}")
    eval_df["predict"] = pred_mask.astype(int)
    modules, true_pos_sns, all_cnt = _split_labels(split_df, eval_df)

    predicted_sns: set[str] = set()
    lead_sec_list: list[float] = []
    lead_pred_sn_list: list[str] = []
    same_or_after_fault_alerts = 0

    for file_name, group in eval_df.groupby("file_name", sort=False):
        file_name = str(file_name)
        if file_name not in modules:
            continue
        pred_rows = group[group["predict"] > 0]
        if pred_rows.empty:
            continue
        pred_ts = int(pred_rows["pred_ts"].min())
        first_failure_ts = int(group["first_failure_ts"].max())
        if first_failure_ts > 0:
            if first_failure_ts > pred_ts:
                predicted_sns.add(file_name)
                lead_sec_list.append(float(first_failure_ts - pred_ts))
                lead_pred_sn_list.append(file_name)
            else:
                same_or_after_fault_alerts += 1
        else:
            predicted_sns.add(file_name)

    hit_sns = true_pos_sns & predicted_sns
    tp = len(hit_sns)
    fp = len(predicted_sns) - tp
    fn = len(true_pos_sns) - tp
    tn = all_cnt - tp - fp - fn
    report = _ofp_report(tp, fp, fn, tn, lead_sec_list, lead_pred_sn_list, len(true_pos_sns))
    report.update(
        {
            "current_threshold": float(current_threshold),
            "ahead_threshold": float(ahead_threshold),
            "prediction_mode": mode,
            "same_or_after_fault_alert_modules": int(same_or_after_fault_alerts),
        }
    )
    return report


def choose_dual_thresholds(
    meta_frame: pd.DataFrame,
    current_scores: np.ndarray,
    ahead_scores: np.ndarray,
    split_df: pd.DataFrame | None,
    grid: np.ndarray,
    optimize_metric: str = "final_score",
    prediction_mode: str = "ahead_only",
) -> tuple[float, float, dict[str, object]]:
    metric_key = "final_score" if optimize_metric == "ofp_final_score" else "f1_score"
    best: dict[str, object] | None = None
    best_pair = (0.5, 0.5)
    mode = str(prediction_mode).lower()
    if mode == "ahead_only":
        candidates = [(1.1, float(ahead_threshold)) for ahead_threshold in grid]
    elif mode == "current_only":
        candidates = [(float(current_threshold), 1.1) for current_threshold in grid]
    elif mode in {"or", "dual_or", "legacy_or"}:
        candidates = [
            (float(current_threshold), float(ahead_threshold))
            for current_threshold in grid
            for ahead_threshold in grid
        ]
    else:
        raise ValueError(f"unknown prediction_mode={prediction_mode!r}")
    for current_threshold, ahead_threshold in candidates:
        metrics = evaluate_dual_ofp_scores(
            meta_frame,
            current_scores,
            ahead_scores,
            float(current_threshold),
            float(ahead_threshold),
            split_df=split_df,
            prediction_mode=mode,
        )
        if best is None:
            best = metrics
            best_pair = (float(current_threshold), float(ahead_threshold))
            continue
        if (
            float(metrics[metric_key]) > float(best[metric_key])
            or (
                float(metrics[metric_key]) == float(best[metric_key])
                and float(metrics["final_score"]) > float(best["final_score"])
            )
            or (
                float(metrics[metric_key]) == float(best[metric_key])
                and float(metrics["final_score"]) == float(best["final_score"])
                and float(metrics["recall"]) > float(best["recall"])
            )
        ):
            best = metrics
            best_pair = (float(current_threshold), float(ahead_threshold))
    return best_pair[0], best_pair[1], best or {}


def sample_eval_frame_and_split(
    frame: pd.DataFrame,
    split_df: pd.DataFrame | None,
    max_windows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Sample complete modules for light validation while preserving OFP semantics."""
    if max_windows <= 0 or frame.empty or len(frame) <= int(max_windows):
        return frame.reset_index(drop=True), split_df
    grouped = (
        frame.groupby("file_name", sort=False)
        .agg(windows=("file_name", "size"), event_label=("event_label", "max"), ahead_pos=("ahead_label", "sum"))
        .reset_index()
    )
    rng = np.random.default_rng(int(seed))
    chosen: list[str] = []
    used = 0

    def take_from(group: pd.DataFrame) -> None:
        nonlocal used
        order = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        for row in order.itertuples(index=False):
            n = int(row.windows)
            if chosen and used + n > int(max_windows):
                continue
            chosen.append(str(row.file_name))
            used += n
            if used >= int(max_windows):
                break

    pos_group = grouped[grouped["event_label"] > 0]
    neg_group = grouped[grouped["event_label"] <= 0]
    take_from(pos_group)
    if used < int(max_windows):
        take_from(neg_group)
    if not chosen:
        chosen = [str(grouped.iloc[0]["file_name"])]
    sampled = frame[frame["file_name"].astype(str).isin(set(chosen))].reset_index(drop=True)
    if split_df is None:
        sampled_split = None
    else:
        sampled_split = split_df[split_df["file_name"].astype(str).isin(set(chosen))].reset_index(drop=True)
    return sampled, sampled_split


def binary_metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    pred = (np.asarray(scores, dtype=float).reshape(-1) >= float(threshold)).astype(int)
    if len(y) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall)}


def _last_output_linear(model: nn.Module, output_dim: int = 2) -> nn.Linear | None:
    last = None
    for module in model.modules():
        if isinstance(module, nn.Linear) and int(module.out_features) == int(output_dim):
            last = module
    return last


class DualHeadLoss(nn.Module):
    """Two-head BCE with an auxiliary union-risk consistency term."""

    def __init__(
        self,
        pos_weight: torch.Tensor,
        head_weight: torch.Tensor,
        union_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.float())
        self.register_buffer("head_weight", head_weight.float())
        self.union_loss_weight = float(union_loss_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2 or logits.shape[1] != 2:
            raise ValueError(f"dual-head logits must have shape (B, 2), got {tuple(logits.shape)}")
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )
        loss = (bce * self.head_weight).mean()
        if self.union_loss_weight > 0:
            union_target = torch.clamp(targets[:, 0] + targets[:, 1], 0, 1)
            current_prob = torch.sigmoid(logits[:, 0])
            ahead_prob = torch.sigmoid(logits[:, 1])
            union_prob = 1.0 - (1.0 - current_prob) * (1.0 - ahead_prob)
            union_loss = F.binary_cross_entropy(union_prob.clamp(1e-6, 1 - 1e-6), union_target.float())
            loss = loss + self.union_loss_weight * union_loss
        return loss


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: str,
    grad_clip: float,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_n = 0
    score_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for xs, masks, ys, _metas in loader:
        xs = xs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True).float()
        if is_train:
            optimizer.zero_grad()
        amp_enabled = bool(use_amp and str(device).startswith("cuda"))
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _primary_logits(model(xs, masks))
                loss = loss_fn(logits, ys)
        if is_train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += float(loss.item()) * xs.size(0)
        total_n += xs.size(0)
        score_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
        target_parts.append(ys.detach().cpu().numpy())
    scores = np.concatenate(score_parts, axis=0) if score_parts else np.zeros((0, 2), dtype=float)
    targets = np.concatenate(target_parts, axis=0) if target_parts else np.zeros((0, 2), dtype=float)
    union_targets = np.clip(targets[:, 0] + targets[:, 1], 0, 1).astype(int) if len(targets) else np.zeros(0, dtype=int)
    return total_loss / max(total_n, 1), scores, targets, union_targets


@torch.no_grad()
def collect_scores_and_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    return_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    score_parts: list[np.ndarray] = []
    emb_parts: list[np.ndarray] = []
    captured: list[torch.Tensor] = []
    handle = None
    if return_embeddings:
        final_linear = _last_output_linear(model, output_dim=2)
        if final_linear is not None:
            def _capture(_module, inputs):
                if inputs:
                    captured.append(inputs[0].detach())

            handle = final_linear.register_forward_pre_hook(_capture)
    try:
        for xs, masks, _ys, _metas in loader:
            xs = xs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            before = len(captured)
            logits = _primary_logits(model(xs, masks))
            score_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
            if return_embeddings:
                if len(captured) > before:
                    emb = captured[-1].detach().float().cpu()
                    emb_parts.append(emb.reshape(emb.shape[0], -1).numpy())
                else:
                    emb_parts.append(logits.detach().float().cpu().reshape(logits.shape[0], -1).numpy())
    finally:
        if handle is not None:
            handle.remove()
    scores = np.concatenate(score_parts, axis=0) if score_parts else np.zeros((0, 2), dtype=float)
    if return_embeddings:
        embeddings = np.concatenate(emb_parts, axis=0) if emb_parts else np.zeros((0, 0), dtype=np.float32)
    else:
        embeddings = None
    return scores, embeddings


def build_model(model_name: str, n_sensors: int, obs_steps: int):
    if model_name == "itransformer":
        from model.Optical_prediction_model.deep_learning.models import ITransformerCfg, ITransformerClassifier

        cfg = ITransformerCfg(seq_len=obs_steps, n_sensors=n_sensors, output_dim=2)
        return ITransformerClassifier(cfg), asdict(cfg), "iTransformerDualHead"
    if model_name == "patchtst":
        from model.Optical_prediction_model.deep_learning.patchtst_wrapper import PatchTSTCfg, PatchTSTClassifier

        cfg = PatchTSTCfg(seq_len=obs_steps, n_sensors=n_sensors, output_dim=2)
        return PatchTSTClassifier(cfg), asdict(cfg), "PatchTSTDualHead"
    if model_name == "moderntcn":
        from model.Optical_prediction_model.deep_learning.moderntcn_wrapper import ModernTCNCfg, ModernTCNClassifier

        cfg = ModernTCNCfg(seq_len=obs_steps, n_sensors=n_sensors, output_dim=2)
        return ModernTCNClassifier(cfg), asdict(cfg), "ModernTCNDualHead"
    if model_name == "fits":
        from model.Optical_prediction_model.deep_learning.interpretable_fits import InterpFITSCfg, InterpFITSClassifier

        cfg = InterpFITSCfg(
            seq_len=obs_steps,
            n_sensors=n_sensors,
            output_dim=2,
            cut_freq=min(20, max(1, obs_steps // 2 + 1)),
        )
        return InterpFITSClassifier(cfg), asdict(cfg), "FITSDualHead"
    raise ValueError(f"unknown model name: {model_name}")


def export_prediction_files(
    meta_frame: pd.DataFrame,
    current_scores: np.ndarray,
    ahead_scores: np.ndarray,
    current_threshold: float,
    ahead_threshold: float,
    output_dir: Path,
    source_dir: Path = base.TRAINING_DIR,
    prediction_mode: str = "ahead_only",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = meta_frame[["file_name", "pred_ts"]].copy()
    pred_df["current_score"] = np.asarray(current_scores, dtype=float)
    pred_df["ahead_score"] = np.asarray(ahead_scores, dtype=float)
    mode = str(prediction_mode).lower()
    if mode == "ahead_only":
        pred_df["predict"] = (pred_df["ahead_score"] >= float(ahead_threshold)).astype(int)
    elif mode == "current_only":
        pred_df["predict"] = (pred_df["current_score"] >= float(current_threshold)).astype(int)
    elif mode in {"or", "dual_or", "legacy_or"}:
        pred_df["predict"] = (
            (pred_df["current_score"] >= float(current_threshold))
            | (pred_df["ahead_score"] >= float(ahead_threshold))
        ).astype(int)
    else:
        raise ValueError(f"unknown prediction_mode={prediction_mode!r}")
    for file_name, group in pred_df.groupby("file_name", sort=False):
        file_name = str(file_name)
        group = group.sort_values("pred_ts")
        if (source_dir / file_name).exists():
            raw_ts = pd.read_csv(source_dir / file_name, usecols=["timestamp"])
            out = raw_ts.copy()
            out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("int64")
            out["predict"] = 0
            out["current_score"] = 0.0
            out["ahead_score"] = 0.0
            agg = group.groupby("pred_ts", as_index=False).agg(
                {"predict": "max", "current_score": "max", "ahead_score": "max"}
            )
            out["predict"] = out["timestamp"].map(dict(zip(agg["pred_ts"], agg["predict"]))).fillna(0).astype(int)
            out["current_score"] = out["timestamp"].map(dict(zip(agg["pred_ts"], agg["current_score"]))).fillna(0.0)
            out["ahead_score"] = out["timestamp"].map(dict(zip(agg["pred_ts"], agg["ahead_score"]))).fillna(0.0)
        else:
            out = (
                group.groupby("pred_ts", as_index=False)
                .agg({"predict": "max", "current_score": "max", "ahead_score": "max"})
                .rename(columns={"pred_ts": "timestamp"})
            )
        out.to_csv(output_dir / file_name, index=False)


def build_model2_feature_matrix(meta_frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Align OFP model2 engineered features to DualTask prediction timestamps."""
    from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
        MODEL2_FEATURES,
        NA_DEFAULT,
        make_model2_features,
    )

    feature_names = [f"model2_{name}" for name in MODEL2_FEATURES]
    if meta_frame.empty:
        return np.empty((0, len(feature_names)), dtype=np.float32), feature_names
    out = np.full((len(meta_frame), len(MODEL2_FEATURES)), float(NA_DEFAULT), dtype=np.float32)
    order_col = "__order__"
    indexed_meta = meta_frame.reset_index(drop=True).reset_index()
    for file_name, group in indexed_meta.groupby("file_name", sort=False):
        file_name = str(file_name)
        path = base.TRAINING_DIR / file_name
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path)
            normal, _extra = make_model2_features(raw, with_label=True)
        except Exception:
            continue
        if normal.empty:
            continue
        for name in MODEL2_FEATURES:
            if name not in normal.columns:
                normal[name] = NA_DEFAULT
        feat = normal[MODEL2_FEATURES].copy()
        feat["__merge_ts__"] = pd.to_numeric(normal["Ts"], errors="coerce").astype("float64")
        feat = feat.dropna(subset=["__merge_ts__"]).sort_values("__merge_ts__")
        if feat.empty:
            continue
        wanted = pd.DataFrame(
            {
                order_col: group["index"].to_numpy(dtype=np.int64),
                "__merge_ts__": pd.to_numeric(group["pred_ts"], errors="coerce").astype("float64").to_numpy(),
            }
        ).dropna(subset=["__merge_ts__"]).sort_values("__merge_ts__")
        if wanted.empty:
            continue
        merged = pd.merge_asof(wanted, feat, on="__merge_ts__", direction="backward")
        values = merged[MODEL2_FEATURES].fillna(float(NA_DEFAULT)).to_numpy(dtype=np.float32)
        out[merged[order_col].to_numpy(dtype=np.int64)] = values
    out = np.nan_to_num(out, nan=float(NA_DEFAULT), posinf=float(NA_DEFAULT), neginf=float(NA_DEFAULT)).astype(np.float32)
    return out, feature_names


def _sample_hybrid_train_rows(labels: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if max_rows <= 0 or n <= int(max_rows):
        return np.arange(n, dtype=np.int64)
    pos = np.flatnonzero(labels > 0)
    neg = np.flatnonzero(labels <= 0)
    rng = np.random.default_rng(int(seed))
    if len(pos) >= int(max_rows):
        return np.sort(rng.choice(pos, size=int(max_rows), replace=False)).astype(np.int64)
    neg_take = min(len(neg), int(max_rows) - len(pos))
    chosen_neg = rng.choice(neg, size=neg_take, replace=False) if neg_take > 0 else np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate([pos, chosen_neg])).astype(np.int64)


def _fit_tree_classifier(x_train: np.ndarray, y_train: np.ndarray, model_name: str, seed: int):
    y_train = np.asarray(y_train, dtype=int)
    n_pos = int((y_train > 0).sum())
    n_neg = int((y_train <= 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError(f"hybrid tree needs both classes, got pos={n_pos} neg={n_neg}")
    name = str(model_name).lower()
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier

            clf = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
                random_state=int(seed),
                scale_pos_weight=max(1.0, n_neg / max(n_pos, 1)),
            )
            clf.fit(x_train, y_train)
            return clf, "xgboost"
        except Exception as exc:
            print(f"[hybrid] xgboost unavailable/failed ({exc}); fallback to RandomForest")
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=int(seed),
    )
    clf.fit(x_train, y_train)
    return clf, "random_forest"


def _tree_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return np.asarray(proba[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def run_hybrid_embedding_tree(
    model: nn.Module,
    datasets: dict[str, DualTaskWindowDataset],
    frames: dict[str, pd.DataFrame],
    split_map: dict[str, pd.DataFrame],
    train_cfg: DualTrainConfig,
    output_dir: Path,
) -> dict[str, object]:
    pred_loaders = {
        name: DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=train_cfg.num_workers,
            collate_fn=collate_dual,
            pin_memory=str(train_cfg.device).startswith("cuda"),
        )
        for name, dataset in datasets.items()
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, loader in pred_loaders.items():
        scores, embeddings = collect_scores_and_embeddings(
            model,
            loader,
            train_cfg.device,
            return_embeddings=True,
        )
        model2_x, model2_names = build_model2_feature_matrix(frames[name])
        emb = embeddings if embeddings is not None else np.zeros((len(frames[name]), 0), dtype=np.float32)
        x = np.concatenate([scores.astype(np.float32), emb.astype(np.float32), model2_x.astype(np.float32)], axis=1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        arrays[name] = {"x": x, "scores": scores}
    target_col = str(train_cfg.hybrid_target)
    if target_col == "module_label":
        target_col = "event_label"
    if target_col not in frames["train"].columns:
        raise ValueError(f"unknown hybrid_target={train_cfg.hybrid_target!r}")
    y_train_all = pd.to_numeric(frames["train"][target_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    selected = _sample_hybrid_train_rows(y_train_all, train_cfg.hybrid_max_train_windows, train_cfg.seed)
    x_train = arrays["train"]["x"][selected]
    y_train = y_train_all[selected]
    tree, tree_label = _fit_tree_classifier(x_train, y_train, train_cfg.hybrid_tree_model, train_cfg.seed)
    val_score = _tree_scores(tree, arrays["val"]["x"])
    test_score = _tree_scores(tree, arrays["test"]["x"])
    grid = threshold_values(train_cfg.threshold_grid)
    _cur_thr, tree_thr, val_metrics = choose_dual_thresholds(
        frames["val"],
        np.zeros_like(val_score),
        val_score,
        split_df=split_map.get("val"),
        grid=grid,
        optimize_metric=train_cfg.threshold_metric,
        prediction_mode="ahead_only",
    )
    test_metrics = evaluate_dual_ofp_scores(
        frames["test"],
        np.zeros_like(test_score),
        test_score,
        1.1,
        tree_thr,
        split_df=split_map.get("test"),
        prediction_mode="ahead_only",
    )
    payload = {
        "tree_model": tree_label,
        "target": str(train_cfg.hybrid_target),
        "train_rows": int(len(selected)),
        "train_pos": int((y_train > 0).sum()),
        "train_neg": int((y_train <= 0).sum()),
        "feature_dim": int(arrays["train"]["x"].shape[1]) if len(arrays["train"]["x"]) else 0,
        "threshold": float(tree_thr),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model2_feature_count": int(len(model2_names)),
    }
    hybrid_dir = output_dir / "results" / "hybrid_tree"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    with open(hybrid_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload


def fit_dual_task_model(
    frames: dict[str, pd.DataFrame],
    split_map: dict[str, pd.DataFrame],
    model_name: str,
    output_dir: Path,
    task_cfg: DualTaskConfig,
    train_cfg: DualTrainConfig,
) -> dict[str, object]:
    set_seed(train_cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_cache = WindowSliceCache()
    norm = subset_norm_stats(
        compute_norm_stats(frames["train"], slice_cache, task_cfg.obs_steps, max_modules=400),
        task_cfg.sensor_columns,
    )
    n_sensors = len(norm.sensors)
    model, model_cfg, model_label = build_model(model_name, n_sensors, task_cfg.obs_steps)
    monitor_val_frame, monitor_val_split = sample_eval_frame_and_split(
        frames["val"],
        split_map.get("val"),
        train_cfg.monitor_val_max_windows,
        seed=train_cfg.seed + 17,
    )
    final_val_frame, final_val_split = sample_eval_frame_and_split(
        frames["val"],
        split_map.get("val"),
        train_cfg.final_val_max_windows,
        seed=train_cfg.seed + 29,
    )

    datasets = {
        name: DualTaskWindowDataset(frame, task_cfg.obs_steps, slice_cache, norm)
        for name, frame in frames.items()
    }
    monitor_val_dataset = DualTaskWindowDataset(monitor_val_frame, task_cfg.obs_steps, slice_cache, norm)
    final_val_dataset = DualTaskWindowDataset(final_val_frame, task_cfg.obs_steps, slice_cache, norm)
    dl_kw = dict(
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_dual,
        pin_memory=str(train_cfg.device).startswith("cuda"),
    )
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **dl_kw),
        "val": DataLoader(monitor_val_dataset, shuffle=False, drop_last=False, **dl_kw),
        "final_val": DataLoader(final_val_dataset, shuffle=False, drop_last=False, **dl_kw),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **dl_kw),
    }

    y_train = frames["train"][["current_label", "ahead_label"]].to_numpy(dtype=np.float32)
    n_pos = y_train.sum(axis=0)
    n_neg = y_train.shape[0] - n_pos
    pos_weight = np.where(n_pos > 0, np.maximum(1.0, n_neg / np.maximum(n_pos, 1.0)), 1.0).astype(np.float32)
    head_weight = np.asarray([train_cfg.current_loss_weight, train_cfg.ahead_loss_weight], dtype=np.float32)

    device = train_cfg.device
    model = model.to(device)
    loss_fn = DualHeadLoss(
        pos_weight=torch.tensor(pos_weight, device=device),
        head_weight=torch.tensor(head_weight, device=device),
        union_loss_weight=train_cfg.union_loss_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(train_cfg.use_amp and str(device).startswith("cuda")),
    )
    grid = threshold_values(train_cfg.threshold_grid)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model_label} params={n_params/1e6:.2f}M sensors={n_sensors} device={device}")
    print(f"[sensors] {', '.join(norm.sensors)}")
    print(
        f"[protocol] prediction_mode={train_cfg.prediction_mode} threshold_metric={train_cfg.threshold_metric} "
        f"monitor_val_windows={len(monitor_val_frame)}/{len(frames['val'])} "
        f"final_val_windows={len(final_val_frame)}/{len(frames['val'])}"
    )
    print(
        "[loss] "
        f"current pos={int(n_pos[0])} neg={int(n_neg[0])} pos_w={pos_weight[0]:.2f}; "
        f"ahead pos={int(n_pos[1])} neg={int(n_neg[1])} pos_w={pos_weight[1]:.2f}"
    )

    best_state = None
    best_score = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.time()
        train_loss, _, _, _ = run_one_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_fn,
            device,
            grad_clip=train_cfg.grad_clip,
            scaler=scaler,
            use_amp=train_cfg.use_amp,
        )
        do_eval = int(train_cfg.eval_every) <= 1 or epoch % int(train_cfg.eval_every) == 0 or epoch == train_cfg.epochs
        if do_eval:
            val_loss, val_scores, val_targets, _ = run_one_epoch(
                model,
                loaders["val"],
                None,
                loss_fn,
                device,
                grad_clip=train_cfg.grad_clip,
                use_amp=train_cfg.use_amp,
            )
            cur_thr, ahead_thr, val_metrics = choose_dual_thresholds(
                monitor_val_frame,
                val_scores[:, 0],
                val_scores[:, 1],
                split_df=monitor_val_split,
                grid=grid,
                optimize_metric=train_cfg.threshold_metric,
                prediction_mode=train_cfg.prediction_mode,
            )
            ahead_row = binary_metrics_from_scores(val_targets[:, 1], val_scores[:, 1], ahead_thr)
            selection_key = "final_score" if train_cfg.threshold_metric == "ofp_final_score" else "f1_score"
            selection_score = float(val_metrics.get(selection_key, 0.0))
            improved = selection_score > best_score + 1e-4
            if improved:
                best_score = selection_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
        else:
            val_loss = float("nan")
            val_metrics = {"final_score": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            ahead_row = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            cur_thr, ahead_thr = 1.1, 0.5
            improved = False
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_current_threshold": float(cur_thr),
            "val_ahead_threshold": float(ahead_thr),
            "val_final_score": float(val_metrics.get("final_score", 0.0)),
            "val_f1_score": float(val_metrics.get("f1_score", 0.0)),
            "val_precision": float(val_metrics.get("precision", 0.0)),
            "val_recall": float(val_metrics.get("recall", 0.0)),
            "val_ahead_row_f1": float(ahead_row["f1"]),
            "epoch_seconds": float(time.time() - t0),
        }
        history.append(record)
        if do_eval:
            print(
                f"ep{epoch:02d} train={train_loss:.4f} val={val_loss:.4f} "
                f"OFP_F1={record['val_f1_score']:.3f} P={record['val_precision']:.3f} "
                f"R={record['val_recall']:.3f} rowF1={record['val_ahead_row_f1']:.3f} "
                f"thr=({cur_thr:.2f},{ahead_thr:.2f}) {time.time() - t0:.1f}s"
                f"{' *' if improved else ''}"
            )
        else:
            print(f"ep{epoch:02d} train={train_loss:.4f} val=skip {time.time() - t0:.1f}s")
        if do_eval and epochs_since_improve >= train_cfg.patience:
            print(f"[early-stop] no validation improvement for {train_cfg.patience} epochs")
            break

    if best_state is None:
        best_epoch = int(train_cfg.epochs)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    _, val_scores, _, _ = run_one_epoch(
        model,
        loaders["final_val"],
        None,
        loss_fn,
        device,
        grad_clip=train_cfg.grad_clip,
        use_amp=train_cfg.use_amp,
    )
    cur_thr, ahead_thr, threshold_metrics = choose_dual_thresholds(
        final_val_frame,
        val_scores[:, 0],
        val_scores[:, 1],
        split_df=final_val_split,
        grid=grid,
        optimize_metric=train_cfg.threshold_metric,
        prediction_mode=train_cfg.prediction_mode,
    )
    _, test_scores, test_targets, _ = run_one_epoch(
        model,
        loaders["test"],
        None,
        loss_fn,
        device,
        grad_clip=train_cfg.grad_clip,
        use_amp=train_cfg.use_amp,
    )
    test_metrics = evaluate_dual_ofp_scores(
        frames["test"],
        test_scores[:, 0],
        test_scores[:, 1],
        cur_thr,
        ahead_thr,
        split_df=split_map.get("test"),
        prediction_mode=train_cfg.prediction_mode,
    )
    if train_cfg.export_predictions:
        export_prediction_files(
            frames["test"],
            test_scores[:, 0],
            test_scores[:, 1],
            cur_thr,
            ahead_thr,
            output_dir / "results" / "ofp_predict_holdout",
            prediction_mode=train_cfg.prediction_mode,
        )
    hybrid_summary = None
    if train_cfg.hybrid_tree:
        hybrid_summary = run_hybrid_embedding_tree(
            model=model,
            datasets=datasets,
            frames=frames,
            split_map=split_map,
            train_cfg=train_cfg,
            output_dir=output_dir,
        )

    summary = {
        "model": model_label,
        "model_name": model_name,
        "model_cfg": model_cfg,
        "task_cfg": task_cfg.to_dict(),
        "train_cfg": asdict(train_cfg),
        "feature_count": int(n_sensors),
        "sensor_names": norm.sensors,
        "n_params": int(n_params),
        "pos_weight": pos_weight.tolist(),
        "head_weight": head_weight.tolist(),
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "threshold_selection": threshold_metrics,
        "current_threshold": float(cur_thr),
        "ahead_threshold": float(ahead_thr),
        "ofp_metrics": test_metrics,
        "hybrid_tree": hybrid_summary,
        "test_target_positive": {
            "current": int(test_targets[:, 0].sum()) if len(test_targets) else 0,
            "ahead": int(test_targets[:, 1].sum()) if len(test_targets) else 0,
        },
        "history": history,
    }
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "model_cfg": model_cfg, "task_cfg": task_cfg.to_dict()}, output_dir / "models" / f"{model_name}_dual.pt")
    return summary
