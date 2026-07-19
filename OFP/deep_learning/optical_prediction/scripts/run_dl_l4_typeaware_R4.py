"""Train and evaluate fault-type-aware optical-module prediction models on R4.

This is the MTHC-inspired extension:
  main task: binary fault prediction
  auxiliary tasks: weak multi-label optical fault-type prediction
  optional regularizers: fault-type attention diversity and sensor-group prior
"""
from __future__ import annotations

import argparse
import json
import math
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from OFP.deep_learning.optical_prediction.common import base_utils as base
from OFP.deep_learning.optical_prediction.common import task_utils as week1
from OFP.deep_learning.optical_prediction.deep_learning.cross_attn_itransformer import (
    CrossAttnITransformerCfg,
    L4_TypeAwareCrossAttnITransformer,
)
from OFP.deep_learning.optical_prediction.deep_learning.typeaware_transformer import (
    TypeAwareCrossAttnTransformer,
    TypeAwareTransformerCfg,
)
from OFP.deep_learning.optical_prediction.deep_learning.mf_transformer_v3 import (
    MFTransformerV3,
    MFTransformerV3Cfg,
)
from OFP.deep_learning.optical_prediction.deep_learning.data import (
    FailureWindowDataset,
    WindowSliceCache,
    compute_norm_stats,
)
from OFP.deep_learning.optical_prediction.deep_learning.fault_type_labels import (
    TYPE_DISPLAY_NAMES,
    TYPE_NAMES,
    WeakTypeLabelMeta,
    annotate_fault_type_frames,
    type_columns,
)
from OFP.deep_learning.optical_prediction.deep_learning.train import (
    TrainCfg,
    choose_threshold,
    event_metrics_from_preds,
    load_task_frames,
    set_seed,
)


EXP_ROOT = PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments"


@dataclass
class TypeLossCfg:
    variant: str = "full"
    type_loss_weight: float = 0.2
    diversity_weight: float = 0.03
    prior_weight: float = 0.05
    negative_type_weight: float = 0.05
    class_balanced_type_loss: bool = True
    concept_pos_weight_min: float = 0.25
    concept_pos_weight_max: float = 12.0
    threshold_quantile: float = 0.90
    fallback_quantile: float = 0.75


def r4_task_cfg() -> week1.Week1TaskConfig:
    return week1.Week1TaskConfig(
        obs_minutes=1440, lead_minutes=60, pred_minutes=60, step_minutes=60,
        feature_profile="fast", train_scope="prefirst", eval_scope="prefirst",
        train_label_mode="first_in_future", eval_label_mode="first_in_future",
        train_faulty_max_windows=96, train_healthy_max_windows=6,
        train_faulty_build_stride_minutes=30, train_healthy_build_stride_minutes=360,
    )


class TypeAwareFailureWindowDataset(FailureWindowDataset):
    def __init__(self, frame, obs_steps, slice_cache, norm, type_names):
        super().__init__(frame, obs_steps, slice_cache, norm)
        self.type_names = type_names
        self.type_cols = type_columns(type_names)

    def __getitem__(self, idx: int):
        x, m, y, meta = super().__getitem__(idx)
        row = self.frame.iloc[idx]
        type_target = torch.tensor(
            [float(row.get(col, 0.0)) for col in self.type_cols],
            dtype=torch.float32,
        )
        type_mask = torch.tensor(float(row.get("weak_type_mask", 0.0)), dtype=torch.float32)
        meta["weak_type_primary"] = str(row.get("weak_type_primary", "none"))
        return x, m, y, type_target, type_mask, meta


def collate_type(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ms = torch.stack([b[1] for b in batch], dim=0)
    ys = torch.stack([b[2] for b in batch], dim=0)
    type_targets = torch.stack([b[3] for b in batch], dim=0)
    type_masks = torch.stack([b[4] for b in batch], dim=0)
    metas = [b[5] for b in batch]
    return xs, ms, ys, type_targets, type_masks, metas


def make_sensor_prior(type_names: list[str], sensors: list[str]) -> torch.Tensor:
    groups = {
        "thermal_anomaly": ["temperature"],
        "current_bias_anomaly": ["current"],
        "tx_power_anomaly": ["currentTXPower"] + [f"currentMultiTXPower{i}" for i in range(1, 5)],
        "rx_power_anomaly": ["currentRXPower"] + [f"currentMultiRXPower{i}" for i in range(1, 5)],
        "lane_imbalance": [f"currentMultiTXPower{i}" for i in range(1, 5)]
                          + [f"currentMultiRXPower{i}" for i in range(1, 5)],
    }
    prior = np.zeros((len(type_names), len(sensors)), dtype=np.float32)
    for row, name in enumerate(type_names):
        group = set(groups.get(name, []))
        for col, sensor in enumerate(sensors):
            if sensor in group:
                prior[row, col] = 1.0
        if prior[row].sum() == 0:
            prior[row] = 1.0
    return torch.from_numpy(prior)


def attention_diversity_loss(attn_c2s: torch.Tensor, type_masks: torch.Tensor) -> torch.Tensor:
    sample_mask = type_masks > 0
    if sample_mask.any():
        attn = attn_c2s[sample_mask]
    else:
        attn = attn_c2s
    if attn.size(0) == 0 or attn.size(1) <= 1:
        return attn_c2s.new_tensor(0.0)
    normed = F.normalize(attn, p=2, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))
    eye = torch.eye(sim.size(1), device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye)
    return (off_diag ** 2).mean()


def sensor_prior_loss(
    attn_c2s: torch.Tensor,
    type_targets: torch.Tensor,
    type_masks: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    sample_weight = type_masks.unsqueeze(-1) * type_targets
    denom = sample_weight.sum()
    if float(denom.detach().cpu()) <= 0:
        return attn_c2s.new_tensor(0.0)
    mass = (attn_c2s * prior.unsqueeze(0)).sum(dim=-1).clamp(min=1e-6)
    return (-(torch.log(mass) * sample_weight).sum() / denom)


def compute_concept_pos_weight(
    frame: pd.DataFrame,
    type_names: list[str],
    min_weight: float = 0.25,
    max_weight: float = 12.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Class-balanced positive weights for weak fault-concept heads.

    We compute balance only on windows with weak fault-concept labels. Healthy
    windows are still used as all-zero concept negatives through
    ``negative_type_weight`` during training, but they are excluded here to
    avoid extreme weights caused by the large healthy/fault imbalance.
    """
    mask = pd.to_numeric(frame.get("weak_type_mask", 0.0), errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    cols = type_columns(type_names)
    if mask.sum() <= 0:
        weights = np.ones(len(type_names), dtype=np.float32)
    else:
        targets = frame.loc[mask, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        total = max(float(targets.shape[0]), 1.0)
        pos = targets.sum(axis=0)
        neg = np.maximum(total - pos, 0.0)
        weights = neg / np.maximum(pos, 1.0)
        weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    return torch.from_numpy(weights), {
        name: float(weights[idx]) for idx, name in enumerate(type_names)
    }


def concept_multitask_loss(
    type_logits: torch.Tensor,
    type_targets: torch.Tensor,
    type_masks: torch.Tensor,
    ys: torch.Tensor,
    type_loss_cfg: TypeLossCfg,
    concept_pos_weight: torch.Tensor,
) -> torch.Tensor:
    pos_weight = concept_pos_weight.to(device=type_logits.device, dtype=type_logits.dtype)
    if not type_loss_cfg.class_balanced_type_loss:
        pos_weight = torch.ones_like(pos_weight)

    raw = F.binary_cross_entropy_with_logits(
        type_logits,
        type_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    supervised_fault_weight = type_masks.unsqueeze(-1)
    healthy_negative_weight = (
        (ys <= 0).float().unsqueeze(-1) * float(type_loss_cfg.negative_type_weight)
    )
    weights = supervised_fault_weight + healthy_negative_weight
    return (raw * weights).sum() / weights.sum().clamp(min=1.0)


def fault_mode_hierarchy_loss(
    event_logits: torch.Tensor,
    type_logits: torch.Tensor,
) -> torch.Tensor:
    """MTHC-style parent-child consistency between event and fault-mode tasks.

    The binary fault-event head is the parent task and the fault-mode heads are
    child tasks. A child fault-mode risk should not exceed the parent event risk.
    """
    event_prob = torch.sigmoid(event_logits)
    max_type_prob = torch.sigmoid(type_logits).max(dim=-1).values
    return F.relu(max_type_prob - event_prob).pow(2).mean()


def task_support_consistency_loss(
    event_logits: torch.Tensor,
    type_logits: torch.Tensor,
    ys: torch.Tensor,
    margin: float = 0.05,
    negative_weight: float = 0.05,
    mode: str = "max",
) -> torch.Tensor:
    """Require positive event risks to be supported by fault-mode tasks.

    This is different from the parent-child hierarchy loss. For positive
    windows, the global event risk should not exceed the combined fault-mode
    support by more than a small margin. For negative windows, fault-mode heads
    are lightly discouraged from firing without an event.
    """
    event_prob = torch.sigmoid(event_logits)
    type_prob = torch.sigmoid(type_logits)
    if mode == "noisy_or":
        support_prob = 1.0 - torch.prod(
            (1.0 - type_prob).clamp(min=1e-6, max=1.0), dim=-1
        )
    elif mode == "top2":
        k = min(2, type_prob.size(-1))
        support_prob = type_prob.topk(k=k, dim=-1).values.mean(dim=-1)
    else:
        support_prob = type_prob.max(dim=-1).values

    pos_mask = (ys > 0).float()
    neg_mask = 1.0 - pos_mask
    pos_loss = (
        F.relu(event_prob - support_prob - float(margin)).pow(2) * pos_mask
    ).sum() / pos_mask.sum().clamp(min=1.0)
    neg_loss = (
        type_prob.pow(2).mean(dim=-1) * neg_mask
    ).sum() / neg_mask.sum().clamp(min=1.0)
    return pos_loss + float(negative_weight) * neg_loss


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss for binary classification with logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Reduces contribution of easy-to-classify samples, focusing on hard boundary cases.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # balance factor for positive class
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits (numerically stable)
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none",
                pos_weight=self.pos_weight,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none",
            )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce
        return loss.mean()


def run_one_epoch(
    model,
    loader,
    optimizer,
    fault_loss_fn,
    type_loss_cfg: TypeLossCfg,
    sensor_prior,
    concept_pos_weight,
    device,
    grad_clip=1.0,
    hierarchy_weight=0.0,
    support_weight=0.0,
    support_margin=0.05,
    support_negative_weight=0.05,
    support_mode="max",
    sensor_contrastive_weight=0.0,
):
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss, total_fault, total_type, total_div, total_prior, total_hier, total_support, total_contrast, total_n = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    )
    scores_all, y_all = [], []

    for xs, ms, ys, type_targets, type_masks, _meta in loader:
        xs = xs.to(device, non_blocking=True)
        ms = ms.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True).float()
        type_targets = type_targets.to(device, non_blocking=True)
        type_masks = type_masks.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(xs, ms)
            logits, attn_c2s, type_logits = outputs[0], outputs[2], outputs[4]
            fault_loss = fault_loss_fn(logits, ys)
            type_loss = concept_multitask_loss(
                type_logits,
                type_targets,
                type_masks,
                ys,
                type_loss_cfg,
                concept_pos_weight,
            )

            div_loss = attention_diversity_loss(attn_c2s, type_masks)
            prior_loss = sensor_prior_loss(attn_c2s, type_targets, type_masks, sensor_prior)

            # MTHC-style hierarchy consistency: event risk is the parent task,
            # and fault-mode risks are child tasks. This works for FTEformer
            # even though outputs[6] is reserved for sensor contrastive loss.
            hier_loss = logits.new_tensor(0.0)
            if hierarchy_weight > 0:
                hier_loss = fault_mode_hierarchy_loss(logits, type_logits)

            support_loss = logits.new_tensor(0.0)
            if support_weight > 0:
                support_loss = task_support_consistency_loss(
                    logits,
                    type_logits,
                    ys,
                    margin=support_margin,
                    negative_weight=support_negative_weight,
                    mode=support_mode,
                )

            # CATCH-style sensor contrastive loss (V3c)
            contrast_loss = logits.new_tensor(0.0)
            if sensor_contrastive_weight > 0 and len(outputs) > 6:
                # outputs[6] is sensor_contrastive_loss (scalar tensor)
                contrast_loss = outputs[6]

            loss = (
                fault_loss
                + type_loss_cfg.type_loss_weight * type_loss
                + type_loss_cfg.diversity_weight * div_loss
                + type_loss_cfg.prior_weight * prior_loss
                + hierarchy_weight * hier_loss
                + support_weight * support_loss
                + sensor_contrastive_weight * contrast_loss
            )

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        n = xs.size(0)
        total_loss += float(loss.item()) * n
        total_fault += float(fault_loss.item()) * n
        total_type += float(type_loss.item()) * n
        total_div += float(div_loss.item()) * n
        total_prior += float(prior_loss.item()) * n
        total_hier += float(hier_loss.item()) * n
        total_support += float(support_loss.item()) * n
        total_contrast += float(contrast_loss.item()) * n
        total_n += n
        scores_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_all.append(ys.detach().cpu().numpy().astype(int))

    denom = max(total_n, 1)
    loss_parts = {
        "loss": total_loss / denom,
        "fault_loss": total_fault / denom,
        "type_loss": total_type / denom,
        "diversity_loss": total_div / denom,
        "prior_loss": total_prior / denom,
        "hier_loss": total_hier / denom,
        "support_loss": total_support / denom,
        "contrast_loss": total_contrast / denom,
    }
    return loss_parts, np.concatenate(scores_all), np.concatenate(y_all)


def collect_outputs(model, loader, device):
    model.eval()
    all_weights, all_scores, all_y, all_type_probs, all_type_targets, all_type_masks = [], [], [], [], [], []
    all_c2s, all_s2c, all_s2t, all_sensor_masks, all_meta = [], [], [], [], []
    with torch.no_grad():
        for xs, ms, ys, type_targets, type_masks, metas in loader:
            xs = xs.to(device, non_blocking=True)
            ms = ms.to(device, non_blocking=True)
            outputs = model(xs, ms)
            logits = outputs[0]
            all_weights.append(outputs[1].cpu().numpy())
            all_c2s.append(outputs[2].cpu().numpy())
            all_s2c.append(outputs[3].cpu().numpy())
            if len(outputs) > 5:
                all_s2t.append(outputs[5].cpu().numpy())
            if len(outputs) > 7 and outputs[7] is not None:
                all_sensor_masks.append(outputs[7].cpu().numpy())
            all_type_probs.append(torch.sigmoid(outputs[4]).cpu().numpy())
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(ys.numpy().astype(int))
            all_type_targets.append(type_targets.numpy().astype(np.float32))
            all_type_masks.append(type_masks.numpy().astype(np.float32))
            all_meta.extend(metas)
    return {
        "weights": np.concatenate(all_weights),
        "scores": np.concatenate(all_scores),
        "y_true": np.concatenate(all_y),
        "type_probs": np.concatenate(all_type_probs),
        "type_targets": np.concatenate(all_type_targets),
        "type_masks": np.concatenate(all_type_masks),
        "concept_to_sensor": np.concatenate(all_c2s),
        "sensor_to_concept": np.concatenate(all_s2c),
        "sensor_to_time": np.concatenate(all_s2t) if all_s2t else None,
        "dynamic_sensor_mask": np.concatenate(all_sensor_masks) if all_sensor_masks else None,
        "meta": all_meta,
    }


def sensor_mask_metrics(sensor_mask: np.ndarray | None, y_true: np.ndarray,
                        y_pred: np.ndarray) -> dict | None:
    if sensor_mask is None:
        return None
    mask = np.asarray(sensor_mask, dtype=np.float32)
    if mask.ndim != 3 or mask.shape[-1] != mask.shape[-2]:
        return {"shape": list(mask.shape)}

    n_sensors = mask.shape[-1]
    offdiag = ~np.eye(n_sensors, dtype=bool)
    offdiag_values = mask[:, offdiag]
    metrics = {
        "shape": list(mask.shape),
        "mean_offdiag_density": float(offdiag_values.mean()),
        "std_offdiag_density": float(offdiag_values.std()),
        "mean_diag": float(np.diagonal(mask, axis1=1, axis2=2).mean()),
    }
    for name, sample_mask in {
        "true_positive_windows": (y_true == 1) & (y_pred == 1),
        "false_positive_windows": (y_true == 0) & (y_pred == 1),
        "negative_windows": y_true == 0,
        "positive_label_windows": y_true == 1,
    }.items():
        if sample_mask.any():
            metrics[f"{name}_mean_offdiag_density"] = float(mask[sample_mask][:, offdiag].mean())
            metrics[f"{name}_count"] = int(sample_mask.sum())
        else:
            metrics[f"{name}_count"] = 0
    return metrics


def support_metrics(scores: np.ndarray, type_probs: np.ndarray, y_true: np.ndarray,
                    y_pred: np.ndarray, mode: str = "max") -> dict:
    type_probs = np.asarray(type_probs, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    noisy_or_support = 1.0 - np.prod(np.clip(1.0 - type_probs, 1e-6, 1.0), axis=-1)
    max_support = type_probs.max(axis=-1)
    top2_support = np.sort(type_probs, axis=-1)[:, -2:].mean(axis=-1)
    if mode == "noisy_or":
        support = noisy_or_support
    elif mode == "top2":
        support = top2_support
    else:
        support = max_support
    gap = scores - support
    out = {
        "support_mode": mode,
        "mean_event_score": float(scores.mean()),
        "mean_support_score": float(support.mean()),
        "mean_noisy_or_support": float(noisy_or_support.mean()),
        "mean_max_support": float(max_support.mean()),
        "mean_top2_support": float(top2_support.mean()),
        "mean_event_minus_support": float(gap.mean()),
        "positive_mean_support": float(support[y_true == 1].mean()) if np.any(y_true == 1) else float("nan"),
        "negative_mean_support": float(support[y_true == 0].mean()) if np.any(y_true == 0) else float("nan"),
    }
    groups = {
        "true_positive": (y_true == 1) & (y_pred == 1),
        "false_positive": (y_true == 0) & (y_pred == 1),
        "false_negative": (y_true == 1) & (y_pred == 0),
        "true_negative": (y_true == 0) & (y_pred == 0),
    }
    for name, mask in groups.items():
        out[f"{name}_count"] = int(mask.sum())
        if np.any(mask):
            out[f"{name}_mean_event_score"] = float(scores[mask].mean())
            out[f"{name}_mean_support_score"] = float(support[mask].mean())
            out[f"{name}_mean_event_minus_support"] = float(gap[mask].mean())
    return out


def type_metrics(type_probs: np.ndarray, type_targets: np.ndarray, type_masks: np.ndarray,
                 type_names: list[str] | None = None) -> dict:
    type_names = type_names or TYPE_NAMES
    mask = type_masks > 0
    if not mask.any():
        return {"type_samples": 0}
    pred = (type_probs[mask] >= 0.5).astype(int)
    true = type_targets[mask].astype(int)
    exact = float((pred == true).all(axis=1).mean())
    per_type = {}
    for idx, name in enumerate(type_names):
        tp = int(((pred[:, idx] == 1) & (true[:, idx] == 1)).sum())
        fp = int(((pred[:, idx] == 1) & (true[:, idx] == 0)).sum())
        fn = int(((pred[:, idx] == 0) & (true[:, idx] == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_type[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(true[:, idx].sum())}
    return {"type_samples": int(mask.sum()), "exact_match": exact, "per_type": per_type}


def concept_similarity(concept_to_sensor: np.ndarray, sample_mask: np.ndarray) -> float:
    if sample_mask.any():
        attn = concept_to_sensor[sample_mask]
    else:
        attn = concept_to_sensor
    if len(attn) == 0:
        return float("nan")
    norm = attn / np.clip(np.linalg.norm(attn, axis=-1, keepdims=True), 1e-8, None)
    sim = np.matmul(norm, np.swapaxes(norm, 1, 2))
    k = sim.shape[1]
    off = sim[:, ~np.eye(k, dtype=bool)]
    return float(np.mean(off))


def train_typeaware_l4(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    sensors: list[str],
    train_cfg: TrainCfg,
    type_loss_cfg: TypeLossCfg,
    label_meta: WeakTypeLabelMeta,
    backbone: str = "itransformer",
    time_pool_size: int = 6,
    type_names: list[str] | None = None,
    focal_loss: bool = False,
    hierarchy_weight: float = 0.0,
    support_weight: float = 0.0,
    support_margin: float = 0.05,
    support_negative_weight: float = 0.05,
    support_mode: str = "max",
) -> dict:
    if type_names is None:
        type_names = TYPE_NAMES
    summary_backbone = "fteformer" if backbone in {"creformer", "fteformer"} else backbone
    set_seed(train_cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_steps = int(frames["train"].iloc[0]["obs_end_idx"] - frames["train"].iloc[0]["obs_start_idx"] + 1)
    slice_cache = WindowSliceCache()
    norm = compute_norm_stats(frames["train"], slice_cache, obs_steps, max_modules=400)

    ds_train = TypeAwareFailureWindowDataset(frames["train"], obs_steps, slice_cache, norm, type_names)
    ds_val = TypeAwareFailureWindowDataset(frames["val"], obs_steps, slice_cache, norm, type_names)
    ds_test = TypeAwareFailureWindowDataset(frames["test"], obs_steps, slice_cache, norm, type_names)

    dl_kw = dict(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
                 collate_fn=collate_type, pin_memory=(train_cfg.device == "cuda"))
    train_loader = DataLoader(ds_train, shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(ds_val, shuffle=False, drop_last=False, **dl_kw)
    test_loader = DataLoader(ds_test, shuffle=False, drop_last=False, **dl_kw)

    device = train_cfg.device
    if backbone == "itransformer":
        cfg = CrossAttnITransformerCfg(
            seq_len=obs_steps,
            n_sensors=len(sensors),
            use_mask_channel=True,
            d_model=128,
            n_heads=8,
            e_layers=3,
            d_ff=256,
            dropout=0.2,
            activation="gelu",
            factor=1,
            use_norm=True,
            concept_names=list(type_names),
            n_concepts=len(type_names),
            concept_dropout=0.1,
            bypass_ratio=0.3,
        )
        model = L4_TypeAwareCrossAttnITransformer(cfg).to(device)
        model_name = "iTransformer+MFP"
        log_prefix = "iTransformer+MFP"
    elif backbone == "transformer":
        cfg = TypeAwareTransformerCfg(
            seq_len=obs_steps,
            n_sensors=len(sensors),
            use_mask_channel=True,
            d_model=128,
            n_heads=8,
            e_layers=3,
            d_ff=256,
            dropout=0.2,
            activation="gelu",
            use_norm=True,
            concept_names=list(type_names),
            n_concepts=len(type_names),
            bypass_ratio=0.3,
            n_mfp_layers=2,
            use_sensor_series_residual=True,
            time_pool_size=time_pool_size,
        )
        model = TypeAwareCrossAttnTransformer(cfg).to(device)
        model_name = "MFP-Former"
        log_prefix = "MFP-Former"
    elif backbone == "mftransformerv3a":
        # Tuned V3: higher concept mix, less leak, no prior_loss needed
        cfg = MFTransformerV3Cfg(
            seq_len=obs_steps,
            n_sensors=len(sensors),
            use_mask_channel=True,
            d_model=128,
            n_heads=8,
            e_layers=3,
            d_ff=256,
            dropout=0.2,
            activation="gelu",
            use_norm=True,
            concept_names=list(type_names),
            n_types=len(type_names),
            bypass_ratio=0.3,
            n_mfp_layers=2,
            use_sensor_series_residual=True,
            time_pool_size=time_pool_size,
            concept_risk_mix=0.50,
            leak_ratio=0.15,
            hierarchy_weight=hierarchy_weight,
        )
        model = MFTransformerV3(cfg).to(device)
        model_name = "MFTransformerV3a"
        log_prefix = "MFTransformerV3a"
    elif backbone in {"creformer", "fteformer"}:
        # FTEformer: fault-type experts + dynamic sensor mask + contrastive loss
        cfg = MFTransformerV3Cfg(
            seq_len=obs_steps,
            n_sensors=len(sensors),
            use_mask_channel=True,
            d_model=128,
            n_heads=8,
            e_layers=3,
            d_ff=256,
            dropout=0.2,
            activation="gelu",
            use_norm=True,
            concept_names=list(type_names),
            n_types=len(type_names),
            bypass_ratio=0.3,
            n_mfp_layers=2,
            use_sensor_series_residual=True,
            time_pool_size=time_pool_size,
            concept_risk_mix=0.50,
            leak_ratio=0.15,
            hierarchy_weight=hierarchy_weight,
            use_dynamic_sensor_mask=True,
            sensor_mask_mode=train_cfg.sensor_mask_mode,
            sensor_mask_temperature=train_cfg.sensor_mask_temperature,
            sensor_contrastive_lambda=0.3,
            sensor_contrastive_weight=train_cfg.sensor_contrastive_weight,
        )
        model = MFTransformerV3(cfg).to(device)
        model_name = "FTEformer"
        log_prefix = "FTEformer"
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{log_prefix}] params={n_params / 1e6:.2f}M, device={device}")

    y_train_all = frames["train"]["label"].to_numpy(dtype=int)
    n_pos = int((y_train_all == 1).sum())
    n_neg = int((y_train_all == 0).sum())
    pos_weight = max(1.0, n_neg / max(n_pos, 1))
    print(f"[loss] fault BCE pos_weight={pos_weight:.2f}")
    print(f"[weak-type] train coverage={label_meta.train_positive_coverage:.3f}, counts={label_meta.train_positive_counts}")
    if focal_loss:
        fault_loss_fn = FocalBCEWithLogitsLoss(
            gamma=2.0, alpha=0.75,
            pos_weight=torch.tensor(pos_weight, device=device),
        )
        print(f"[loss] using Focal Loss (gamma=2.0, alpha=0.75)")
    else:
        fault_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    sensor_prior = make_sensor_prior(type_names, sensors).to(device)
    concept_pos_weight, concept_pos_weight_map = compute_concept_pos_weight(
        frames["train"],
        type_names,
        min_weight=type_loss_cfg.concept_pos_weight_min,
        max_weight=type_loss_cfg.concept_pos_weight_max,
    )
    concept_pos_weight = concept_pos_weight.to(device)
    print(f"[loss] class-balanced concept pos_weight={concept_pos_weight_map}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)

    def lr_lambda(epoch):
        if epoch < train_cfg.warmup_epochs:
            return float(epoch + 1) / float(max(1, train_cfg.warmup_epochs))
        prog = (epoch - train_cfg.warmup_epochs) / max(1, train_cfg.epochs - train_cfg.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_state, best_val_f1, best_val_auc = None, -1.0, -1.0
    history, epochs_since_improve = [], 0

    hier_w = float(cfg.hierarchy_weight) if hasattr(cfg, "hierarchy_weight") else 0.0
    scw = float(cfg.sensor_contrastive_weight) if hasattr(cfg, "sensor_contrastive_weight") else 0.0
    support_w = float(support_weight)

    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.time()
        tr_parts, _, _ = run_one_epoch(
            model, train_loader, optimizer, fault_loss_fn, type_loss_cfg,
            sensor_prior, concept_pos_weight, device, grad_clip=train_cfg.grad_clip,
            hierarchy_weight=hier_w, support_weight=support_w,
            support_margin=support_margin, support_negative_weight=support_negative_weight,
            support_mode=support_mode,
            sensor_contrastive_weight=scw,
        )
        scheduler.step()
        va_parts, va_scores, va_y = run_one_epoch(
            model, val_loader, None, fault_loss_fn, type_loss_cfg,
            sensor_prior, concept_pos_weight, device,
            hierarchy_weight=hier_w, support_weight=support_w,
            support_margin=support_margin, support_negative_weight=support_negative_weight,
            support_mode=support_mode,
            sensor_contrastive_weight=scw,
        )
        _, thr_meta = choose_threshold(va_y, va_scores)
        try:
            from sklearn.metrics import roc_auc_score
            va_auc = float(roc_auc_score(va_y, va_scores)) if len(set(va_y)) > 1 else float("nan")
        except Exception:
            va_auc = float("nan")
        improved = thr_meta["f1"] > best_val_f1 + 1e-4
        if improved:
            best_val_f1 = thr_meta["f1"]
            best_val_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        row = {
            "epoch": epoch,
            "train_loss": tr_parts["loss"],
            "train_fault_loss": tr_parts["fault_loss"],
            "train_type_loss": tr_parts["type_loss"],
            "train_diversity_loss": tr_parts["diversity_loss"],
            "train_prior_loss": tr_parts["prior_loss"],
            "train_hier_loss": tr_parts["hier_loss"],
            "train_support_loss": tr_parts["support_loss"],
            "train_contrast_loss": tr_parts["contrast_loss"],
            "val_loss": va_parts["loss"],
            "val_fault_loss": va_parts["fault_loss"],
            "val_type_loss": va_parts["type_loss"],
            "val_diversity_loss": va_parts["diversity_loss"],
            "val_prior_loss": va_parts["prior_loss"],
            "val_hier_loss": va_parts["hier_loss"],
            "val_support_loss": va_parts["support_loss"],
            "val_contrast_loss": va_parts["contrast_loss"],
            "val_f1": thr_meta["f1"],
            "val_precision": thr_meta["precision"],
            "val_recall": thr_meta["recall"],
            "val_threshold": thr_meta["threshold"],
            "val_auc": va_auc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"  ep{epoch:02d} | loss={tr_parts['loss']:.4f} "
            f"(fault={tr_parts['fault_loss']:.3f}, type={tr_parts['type_loss']:.3f}, "
            f"div={tr_parts['diversity_loss']:.3f}, prior={tr_parts['prior_loss']:.3f}, "
            f"support={tr_parts['support_loss']:.3f}, mask={tr_parts['contrast_loss']:.3f}) "
            f"va_F1={thr_meta['f1']:.3f} thr={thr_meta['threshold']:.2f} "
            f"AUC={va_auc:.3f} | {time.time() - t0:.1f}s{' *' if improved else ''}"
        )
        if epochs_since_improve >= train_cfg.patience:
            print(f"  early-stop at ep{epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    _, va_scores, va_y = run_one_epoch(
        model, val_loader, None, fault_loss_fn, type_loss_cfg,
        sensor_prior, concept_pos_weight, device,
            hierarchy_weight=hier_w, support_weight=support_w,
            support_margin=support_margin, support_negative_weight=support_negative_weight,
            support_mode=support_mode,
            sensor_contrastive_weight=scw,
    )
    threshold, threshold_meta = choose_threshold(va_y, va_scores)
    test_out = collect_outputs(model, test_loader, device)
    te_scores = test_out["scores"]
    te_y = test_out["y_true"]
    test_pred = (te_scores >= threshold).astype(int)

    window_metrics = base.evaluate_binary_scores(te_y, te_scores, threshold)
    event_metrics = event_metrics_from_preds(frames["test"], test_pred)

    pos_mask = test_pred == 1
    if pos_mask.any():
        global_sensor_importance = test_out["weights"][pos_mask].mean(axis=0).tolist()
    else:
        global_sensor_importance = test_out["weights"].mean(axis=0).tolist()
    sensor_ranking = sorted(zip(sensors, global_sensor_importance), key=lambda x: x[1], reverse=True)
    type_summary = type_metrics(test_out["type_probs"], test_out["type_targets"], test_out["type_masks"],
                                type_names=type_names)
    support_summary = support_metrics(te_scores, test_out["type_probs"], te_y, test_pred,
                                      mode=support_mode)
    concept_sim = concept_similarity(test_out["concept_to_sensor"], pos_mask)
    mask_summary = sensor_mask_metrics(test_out.get("dynamic_sensor_mask"), te_y, test_pred)
    concept_risk_mix = None
    if hasattr(model, "concept_risk_mix_logit"):
        concept_risk_mix = float(torch.sigmoid(model.concept_risk_mix_logit.detach()).cpu().item())

    summary = {
        "model": model_name,
        "variant": type_loss_cfg.variant,
        "backbone": summary_backbone,
        "implementation_backbone": backbone,
        "model_cfg": asdict(cfg),
        "fault_type_names": type_names,
        "train_cfg": asdict(train_cfg),
        "type_loss_cfg": asdict(type_loss_cfg),
        "support_loss_cfg": {
            "support_weight": support_w,
            "support_margin": float(support_margin),
            "support_negative_weight": float(support_negative_weight),
            "support_mode": support_mode,
        },
        "concept_pos_weight": concept_pos_weight_map,
        "weak_type_label_meta": asdict(label_meta),
        "type_names_used": type_names,
        "type_display_names": {k: v for k, v in TYPE_DISPLAY_NAMES.items()
                               if k in type_names},
        "n_params": n_params,
        "feature_count": len(sensors),
        "sensors": sensors,
        "threshold_selection": threshold_meta,
        "threshold": threshold,
        "best_val_f1": best_val_f1,
        "best_val_auc": best_val_auc,
        "window_metrics": {k: float(v) if isinstance(v, (int, float, np.floating))
                           else v for k, v in window_metrics.items()
                           if k not in {"y_pred", "y_score"}},
        "event_metrics": event_metrics,
        "type_metrics": type_summary,
        "support_metrics": support_summary,
        "dynamic_sensor_mask_metrics": mask_summary,
        "concept_risk_mix": concept_risk_mix,
        "concept_sensor_mean_offdiag_cosine": concept_sim,
        "sensor_importance": {
            "ranking": [{"sensor": s, "weight": round(w, 6)} for s, w in sensor_ranking],
            "global_weights": {s: round(w, 6) for s, w in zip(sensors, global_sensor_importance)},
        },
        "history": history,
    }

    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "model_cfg": asdict(cfg)},
               output_dir / "models" / "model.pt")

    meta_file = np.asarray([m["file_name"] for m in test_out["meta"]])
    meta_primary = np.asarray([m.get("weak_type_primary", "none") for m in test_out["meta"]])
    np.savez_compressed(
        output_dir / "results" / "sensor_weights.npz",
        weights=test_out["weights"], y_true=te_y, y_pred=test_pred, scores=te_scores,
        type_probs=test_out["type_probs"], type_targets=test_out["type_targets"],
        type_masks=test_out["type_masks"], file_name=meta_file, weak_type_primary=meta_primary,
    )
    maps_payload = {
        "concept_to_sensor": test_out["concept_to_sensor"],
        "sensor_to_concept": test_out["sensor_to_concept"],
    }
    if test_out.get("sensor_to_time") is not None:
        maps_payload["sensor_to_time"] = test_out["sensor_to_time"]
    if test_out.get("dynamic_sensor_mask") is not None:
        maps_payload["dynamic_sensor_mask"] = test_out["dynamic_sensor_mask"]
    np.savez_compressed(output_dir / "results" / "cross_attn_maps.npz", **maps_payload)

    print(
        f"\n[{log_prefix}:{type_loss_cfg.variant}] Test F1={window_metrics.get('f1', 0):.4f} "
        f"P={window_metrics.get('precision', 0):.4f} R={window_metrics.get('recall', 0):.4f} "
        f"Hit@Eval={event_metrics.get('event_hit_rate_evaluable', 0):.4f} "
        f"Hit@All={event_metrics.get('all_fault_module_hit_rate', 0):.4f} "
        f"concept_sim={concept_sim:.4f}"
    )
    print(f"[{log_prefix}] Sensor ranking:")
    for rank, (sensor, weight) in enumerate(sensor_ranking[:5], 1):
        print(f"  #{rank} {sensor}: {weight:.4f}")
    return summary


def variant_cfg(name: str, threshold_quantile: float, fallback_quantile: float) -> TypeLossCfg:
    cfg = TypeLossCfg(
        variant=name,
        threshold_quantile=threshold_quantile,
        fallback_quantile=fallback_quantile,
    )
    if name == "type":
        cfg.diversity_weight = 0.0
        cfg.prior_weight = 0.0
    elif name == "type_div":
        cfg.diversity_weight = 0.03
        cfg.prior_weight = 0.0
    elif name == "type_prior":
        cfg.diversity_weight = 0.0
        cfg.prior_weight = 0.05
    elif name == "full":
        cfg.diversity_weight = 0.03
        cfg.prior_weight = 0.05
    else:
        raise ValueError(f"Unknown variant: {name}")
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["type", "type_div", "type_prior", "full"], default="full")
    parser.add_argument("--backbone", choices=["itransformer", "transformer", "mftransformerv3a", "creformer", "fteformer"], default="itransformer")
    parser.add_argument("--time_pool_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold_quantile", type=float, default=0.90)
    parser.add_argument("--fallback_quantile", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--focal_loss", action="store_true", default=False)
    # FTEformer dynamic sensor mask tuning
    parser.add_argument("--sensor_mask_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--sensor_contrastive_weight", type=float, default=0.15)
    parser.add_argument("--sensor_mask_temperature", type=float, default=0.3)
    parser.add_argument("--hierarchy_weight", type=float, default=0.0)
    parser.add_argument("--support_weight", type=float, default=0.0)
    parser.add_argument("--support_margin", type=float, default=0.05)
    parser.add_argument("--support_negative_weight", type=float, default=0.05)
    parser.add_argument("--support_mode", choices=["max", "top2", "noisy_or"], default="max")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    task_cfg = r4_task_cfg()
    sensors = base.SENSORS
    print("=" * 80)
    print(f"Fault-type expert {args.backbone} - R4 ({args.variant})")
    print("=" * 80)
    print("[task] loading R4 frames ...")
    frames = load_task_frames(task_cfg=task_cfg, tps_lead_minutes=task_cfg.lead_minutes)
    print(f"[data] train={len(frames['train'])} val={len(frames['val'])} test={len(frames['test'])}")

    type_loss_cfg = variant_cfg(args.variant, args.threshold_quantile, args.fallback_quantile)
    active_type_names = list(TYPE_NAMES)
    print(f"[fault-types] {active_type_names}")
    frames, label_meta = annotate_fault_type_frames(
        frames,
        threshold_quantile=type_loss_cfg.threshold_quantile,
        fallback_quantile=type_loss_cfg.fallback_quantile,
        type_names=active_type_names,
    )

    train_cfg = TrainCfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-2,
        patience=args.patience,
        warmup_epochs=3,
        grad_clip=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=0,
        seed=args.seed,
        sensor_mask_mode=args.sensor_mask_mode,
        sensor_contrastive_weight=args.sensor_contrastive_weight,
        sensor_mask_temperature=args.sensor_mask_temperature,
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        EXP_ROOT / {
            "itransformer": f"dl_interp_L4_typeaware_R4_{args.variant}",
            "transformer": f"dl_mfpformer_R4_{args.variant}",
            "mftransformerv3a": f"dl_mftransformerv3a_R4_{args.variant}",
            "creformer": f"dl_fteformer_R4_{args.variant}",
            "fteformer": f"dl_fteformer_R4_{args.variant}",
        }[args.backbone]
    )
    if args.focal_loss and not args.output_dir:
        output_dir = output_dir.parent / (output_dir.name + "_focal")
    train_typeaware_l4(
        frames,
        output_dir,
        sensors,
        train_cfg,
        type_loss_cfg,
        label_meta,
        backbone=args.backbone,
        time_pool_size=args.time_pool_size,
        type_names=active_type_names,
        focal_loss=args.focal_loss,
        hierarchy_weight=args.hierarchy_weight,
        support_weight=args.support_weight,
        support_margin=args.support_margin,
        support_negative_weight=args.support_negative_weight,
        support_mode=args.support_mode,
    )


if __name__ == "__main__":
    main()
