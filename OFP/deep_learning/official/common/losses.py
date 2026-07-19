from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FTEformerLossCfg:
    type_loss_weight: float = 0.2
    diversity_weight: float = 0.03
    prior_weight: float = 0.05
    hierarchy_weight: float = 0.01
    support_weight: float = 0.03
    support_margin: float = 0.05
    support_negative_weight: float = 0.05
    negative_type_weight: float = 0.05
    sensor_contrastive_weight: float = 0.03


def primary_logit(output: torch.Tensor | tuple) -> torch.Tensor:
    logits = output[0] if isinstance(output, tuple) else output
    if logits.ndim > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    return logits


def primary_horizon_tensor(values: torch.Tensor, primary_index: int = -1) -> torch.Tensor:
    if values.ndim > 1:
        return values[:, int(primary_index)]
    return values


def split_alarm_aux_logits(logits: torch.Tensor, aux_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split model outputs into an event alarm logit and auxiliary horizon logits.

    First-warning models emit `[alarm, horizon...]`.  In the current
    OFP-compatible 24h-lookback setting this is `[alarm, h1]`.  Older
    checkpoints emitted only the horizon logits; for those, the primary horizon
    remains the alarm proxy for backward compatibility.
    """
    aux_dim = int(aux_dim)
    if logits.ndim > 1 and logits.shape[-1] == aux_dim + 1:
        return logits[..., 0], logits[..., 1:]
    if logits.ndim > 1 and logits.shape[-1] == aux_dim:
        return primary_horizon_tensor(logits), logits
    if logits.ndim > 1 and logits.shape[-1] == 1:
        squeezed = logits.squeeze(-1)
        return squeezed, squeezed.unsqueeze(-1)
    return logits, logits.unsqueeze(-1) if logits.ndim == 1 else logits


def first_warning_event_loss(
    alarm_logits: torch.Tensor,
    hit_mask: torch.Tensor,
    module_targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable first-warning loss.

    `alarm_logits[b, t]` is interpreted as the probability of firing at sampled
    time `t`.  For faulty modules, the loss maximizes the probability that the
    first alarm falls inside `hit_mask` (pre-first-anomaly positions).  For
    normal modules, it maximizes the probability of no alarm over valid sampled
    positions.
    """
    if alarm_logits.ndim == 1:
        alarm_logits = alarm_logits.unsqueeze(0)
    if hit_mask.ndim == 1:
        hit_mask = hit_mask.unsqueeze(0)
    if valid_mask is None:
        valid_mask = torch.ones_like(hit_mask, dtype=torch.bool)
    elif valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)

    logits = alarm_logits.float()
    hit_mask = hit_mask.to(device=logits.device, dtype=torch.bool)
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    targets = module_targets.to(device=logits.device, dtype=torch.float32).view(-1)

    log_yes = F.logsigmoid(logits).masked_fill(~valid_mask, -1.0e9)
    log_no = F.logsigmoid(-logits).masked_fill(~valid_mask, 0.0)
    prefix_no = torch.cumsum(log_no, dim=1) - log_no
    log_first = log_yes + prefix_no

    losses: list[torch.Tensor] = []
    pos_rows = (targets > 0.5) & (hit_mask & valid_mask).any(dim=1)
    if pos_rows.any():
        pos_log_first = log_first[pos_rows].masked_fill(~(hit_mask[pos_rows] & valid_mask[pos_rows]), -1.0e9)
        log_hit = torch.logsumexp(pos_log_first, dim=1)
        losses.append(-log_hit)

    normal_rows = targets <= 0.5
    if normal_rows.any():
        log_no_alarm = log_no[normal_rows].sum(dim=1)
        losses.append(-log_no_alarm)

    if not losses:
        return logits.new_tensor(0.0)
    return torch.cat([loss.reshape(-1) for loss in losses]).mean()


def attention_diversity_loss(attn_c2s: torch.Tensor, type_masks: torch.Tensor) -> torch.Tensor:
    sample_mask = type_masks > 0
    attn = attn_c2s[sample_mask] if sample_mask.any() else attn_c2s
    if attn.size(0) == 0 or attn.size(1) <= 1:
        return attn_c2s.new_tensor(0.0)
    normed = F.normalize(attn, p=2, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))
    eye = torch.eye(sim.size(1), device=sim.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_select(~eye).pow(2).mean()


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


def concept_multitask_loss(
    type_logits: torch.Tensor,
    type_targets: torch.Tensor,
    type_masks: torch.Tensor,
    ys: torch.Tensor,
    cfg: FTEformerLossCfg,
) -> torch.Tensor:
    ys_primary = primary_horizon_tensor(ys)
    raw = F.binary_cross_entropy_with_logits(type_logits, type_targets, reduction="none")
    supervised_fault_weight = type_masks.unsqueeze(-1)
    healthy_negative_weight = (ys_primary <= 0).float().unsqueeze(-1) * float(cfg.negative_type_weight)
    weights = supervised_fault_weight + healthy_negative_weight
    return (raw * weights).sum() / weights.sum().clamp(min=1.0)


def hierarchy_loss(event_logits: torch.Tensor, type_logits: torch.Tensor) -> torch.Tensor:
    event_prob = torch.sigmoid(primary_horizon_tensor(event_logits))
    max_type_prob = torch.sigmoid(type_logits).max(dim=-1).values
    return F.relu(max_type_prob - event_prob).pow(2).mean()


def support_loss(
    event_logits: torch.Tensor,
    type_logits: torch.Tensor,
    ys: torch.Tensor,
    margin: float,
    negative_weight: float,
) -> torch.Tensor:
    event_primary = primary_horizon_tensor(event_logits)
    ys_primary = primary_horizon_tensor(ys)
    event_prob = torch.sigmoid(event_primary)
    type_prob = torch.sigmoid(type_logits)
    support_prob = type_prob.max(dim=-1).values
    pos_mask = (ys_primary > 0).float()
    neg_mask = 1.0 - pos_mask
    pos = (F.relu(event_prob - support_prob - float(margin)).pow(2) * pos_mask).sum()
    pos = pos / pos_mask.sum().clamp(min=1.0)
    neg = (type_prob.pow(2).mean(dim=-1) * neg_mask).sum()
    neg = neg / neg_mask.sum().clamp(min=1.0)
    return pos + float(negative_weight) * neg


def make_sensor_prior(type_names: list[str], sensors: list[str]) -> torch.Tensor:
    groups = {
        "thermal_anomaly": ["temperature"],
        "current_bias_anomaly": ["current"],
        "tx_power_anomaly": ["currentTXPower"] + [f"currentMultiTXPower{i}" for i in range(1, 5)],
        "rx_power_anomaly": ["currentRXPower"] + [f"currentMultiRXPower{i}" for i in range(1, 5)],
        "lane_imbalance": [f"currentMultiTXPower{i}" for i in range(1, 5)]
        + [f"currentMultiRXPower{i}" for i in range(1, 5)],
    }
    prior = torch.zeros((len(type_names), len(sensors)), dtype=torch.float32)
    for row, name in enumerate(type_names):
        group = set(groups.get(name, []))
        for col, sensor in enumerate(sensors):
            if sensor in group:
                prior[row, col] = 1.0
        if prior[row].sum() <= 0:
            prior[row] = 1.0
    return prior


def fteformer_total_loss(
    outputs: tuple,
    ys: torch.Tensor,
    fault_loss: torch.Tensor,
    type_targets: torch.Tensor,
    type_masks: torch.Tensor,
    sensor_prior: torch.Tensor,
    cfg: FTEformerLossCfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = primary_logit(outputs)
    logits_primary, _aux_logits = split_alarm_aux_logits(logits, aux_dim=ys.shape[-1] if ys.ndim > 1 else 1)
    ys_primary = primary_horizon_tensor(ys)
    attn_c2s = outputs[2]
    type_logits = outputs[4]

    type_loss = concept_multitask_loss(type_logits, type_targets, type_masks, ys, cfg)
    div_loss = attention_diversity_loss(attn_c2s, type_masks)
    prior_loss = sensor_prior_loss(attn_c2s, type_targets, type_masks, sensor_prior)
    hier = hierarchy_loss(logits_primary, type_logits)
    support = support_loss(
        logits_primary,
        type_logits,
        ys_primary,
        margin=cfg.support_margin,
        negative_weight=cfg.support_negative_weight,
    )
    contrast = outputs[6] if len(outputs) > 6 and torch.is_tensor(outputs[6]) else logits.new_tensor(0.0)
    total = (
        fault_loss
        + cfg.type_loss_weight * type_loss
        + cfg.diversity_weight * div_loss
        + cfg.prior_weight * prior_loss
        + cfg.hierarchy_weight * hier
        + cfg.support_weight * support
        + cfg.sensor_contrastive_weight * contrast
    )
    parts = {
        "fault_loss": float(fault_loss.detach().cpu()),
        "type_loss": float(type_loss.detach().cpu()),
        "diversity_loss": float(div_loss.detach().cpu()),
        "prior_loss": float(prior_loss.detach().cpu()),
        "hierarchy_loss": float(hier.detach().cpu()),
        "support_loss": float(support.detach().cpu()),
        "contrast_loss": float(contrast.detach().cpu()),
    }
    return total, parts
