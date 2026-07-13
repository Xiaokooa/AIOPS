"""Module-balanced supervision and Future-Guided Learning losses."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F


Coverage = Dict[str, int]
LossOrCoverage = Union[Tensor, Tuple[Tensor, Coverage]]


@dataclass
class FutureGuidedLossOutput:
    """Structured loss result consumed by training and experiment logging."""

    total: Tensor
    cross_entropy: Tensor
    fgl_kl: Tensor
    coverage: Coverage


def _validate_logits(logits: Tensor, name: str) -> None:
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [batch, time, 2]")
    if not logits.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def _as_bool_mask(mask: Tensor, shape: torch.Size, name: str) -> Tensor:
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    return mask.to(dtype=torch.bool)


def _module_normalized_mean(
    values: Tensor,
    mask: Tensor,
    *,
    include_uncovered_modules: bool = False,
) -> Tuple[Tensor, int]:
    """Average rows within modules, then average modules with coverage."""

    counts = mask.sum(dim=1)
    covered_modules = counts > 0
    if not bool(covered_modules.any()):
        # Keep a differentiable connection to ``values`` for empty minibatches.
        return values.sum() * 0.0, 0
    per_module = (values * mask.to(dtype=values.dtype)).sum(dim=1)
    per_module = per_module / counts.clamp_min(1).to(dtype=values.dtype)
    # CE covers every training module, whereas some modules have no valid
    # future teacher row. For KL, retaining those modules as explicit zeros
    # gives every module a stable global weight independent of minibatch
    # composition or length bucketing.
    loss = (
        per_module.mean()
        if include_uncovered_modules
        else per_module[covered_modules].mean()
    )
    return loss, int(covered_modules.sum().detach().cpu().item())


def module_normalized_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    pos_weight: float | Tensor | None = None,
    return_coverage: bool = False,
) -> LossOrCoverage:
    """Two-class CE with equal influence for every covered module.

    Rows are averaged inside each module first, so a long module cannot dominate
    a short one. ``pos_weight`` follows the usual weighted-CE convention: it
    multiplies positive-row losses while the denominator remains the number of
    valid rows in that module.
    """

    _validate_logits(logits, "logits")
    expected_shape = logits.shape[:2]
    if target.shape != expected_shape:
        raise ValueError("target must have shape [batch, time]")
    valid = _as_bool_mask(mask, expected_shape, "mask").to(device=logits.device)
    target = target.to(device=logits.device, dtype=torch.long)
    if bool(valid.any()):
        covered_targets = target[valid]
        if bool(((covered_targets < 0) | (covered_targets > 1)).any()):
            raise ValueError("valid target values must be 0 or 1")

    safe_target = torch.where(valid, target, torch.zeros_like(target))
    row_loss = F.cross_entropy(
        logits.reshape(-1, 2), safe_target.reshape(-1), reduction="none"
    ).reshape(expected_shape)
    if pos_weight is not None:
        weight = torch.as_tensor(
            pos_weight, dtype=logits.dtype, device=logits.device
        )
        if weight.numel() != 1 or not bool(torch.isfinite(weight)) or float(weight) <= 0:
            raise ValueError("pos_weight must be a finite positive scalar")
        row_loss = row_loss * torch.where(
            safe_target == 1, weight, torch.ones_like(row_loss)
        )

    loss, module_count = _module_normalized_mean(row_loss, valid)
    coverage: Coverage = {
        "ce_modules": module_count,
        "ce_rows": int(valid.sum().detach().cpu().item()),
        "ce_positive_rows": int(((safe_target == 1) & valid).sum().detach().cpu().item()),
    }
    if return_coverage:
        return loss, coverage
    return loss


def fgl_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    teacher_index: Tensor,
    fgl_mask: Tensor,
    temperature: float = 4.0,
    return_coverage: bool = False,
) -> LossOrCoverage:
    """Module-normalized KL from a future teacher row to a student row.

    ``teacher_index[b, t]`` points to the future teacher output aligned with
    student row ``(b, t)``. Only entries selected by ``fgl_mask`` participate.
    Teacher probabilities are detached here as a second safety boundary in
    addition to freezing the teacher in the training loop.
    """

    _validate_logits(student_logits, "student_logits")
    _validate_logits(teacher_logits, "teacher_logits")
    if student_logits.shape[0] != teacher_logits.shape[0]:
        raise ValueError("student and teacher batch dimensions must match")
    student_shape = student_logits.shape[:2]
    if teacher_index.shape != student_shape:
        raise ValueError("teacher_index must have shape [batch, student_time]")
    valid = _as_bool_mask(fgl_mask, student_shape, "fgl_mask").to(
        device=student_logits.device
    )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    indices = teacher_index.to(device=student_logits.device, dtype=torch.long)
    teacher_time = int(teacher_logits.shape[1])
    invalid_selected = valid & ((indices < 0) | (indices >= teacher_time))
    if bool(invalid_selected.any()):
        raise ValueError("fgl_mask selects an out-of-range teacher_index")
    safe_indices = indices.clamp(min=0, max=max(teacher_time - 1, 0))
    gather_index = safe_indices.unsqueeze(-1).expand(-1, -1, 2)
    aligned_teacher = torch.gather(
        teacher_logits.detach().to(device=student_logits.device),
        dim=1,
        index=gather_index,
    )

    scale = float(temperature)
    student_log_probability = F.log_softmax(student_logits / scale, dim=-1)
    teacher_log_probability = F.log_softmax(aligned_teacher / scale, dim=-1)
    teacher_probability = teacher_log_probability.exp()
    # Writing KL explicitly makes identical logits yield an exact zero instead
    # of the small negative round-off occasionally produced by ``F.kl_div``.
    row_kl = (
        teacher_probability
        * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1) * (scale**2)
    loss, module_count = _module_normalized_mean(
        row_kl, valid, include_uncovered_modules=True
    )
    coverage: Coverage = {
        "fgl_modules": module_count,
        "fgl_pairs": int(valid.sum().detach().cpu().item()),
    }
    if return_coverage:
        return loss, coverage
    return loss


def combined_future_guided_loss(
    student_logits: Tensor,
    target: Tensor,
    target_mask: Tensor,
    teacher_logits: Tensor,
    teacher_index: Tensor,
    fgl_mask: Tensor,
    alpha: float = 0.7,
    temperature: float = 4.0,
    pos_weight: float | Tensor | None = None,
    fgl_coverage_scale: float = 1.0,
) -> FutureGuidedLossOutput:
    """Return ``alpha * CE + (1-alpha) * FGL-KL`` and audit counts."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    scale = float(fgl_coverage_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("fgl_coverage_scale must be finite and positive")
    ce_result = module_normalized_cross_entropy(
        logits=student_logits,
        target=target,
        mask=target_mask,
        pos_weight=pos_weight,
        return_coverage=True,
    )
    kl_result = fgl_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_index=teacher_index,
        fgl_mask=fgl_mask,
        temperature=temperature,
        return_coverage=True,
    )
    cross_entropy, ce_coverage = ce_result
    raw_fgl_kl, fgl_coverage = kl_result
    fgl_kl = raw_fgl_kl * scale
    total = float(alpha) * cross_entropy + (1.0 - float(alpha)) * fgl_kl
    return FutureGuidedLossOutput(
        total=total,
        cross_entropy=cross_entropy,
        fgl_kl=fgl_kl,
        coverage={**ce_coverage, **fgl_coverage},
    )


def combined_fgl_loss(*args, **kwargs) -> FutureGuidedLossOutput:
    """Concise alias retained for training-loop readability."""

    return combined_future_guided_loss(*args, **kwargs)


__all__ = [
    "FutureGuidedLossOutput",
    "combined_fgl_loss",
    "combined_future_guided_loss",
    "fgl_kl_loss",
    "module_normalized_cross_entropy",
]
