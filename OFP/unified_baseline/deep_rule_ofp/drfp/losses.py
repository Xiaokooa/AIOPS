"""Losses for monotone multi-horizon DRFP outputs."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor, nn


FULL_VARIANTS = frozenset({"fixed_fusion", "gated_fusion"})


def _weight_tensor(
    value: Optional[Union[float, Sequence[float], Tensor]],
    width: int,
    default: float,
) -> Tensor:
    if value is None:
        tensor = torch.full((width,), float(default), dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.repeat(width)
    if tensor.ndim != 1 or tensor.numel() != width:
        raise ValueError("weight must be a scalar or have one value per horizon")
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError("weights must be finite and non-negative")
    return tensor


def multi_horizon_weighted_bce(
    probabilities: Tensor,
    targets: Tensor,
    valid_mask: Optional[Tensor] = None,
    pos_weight: Optional[Union[float, Sequence[float], Tensor]] = None,
    horizon_weight: Optional[Union[float, Sequence[float], Tensor]] = None,
    epsilon: float = 1.0e-7,
) -> Tensor:
    """Weighted BCE over cumulative probabilities at multiple horizons.

    ``pos_weight`` controls class imbalance separately at each horizon.
    ``horizon_weight`` controls the relative contribution of horizons.  A
    validity mask can exclude right-censored horizon labels.
    """

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, horizon]")
    if targets.shape != probabilities.shape:
        raise ValueError("targets must have the same shape as probabilities")
    width = probabilities.shape[-1]
    positive = _weight_tensor(pos_weight, width, 1.0).to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    horizon = _weight_tensor(horizon_weight, width, 1.0).to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    target = targets.to(device=probabilities.device, dtype=probabilities.dtype)
    if not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError("targets must be finite values in [0, 1]")
    if valid_mask is None:
        mask = torch.ones_like(probabilities)
    else:
        if valid_mask.shape != probabilities.shape:
            raise ValueError("valid_mask must have the same shape as probabilities")
        mask = valid_mask.to(device=probabilities.device, dtype=probabilities.dtype)
        mask = torch.nan_to_num(mask, nan=0.0).clamp(0.0, 1.0)

    probability = probabilities.clamp(epsilon, 1.0 - epsilon)
    elementwise = -(
        positive.unsqueeze(0) * target * torch.log(probability)
        + (1.0 - target) * torch.log1p(-probability)
    )
    weights = mask * horizon.unsqueeze(0)
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("valid_mask excludes every target")
    return (elementwise * weights).sum() / denominator


class DRFPLoss(nn.Module):
    """Variant-aware objective without unvalidated pseudo-rule terms.

    Single-branch variants optimize their active branch once.  Full fusion
    variants optimize the fused output as the primary prediction and retain
    smaller deep/rule auxiliary losses so neither branch can collapse.
    """

    def __init__(
        self,
        horizon_count: int = 4,
        pos_weight: Optional[Union[float, Sequence[float], Tensor]] = None,
        horizon_weight: Optional[Union[float, Sequence[float], Tensor]] = None,
        fused_weight: float = 1.0,
        deep_aux_weight: float = 0.25,
        rule_aux_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if horizon_count <= 0:
            raise ValueError("horizon_count must be positive")
        if fused_weight <= 0 or deep_aux_weight < 0 or rule_aux_weight < 0:
            raise ValueError("fused weight must be positive and auxiliary weights non-negative")
        self.horizon_count = int(horizon_count)
        self.fused_weight = float(fused_weight)
        self.deep_aux_weight = float(deep_aux_weight)
        self.rule_aux_weight = float(rule_aux_weight)
        self.register_buffer(
            "pos_weight",
            _weight_tensor(pos_weight, self.horizon_count, 1.0),
        )
        self.register_buffer(
            "horizon_weight",
            _weight_tensor(horizon_weight, self.horizon_count, 1.0),
        )

    def _branch_loss(
        self,
        probabilities: Optional[Tensor],
        targets: Tensor,
        valid_mask: Optional[Tensor],
        branch_name: str,
    ) -> Tensor:
        if probabilities is None:
            raise ValueError("%s probabilities are unavailable" % branch_name)
        if probabilities.shape[-1] != self.horizon_count:
            raise ValueError("unexpected number of horizons in %s" % branch_name)
        return multi_horizon_weighted_bce(
            probabilities,
            targets,
            valid_mask=valid_mask,
            pos_weight=self.pos_weight,
            horizon_weight=self.horizon_weight,
        )

    def components(
        self,
        outputs: Dict[str, object],
        targets: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        variant = str(outputs.get("variant", ""))
        result: Dict[str, Tensor] = {}
        if variant in FULL_VARIANTS:
            fused = self._branch_loss(
                outputs.get("fused_probabilities"),  # type: ignore[arg-type]
                targets,
                valid_mask,
                "fused",
            )
            deep = self._branch_loss(
                outputs.get("deep_probabilities"),  # type: ignore[arg-type]
                targets,
                valid_mask,
                "deep",
            )
            rule = self._branch_loss(
                outputs.get("rule_probabilities"),  # type: ignore[arg-type]
                targets,
                valid_mask,
                "rule",
            )
            result.update(
                {
                    "fused": fused,
                    "deep_aux": deep,
                    "rule_aux": rule,
                }
            )
            weight_sum = self.fused_weight + self.deep_aux_weight + self.rule_aux_weight
            total = (
                self.fused_weight * fused
                + self.deep_aux_weight * deep
                + self.rule_aux_weight * rule
            ) / weight_sum
        elif variant == "rule_only":
            rule = self._branch_loss(
                outputs.get("rule_probabilities"),  # type: ignore[arg-type]
                targets,
                valid_mask,
                "rule",
            )
            result["rule"] = rule
            total = rule
        elif variant in {"temporal_only", "deep_raw_stat"}:
            deep = self._branch_loss(
                outputs.get("deep_probabilities"),  # type: ignore[arg-type]
                targets,
                valid_mask,
                "deep",
            )
            result["deep"] = deep
            total = deep
        else:
            raise ValueError("outputs contain unknown variant %r" % variant)
        result["total"] = total
        return result

    def forward(
        self,
        outputs: Dict[str, object],
        targets: Tensor,
        valid_mask: Optional[Tensor] = None,
        return_components: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        values = self.components(outputs, targets, valid_mask)
        return values if return_components else values["total"]
