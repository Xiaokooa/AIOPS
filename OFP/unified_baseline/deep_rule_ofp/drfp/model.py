"""Neural building blocks for the Deep-Rule Failure Prediction network.

The model exposes interval-hazard logits and cumulative failure probabilities
at the configured horizons.  The cumulative transform is structural, so the
reported probabilities are monotone without a post-hoc correction.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_HORIZONS: Tuple[int, ...] = (16, 24, 72, 120)
VALID_VARIANTS = frozenset(
    {
        "temporal_only",
        "deep_raw_stat",
        "rule_only",
        "fixed_fusion",
        "gated_fusion",
    }
)


def cumulative_from_hazards(hazard_logits: Tensor) -> Tensor:
    """Convert interval-hazard logits to cumulative event probabilities.

    For interval hazards ``h_k``, the cumulative probability at horizon ``k``
    is ``1 - product_{j<=k}(1-h_j)``.  Consequently each output row is
    non-decreasing along its last dimension.
    """

    if hazard_logits.ndim < 1 or hazard_logits.shape[-1] < 1:
        raise ValueError("hazard_logits must have a non-empty final dimension")
    hazards = torch.sigmoid(hazard_logits)
    survival = torch.cumprod(1.0 - hazards, dim=-1)
    return 1.0 - survival


def _validate_2d(name: str, value: Tensor, width: int) -> None:
    if value.ndim != 2 or value.shape[-1] != width:
        raise ValueError(
            "%s must have shape [batch, %d], got %s"
            % (name, width, tuple(value.shape))
        )


class TemporalEncoder(nn.Module):
    """Causal-window encoder using patch projection and a Transformer.

    The input channel set is deliberately explicit: masked raw values, the
    raw observation mask, and ``log1p`` time since the last observation.  The
    encoder never interpolates or reads values beyond the supplied history.
    """

    def __init__(
        self,
        raw_dim: int = 12,
        sequence_length: int = 168,
        patch_length: int = 8,
        patch_stride: int = 4,
        d_model: int = 64,
        attention_heads: int = 4,
        transformer_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if raw_dim <= 0 or sequence_length <= 0 or patch_length <= 0:
            raise ValueError("raw_dim, sequence_length, and patch_length must be positive")
        if patch_length > sequence_length:
            raise ValueError("patch_length cannot exceed sequence_length")
        if patch_stride <= 0 or patch_stride > patch_length:
            raise ValueError("patch_stride must be in [1, patch_length]")
        if d_model <= 0 or d_model % attention_heads != 0:
            raise ValueError("d_model must be positive and divisible by attention_heads")

        self.raw_dim = int(raw_dim)
        self.sequence_length = int(sequence_length)
        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.d_model = int(d_model)
        input_channels = 2 * self.raw_dim + 1
        self.patch_projection = nn.Conv1d(
            input_channels,
            self.d_model,
            kernel_size=self.patch_length,
            stride=self.patch_stride,
        )
        max_patches = 1 + (self.sequence_length - self.patch_length) // self.patch_stride
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_patches + 1, self.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _positions(self, patch_count: int) -> Tensor:
        if patch_count + 1 == self.position_embedding.shape[1]:
            return self.position_embedding
        cls_position = self.position_embedding[:, :1]
        patch_positions = self.position_embedding[:, 1:].transpose(1, 2)
        patch_positions = F.interpolate(
            patch_positions,
            size=patch_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return torch.cat([cls_position, patch_positions], dim=1)

    def forward(self, raw: Tensor, raw_mask: Tensor, delta_hours: Tensor) -> Tensor:
        if raw.ndim != 3 or raw.shape[-1] != self.raw_dim:
            raise ValueError(
                "raw must have shape [batch, time, %d], got %s"
                % (self.raw_dim, tuple(raw.shape))
            )
        if raw_mask.shape != raw.shape:
            raise ValueError("raw_mask must have the same shape as raw")
        if (
            delta_hours.ndim != 3
            or delta_hours.shape[:2] != raw.shape[:2]
            or delta_hours.shape[-1] != 1
        ):
            raise ValueError("delta_hours must have shape [batch, time, 1]")

        mask = torch.nan_to_num(raw_mask.to(dtype=raw.dtype), nan=0.0).clamp(0.0, 1.0)
        values = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) * mask
        delta = torch.nan_to_num(
            delta_hours.to(dtype=raw.dtype),
            nan=0.0,
            posinf=1.0e6,
            neginf=0.0,
        ).clamp(min=0.0, max=1.0e6)
        encoded_delta = torch.log1p(delta)
        x = torch.cat([values, mask, encoded_delta], dim=-1).transpose(1, 2)

        if x.shape[-1] < self.patch_length:
            raise ValueError("temporal history is shorter than patch_length")
        patches = self.patch_projection(x).transpose(1, 2)
        cls = self.cls_token.expand(raw.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = self.input_dropout(tokens + self._positions(patches.shape[1]))
        encoded = self.transformer(tokens)
        return self.output_norm(encoded[:, 0])


class StatEncoder(nn.Module):
    """Encode statistical values jointly with their availability mask."""

    def __init__(
        self,
        stats_dim: int = 76,
        hidden_dim: int = 96,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stats_dim = int(stats_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * self.stats_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, stats: Tensor, stats_mask: Tensor) -> Tensor:
        _validate_2d("stats", stats, self.stats_dim)
        if stats_mask.shape != stats.shape:
            raise ValueError("stats_mask must have the same shape as stats")
        mask = torch.nan_to_num(stats_mask.to(dtype=stats.dtype), nan=0.0).clamp(0.0, 1.0)
        values = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0) * mask
        return self.network(torch.cat([values, mask], dim=-1))


class RuleEncoder(nn.Module):
    """Independent encoder for predictive-rule signals and their masks."""

    def __init__(
        self,
        rule_dim: int = 156,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rule_dim = int(rule_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * self.rule_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, rule: Tensor, rule_mask: Optional[Tensor] = None) -> Tensor:
        _validate_2d("rule", rule, self.rule_dim)
        if rule_mask is None:
            mask = torch.ones_like(rule)
        else:
            if rule_mask.shape != rule.shape:
                raise ValueError("rule_mask must have the same shape as rule")
            mask = torch.nan_to_num(rule_mask.to(dtype=rule.dtype), nan=0.0).clamp(0.0, 1.0)
        values = torch.nan_to_num(rule, nan=0.0, posinf=0.0, neginf=0.0) * mask
        return self.network(torch.cat([values, mask], dim=-1))


class DRFPNet(nn.Module):
    """Deep-rule network with ablatable branches and hazard-level fusion."""

    def __init__(
        self,
        variant: str = "gated_fusion",
        raw_dim: int = 12,
        sequence_length: int = 168,
        stats_dim: int = 76,
        rule_dim: int = 156,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        patch_length: int = 8,
        patch_stride: int = 4,
        d_model: int = 64,
        latent_dim: int = 64,
        attention_heads: int = 4,
        transformer_layers: int = 2,
        statistical_hidden: int = 128,
        rule_hidden: int = 128,
        ff_dim: int = 128,
        dropout: float = 0.1,
        fixed_gate: float = 0.5,
    ) -> None:
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(
                "unknown variant %r; expected one of %s"
                % (variant, sorted(VALID_VARIANTS))
            )
        horizon_tuple = tuple(int(value) for value in horizons)
        if not horizon_tuple or any(value <= 0 for value in horizon_tuple):
            raise ValueError("horizons must contain positive values")
        if any(left >= right for left, right in zip(horizon_tuple, horizon_tuple[1:])):
            raise ValueError("horizons must be strictly increasing")
        if not 0.0 <= fixed_gate <= 1.0:
            raise ValueError("fixed_gate must be in [0, 1]")

        self.variant = variant
        self.horizons = horizon_tuple
        self.raw_dim = int(raw_dim)
        self.sequence_length = int(sequence_length)
        self.stats_dim = int(stats_dim)
        self.rule_dim = int(rule_dim)
        self.latent_dim = int(latent_dim)
        self.fixed_gate = float(fixed_gate)
        self.use_deep = variant != "rule_only"
        self.use_stats = variant in {
            "deep_raw_stat",
            "fixed_fusion",
            "gated_fusion",
        }
        self.use_rule = variant in {"rule_only", "fixed_fusion", "gated_fusion"}

        if self.use_deep:
            self.temporal_encoder: Optional[TemporalEncoder] = TemporalEncoder(
                raw_dim=self.raw_dim,
                sequence_length=self.sequence_length,
                patch_length=patch_length,
                patch_stride=patch_stride,
                d_model=d_model,
                attention_heads=attention_heads,
                transformer_layers=transformer_layers,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.temporal_projection: Optional[nn.Module] = (
                nn.Identity()
                if d_model == self.latent_dim
                else nn.Linear(d_model, self.latent_dim)
            )
            if self.use_stats:
                self.stat_encoder: Optional[StatEncoder] = StatEncoder(
                    stats_dim=self.stats_dim,
                    hidden_dim=statistical_hidden,
                    output_dim=self.latent_dim,
                    dropout=dropout,
                )
                self.deep_fusion: Optional[nn.Module] = nn.Sequential(
                    nn.Linear(2 * self.latent_dim, self.latent_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.latent_dim),
                    nn.Dropout(dropout),
                )
            else:
                self.stat_encoder = None
                self.deep_fusion = None
            self.deep_hazard_head: Optional[nn.Linear] = nn.Linear(
                self.latent_dim, len(self.horizons)
            )
        else:
            self.temporal_encoder = None
            self.temporal_projection = None
            self.stat_encoder = None
            self.deep_fusion = None
            self.deep_hazard_head = None

        if self.use_rule:
            self.rule_encoder: Optional[RuleEncoder] = RuleEncoder(
                rule_dim=self.rule_dim,
                hidden_dim=rule_hidden,
                output_dim=self.latent_dim,
                dropout=dropout,
            )
            self.rule_hazard_head: Optional[nn.Linear] = nn.Linear(
                self.latent_dim, len(self.horizons)
            )
        else:
            self.rule_encoder = None
            self.rule_hazard_head = None

        if variant == "gated_fusion":
            # Four rule-quality signals and four corresponding availability
            # indicators are supplied explicitly to the reliability gate.
            self.gate_network: Optional[nn.Module] = nn.Sequential(
                nn.Linear(2 * self.latent_dim + 8, self.latent_dim),
                nn.GELU(),
                nn.LayerNorm(self.latent_dim),
                nn.Dropout(dropout),
                nn.Linear(self.latent_dim, len(self.horizons)),
            )
        else:
            self.gate_network = None

    def _encode_deep(
        self,
        raw: Tensor,
        raw_mask: Tensor,
        delta_hours: Tensor,
        stats: Optional[Tensor],
        stats_mask: Optional[Tensor],
    ) -> Tensor:
        if self.temporal_encoder is None or self.temporal_projection is None:
            raise RuntimeError("deep branch is inactive")
        temporal = self.temporal_projection(
            self.temporal_encoder(raw, raw_mask, delta_hours)
        )
        if not self.use_stats:
            return temporal
        if stats is None or stats_mask is None:
            raise ValueError("stats and stats_mask are required for this variant")
        if self.stat_encoder is None or self.deep_fusion is None:
            raise RuntimeError("statistical branch was not initialized")
        statistical = self.stat_encoder(stats, stats_mask)
        return self.deep_fusion(torch.cat([temporal, statistical], dim=-1))

    def _rule_quality(
        self, rule: Tensor, rule_mask: Optional[Tensor]
    ) -> Tensor:
        if rule_mask is None:
            mask = torch.ones_like(rule)
        else:
            if rule_mask.shape != rule.shape:
                raise ValueError("rule_mask must have the same shape as rule")
            mask = torch.nan_to_num(rule_mask.to(dtype=rule.dtype), nan=0.0).clamp(0.0, 1.0)
        quality = torch.nan_to_num(
            rule[:, -4:], nan=0.0, posinf=0.0, neginf=0.0
        ) * mask[:, -4:]
        return torch.cat([quality, mask[:, -4:]], dim=-1)

    def forward(
        self,
        raw: Optional[Tensor] = None,
        raw_mask: Optional[Tensor] = None,
        delta_hours: Optional[Tensor] = None,
        stats: Optional[Tensor] = None,
        stats_mask: Optional[Tensor] = None,
        rule: Optional[Tensor] = None,
        rule_mask: Optional[Tensor] = None,
    ) -> Dict[str, Union[str, Tensor, None]]:
        deep_latent: Optional[Tensor] = None
        rule_latent: Optional[Tensor] = None
        deep_logits: Optional[Tensor] = None
        rule_logits: Optional[Tensor] = None

        if self.use_deep:
            if raw is None or raw_mask is None or delta_hours is None:
                raise ValueError("raw, raw_mask, and delta_hours are required")
            deep_latent = self._encode_deep(
                raw, raw_mask, delta_hours, stats, stats_mask
            )
            if self.deep_hazard_head is None:
                raise RuntimeError("deep hazard head was not initialized")
            deep_logits = self.deep_hazard_head(deep_latent)

        if self.use_rule:
            if rule is None:
                raise ValueError("rule is required for this variant")
            if self.rule_encoder is None or self.rule_hazard_head is None:
                raise RuntimeError("rule branch was not initialized")
            rule_latent = self.rule_encoder(rule, rule_mask)
            rule_logits = self.rule_hazard_head(rule_latent)

        if self.variant in {"temporal_only", "deep_raw_stat"}:
            if deep_logits is None:
                raise RuntimeError("deep output is unavailable")
            fused_logits = deep_logits
            gate = torch.zeros(
                (deep_logits.shape[0], len(self.horizons)),
                dtype=deep_logits.dtype,
                device=deep_logits.device,
            )
        elif self.variant == "rule_only":
            if rule_logits is None:
                raise RuntimeError("rule output is unavailable")
            fused_logits = rule_logits
            gate = torch.ones(
                (rule_logits.shape[0], len(self.horizons)),
                dtype=rule_logits.dtype,
                device=rule_logits.device,
            )
        else:
            if deep_logits is None or rule_logits is None:
                raise RuntimeError("both branches are required for fusion")
            if self.variant == "fixed_fusion":
                gate = torch.full(
                    (deep_logits.shape[0], len(self.horizons)),
                    self.fixed_gate,
                    dtype=deep_logits.dtype,
                    device=deep_logits.device,
                )
            else:
                if (
                    self.gate_network is None
                    or deep_latent is None
                    or rule_latent is None
                    or rule is None
                ):
                    raise RuntimeError("gated fusion inputs are unavailable")
                gate_features = torch.cat(
                    [
                        deep_latent,
                        rule_latent,
                        self._rule_quality(rule, rule_mask),
                    ],
                    dim=-1,
                )
                gate = torch.sigmoid(self.gate_network(gate_features))
            fused_logits = (1.0 - gate) * deep_logits + gate * rule_logits

        deep_hazards = torch.sigmoid(deep_logits) if deep_logits is not None else None
        rule_hazards = torch.sigmoid(rule_logits) if rule_logits is not None else None
        fused_hazards = torch.sigmoid(fused_logits)
        deep_probabilities = (
            cumulative_from_hazards(deep_logits) if deep_logits is not None else None
        )
        rule_probabilities = (
            cumulative_from_hazards(rule_logits) if rule_logits is not None else None
        )
        fused_probabilities = cumulative_from_hazards(fused_logits)
        return {
            "variant": self.variant,
            "deep_latent": deep_latent,
            "rule_latent": rule_latent,
            "deep_hazard_logits": deep_logits,
            "rule_hazard_logits": rule_logits,
            "fused_hazard_logits": fused_logits,
            "deep_hazards": deep_hazards,
            "rule_hazards": rule_hazards,
            "fused_hazards": fused_hazards,
            "deep_probabilities": deep_probabilities,
            "rule_probabilities": rule_probabilities,
            "fused_probabilities": fused_probabilities,
            # Stable aliases used by generic training/evaluation code.
            "hazard_logits": fused_logits,
            "probabilities": fused_probabilities,
            "gate": gate,
        }


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of scalar parameters updated by an optimizer."""

    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def parameter_statistics(module: nn.Module) -> Dict[str, int]:
    """Return total and trainable scalar parameter counts."""

    total = int(sum(parameter.numel() for parameter in module.parameters()))
    trainable = count_trainable_parameters(module)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
