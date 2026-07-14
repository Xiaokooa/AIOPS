"""Causal native-cadence sequence model used by FGL-OFP.

The model deliberately keeps the temporal contract small and auditable:

* every output row is computed from that row and earlier rows only;
* the temporal trunk consumes the 12 raw measurements, their observation
  masks, and one delta-time channel;
* the optional rule adapter applies a bounded pointwise residual to temporal
  logits using only causal signed rule margins;
* padding is explicitly zeroed after every residual block; and
* normalization is per time point (``LayerNorm``), never across time or across
  modules in a batch.

The teacher and student use the same class, but are instantiated independently.
Future information is therefore introduced only by the FGL alignment performed
by the loss during training, not by the student architecture.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rules import RULE_FEATURE_COUNT


RAW_FEATURE_COUNT = 12
OUTPUT_CLASS_COUNT = 2


class DepthwiseSeparableCausalConv1d(nn.Module):
    """Depthwise-separable convolution with explicit left-only padding."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        self.depthwise = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            groups=self.channels,
        )
        self.pointwise = nn.Conv1d(self.channels, self.channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("causal convolution expects [batch, channels, time]")
        if inputs.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, received {inputs.shape[1]}"
            )
        padded = F.pad(inputs, (self.left_padding, 0))
        return self.pointwise(self.depthwise(padded))


class CausalResidualBlock(nn.Module):
    """One pre-normalized causal residual block.

    ``LayerNorm`` is applied to the channel vector independently at every
    ``(module, time)`` location. This avoids batch-statistic leakage and makes
    train and inference behavior identical for a given prefix.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.convolution = DepthwiseSeparableCausalConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, padding_mask: Tensor) -> Tensor:
        valid = padding_mask.unsqueeze(-1).to(dtype=inputs.dtype)
        normalized = self.norm(inputs * valid)
        convolved = self.convolution(normalized.transpose(1, 2)).transpose(1, 2)
        update = self.dropout(self.activation(convolved))
        return (inputs + update) * valid


class CausalSeq2SeqTCN(nn.Module):
    """Causal TCN producing two-class logits at every native-cadence row.

    Parameters mirror :class:`fgofp.config.ModelConfig`. The default input has
    25 channels: 12 raw features, 12 raw-observation indicators, and one elapsed
    cadence-step value.
    """

    def __init__(
        self,
        raw_feature_count: int = RAW_FEATURE_COUNT,
        hidden_channels: int = 48,
        dilation_blocks: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if raw_feature_count != RAW_FEATURE_COUNT:
            raise ValueError(
                f"OFP v2 fixes raw_feature_count={RAW_FEATURE_COUNT}, "
                f"received {raw_feature_count}"
            )
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if dilation_blocks <= 0:
            raise ValueError("dilation_blocks must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.raw_feature_count = int(raw_feature_count)
        self.hidden_channels = int(hidden_channels)
        self.dilation_blocks = int(dilation_blocks)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(2**index for index in range(self.dilation_blocks))

        input_channels = 2 * self.raw_feature_count + 1
        self.input_projection = nn.Linear(input_channels, self.hidden_channels)
        self.blocks = nn.ModuleList(
            [
                CausalResidualBlock(
                    channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_channels)
        self.classifier = nn.Linear(self.hidden_channels, OUTPUT_CLASS_COUNT)
        self.requires_rule_inputs = False
        self.architecture = "causal_depthwise_tcn"

    @classmethod
    def from_config(
        cls,
        model_config: Any,
        raw_feature_count: int = RAW_FEATURE_COUNT,
    ) -> "CausalSeq2SeqTCN":
        """Build from a ``ModelConfig``-like object without importing config."""

        return cls(
            raw_feature_count=raw_feature_count,
            hidden_channels=int(model_config.hidden_channels),
            dilation_blocks=int(model_config.dilation_blocks),
            kernel_size=int(model_config.kernel_size),
            dropout=float(model_config.dropout),
        )

    @property
    def receptive_field_steps(self) -> int:
        """Number of input rows that can affect one output row."""

        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def _validate_and_combine_inputs(
        self,
        raw: Tensor,
        raw_mask: Tensor,
        delta_steps: Tensor,
        padding_mask: Tensor | None,
    ) -> Tuple[Tensor, Tensor]:
        if raw.ndim != 3:
            raise ValueError("raw must have shape [batch, time, 12]")
        if raw.shape[-1] != self.raw_feature_count:
            raise ValueError(
                f"raw must contain {self.raw_feature_count} features, "
                f"received {raw.shape[-1]}"
            )
        if raw_mask.shape != raw.shape:
            raise ValueError("raw_mask must have the same shape as raw")

        if delta_steps.ndim == 2:
            delta_steps = delta_steps.unsqueeze(-1)
        if delta_steps.ndim != 3 or delta_steps.shape[-1] != 1:
            raise ValueError("delta_steps must have shape [batch, time] or [batch, time, 1]")
        if delta_steps.shape[:2] != raw.shape[:2]:
            raise ValueError("delta_steps batch/time dimensions must match raw")

        if padding_mask is None:
            padding_mask = torch.ones(
                raw.shape[:2], dtype=torch.bool, device=raw.device
            )
        if padding_mask.shape != raw.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, time]")
        padding_mask = padding_mask.to(device=raw.device, dtype=torch.bool)

        observed = raw_mask.to(device=raw.device, dtype=torch.bool)
        observed = observed & padding_mask.unsqueeze(-1)
        clean_raw = torch.where(observed, raw, torch.zeros_like(raw))
        mask_channel = observed.to(dtype=raw.dtype)
        clean_delta = torch.nan_to_num(
            delta_steps.to(device=raw.device, dtype=raw.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        clean_delta = clean_delta * padding_mask.unsqueeze(-1).to(dtype=raw.dtype)
        combined = torch.cat((clean_raw, mask_channel, clean_delta), dim=-1)
        return combined, padding_mask

    def encode(
        self,
        raw: Tensor,
        raw_mask: Tensor,
        delta_steps: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return causal temporal states and the validated real-row mask."""

        combined, valid_rows = self._validate_and_combine_inputs(
            raw=raw,
            raw_mask=raw_mask,
            delta_steps=delta_steps,
            padding_mask=padding_mask,
        )
        valid = valid_rows.unsqueeze(-1).to(dtype=raw.dtype)
        hidden = self.input_projection(combined) * valid
        for block in self.blocks:
            hidden = block(hidden, valid_rows)
        hidden = self.output_norm(hidden) * valid
        return hidden, valid_rows

    def forward(
        self,
        raw: Tensor,
        raw_mask: Tensor,
        delta_steps: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        hidden, valid_rows = self.encode(
            raw=raw,
            raw_mask=raw_mask,
            delta_steps=delta_steps,
            padding_mask=padding_mask,
        )
        valid = valid_rows.unsqueeze(-1).to(dtype=hidden.dtype)
        logits = self.classifier(hidden) * valid
        return logits


class RuleGuidedResidualTCN(CausalSeq2SeqTCN):
    """Temporal TCN with a zero-initialized bounded expert-rule correction.

    The deterministic RuleModel is *not* optimized by CE.  Its continuous
    signed margins are encoded pointwise and may only adjust the temporal
    class-logit difference by ``[-rule_residual_scale, +rule_residual_scale]``.
    The final residual layer starts at zero, so initialization is exactly the
    temporal model and missing rule evidence always reduces exactly to it.
    """

    def __init__(
        self,
        raw_feature_count: int = RAW_FEATURE_COUNT,
        hidden_channels: int = 48,
        dilation_blocks: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.1,
        rule_hidden_channels: int = 32,
        rule_residual_scale: float = 4.0,
    ) -> None:
        super().__init__(
            raw_feature_count=raw_feature_count,
            hidden_channels=hidden_channels,
            dilation_blocks=dilation_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        if int(rule_hidden_channels) < 4:
            raise ValueError("rule_hidden_channels must be at least 4")
        if float(rule_residual_scale) <= 0:
            raise ValueError("rule_residual_scale must be positive")
        self.rule_hidden_channels = int(rule_hidden_channels)
        self.rule_residual_scale = float(rule_residual_scale)
        # margin + validity per rule, followed by max margin, positive-rule
        # fraction, and valid-rule coverage.
        rule_input_channels = 2 * RULE_FEATURE_COUNT + 3
        self.rule_encoder = nn.Sequential(
            nn.Linear(rule_input_channels, self.rule_hidden_channels),
            nn.LayerNorm(self.rule_hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        residual_hidden = max(self.hidden_channels, self.rule_hidden_channels)
        self.residual_head = nn.Sequential(
            nn.Linear(
                self.hidden_channels + self.rule_hidden_channels + 2,
                residual_hidden,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.requires_rule_inputs = True
        self.architecture = "rule_guided_residual_tcn"

    @classmethod
    def from_config(
        cls,
        model_config: Any,
        raw_feature_count: int = RAW_FEATURE_COUNT,
    ) -> "RuleGuidedResidualTCN":
        return cls(
            raw_feature_count=raw_feature_count,
            hidden_channels=int(model_config.hidden_channels),
            dilation_blocks=int(model_config.dilation_blocks),
            kernel_size=int(model_config.kernel_size),
            dropout=float(model_config.dropout),
            rule_hidden_channels=int(model_config.rule_hidden_channels),
            rule_residual_scale=float(model_config.rule_residual_scale),
        )

    def _rule_representation(
        self,
        rule_margin: Tensor,
        rule_mask: Tensor,
        rule_hard: Tensor,
        valid_rows: Tensor,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        expected = (*valid_rows.shape, RULE_FEATURE_COUNT)
        if tuple(rule_margin.shape) != expected:
            raise ValueError(
                "rule_margin must have shape "
                f"[batch, time, {RULE_FEATURE_COUNT}]"
            )
        if rule_mask.shape != rule_margin.shape:
            raise ValueError("rule_mask must have the same shape as rule_margin")
        if rule_hard.ndim == 3 and rule_hard.shape[-1] == 1:
            rule_hard = rule_hard.squeeze(-1)
        if rule_hard.shape != valid_rows.shape:
            raise ValueError("rule_hard must have shape [batch, time]")

        mask = rule_mask.to(device=rule_margin.device, dtype=torch.bool)
        mask &= valid_rows.unsqueeze(-1)
        clean_margin = torch.nan_to_num(
            rule_margin.to(dtype=dtype), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(min=-10.0, max=10.0)
        clean_margin = clean_margin * mask.to(dtype=dtype)
        coverage = mask.to(dtype=dtype).mean(dim=-1, keepdim=True)
        masked_for_max = torch.where(
            mask,
            clean_margin,
            torch.full_like(clean_margin, -torch.inf),
        )
        maximum = masked_for_max.max(dim=-1, keepdim=True).values
        maximum = torch.where(coverage > 0, maximum, torch.zeros_like(maximum))
        positive_fraction = (
            (mask & (clean_margin > 0)).to(dtype=dtype).sum(dim=-1, keepdim=True)
            / float(RULE_FEATURE_COUNT)
        )
        encoded = self.rule_encoder(
            torch.cat(
                (
                    clean_margin,
                    mask.to(dtype=dtype),
                    maximum,
                    positive_fraction,
                    coverage,
                ),
                dim=-1,
            )
        )
        hard = (
            rule_hard.to(device=rule_margin.device, dtype=dtype).unsqueeze(-1)
            * valid_rows.unsqueeze(-1).to(dtype=dtype)
        )
        return encoded, coverage, hard

    def forward(
        self,
        raw: Tensor,
        raw_mask: Tensor,
        delta_steps: Tensor,
        padding_mask: Tensor | None = None,
        *,
        rule_margin: Tensor,
        rule_mask: Tensor,
        rule_hard: Tensor,
    ) -> Tensor:
        hidden, valid_rows = self.encode(
            raw=raw,
            raw_mask=raw_mask,
            delta_steps=delta_steps,
            padding_mask=padding_mask,
        )
        valid = valid_rows.unsqueeze(-1).to(dtype=hidden.dtype)
        temporal_logits = self.classifier(hidden) * valid
        rule_latent, coverage, hard = self._rule_representation(
            rule_margin=rule_margin,
            rule_mask=rule_mask,
            rule_hard=rule_hard,
            valid_rows=valid_rows,
            dtype=hidden.dtype,
        )
        residual_input = torch.cat((hidden, rule_latent, coverage, hard), dim=-1)
        delta = self.rule_residual_scale * torch.tanh(
            self.residual_head(residual_input)
        )
        delta = delta * coverage * valid
        correction = torch.cat((-0.5 * delta, 0.5 * delta), dim=-1)
        return (temporal_logits + correction) * valid


def build_teacher_student(
    model_config: Any,
    raw_feature_count: int = RAW_FEATURE_COUNT,
) -> Tuple[nn.Module, nn.Module]:
    """Return independently initialized student and teacher networks."""

    architecture = str(
        getattr(model_config, "architecture", "causal_depthwise_tcn")
    )
    if architecture == "causal_depthwise_tcn":
        model_class = CausalSeq2SeqTCN
    elif architecture == "rule_guided_residual_tcn":
        model_class = RuleGuidedResidualTCN
    else:
        raise ValueError(f"unsupported model architecture={architecture!r}")
    student = model_class.from_config(model_config, raw_feature_count)
    teacher = model_class.from_config(model_config, raw_feature_count)
    return student, teacher


__all__ = [
    "CausalSeq2SeqTCN",
    "CausalResidualBlock",
    "DepthwiseSeparableCausalConv1d",
    "OUTPUT_CLASS_COUNT",
    "RAW_FEATURE_COUNT",
    "RuleGuidedResidualTCN",
    "build_teacher_student",
]
