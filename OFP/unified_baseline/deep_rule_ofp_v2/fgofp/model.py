"""Causal native-cadence sequence model used by FGL-OFP.

The model deliberately keeps the temporal contract small and auditable:

* every output row is computed from that row and earlier rows only;
* the 12 raw measurements, their observation masks, and one delta-time channel
  are the only inputs;
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

    def forward(
        self,
        raw: Tensor,
        raw_mask: Tensor,
        delta_steps: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
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
        logits = self.classifier(hidden) * valid
        return logits


def build_teacher_student(
    model_config: Any,
    raw_feature_count: int = RAW_FEATURE_COUNT,
) -> Tuple[CausalSeq2SeqTCN, CausalSeq2SeqTCN]:
    """Return independently initialized student and teacher networks."""

    student = CausalSeq2SeqTCN.from_config(model_config, raw_feature_count)
    teacher = CausalSeq2SeqTCN.from_config(model_config, raw_feature_count)
    return student, teacher


__all__ = [
    "CausalSeq2SeqTCN",
    "CausalResidualBlock",
    "DepthwiseSeparableCausalConv1d",
    "OUTPUT_CLASS_COUNT",
    "RAW_FEATURE_COUNT",
    "build_teacher_student",
]
