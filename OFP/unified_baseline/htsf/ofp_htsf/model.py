"""Small explicit HTSF blocks with equal-dimensional learned representations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .config import HTSFConfig
from .data import ENGINEERED_FEATURES
from .variants import VariantSpec


class PatchTSTStyleEncoder(nn.Module):
    """Channel-independent causal patch transformer with validity masks.

    This is a transparent in-repository PatchTST-style encoder, not an import
    of the entangled teacher runner.  Each raw channel shares the same patch
    projection and Transformer; channel representations are pooled only after
    temporal encoding.
    """

    def __init__(self, config: HTSFConfig, n_channels: int = 12) -> None:
        super().__init__()
        window = config.window
        representation = config.representation
        self.sequence_length = int(window.sequence_length)
        self.patch_length = int(window.patch_length)
        self.patch_stride = int(window.patch_stride)
        self.n_channels = int(n_channels)
        patch_starts = list(
            range(
                0,
                self.sequence_length - self.patch_length + 1,
                self.patch_stride,
            )
        )
        endpoint_start = self.sequence_length - self.patch_length
        if patch_starts[-1] != endpoint_start:
            patch_starts.append(endpoint_start)
        self.patch_starts = tuple(patch_starts)
        self.n_patches = len(self.patch_starts)
        d_model = int(representation.d_model)
        self.patch_projection = nn.Linear(self.patch_length * 2, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        self.channel_embedding = nn.Parameter(torch.zeros(1, self.n_channels, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(representation.attention_heads),
            dim_feedforward=d_model * 4,
            dropout=float(representation.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(representation.transformer_layers),
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.channel_score = nn.Linear(d_model, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(representation.latent_dim)),
            nn.GELU(),
            nn.Dropout(float(representation.dropout)),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.channel_embedding, std=0.02)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or valid_mask.shape != values.shape:
            raise ValueError("temporal values/mask must be aligned [batch,time,channel]")
        if values.shape[1] != self.sequence_length or values.shape[2] != self.n_channels:
            raise ValueError("temporal input shape does not match the frozen model window")
        values_by_channel = values.transpose(1, 2)
        mask_by_channel = valid_mask.transpose(1, 2)
        value_patches = torch.stack(
            [
                values_by_channel[:, :, start : start + self.patch_length]
                for start in self.patch_starts
            ],
            dim=2,
        )
        mask_patches = torch.stack(
            [
                mask_by_channel[:, :, start : start + self.patch_length]
                for start in self.patch_starts
            ],
            dim=2,
        )
        patch_input = torch.cat([value_patches, mask_patches], dim=-1)
        tokens = self.patch_projection(patch_input)
        batch, channels, patches, width = tokens.shape
        tokens = tokens.reshape(batch * channels, patches, width)
        patch_valid = mask_patches.any(dim=-1).reshape(batch * channels, patches)
        cls = self.cls_token.expand(batch * channels, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position[:, : patches + 1]
        padding_mask = torch.cat(
            [
                torch.zeros(batch * channels, 1, dtype=torch.bool, device=values.device),
                ~patch_valid,
            ],
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        channel_latent = encoded[:, 0].reshape(batch, channels, width)
        channel_latent = channel_latent + self.channel_embedding[:, :channels]
        channel_valid = valid_mask.any(dim=1)
        has_valid_channel = channel_valid.any(dim=1, keepdim=True)
        safe_channel_valid = torch.where(
            has_valid_channel,
            channel_valid,
            torch.ones_like(channel_valid),
        )
        scores = self.channel_score(channel_latent).squeeze(-1)
        scores = scores.masked_fill(~safe_channel_valid, torch.finfo(scores.dtype).min)
        channel_weight = torch.softmax(scores, dim=1)
        pooled = torch.sum(channel_weight.unsqueeze(-1) * channel_latent, dim=1)
        return self.output(pooled)


class EngineeredEncoder(nn.Module):
    def __init__(self, config: HTSFConfig, n_features: int = len(ENGINEERED_FEATURES)) -> None:
        super().__init__()
        hidden = int(config.representation.engineered_hidden)
        latent = int(config.representation.latent_dim)
        dropout = float(config.representation.dropout)
        self.network = nn.Sequential(
            nn.Linear(int(n_features) * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.LayerNorm(latent),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or valid_mask.shape != values.shape:
            raise ValueError("engineered values/mask must be aligned [batch,feature]")
        return self.network(torch.cat([values, valid_mask], dim=1))


class DualViewFusion(nn.Module):
    def __init__(self, config: HTSFConfig, mode: str) -> None:
        super().__init__()
        if mode not in {"concat_projection", "cross_attention", "gated_cross_attention"}:
            raise ValueError(f"unknown dual-view fusion mode={mode!r}")
        self.mode = mode
        latent = int(config.representation.latent_dim)
        dropout = float(config.representation.dropout)
        self.token_norm = (
            nn.LayerNorm(latent) if mode in {"cross_attention", "gated_cross_attention"} else None
        )
        self.attention = (
            nn.MultiheadAttention(
                embed_dim=latent,
                num_heads=int(config.representation.attention_heads),
                dropout=dropout,
                batch_first=True,
            )
            if mode in {"cross_attention", "gated_cross_attention"}
            else None
        )
        self.gate = (
            nn.Sequential(
                nn.LayerNorm(latent * 2),
                nn.Linear(latent * 2, latent),
                nn.Sigmoid(),
            )
            if mode == "gated_cross_attention"
            else None
        )
        self.concat_projection = nn.Sequential(
            nn.LayerNorm(latent * 2),
            nn.Linear(latent * 2, latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        temporal: torch.Tensor,
        engineered: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.mode == "concat_projection":
            fused = self.concat_projection(torch.cat([temporal, engineered], dim=1))
            return fused, {
                "attention": torch.empty(0, device=temporal.device),
                "gate": torch.full_like(temporal, 0.5),
            }
        tokens = torch.stack([temporal, engineered], dim=1)
        if self.token_norm is None or self.attention is None:
            raise RuntimeError("attention modules are absent for an attention fusion mode")
        aligned = self.token_norm(tokens)
        attended, attention = self.attention(
            aligned,
            aligned,
            aligned,
            need_weights=True,
            average_attn_weights=False,
        )
        temporal_context = 0.5 * (temporal + attended[:, 0])
        engineered_context = 0.5 * (engineered + attended[:, 1])
        if self.mode == "cross_attention":
            gate = torch.full_like(temporal, 0.5)
            gated = gate * temporal_context + (1.0 - gate) * engineered_context
            fused = self.concat_projection(
                torch.cat([gated, attended.mean(dim=1)], dim=1)
            )
        else:
            if self.gate is None:
                raise RuntimeError("learned gate is absent for gated_cross_attention")
            gate = self.gate(torch.cat([temporal, engineered], dim=1))
            gated = gate * temporal_context + (1.0 - gate) * engineered_context
            fused = self.concat_projection(torch.cat([gated, attended.mean(dim=1)], dim=1))
        return fused, {"attention": attention, "gate": gate}


@dataclass
class EncoderOutput:
    representations: dict[str, torch.Tensor]
    auxiliary_logit: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class HTSFEncoder(nn.Module):
    def __init__(self, config: HTSFConfig, variant: VariantSpec) -> None:
        super().__init__()
        if not variant.needs_encoder:
            raise ValueError("sampled_b2_xgb has no neural representation stage")
        self.variant = variant
        self.temporal = PatchTSTStyleEncoder(config) if variant.use_temporal else None
        self.engineered = EngineeredEncoder(config) if variant.use_engineered else None
        self.fusion = (
            DualViewFusion(config, variant.fusion)
            if variant.use_temporal and variant.use_engineered
            else None
        )
        latent = int(config.representation.latent_dim)
        self.auxiliary_classifier = nn.Linear(latent, 1)

    def encode(
        self,
        raw: torch.Tensor,
        raw_mask: torch.Tensor,
        engineered: torch.Tensor,
        engineered_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        outputs: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, torch.Tensor] = {}
        if self.temporal is not None:
            outputs["temporal"] = self.temporal(raw, raw_mask)
        if self.engineered is not None:
            outputs["engineered"] = self.engineered(engineered, engineered_mask)
        if self.fusion is not None:
            outputs["fused"], diagnostics = self.fusion(
                outputs["temporal"],
                outputs["engineered"],
            )
            outputs["representation"] = outputs["fused"]
        elif self.variant.use_temporal:
            outputs["representation"] = outputs["temporal"]
        else:
            outputs["representation"] = outputs["engineered"]
        return outputs, diagnostics

    def forward(
        self,
        raw: torch.Tensor,
        raw_mask: torch.Tensor,
        engineered: torch.Tensor,
        engineered_mask: torch.Tensor,
    ) -> EncoderOutput:
        representations, diagnostics = self.encode(
            raw,
            raw_mask,
            engineered,
            engineered_mask,
        )
        logit = self.auxiliary_classifier(representations["representation"]).squeeze(-1)
        return EncoderOutput(representations, logit, diagnostics)


def model_manifest(model: HTSFEncoder, config: HTSFConfig, variant: VariantSpec) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "mechanism": variant.mechanism,
        "temporal_branch": variant.use_temporal,
        "engineered_branch": variant.use_engineered,
        "fusion": variant.fusion,
        "decision_inputs": list(variant.decision_inputs),
        "learned_representation_dim": int(config.representation.latent_dim),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "auxiliary_head_final_decision": False,
    }
