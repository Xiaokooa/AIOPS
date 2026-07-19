"""MFP-Former for interpretable optical-module failure prediction.

MFP-Former uses a standard time-token Transformer encoder as the temporal
backbone. Sensor tokens are distilled from encoded time tokens by learnable
sensor queries, then refined by Multi-grained Fault Perception (MFP)
concept-sensor cross-attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from OFP.deep_learning.optical_prediction.deep_learning.cross_attn_itransformer import (
    MFPLayer,
    _mask_aware_norm,
)


@dataclass
class TypeAwareTransformerCfg:
    seq_len: int = 288
    n_sensors: int = 12
    use_mask_channel: bool = True
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    activation: str = "gelu"
    use_norm: bool = True
    concept_names: List[str] = field(default_factory=lambda: [
        "thermal_anomaly",
        "current_bias_anomaly",
        "tx_power_anomaly",
        "rx_power_anomaly",
        "lane_imbalance",
    ])
    n_concepts: int = 5
    bypass_ratio: float = 0.3
    n_mfp_layers: int = 2
    use_sensor_series_residual: bool = True
    time_pool_size: int = 6
    sensor_residual_scale: float = 1.0
    concept_risk_mix: float = 0.35
    prior_bias_strength: float = 3.0  # initial negative bias for non-prior sensors
    sensor_names: List[str] = field(default_factory=lambda: [
        "temperature", "current", "currentTXPower", "currentRXPower",
        "currentMultiRXPower1", "currentMultiRXPower2",
        "currentMultiRXPower3", "currentMultiRXPower4",
        "currentMultiTXPower1", "currentMultiTXPower2",
        "currentMultiTXPower3", "currentMultiTXPower4",
    ])


class TypeAwareCrossAttnTransformer(nn.Module):
    """MFP-Former: time-token Transformer with multi-task fault concepts.

    Forward returns:
    (logit, sensor_importance, concept_sensor_attn, sensor_concept_attn,
    concept_logits, sensor_time_attn). The final binary risk logit combines a
    global risk head with a Noisy-OR aggregation over concept logits, making
    fault concepts part of the prediction path instead of only an auxiliary
    explanation target.
    """

    def __init__(self, cfg: TypeAwareTransformerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask_channel = cfg.use_mask_channel
        self.use_norm = cfg.use_norm

        in_dim = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        self.max_time_tokens = int(math.ceil(cfg.seq_len / max(1, cfg.time_pool_size)))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_time_tokens, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.e_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        self.sensor_queries = nn.Parameter(torch.randn(cfg.n_sensors, cfg.d_model) * 0.02)
        self.sensor_time_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.sensor_token_norm = nn.LayerNorm(cfg.d_model)

        if cfg.use_sensor_series_residual:
            # Lightweight per-sensor statistics (6 features) instead of full
            # raw sequence, to preserve sensor identity without creating
            # a shortcut that bypasses the temporal Transformer.
            n_stats = 6  # mean, std, min, max, slope, last
            self.sensor_series_proj = nn.Sequential(
                nn.Linear(n_stats, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
        else:
            self.sensor_series_proj = None

        self.concept_queries = nn.Parameter(torch.empty(cfg.n_concepts, cfg.d_model))
        nn.init.xavier_uniform_(self.concept_queries.data.unsqueeze(0))
        self.mfp_layers = nn.ModuleList([
            MFPLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_concepts, cfg.dropout)
            for _ in range(cfg.n_mfp_layers)
        ])

        # Prior-based attention bias: structurally discourages concepts from
        # attending to unrelated sensors. Learnable so it can relax if needed.
        prior_matrix = self._build_prior_matrix(cfg)
        # bias = 0 for prior-matched pairs, -strength for non-matched
        init_bias = -cfg.prior_bias_strength * (1.0 - prior_matrix)
        self.attn_bias = nn.Parameter(torch.from_numpy(init_bias))

        self.type_head = nn.Linear(cfg.d_model, 1)
        nn.init.constant_(self.type_head.bias, -2.0)
        self.time_pool_score = nn.Linear(cfg.d_model, 1, bias=False)
        bypass_dim = max(4, cfg.n_concepts // 2)
        self.bypass_proj = nn.Sequential(
            nn.Linear(cfg.d_model, bypass_dim),
            nn.GELU(),
        )

        head_in = cfg.d_model + bypass_dim
        self.global_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(8, head_in // 2)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(max(8, head_in // 2), 1),
        )
        mix = min(max(float(cfg.concept_risk_mix), 1e-4), 1.0 - 1e-4)
        self.concept_risk_mix_logit = nn.Parameter(
            torch.tensor(math.log(mix / (1.0 - mix)), dtype=torch.float32)
        )

    @staticmethod
    def _build_prior_matrix(cfg: TypeAwareTransformerCfg) -> np.ndarray:
        """Build (K, M) binary prior: 1 where concept should attend sensor."""
        groups = {
            "thermal_anomaly": {"temperature"},
            "current_bias_anomaly": {"current"},
            "tx_power_anomaly": {"currentTXPower"} | {f"currentMultiTXPower{i}" for i in range(1, 5)},
            "rx_power_anomaly": {"currentRXPower"} | {f"currentMultiRXPower{i}" for i in range(1, 5)},
            "lane_imbalance": {f"currentMultiTXPower{i}" for i in range(1, 5)}
                             | {f"currentMultiRXPower{i}" for i in range(1, 5)},
        }
        prior = np.zeros((cfg.n_concepts, cfg.n_sensors), dtype=np.float32)
        for row, name in enumerate(cfg.concept_names):
            group = groups.get(name, set())
            for col, sensor in enumerate(cfg.sensor_names):
                if sensor in group:
                    prior[row, col] = 1.0
            if prior[row].sum() == 0:
                prior[row] = 1.0  # fallback: no bias if no prior
        return prior

    def _pool_time_tokens(self, inp: torch.Tensor) -> torch.Tensor:
        pool = max(1, int(self.cfg.time_pool_size))
        if pool <= 1:
            return inp
        batch, length, channels = inp.shape
        pad_len = (pool - (length % pool)) % pool
        if pad_len:
            pad = inp.new_zeros(batch, pad_len, channels)
            inp = torch.cat([inp, pad], dim=1)
        return inp.view(batch, -1, pool, channels).mean(dim=2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.use_norm:
            x_model = _mask_aware_norm(x, mask)
        else:
            x_model = x * mask

        if self.use_mask_channel:
            inp = torch.cat([x_model, mask], dim=-1)
        else:
            inp = x_model

        inp = self._pool_time_tokens(inp)
        length = inp.size(1)
        time_tokens = self.input_proj(inp)
        time_tokens = time_tokens + self.pos_embedding[:, :length, :]
        time_tokens = self.time_encoder(time_tokens)

        batch = x.size(0)
        sensor_queries = self.sensor_queries.unsqueeze(0).expand(batch, -1, -1)
        sensor_tokens, sensor_time_attn = self.sensor_time_attn(
            query=sensor_queries,
            key=time_tokens,
            value=time_tokens,
        )

        if self.sensor_series_proj is not None:
            # Lightweight per-sensor statistics for sensor identity.
            sensor_vals = x_model.transpose(1, 2)  # (B, M, T)
            sensor_mask = mask.transpose(1, 2)      # (B, M, T)
            denom = sensor_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            s_mean = (sensor_vals * sensor_mask).sum(dim=-1, keepdim=True) / denom
            s_var = ((sensor_vals - s_mean) ** 2 * sensor_mask).sum(dim=-1, keepdim=True) / denom
            s_std = torch.sqrt(s_var + 1e-5)
            s_min = sensor_vals.masked_fill(sensor_mask < 0.5, 1e9).min(dim=-1, keepdim=True).values
            s_max = sensor_vals.masked_fill(sensor_mask < 0.5, -1e9).max(dim=-1, keepdim=True).values
            s_last = sensor_vals[:, :, -1:]
            # slope: linear trend (last - first valid) / length
            s_first = sensor_vals[:, :, :1]
            s_slope = (s_last - s_first) / sensor_vals.size(-1)
            stats = torch.cat([s_mean, s_std, s_min, s_max, s_slope, s_last], dim=-1)  # (B, M, 6)
            sensor_tokens = sensor_tokens + self.sensor_series_proj(stats)
        sensor_tokens = self.sensor_token_norm(sensor_tokens)

        concept_tokens = self.concept_queries.unsqueeze(0).expand(batch, -1, -1)
        attn_c2s_last = None
        attn_s2c_last = None
        # Expand attn_bias for multi-head attention: (K, M)
        bias = self.attn_bias
        for mfp in self.mfp_layers:
            sensor_tokens, concept_tokens, attn_c2s_last, attn_s2c_last = mfp(
                sensor_tokens, concept_tokens, attn_bias=bias
            )

        type_logits = self.type_head(concept_tokens).squeeze(-1)
        concept_gate = torch.sigmoid(type_logits)
        pooled = sensor_tokens.mean(dim=1)
        time_alpha = torch.softmax(self.time_pool_score(time_tokens), dim=1)
        time_context = (time_tokens * time_alpha).sum(dim=1)
        bypass = self.bypass_proj(time_context) * self.cfg.bypass_ratio
        global_logit = self.global_head(torch.cat([pooled, bypass], dim=-1)).squeeze(-1)

        # Multi-task risk path: mix global and concept risks in probability
        # space for better score calibration, then convert back to logit.
        global_prob = torch.sigmoid(global_logit)
        any_concept_prob = 1.0 - torch.prod(
            (1.0 - concept_gate).clamp(min=1e-6, max=1.0),
            dim=-1,
        )
        concept_mix = torch.sigmoid(self.concept_risk_mix_logit)
        fused_prob = (1.0 - concept_mix) * global_prob + concept_mix * any_concept_prob
        logit = torch.logit(fused_prob.clamp(min=1e-6, max=1.0 - 1e-6))

        sensor_importance = torch.bmm(concept_gate.unsqueeze(1), attn_c2s_last).squeeze(1)
        gate_sum = concept_gate.sum(dim=-1, keepdim=True)
        fallback = attn_c2s_last.mean(dim=1)
        sensor_importance = torch.where(
            gate_sum > 1e-6,
            sensor_importance / gate_sum.clamp(min=1e-6),
            fallback,
        )
        sensor_importance = sensor_importance / sensor_importance.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)

        return (
            logit,
            sensor_importance,
            attn_c2s_last,
            attn_s2c_last,
            type_logits,
            sensor_time_attn,
        )
