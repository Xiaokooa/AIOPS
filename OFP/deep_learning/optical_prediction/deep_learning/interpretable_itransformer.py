"""Interpretable iTransformer variants for fault prediction.

Three levels of built-in interpretability:
  L1 – AttentionPool iTransformer: replace mean-pool with learned attention pooling
  L2 – SparseGate iTransformer:   top-K sensor gating for sparse explanations
  L3 – ConceptBottleneck iTransformer: intermediate concept layer with semantic meaning

All variants output (logit, sensor_weights) so downstream code can inspect
which sensors drove each prediction.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

ITF_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "iTransformer"
if str(ITF_ROOT) not in sys.path:
    sys.path.insert(0, str(ITF_ROOT))

from layers.Transformer_EncDec import Encoder, EncoderLayer  # type: ignore
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # type: ignore
from layers.Embed import DataEmbedding_inverted  # type: ignore


# ── shared config ────────────────────────────────────────────────────
@dataclass
class InterpITransformerCfg:
    seq_len: int = 288
    n_sensors: int = 12
    use_mask_channel: bool = True
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    activation: str = "gelu"
    factor: int = 1
    use_norm: bool = True
    embed: str = "fixed"
    freq: str = "h"
    # L2 specific (kept for potential future use)
    top_k: int = 3
    gate_temp: float = 1.0
    # L1v2 specific
    attn_residual_ratio: float = 0.5
    # L2 specific (top_k, gate_temp already above)
    # L3 specific
    concept_names: List[str] = field(default_factory=lambda: [
        "temp_anomaly", "tx_degradation", "rx_degradation", "current_anomaly",
    ])
    n_concepts: int = 4
    bypass_ratio: float = 0.3


# ── shared encoder builder ──────────────────────────────────────────
def _build_encoder(cfg: InterpITransformerCfg, output_attention: bool = False):
    """Build iTransformer embedding + encoder, shared across all variants."""
    n_channels = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)
    enc_embedding = DataEmbedding_inverted(
        cfg.seq_len, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout,
    )
    encoder = Encoder(
        [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        False, cfg.factor,
                        attention_dropout=cfg.dropout,
                        output_attention=output_attention,
                    ),
                    cfg.d_model, cfg.n_heads,
                ),
                cfg.d_model, cfg.d_ff,
                dropout=cfg.dropout, activation=cfg.activation,
            )
            for _ in range(cfg.e_layers)
        ],
        norm_layer=nn.LayerNorm(cfg.d_model),
    )
    return enc_embedding, encoder, n_channels


def _mask_aware_norm(x: torch.Tensor, mask: torch.Tensor):
    """Per-window z-score on valid entries."""
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / denom
    var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + 1e-5)
    return (x - mean) / std * mask




# =====================================================================
# L1 – Attention-Pooling iTransformer
# =====================================================================
class L1_AttentionPoolITransformer(nn.Module):
    """Replaces mean-pool with a learned attention gate over sensor tokens.

    sensor_weights α ∈ R^N (softmax-normalized) are produced per sample,
    directly indicating each sensor's contribution.
    """

    def __init__(self, cfg: InterpITransformerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask_channel = cfg.use_mask_channel
        self.use_norm = cfg.use_norm
        self.enc_embedding, self.encoder, self.n_channels = _build_encoder(cfg, output_attention=True)

        # Attention pooling: project each sensor token to a scalar score
        self.attn_query = nn.Linear(cfg.d_model, 1, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, L, N), mask: (B, L, N)
        if self.use_mask_channel:
            inp = torch.cat([x, mask], dim=-1)
        else:
            inp = x
        if self.use_norm:
            x_norm = _mask_aware_norm(x, mask)
            inp = torch.cat([x_norm, mask], dim=-1) if self.use_mask_channel else x_norm

        enc_out = self.enc_embedding(inp, None)          # (B, C, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # (B, C, d_model)

        # Attention pooling over sensor tokens (first N channels are sensors)
        n_s = self.cfg.n_sensors
        sensor_tokens = enc_out[:, :n_s, :]               # (B, N, d_model)
        scores = self.attn_query(sensor_tokens).squeeze(-1)  # (B, N)
        alpha = F.softmax(scores, dim=-1)                 # (B, N) sensor weights
        pooled = (sensor_tokens * alpha.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        logit = self.head(pooled).squeeze(-1)             # (B,)
        return logit, alpha, attns



# =====================================================================
# L3 – Concept-Bottleneck iTransformer
# =====================================================================
class L3_ConceptBottleneckITransformer(nn.Module):
    """Concept bottleneck: sensor tokens → K interpretable concept scores → prediction.

    Concepts are semantically meaningful (e.g. 'temperature anomaly',
    'TX power degradation') and each one attends to a subset of sensors.
    """

    def __init__(self, cfg: InterpITransformerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask_channel = cfg.use_mask_channel
        self.use_norm = cfg.use_norm
        self.enc_embedding, self.encoder, self.n_channels = _build_encoder(cfg, output_attention=True)
        self.concept_names = cfg.concept_names
        n_concepts = cfg.n_concepts

        # Each concept has a learned query that attends to sensor tokens
        self.concept_queries = nn.Parameter(torch.randn(n_concepts, cfg.d_model) * 0.02)
        self.concept_key = nn.Linear(cfg.d_model, cfg.d_model)
        self.concept_proj = nn.Linear(cfg.d_model, 1)  # concept activation score

        self.head = nn.Sequential(
            nn.LayerNorm(n_concepts),
            nn.Linear(n_concepts, n_concepts * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(n_concepts * 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.use_mask_channel:
            inp = torch.cat([x, mask], dim=-1)
        else:
            inp = x
        if self.use_norm:
            x_norm = _mask_aware_norm(x, mask)
            inp = torch.cat([x_norm, mask], dim=-1) if self.use_mask_channel else x_norm

        enc_out = self.enc_embedding(inp, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        n_s = self.cfg.n_sensors
        sensor_tokens = enc_out[:, :n_s, :]               # (B, N, d_model)
        keys = self.concept_key(sensor_tokens)             # (B, N, d_model)

        B = sensor_tokens.size(0)
        queries = self.concept_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)

        # Cross-attention: concepts query sensors
        scale = self.cfg.d_model ** 0.5
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / scale  # (B, K, N)
        concept_sensor_weights = F.softmax(attn_scores, dim=-1)          # (B, K, N)

        # Weighted sum of sensor tokens for each concept
        concept_repr = torch.bmm(concept_sensor_weights, sensor_tokens)  # (B, K, d_model)
        concept_activations = self.concept_proj(concept_repr).squeeze(-1)  # (B, K)

        # Aggregate sensor weights across concepts for overall sensor importance
        # weight each concept's sensor attention by the concept's activation
        concept_act_weights = torch.sigmoid(concept_activations)          # (B, K)
        sensor_importance = torch.bmm(
            concept_act_weights.unsqueeze(1), concept_sensor_weights
        ).squeeze(1)  # (B, N)
        sensor_importance = sensor_importance / sensor_importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        logit = self.head(concept_activations).squeeze(-1)  # (B,)

        return logit, sensor_importance, attns, concept_activations, concept_sensor_weights


# =====================================================================
# L1v2 – Residual Attention-Pooling iTransformer
# =====================================================================
class L1v2_ResidualAttnPoolITransformer(nn.Module):
    """Attention pooling with residual connection from mean-pool.

    pooled = (1 - r) * mean_pool + r * attn_pool
    This prevents information loss from over-selective attention.
    Multi-head attention query for richer sensor scoring.
    """

    def __init__(self, cfg: InterpITransformerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask_channel = cfg.use_mask_channel
        self.use_norm = cfg.use_norm
        self.ratio = cfg.attn_residual_ratio
        self.enc_embedding, self.encoder, self.n_channels = _build_encoder(cfg, output_attention=True)

        # Multi-head attention pooling (4 heads → average)
        self.n_attn_heads = 4
        self.attn_queries = nn.Linear(cfg.d_model, self.n_attn_heads, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.use_mask_channel:
            inp = torch.cat([x, mask], dim=-1)
        else:
            inp = x
        if self.use_norm:
            x_norm = _mask_aware_norm(x, mask)
            inp = torch.cat([x_norm, mask], dim=-1) if self.use_mask_channel else x_norm

        enc_out = self.enc_embedding(inp, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        n_s = self.cfg.n_sensors
        sensor_tokens = enc_out[:, :n_s, :]               # (B, N, d_model)

        # Mean pooling (baseline path)
        mean_pooled = sensor_tokens.mean(dim=1)            # (B, d_model)

        # Multi-head attention pooling
        scores = self.attn_queries(sensor_tokens)           # (B, N, H)
        alpha_heads = F.softmax(scores, dim=1)              # (B, N, H) softmax over sensors
        alpha = alpha_heads.mean(dim=-1)                    # (B, N) average across heads
        attn_pooled = (sensor_tokens * alpha.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        # Residual combination
        pooled = (1 - self.ratio) * mean_pooled + self.ratio * attn_pooled

        logit = self.head(pooled).squeeze(-1)
        return logit, alpha, attns


# =====================================================================
# L3v2 – Wide Concept Bottleneck with Bypass
# =====================================================================
class L3v2_WideConceptBypassITransformer(nn.Module):
    """Wider concept bottleneck (n_concepts = n_sensors by default)
    with a bypass residual from mean-pool.

    logit = head( concat(concept_activations, bypass_repr) )
    The bypass ensures gradient flow; concepts provide interpretability.
    """

    def __init__(self, cfg: InterpITransformerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask_channel = cfg.use_mask_channel
        self.use_norm = cfg.use_norm
        self.bypass_ratio = cfg.bypass_ratio
        self.enc_embedding, self.encoder, self.n_channels = _build_encoder(cfg, output_attention=True)
        self.concept_names = cfg.concept_names
        n_concepts = cfg.n_concepts

        # Multi-head concept queries
        self.concept_queries = nn.Parameter(torch.randn(n_concepts, cfg.d_model) * 0.02)
        self.concept_key = nn.Linear(cfg.d_model, cfg.d_model)
        self.concept_value = nn.Linear(cfg.d_model, cfg.d_model)
        self.concept_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        # Bypass: project mean-pooled encoder output to a small vector
        bypass_dim = max(4, n_concepts // 2)
        self.bypass_proj = nn.Sequential(
            nn.Linear(cfg.d_model, bypass_dim),
            nn.GELU(),
        )

        # Final head takes concept activations + bypass
        self.head = nn.Sequential(
            nn.LayerNorm(n_concepts + bypass_dim),
            nn.Linear(n_concepts + bypass_dim, (n_concepts + bypass_dim) * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear((n_concepts + bypass_dim) * 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.use_mask_channel:
            inp = torch.cat([x, mask], dim=-1)
        else:
            inp = x
        if self.use_norm:
            x_norm = _mask_aware_norm(x, mask)
            inp = torch.cat([x_norm, mask], dim=-1) if self.use_mask_channel else x_norm

        enc_out = self.enc_embedding(inp, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        n_s = self.cfg.n_sensors
        sensor_tokens = enc_out[:, :n_s, :]               # (B, N, d_model)
        keys = self.concept_key(sensor_tokens)             # (B, N, d_model)
        values = self.concept_value(sensor_tokens)         # (B, N, d_model)

        B = sensor_tokens.size(0)
        queries = self.concept_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)

        # Cross-attention: concepts query sensors
        scale = self.cfg.d_model ** 0.5
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / scale  # (B, K, N)
        concept_sensor_weights = F.softmax(attn_scores, dim=-1)          # (B, K, N)

        # Weighted sum of sensor values for each concept
        concept_repr = torch.bmm(concept_sensor_weights, values)  # (B, K, d_model)
        concept_activations = self.concept_proj(concept_repr).squeeze(-1)  # (B, K)

        # Bypass path: mean-pool encoder output, scaled by bypass_ratio
        bypass = self.bypass_proj(sensor_tokens.mean(dim=1))  # (B, bypass_dim)
        bypass = bypass * self.bypass_ratio

        # Concatenate and classify
        combined = torch.cat([concept_activations, bypass], dim=-1)
        logit = self.head(combined).squeeze(-1)  # (B,)

        # Aggregate sensor weights across concepts for overall sensor importance
        concept_act_weights = torch.sigmoid(concept_activations)          # (B, K)
        sensor_importance = torch.bmm(
            concept_act_weights.unsqueeze(1), concept_sensor_weights
        ).squeeze(1)  # (B, N)
        sensor_importance = sensor_importance / sensor_importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return logit, sensor_importance, attns, concept_activations, concept_sensor_weights
