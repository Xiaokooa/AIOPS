"""MFTransformerV3: Fault-Type Expert Transformer.

Architectural improvements over V2 targeting higher F1:
  1. **Hard-mask + Leak attention**: Each fault-type expert token is hard-routed to its
     sensor group (80%), with 20% leak to all sensors. Forces genuine
     expert specialization instead of soft attention bias.
  2. **No detection token**: Removes the hierarchy bottleneck that caused
     calibration instability in V2.  Global path serves detection role.
  3. **Per-fault-type expert heads**: Each expert has its own 2-layer MLP
     producing (risk, confidence), allowing independent judgment.
  4. **Confidence-weighted expert fusion**: Replace fragile Noisy-OR with
     confidence-weighted average of per-type risks.
  5. **Sensor interaction layer**: Lightweight cross-sensor attention before
     fault-type expert routing to capture inter-sensor correlations.

Output tuple (backward-compatible with training loop):
  0: logit            (B,)      -- binary risk
  1: sensor_importance (B, M)   -- fault-type-gated sensor attribution
  2: attn_c2s         (B, K, M) -- concept→sensor attention (last layer)
  3: attn_s2c         (B, M, K) -- sensor→concept attention (last layer)
  4: type_logits      (B, K)    -- fault-type logits (per-expert risks)
  5: sensor_time_attn (B, M, T')-- sensor-time attribution
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from OFP.deep_learning.optical_prediction.deep_learning.cross_attn_itransformer import (
    FeatureAbstractor,
    AdaptiveLayerNorm,
    _mask_aware_norm,
)
from OFP.deep_learning.optical_prediction.deep_learning.fault_type_labels import TYPE_NAMES


# ── Config ────────────────────────────────────────────────────────────
@dataclass
class MFTransformerV3Cfg:
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
    concept_names: List[str] = field(default_factory=lambda: list(TYPE_NAMES))
    n_types: int = 5
    bypass_ratio: float = 0.3
    n_mfp_layers: int = 2
    use_sensor_series_residual: bool = True
    time_pool_size: int = 6
    concept_risk_mix: float = 0.35
    leak_ratio: float = 0.2          # fraction of attention allowed to leak outside group
    hierarchy_weight: float = 0.0    # no hierarchy loss for V3 (no det token)
    fusion_mode: str = "confidence"   # "confidence" or "noisy_or"
    sensor_names: List[str] = field(default_factory=lambda: [
        "temperature", "current", "currentTXPower", "currentRXPower",
        "currentMultiRXPower1", "currentMultiRXPower2",
        "currentMultiRXPower3", "currentMultiRXPower4",
        "currentMultiTXPower1", "currentMultiTXPower2",
        "currentMultiTXPower3", "currentMultiTXPower4",
    ])
    # ── V3c: Dynamic sensor mask + contrastive loss ──
    use_dynamic_sensor_mask: bool = False
    sensor_mask_mode: str = "hard"  # "hard" (Gumbel-Bernoulli) or "soft" (continuous sigmoid)
    sensor_mask_temperature: float = 0.5
    sensor_contrastive_lambda: float = 0.3  # regularization strength
    sensor_contrastive_weight: float = 0.05  # weight in total loss
    output_dim: int = 1


# ── Hard-mask + Leak Fault-Type Cross-Attention ───────────────────────
class HardRoutedConceptAttention(nn.Module):
    """Fault-type-to-sensor cross-attention with hard routing + leak.

    Each fault-type expert token primarily attends to its designated sensor group,
    with a small leak ratio allowing it to see all sensors.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 n_concepts: int, n_sensors: int, leak_ratio: float = 0.2):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_sensors = n_sensors
        self.leak_ratio = leak_ratio

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, concept_tokens: torch.Tensor, sensor_tokens: torch.Tensor,
                group_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        concept_tokens: (B, K, d)
        sensor_tokens:  (B, M, d)
        group_mask:     (K, M) float, 1.0 for in-group, 0.0 for out-of-group

        Returns: (output, attn_weights)
            output: (B, K, d)
            attn_weights: (B, K, M) averaged over heads
        """
        B, K, d = concept_tokens.shape
        M = sensor_tokens.size(1)
        H = self.n_heads

        Q = self.q_proj(concept_tokens).view(B, K, H, self.head_dim).transpose(1, 2)  # (B, H, K, hd)
        Kv = self.k_proj(sensor_tokens).view(B, M, H, self.head_dim).transpose(1, 2)  # (B, H, M, hd)
        V = self.v_proj(sensor_tokens).view(B, M, H, self.head_dim).transpose(1, 2)   # (B, H, M, hd)

        # Raw attention scores: (B, H, K, M)
        raw_scores = torch.matmul(Q, Kv.transpose(-2, -1)) * self.scale

        # group_mask: (K, M) → (1, 1, K, M)
        gm = group_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, K, M)

        # In-group attention: mask out-of-group with -inf
        in_group_scores = raw_scores.masked_fill(gm < 0.5, float('-inf'))
        in_group_attn = F.softmax(in_group_scores, dim=-1)
        in_group_attn = in_group_attn.nan_to_num(0.0)  # handle all-masked rows

        # All-sensor attention (leak): full softmax
        all_attn = F.softmax(raw_scores, dim=-1)

        # Blend: (1 - leak) * in_group + leak * all
        leak = self.leak_ratio
        blended_attn = (1.0 - leak) * in_group_attn + leak * all_attn
        blended_attn = self.dropout(blended_attn)

        # Weighted sum
        out = torch.matmul(blended_attn, V)  # (B, H, K, hd)
        out = out.transpose(1, 2).contiguous().view(B, K, d)
        out = self.out_proj(out)

        # Average attention weights over heads for interpretability
        attn_weights = blended_attn.mean(dim=1)  # (B, K, M)

        return out, attn_weights


# ── Sensor Interaction Layer ──────────────────────────────────────────
class SensorInteractionLayer(nn.Module):
    """Lightweight self-attention over sensor tokens to capture cross-sensor
    correlations (e.g. TX power drop + temperature rise)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, sensor_tokens: torch.Tensor) -> torch.Tensor:
        # Self-attention among sensors
        sa_out, _ = self.self_attn(sensor_tokens, sensor_tokens, sensor_tokens)
        sensor_tokens = self.norm1(sa_out + sensor_tokens)
        sensor_tokens = self.norm2(self.ffn(sensor_tokens) + sensor_tokens)
        return sensor_tokens


# ── Dynamic Masked Sensor Interaction (CATCH-inspired) ────────────────
class DynamicSensorMaskGenerator(nn.Module):
    """Generate a dynamic mask over sensor-sensor interactions.

    Two modes:
      - "hard": Gumbel-Bernoulli sampling (discrete binary mask, straight-through)
      - "soft": continuous sigmoid probabilities (smoother gradients, more stable)
    """

    def __init__(self, d_model: int, n_sensors: int, mode: str = "hard"):
        super().__init__()
        self.n_sensors = n_sensors
        self.mode = mode
        # Project each sensor token to a distribution over other sensors
        self.mask_proj = nn.Linear(d_model, n_sensors, bias=False)
        # Initialize near-zero so mask starts close to uniform
        nn.init.zeros_(self.mask_proj.weight)

    def forward(self, sensor_tokens: torch.Tensor, temperature: float = 0.5
                ) -> torch.Tensor:
        """Generate mask from sensor tokens.

        Args:
            sensor_tokens: (B, M, d)
            temperature: Gumbel-Softmax temperature (hard) or sharpness (soft)

        Returns:
            mask: (B, M, M) mask values in [0, 1] with diagonal = 1
        """
        # (B, M, M) - each row i gives logits for which sensors i should attend to
        logits = self.mask_proj(sensor_tokens)  # (B, M, M)
        # Sensor interaction is treated as an undirected reliability pathway in
        # the paper, so keep the learned mask symmetric before sampling.
        logits = 0.5 * (logits + logits.transpose(-1, -2))

        if self.mode == "soft":
            # Continuous soft mask: sharpen with temperature
            mask = torch.sigmoid(logits / temperature)
        else:
            prob = torch.sigmoid(logits)
            mask = self._gumbel_bernoulli(prob, temperature)

        # Enforce self-connection (diagonal = 1)
        eye = torch.eye(self.n_sensors, device=mask.device).unsqueeze(0)  # (1, M, M)
        inv_eye = 1.0 - eye
        mask = mask * inv_eye + eye  # off-diag from learned, diag = 1

        return mask

    def _gumbel_bernoulli(self, prob: torch.Tensor, temperature: float
                          ) -> torch.Tensor:
        """Differentiable Bernoulli sampling via Gumbel-Softmax."""
        if not self.training:
            # Deterministic at eval: threshold at 0.5
            return (prob > 0.5).float()

        # Create 2-class logits whose softmax positive probability is p.
        eps = 1e-6
        positive = torch.log(prob.clamp(min=eps))
        negative = torch.log((1.0 - prob).clamp(min=eps))
        two_class = torch.stack([positive, negative], dim=-1)  # (..., 2)

        # Gumbel-Softmax with hard=True for straight-through
        hard_sample = F.gumbel_softmax(two_class, tau=temperature, hard=True)
        return hard_sample[..., 0]  # take the "positive" class


class DynamicMaskedSensorInteraction(nn.Module):
    """Sensor self-attention with CATCH-style dynamic binary mask.

    Combines:
      - Dynamic mask generation (Gumbel-Bernoulli)
      - Masked self-attention among sensors
      - Contrastive loss computation for mask guidance
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_sensors: int, dropout: float, temperature: float = 0.5,
                 contrastive_lambda: float = 0.3, mask_mode: str = "hard"):
        super().__init__()
        self.n_heads = n_heads
        self.n_sensors = n_sensors
        self.temperature = temperature
        self.contrastive_lambda = contrastive_lambda
        self.mask_mode = mask_mode

        self.mask_generator = DynamicSensorMaskGenerator(d_model, n_sensors, mode=mask_mode)

        # Self-attention with custom mask application
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, sensor_tokens: torch.Tensor,
                group_prior: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic masking.

        Args:
            sensor_tokens: (B, M, d)
            group_prior: (M, M) binary prior from fault-type groups (optional)

        Returns:
            sensor_tokens: (B, M, d) refined
            sensor_mask: (B, M, M) discovered mask
            contrastive_loss: scalar loss for mask guidance
        """
        B, M, d = sensor_tokens.shape
        H = self.n_heads

        # Generate dynamic mask
        sensor_mask = self.mask_generator(sensor_tokens, self.temperature)
        # sensor_mask: (B, M, M)

        # Compute contrastive loss
        contrastive_loss = self._contrastive_loss(sensor_tokens, sensor_mask,
                                                  group_prior)

        # Masked self-attention
        Q = self.q_proj(sensor_tokens).view(B, M, H, self.head_dim).transpose(1, 2)
        K = self.k_proj(sensor_tokens).view(B, M, H, self.head_dim).transpose(1, 2)
        V = self.v_proj(sensor_tokens).view(B, M, H, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, M, M)

        # Apply mask to attention scores
        mask_expanded = sensor_mask.unsqueeze(1)  # (B, 1, M, M)
        if self.mask_mode == "soft":
            # Soft: additive log-mask for smooth gradients
            scores = scores + torch.log(mask_expanded.clamp(min=1e-6))
        else:
            # Hard: binary mask with -inf
            scores = scores.masked_fill(mask_expanded < 0.5, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)  # handle all-masked rows
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, M, hd)
        out = out.transpose(1, 2).contiguous().view(B, M, d)
        out = self.out_proj(out)

        # Residual + FFN
        sensor_tokens = self.norm1(out + sensor_tokens)
        sensor_tokens = self.norm2(self.ffn(sensor_tokens) + sensor_tokens)

        return sensor_tokens, sensor_mask, contrastive_loss

    def _contrastive_loss(self, sensor_tokens: torch.Tensor,
                          sensor_mask: torch.Tensor,
                          group_prior: torch.Tensor | None) -> torch.Tensor:
        """CATCH-inspired dynamical contrastive loss.

        Two terms:
        1. Clustering loss: encourage prior-related sensor pairs to have high
           cosine similarity in representation space.
        2. Regularization: guide the learned mask toward prior structure
           (not too dense, not too sparse).
        """
        B, M, d = sensor_tokens.shape

        # Cosine similarity matrix among sensors
        tokens_norm = F.normalize(sensor_tokens, dim=-1)  # (B, M, d)
        cosine_sim = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # (B, M, M)

        # Clustering loss: prior-related pairs should be similar.  Using the
        # learned mask itself as the positive selector makes the all-one mask a
        # trivial optimum, so the mask is regularized separately below.
        eye = torch.eye(M, device=sensor_mask.device).unsqueeze(0)
        if group_prior is not None:
            positive_selector = group_prior.unsqueeze(0).expand(B, -1, -1)
        else:
            positive_selector = eye
        pos_sim = torch.exp(cosine_sim / self.temperature) * positive_selector
        all_sim = torch.exp(cosine_sim / self.temperature)

        # Avoid log(0)
        cluster_loss = -torch.log(
            pos_sim.sum(dim=-1).clamp(min=1e-8) /
            all_sim.sum(dim=-1).clamp(min=1e-8)
        ).mean()

        # Regularization: guide mask toward group prior without allowing dense
        # masks to win simply by covering every sensor pair.
        offdiag = 1.0 - eye
        if group_prior is not None:
            prior = group_prior.unsqueeze(0).expand(B, -1, -1)
            false_positive = sensor_mask * (1.0 - prior) * offdiag
            false_negative = (1.0 - sensor_mask) * prior * offdiag
            fp_denom = ((1.0 - prior) * offdiag).sum(dim=(-2, -1)).clamp(min=1.0)
            fn_denom = (prior * offdiag).sum(dim=(-2, -1)).clamp(min=1.0)
            fp_loss = (false_positive.sum(dim=(-2, -1)) / fp_denom).mean()
            fn_loss = (false_negative.sum(dim=(-2, -1)) / fn_denom).mean()
            mask_density = (sensor_mask * offdiag).sum(dim=(-2, -1)) / offdiag.sum(dim=(-2, -1)).clamp(min=1.0)
            prior_density = (prior * offdiag).sum(dim=(-2, -1)) / offdiag.sum(dim=(-2, -1)).clamp(min=1.0)
            density_loss = (mask_density - prior_density).abs().mean()
            reg_loss = fp_loss + 0.5 * fn_loss + density_loss
        else:
            denom = offdiag.sum(dim=(-2, -1)).clamp(min=1.0)
            reg_loss = ((sensor_mask * offdiag).sum(dim=(-2, -1)) / denom).mean()

        return cluster_loss + self.contrastive_lambda * reg_loss


# ── Fault-Type Expert Routing Layer ──────────────────────────────────
class CREMFPLayer(nn.Module):
    """Fault-type expert routing layer.

    No detection token. Fault-type expert tokens use hard-routed attention.
    Sensor tokens refine via self-attention and expert feedback.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_concepts: int, n_sensors: int, dropout: float,
                 leak_ratio: float = 0.2) -> None:
        super().__init__()

        # ── MFS: Macro-level Fault Semantic guidance ──
        self.feature_abstractor = FeatureAbstractor(d_model, d_model, dropout)
        self.adaln = AdaptiveLayerNorm(d_model, cond_dim=d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_sa = nn.LayerNorm(d_model)

        # ── MPS Step 1: fault-type expert tokens query sensors (hard-routed) ──
        self.concept_cross_attn = HardRoutedConceptAttention(
            d_model, n_heads, dropout,
            n_concepts, n_sensors, leak_ratio,
        )
        self.norm_c2s = nn.LayerNorm(d_model)
        self.ffn_concept = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm_c_ffn = nn.LayerNorm(d_model)

        # ── MPS Step 2: sensors query fault-type expert tokens ──
        self.cross_attn_s2c = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_s2c = nn.LayerNorm(d_model)
        self.ffn_sensor = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm_s_ffn = nn.LayerNorm(d_model)

    def forward(self, sensor_tokens: torch.Tensor,
                concept_tokens: torch.Tensor,
                group_mask: torch.Tensor):
        """
        sensor_tokens:  (B, M, d)
        concept_tokens: (B, K, d)
        group_mask:     (K, M)

        Returns:
            sensor_out, concept_out, attn_c2s, attn_s2c
        """
        # ── MFS: global conditioning + self-attention ──
        global_cond = self.feature_abstractor(sensor_tokens)
        z_cond = self.adaln(sensor_tokens, global_cond)
        z_sa, _ = self.self_attn(z_cond, z_cond, z_cond)
        sensor_tokens = self.norm_sa(z_sa + sensor_tokens)

        # ── MPS Step 1: fault-type expert tokens query sensors (hard-routed) ──
        c_repr, attn_c2s = self.concept_cross_attn(
            concept_tokens, sensor_tokens, group_mask,
        )
        concept_tokens = self.norm_c2s(c_repr + concept_tokens)
        concept_tokens = self.norm_c_ffn(concept_tokens + self.ffn_concept(concept_tokens))

        # ── MPS Step 2: sensors query fault-type expert tokens ──
        s_repr, attn_s2c = self.cross_attn_s2c(
            query=sensor_tokens, key=concept_tokens, value=concept_tokens,
        )
        sensor_tokens = self.norm_s2c(s_repr + sensor_tokens)
        sensor_tokens = self.norm_s_ffn(sensor_tokens + self.ffn_sensor(sensor_tokens))

        return sensor_tokens, concept_tokens, attn_c2s, attn_s2c


# ── Per-Fault-Type Expert Head ───────────────────────────────────────
class ConceptExpertHead(nn.Module):
    """Independent expert MLP for one fault type. Produces (risk, confidence)."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        hidden = max(16, d_model // 4)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Initialize with negative bias for conservative initial predictions
        nn.init.constant_(self.risk_head[-1].bias, -2.0)
        nn.init.constant_(self.conf_head[-1].bias, 0.0)

    def forward(self, concept_token: torch.Tensor):
        """
        concept_token: (B, d)
        Returns: risk_logit (B, 1), confidence_logit (B, 1)
        """
        return self.risk_head(concept_token), self.conf_head(concept_token)


# ── MFTransformerV3 ──────────────────────────────────────────────────
class MFTransformerV3(nn.Module):
    """Fault-Type Expert Transformer for fault prediction.

    Key innovations:
      - Hard-routed fault-type attention with leak for genuine specialization
      - Per-type expert heads with confidence for stable fusion
      - Sensor interaction layer for cross-sensor pattern capture
      - No detection token: simpler, avoids hierarchy bottleneck
    """

    def __init__(self, cfg: MFTransformerV3Cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask_channel = cfg.use_mask_channel
        self.use_norm = cfg.use_norm

        # ── Time-token Transformer encoder ──
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

        # ── Sensor query extraction ──
        self.sensor_queries = nn.Parameter(torch.randn(cfg.n_sensors, cfg.d_model) * 0.02)
        self.sensor_time_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.sensor_token_norm = nn.LayerNorm(cfg.d_model)

        # ── Sensor series residual ──
        if cfg.use_sensor_series_residual:
            n_stats = 6
            self.sensor_series_proj = nn.Sequential(
                nn.Linear(n_stats, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
        else:
            self.sensor_series_proj = None

        # ── Sensor interaction layer ──
        if cfg.use_dynamic_sensor_mask:
            self.sensor_interaction = DynamicMaskedSensorInteraction(
                cfg.d_model, cfg.n_heads, cfg.d_ff,
                cfg.n_sensors, cfg.dropout,
                temperature=cfg.sensor_mask_temperature,
                contrastive_lambda=cfg.sensor_contrastive_lambda,
                mask_mode=cfg.sensor_mask_mode,
            )
        else:
            self.sensor_interaction = SensorInteractionLayer(
                cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            )

        # ── Fault-type expert tokens (no detection token) ──
        self.concept_queries = nn.Parameter(torch.empty(cfg.n_types, cfg.d_model))
        nn.init.xavier_uniform_(self.concept_queries.data.unsqueeze(0))

        # ── Group mask (hard routing prior) ──
        group_mask = self._build_group_mask(cfg)
        self.register_buffer("group_mask", torch.from_numpy(group_mask))

        # ── Sensor group prior for contrastive loss (M, M) ──
        if cfg.use_dynamic_sensor_mask:
            sensor_prior = self._build_sensor_group_prior(cfg)
            self.register_buffer("sensor_group_prior", torch.from_numpy(sensor_prior))
        else:
            self.sensor_group_prior = None

        # ── Stacked fault-type expert routing layers ──
        self.cre_layers = nn.ModuleList([
            CREMFPLayer(
                cfg.d_model, cfg.n_heads, cfg.d_ff,
                cfg.n_types, cfg.n_sensors, cfg.dropout,
                leak_ratio=cfg.leak_ratio,
            )
            for _ in range(cfg.n_mfp_layers)
        ])

        # ── Per-fault-type expert heads ──
        self.expert_heads = nn.ModuleList([
            ConceptExpertHead(cfg.d_model, cfg.dropout)
            for _ in range(cfg.n_types)
        ])

        # ── Time pool for bypass ──
        self.time_pool_score = nn.Linear(cfg.d_model, 1, bias=False)
        bypass_dim = max(4, cfg.n_types // 2)
        self.bypass_proj = nn.Sequential(
            nn.Linear(cfg.d_model, bypass_dim),
            nn.GELU(),
        )

        # ── Global classification head ──
        head_in = cfg.d_model + bypass_dim
        self.global_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(8, head_in // 2)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(max(8, head_in // 2), cfg.output_dim),
        )

        # ── Learnable fault-type risk mix ──
        mix = min(max(float(cfg.concept_risk_mix), 1e-4), 1.0 - 1e-4)
        self.concept_risk_mix_logit = nn.Parameter(
            torch.tensor(math.log(mix / (1.0 - mix)), dtype=torch.float32)
        )

    @staticmethod
    def _build_group_mask(cfg: MFTransformerV3Cfg) -> np.ndarray:
        """Build (K, M) binary group mask: 1 = fault type's primary sensor group."""
        groups = {
            "thermal_anomaly": {"temperature"},
            "current_bias_anomaly": {"current"},
            "tx_power_anomaly": {"currentTXPower"} | {f"currentMultiTXPower{i}" for i in range(1, 5)},
            "rx_power_anomaly": {"currentRXPower"} | {f"currentMultiRXPower{i}" for i in range(1, 5)},
            "lane_imbalance": {f"currentMultiTXPower{i}" for i in range(1, 5)}
                             | {f"currentMultiRXPower{i}" for i in range(1, 5)},
        }
        mask = np.zeros((cfg.n_types, cfg.n_sensors), dtype=np.float32)
        for row, name in enumerate(cfg.concept_names):
            group = groups.get(name, set())
            for col, sensor in enumerate(cfg.sensor_names):
                if sensor in group:
                    mask[row, col] = 1.0
            # Fallback: if a fault type has no assigned sensors, attend to all
            if mask[row].sum() == 0:
                mask[row] = 1.0
        return mask

    @staticmethod
    def _build_sensor_group_prior(cfg: MFTransformerV3Cfg) -> np.ndarray:
        """Build (M, M) sensor co-occurrence prior from fault-type groups.

        Two sensors are co-occurring (prior=1) if they belong to the same
        fault-type group.  This provides guidance for the dynamic mask:
        sensors within the same group should interact more.
        """
        groups = {
            "thermal_anomaly": {"temperature"},
            "current_bias_anomaly": {"current"},
            "tx_power_anomaly": {"currentTXPower"} | {f"currentMultiTXPower{i}" for i in range(1, 5)},
            "rx_power_anomaly": {"currentRXPower"} | {f"currentMultiRXPower{i}" for i in range(1, 5)},
            "lane_imbalance": {f"currentMultiTXPower{i}" for i in range(1, 5)}
                             | {f"currentMultiRXPower{i}" for i in range(1, 5)},
        }
        M = cfg.n_sensors
        prior = np.eye(M, dtype=np.float32)  # diagonal always 1
        for name in cfg.concept_names:
            group_sensors = groups.get(name, set())
            indices = [i for i, s in enumerate(cfg.sensor_names) if s in group_sensors]
            for i in indices:
                for j in indices:
                    prior[i, j] = 1.0
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

        # ── Time encoding ──
        inp = self._pool_time_tokens(inp)
        length = inp.size(1)
        time_tokens = self.input_proj(inp)
        time_tokens = time_tokens + self.pos_embedding[:, :length, :]
        time_tokens = self.time_encoder(time_tokens)

        # ── Sensor query extraction ──
        batch = x.size(0)
        sensor_queries = self.sensor_queries.unsqueeze(0).expand(batch, -1, -1)
        sensor_tokens, sensor_time_attn = self.sensor_time_attn(
            query=sensor_queries,
            key=time_tokens,
            value=time_tokens,
        )

        # ── Sensor series residual ──
        if self.sensor_series_proj is not None:
            sensor_vals = x_model.transpose(1, 2)
            sensor_mask = mask.transpose(1, 2)
            denom = sensor_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            s_mean = (sensor_vals * sensor_mask).sum(dim=-1, keepdim=True) / denom
            s_var = ((sensor_vals - s_mean) ** 2 * sensor_mask).sum(dim=-1, keepdim=True) / denom
            s_std = torch.sqrt(s_var + 1e-5)
            s_min = sensor_vals.masked_fill(sensor_mask < 0.5, 1e9).min(dim=-1, keepdim=True).values
            s_max = sensor_vals.masked_fill(sensor_mask < 0.5, -1e9).max(dim=-1, keepdim=True).values
            s_last = sensor_vals[:, :, -1:]
            s_first = sensor_vals[:, :, :1]
            s_slope = (s_last - s_first) / sensor_vals.size(-1)
            stats = torch.cat([s_mean, s_std, s_min, s_max, s_slope, s_last], dim=-1)
            sensor_tokens = sensor_tokens + self.sensor_series_proj(stats)
        sensor_tokens = self.sensor_token_norm(sensor_tokens)

        # ── Sensor interaction (cross-sensor correlations) ──
        sensor_contrastive_loss = sensor_tokens.new_tensor(0.0)
        discovered_sensor_mask = None
        if self.cfg.use_dynamic_sensor_mask:
            sensor_tokens, discovered_sensor_mask, sensor_contrastive_loss = (
                self.sensor_interaction(sensor_tokens, self.sensor_group_prior)
            )
        else:
            sensor_tokens = self.sensor_interaction(sensor_tokens)

        # ── Fault-type expert routing layers ──
        concept_tokens = self.concept_queries.unsqueeze(0).expand(batch, -1, -1)
        attn_c2s_last = None
        attn_s2c_last = None
        for layer in self.cre_layers:
            sensor_tokens, concept_tokens, attn_c2s_last, attn_s2c_last = layer(
                sensor_tokens, concept_tokens, self.group_mask,
            )

        # ── Per-fault-type expert outputs ──
        K = self.cfg.n_types
        risk_logits = []
        conf_logits = []
        for k in range(K):
            r_k, c_k = self.expert_heads[k](concept_tokens[:, k, :])  # (B, 1) each
            risk_logits.append(r_k)
            conf_logits.append(c_k)
        risk_logits = torch.cat(risk_logits, dim=-1)  # (B, K)
        conf_logits = torch.cat(conf_logits, dim=-1)  # (B, K)

        concept_risk_probs = torch.sigmoid(risk_logits)   # (B, K)
        concept_confidence = torch.softmax(conf_logits, dim=-1)  # (B, K) normalized

        # Fault-type expert fusion
        if self.cfg.fusion_mode == "noisy_or":
            # Noisy-OR: P(any fault type) = 1 - prod(1 - risk_k)
            concept_risk = 1.0 - torch.prod(
                (1.0 - concept_risk_probs).clamp(min=1e-6, max=1.0), dim=-1
            )
        else:
            # Confidence-weighted average
            concept_risk = (concept_confidence * concept_risk_probs).sum(dim=-1)  # (B,)

        # ── Global path: sensor pool + bypass ──
        pooled = sensor_tokens.mean(dim=1)
        time_alpha = torch.softmax(self.time_pool_score(time_tokens), dim=1)
        time_context = (time_tokens * time_alpha).sum(dim=1)
        bypass = self.bypass_proj(time_context) * self.cfg.bypass_ratio
        global_logit = self.global_head(torch.cat([pooled, bypass], dim=-1))
        if self.cfg.output_dim == 1:
            global_logit = global_logit.squeeze(-1)

        # ── Fusion: global + fault-type expert path ──
        global_prob = torch.sigmoid(global_logit)
        concept_mix = torch.sigmoid(self.concept_risk_mix_logit)
        concept_risk_for_fusion = concept_risk if self.cfg.output_dim == 1 else concept_risk.unsqueeze(-1)
        fused_prob = (1.0 - concept_mix) * global_prob + concept_mix * concept_risk_for_fusion
        logit = torch.logit(fused_prob.clamp(min=1e-6, max=1.0 - 1e-6))

        # ── Sensor importance: concept-weighted attention ──
        # Use concept_risk_probs as weights (not confidence)
        concept_gate = concept_risk_probs  # (B, K)
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

        # type_logits for compatibility: use risk_logits as the "type logits"
        type_logits = risk_logits

        return (
            logit,              # 0: binary risk logit
            sensor_importance,  # 1: sensor importance
            attn_c2s_last,      # 2: concept→sensor attention (B, K, M)
            attn_s2c_last,      # 3: sensor→concept attention (B, M, K)
            type_logits,        # 4: type logits = per-expert risk logits (B, K)
            sensor_time_attn,   # 5: sensor-time attention (B, M, T')
            sensor_contrastive_loss,  # 6: CATCH-style contrastive loss (scalar)
            discovered_sensor_mask,   # 7: dynamic sensor mask (B, M, M) or None
        )
