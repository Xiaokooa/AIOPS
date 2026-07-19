"""Interpretable FITS: Frequency-domain classifier with dual explainability.

Provides:
  1. Sensor importance weights (which sensors matter for the prediction)
  2. Frequency importance weights (which frequency components drive the decision)

Architecture:
  Input (B, L, N) → Mask-aware RIN → FFT → LPF → Freq Linear →
  → Sensor Attention Pooling (→ sensor_weights) +
    Frequency Attention Pooling (→ freq_weights)
  → Classification Head → logit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InterpFITSCfg:
    seq_len: int = 288
    n_sensors: int = 12
    use_mask_channel: bool = True
    cut_freq: int = 30         # dominant frequencies to keep
    d_hidden: int = 128        # internal hidden dim
    dropout: float = 0.2
    individual: bool = False   # per-channel freq layer
    sensor_names: List[str] = field(default_factory=list)
    output_dim: int = 1


class InterpFITSClassifier(nn.Module):
    """FITS with L1-style sensor attention + frequency attention.

    Forward returns:
        (logit, sensor_weights, freq_weights, freq_features)
    where:
        - sensor_weights: (B, N) — per-sensor importance (sums to 1)
        - freq_weights:   (B, F) — per-frequency importance (sums to 1)
        - freq_features:  (B, F, N) — magnitude spectrum after freq linear (for viz)
    """

    def __init__(self, cfg: InterpFITSCfg):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.n_sensors = cfg.n_sensors
        self.dominance_freq = cfg.cut_freq

        # Frequency-domain linear (core FITS layer)
        if cfg.individual:
            self.freq_layer = nn.ModuleList([
                nn.Linear(cfg.cut_freq, cfg.cut_freq).to(torch.cfloat)
                for _ in range(cfg.n_sensors)
            ])
        else:
            self.freq_layer = nn.Linear(cfg.cut_freq, cfg.cut_freq).to(torch.cfloat)

        # Project complex freq features to real representation per sensor
        # Each sensor gets a feature vector of size cut_freq*2 (real+imag)
        # We project to d_hidden for attention computation
        self.sensor_proj = nn.Sequential(
            nn.Linear(cfg.cut_freq * 2, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Sensor attention: learned query over sensor dimension
        self.sensor_attn_query = nn.Linear(cfg.d_hidden, 1, bias=False)

        # Frequency attention: learned query over frequency dimension
        # Each freq bin gets a feature vector of size n_sensors*2 (real+imag across sensors)
        self.freq_proj = nn.Sequential(
            nn.Linear(cfg.n_sensors * 2, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.freq_attn_query = nn.Linear(cfg.d_hidden, 1, bias=False)

        # Classification head: takes sensor-pooled + freq-pooled features
        head_in = cfg.d_hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_in // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_in // 2, cfg.output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x    : (B, L, N)
        mask : (B, L, N)
        Returns: (logit, sensor_weights, freq_weights, freq_magnitude)
        """
        B, L, N = x.shape

        # ── Mask-aware RIN ──
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        x_mean = (x * mask).sum(dim=1, keepdim=True) / denom
        x_centered = (x - x_mean) * mask
        x_var = (x_centered ** 2 * mask).sum(dim=1, keepdim=True) / denom + 1e-5
        x_normed = x_centered / torch.sqrt(x_var)

        # ── FFT + LPF ──
        specx = torch.fft.rfft(x_normed, dim=1)           # (B, L//2+1, N)
        low_specx = specx[:, :self.dominance_freq, :]       # (B, F, N)

        # ── Frequency linear transformation ──
        if self.cfg.individual:
            out_spec = torch.zeros_like(low_specx)
            for i, layer in enumerate(self.freq_layer):
                out_spec[:, :, i] = layer(low_specx[:, :, i])
        else:
            out_spec = self.freq_layer(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        # out_spec: (B, F, N) complex

        # ── Extract real features ──
        feat_real = out_spec.real   # (B, F, N)
        feat_imag = out_spec.imag   # (B, F, N)
        freq_magnitude = torch.sqrt(feat_real ** 2 + feat_imag ** 2)  # (B, F, N)

        # ── Sensor attention pathway ──
        # Per-sensor feature: concat real+imag across freq bins → (B, N, F*2)
        sensor_feat = torch.cat([
            feat_real.permute(0, 2, 1),   # (B, N, F)
            feat_imag.permute(0, 2, 1),   # (B, N, F)
        ], dim=-1)  # (B, N, F*2)
        sensor_repr = self.sensor_proj(sensor_feat)         # (B, N, d_hidden)
        sensor_scores = self.sensor_attn_query(sensor_repr).squeeze(-1)  # (B, N)
        sensor_weights = F.softmax(sensor_scores, dim=-1)   # (B, N)
        sensor_pooled = (sensor_repr * sensor_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_hidden)

        # ── Frequency attention pathway ──
        # Per-frequency feature: concat real+imag across sensors → (B, F, N*2)
        freq_feat = torch.cat([feat_real, feat_imag], dim=-1)  # (B, F, N*2)
        freq_repr = self.freq_proj(freq_feat)               # (B, F, d_hidden)
        freq_scores = self.freq_attn_query(freq_repr).squeeze(-1)  # (B, F)
        freq_weights = F.softmax(freq_scores, dim=-1)       # (B, F)
        freq_pooled = (freq_repr * freq_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_hidden)

        # ── Classification ──
        combined = torch.cat([sensor_pooled, freq_pooled], dim=-1)  # (B, d_hidden*2)
        logit = self.head(combined)
        if self.cfg.output_dim == 1:
            logit = logit.squeeze(-1)                      # (B,)

        return logit, sensor_weights, freq_weights, freq_magnitude
