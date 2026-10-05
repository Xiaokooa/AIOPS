from __future__ import annotations
from dataclasses import dataclass,field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FITSCfg:
    seq_len: int = 288
    n_sensors: int = 12
    use_mask_channel: bool = True
    cut_freq: int = 30
    d_hidden: int = 128
    dropout: float = 0.2
    individual: bool = False
    sensor_names: List[str] = field(default_factory=list)
    output_dim: int = 1

class FITSClassifier(nn.Module):
    """FITS with L1-style sensor attention + frequency attention.

    Forward returns:
        (logit, sensor_weights, freq_weights, freq_features)
    where:
        - sensor_weights: (B, N) — per-sensor importance (sums to 1)
        - freq_weights:   (B, F) — per-frequency importance (sums to 1)
        - freq_features:  (B, F, N) — magnitude spectrum after freq linear (for viz)
    """

    def __init__(self, cfg: FITSCfg):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.n_sensors = cfg.n_sensors
        self.dominance_freq = cfg.cut_freq
        if cfg.individual:
            self.freq_layer = nn.ModuleList([nn.Linear(cfg.cut_freq, cfg.cut_freq).to(torch.cfloat) for _ in range(cfg.n_sensors)])
        else:
            self.freq_layer = nn.Linear(cfg.cut_freq, cfg.cut_freq).to(torch.cfloat)
        self.sensor_proj = nn.Sequential(nn.Linear(cfg.cut_freq * 2, cfg.d_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.sensor_attn_query = nn.Linear(cfg.d_hidden, 1, bias=False)
        self.freq_proj = nn.Sequential(nn.Linear(cfg.n_sensors * 2, cfg.d_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.freq_attn_query = nn.Linear(cfg.d_hidden, 1, bias=False)
        head_in = cfg.d_hidden * 2
        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, head_in // 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(head_in // 2, cfg.output_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x    : (B, L, N)
        mask : (B, L, N)
        Returns: (logit, sensor_weights, freq_weights, freq_magnitude)
        """
        B, L, N = x.shape
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        x_mean = (x * mask).sum(dim=1, keepdim=True) / denom
        x_centered = (x - x_mean) * mask
        x_var = (x_centered ** 2 * mask).sum(dim=1, keepdim=True) / denom + 1e-05
        x_normed = x_centered / torch.sqrt(x_var)
        specx = torch.fft.rfft(x_normed, dim=1)
        low_specx = specx[:, :self.dominance_freq, :]
        if self.cfg.individual:
            out_spec = torch.zeros_like(low_specx)
            for i, layer in enumerate(self.freq_layer):
                out_spec[:, :, i] = layer(low_specx[:, :, i])
        else:
            out_spec = self.freq_layer(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        feat_real = out_spec.real
        feat_imag = out_spec.imag
        freq_magnitude = torch.sqrt(feat_real ** 2 + feat_imag ** 2)
        sensor_feat = torch.cat([feat_real.permute(0, 2, 1), feat_imag.permute(0, 2, 1)], dim=-1)
        sensor_repr = self.sensor_proj(sensor_feat)
        sensor_scores = self.sensor_attn_query(sensor_repr).squeeze(-1)
        sensor_weights = F.softmax(sensor_scores, dim=-1)
        sensor_pooled = (sensor_repr * sensor_weights.unsqueeze(-1)).sum(dim=1)
        freq_feat = torch.cat([feat_real, feat_imag], dim=-1)
        freq_repr = self.freq_proj(freq_feat)
        freq_scores = self.freq_attn_query(freq_repr).squeeze(-1)
        freq_weights = F.softmax(freq_scores, dim=-1)
        freq_pooled = (freq_repr * freq_weights.unsqueeze(-1)).sum(dim=1)
        combined = torch.cat([sensor_pooled, freq_pooled], dim=-1)
        logit = self.head(combined)
        if self.cfg.output_dim == 1:
            logit = logit.squeeze(-1)
        return (logit, sensor_weights, freq_weights, freq_magnitude)
