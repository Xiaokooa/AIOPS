"""Classification wrappers around iTransformer / PatchTST.

These models were originally designed for forecasting. We:
  * keep the encoder unchanged,
  * concat a per-timestep `mask` channel as auxiliary covariates,
  * replace the forecasting projector with a binary classification head
    that pools encoder tokens.

Input shape:  (B, L, N)  -- raw normalized time series
Mask shape :  (B, L, N)  -- same shape, 1 where valid
Output     :  logits (B,) or (B, output_dim)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# Make iTransformer importable without polluting global path permanently.
ITF_ROOT = Path(__file__).resolve().parents[3] / "model" / "iTransformer-main"
if str(ITF_ROOT) not in sys.path:
    sys.path.insert(0, str(ITF_ROOT))

# These imports come from iTransformer-main/{layers, model}/.
from layers.Transformer_EncDec import Encoder, EncoderLayer  # type: ignore
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # type: ignore
from layers.Embed import DataEmbedding_inverted  # type: ignore


@dataclass
class ITransformerCfg:
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
    output_attention: bool = False
    embed: str = "fixed"
    freq: str = "h"
    output_dim: int = 1


class ITransformerClassifier(nn.Module):
    """iTransformer-style encoder + binary classification head.

    DataEmbedding_inverted turns the (B, L, N) input into N variate tokens of
    dim d_model. We pool across the variate tokens (mean) and project to a
    single logit.
    """

    def __init__(self, cfg: ITransformerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask_channel = cfg.use_mask_channel
        # Effective channel count = sensors (+ sensors mask if enabled)
        self.n_channels = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)
        self.use_norm = cfg.use_norm

        self.enc_embedding = DataEmbedding_inverted(
            cfg.seq_len, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            cfg.factor,
                            attention_dropout=cfg.dropout,
                            output_attention=False,
                        ),
                        cfg.d_model,
                        cfg.n_heads,
                    ),
                    cfg.d_model,
                    cfg.d_ff,
                    dropout=cfg.dropout,
                    activation=cfg.activation,
                )
                for _ in range(cfg.e_layers)
            ],
            norm_layer=nn.LayerNorm(cfg.d_model),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, L, N) normalized values, NaN already zero-filled.
        mask : (B, L, N) 1 = valid, 0 = missing (after short-gap interpolation).
        return logits (B,) for binary mode or (B, output_dim).
        """
        if self.use_mask_channel:
            inp = torch.cat([x, mask], dim=-1)            # (B, L, 2N)
        else:
            inp = x
        if self.use_norm:
            # Per-channel z-score across L on valid entries (mask-aware).
            # The dataset already z-scored sensor channels globally; this
            # extra layer normalizes per-window which iTransformer relies on.
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (x * mask).sum(dim=1, keepdim=True) / denom
            var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
            std = torch.sqrt(var + 1e-5)
            x_norm = (x - mean) / std * mask
            if self.use_mask_channel:
                inp = torch.cat([x_norm, mask], dim=-1)
            else:
                inp = x_norm

        # DataEmbedding_inverted expects (B, L, channels); produces (B, channels, d_model)
        enc_out = self.enc_embedding(inp, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # Pool over sensor tokens only (first n_sensors channels), ignoring mask tokens.
        pooled = enc_out[:, :self.cfg.n_sensors, :].mean(dim=1)  # (B, d_model)
        logit = self.head(pooled)
        if self.cfg.output_dim == 1:
            logit = logit.squeeze(-1)                    # (B,)
        return logit


@dataclass
class ITransformerOFPCfg(ITransformerCfg):
    stat_count: int = 10


class ITransformerOFPClassifier(ITransformerClassifier):
    """iTransformer adapted to OFP first-warning classification.

    The encoder still operates on temporal DOM windows. The head fuses an
    attention-pooled iTransformer representation with window statistics that
    mirror reliability features used by tree baselines.
    """

    def __init__(self, cfg: ITransformerOFPCfg) -> None:
        super().__init__(cfg)
        stat_dim = cfg.n_sensors * cfg.stat_count
        self.sensor_pool = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1, bias=False),
        )
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.output_dim),
        )

    def _window_stats(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask > 0
        count = mask.sum(dim=1).clamp(min=1.0)
        has_valid = valid.any(dim=1)

        mean = (x * mask).sum(dim=1) / count
        centered = (x - mean.unsqueeze(1)) * mask
        var = (centered.square()).sum(dim=1) / count
        std = torch.sqrt(var + 1e-5)

        inf = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
        min_v = x.masked_fill(~valid, inf).amin(dim=1)
        max_v = x.masked_fill(~valid, -inf).amax(dim=1)
        zeros = torch.zeros_like(mean)
        min_v = torch.where(has_valid, min_v, zeros)
        max_v = torch.where(has_valid, max_v, zeros)
        value_range = max_v - min_v

        bsz, seq_len, n_sensors = x.shape
        idx = torch.arange(seq_len, device=x.device).view(1, seq_len, 1).expand(bsz, seq_len, n_sensors)
        first_idx = torch.where(valid, idx, torch.full_like(idx, seq_len)).amin(dim=1).clamp(max=seq_len - 1)
        last_idx = torch.where(valid, idx, torch.full_like(idx, -1)).amax(dim=1).clamp(min=0)
        first_v = x.gather(1, first_idx.unsqueeze(1)).squeeze(1)
        last_v = x.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        first_v = torch.where(has_valid, first_v, zeros)
        last_v = torch.where(has_valid, last_v, zeros)
        delta = last_v - first_v
        span = (last_idx - first_idx).clamp(min=1).to(x.dtype)
        slope = delta / span

        z = centered / std.unsqueeze(1).clamp(min=1e-5)
        skew = (z.pow(3) * mask).sum(dim=1) / count
        kurt = (z.pow(4) * mask).sum(dim=1) / count

        stats = torch.stack(
            [mean, std, min_v, max_v, value_range, last_v, delta, slope, skew, kurt],
            dim=-1,
        )
        return stats.flatten(start_dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        stat_feat = self._window_stats(x, mask)

        if self.use_mask_channel:
            inp = torch.cat([x, mask], dim=-1)
        else:
            inp = x
        if self.use_norm:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (x * mask).sum(dim=1, keepdim=True) / denom
            var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
            std = torch.sqrt(var + 1e-5)
            x_norm = (x - mean) / std * mask
            if self.use_mask_channel:
                inp = torch.cat([x_norm, mask], dim=-1)
            else:
                inp = x_norm

        enc_out = self.enc_embedding(inp, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        sensor_tokens = enc_out[:, :self.cfg.n_sensors, :]
        attn = torch.softmax(self.sensor_pool(sensor_tokens).squeeze(-1), dim=1)
        pooled = (sensor_tokens * attn.unsqueeze(-1)).sum(dim=1)
        stat_emb = self.stat_proj(stat_feat)
        logit = self.head(torch.cat([pooled, stat_emb], dim=-1))
        if self.cfg.output_dim == 1:
            logit = logit.squeeze(-1)
        return logit
