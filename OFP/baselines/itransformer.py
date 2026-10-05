from __future__ import annotations
from dataclasses import dataclass,field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendor import ROOT,load_layers
_layers=load_layers(ROOT/'iTransformer',['utils.masking','layers.Transformer_EncDec','layers.SelfAttention_Family','layers.Embed'])
Encoder=_layers['layers.Transformer_EncDec'].Encoder
EncoderLayer=_layers['layers.Transformer_EncDec'].EncoderLayer
FullAttention=_layers['layers.SelfAttention_Family'].FullAttention
AttentionLayer=_layers['layers.SelfAttention_Family'].AttentionLayer
DataEmbedding_inverted=_layers['layers.Embed'].DataEmbedding_inverted

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
    activation: str = 'gelu'
    factor: int = 1
    use_norm: bool = True
    output_attention: bool = False
    embed: str = 'fixed'
    freq: str = 'h'
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
        self.n_channels = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)
        self.use_norm = cfg.use_norm
        self.enc_embedding = DataEmbedding_inverted(cfg.seq_len, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout)
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, cfg.factor, attention_dropout=cfg.dropout, output_attention=False), cfg.d_model, cfg.n_heads), cfg.d_model, cfg.d_ff, dropout=cfg.dropout, activation=cfg.activation) for _ in range(cfg.e_layers)], norm_layer=nn.LayerNorm(cfg.d_model))
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model // 2, cfg.output_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, L, N) normalized values, NaN already zero-filled.
        mask : (B, L, N) 1 = valid, 0 = missing (after short-gap interpolation).
        return logits (B,) for binary mode or (B, output_dim).
        """
        if self.use_mask_channel:
            inp = torch.cat([x, mask], dim=-1)
        else:
            inp = x
        if self.use_norm:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (x * mask).sum(dim=1, keepdim=True) / denom
            var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
            std = torch.sqrt(var + 1e-05)
            x_norm = (x - mean) / std * mask
            if self.use_mask_channel:
                inp = torch.cat([x_norm, mask], dim=-1)
            else:
                inp = x_norm
        enc_out = self.enc_embedding(inp, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        pooled = enc_out[:, :self.cfg.n_sensors, :].mean(dim=1)
        logit = self.head(pooled)
        if self.cfg.output_dim == 1:
            logit = logit.squeeze(-1)
        return logit
