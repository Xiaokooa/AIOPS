from __future__ import annotations
from dataclasses import dataclass,field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendor import ROOT,load_layers
_layers=load_layers(ROOT/'PatchTST/PatchTST-main/PatchTST_supervised',['layers.PatchTST_layers','layers.RevIN','layers.PatchTST_backbone'])
TSTiEncoder=_layers['layers.PatchTST_backbone'].TSTiEncoder
RevIN=_layers['layers.RevIN'].RevIN

@dataclass
class PatchTSTCfg:
    seq_len: int = 288
    n_sensors: int = 12
    patch_len: int = 16
    stride: int = 8
    use_mask_channel: bool = True
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    attn_dropout: float = 0.0
    revin: bool = True
    output_dim: int = 1

class PatchTSTClassifier(nn.Module):
    """Channel-independent patched-Transformer for binary failure prediction."""

    def __init__(self, cfg: PatchTSTCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask = cfg.use_mask_channel
        self.n_vars = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)
        patch_num = int((cfg.seq_len - cfg.patch_len) / cfg.stride + 1)
        self.padding_patch = 'end'
        patch_num += 1
        self.patch_num = patch_num
        self.pad = nn.ReplicationPad1d((0, cfg.stride))
        self.revin = cfg.revin
        if self.revin:
            self.revin_layer = RevIN(cfg.n_sensors, affine=True, subtract_last=False)
        self.backbone = TSTiEncoder(c_in=self.n_vars, patch_num=patch_num, patch_len=cfg.patch_len, n_layers=cfg.e_layers, d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff, attn_dropout=cfg.attn_dropout, dropout=cfg.dropout)
        head_nf = self.n_vars * cfg.d_model * patch_num
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.LayerNorm(head_nf), nn.Dropout(cfg.dropout), nn.Linear(head_nf, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.output_dim))

    def _patchify(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pad(z)
        z = z.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)
        return z.permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.revin_layer(x, 'norm')
        if self.use_mask:
            inp = torch.cat([x, mask], dim=-1)
        else:
            inp = x
        inp = inp.permute(0, 2, 1)
        z = self._patchify(inp)
        z = self.backbone(z)
        logit = self.head(z)
        if self.cfg.output_dim == 1:
            logit = logit.squeeze(-1)
        return logit
