from __future__ import annotations
from dataclasses import dataclass,field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendor import ROOT,load_layers
_layers=load_layers(ROOT/'ModernTCN/ModernTCN-classification',['layers.RevIN','models.ModernTCN_Layer','models.ModernTCN'])
OfficialModernTCN=_layers['models.ModernTCN'].ModernTCN

@dataclass
class ModernTCNCfg:
    seq_len: int = 288
    n_sensors: int = 12
    patch_size: int = 16
    patch_stride: int = 8
    downsample_ratio: int = 2
    ffn_ratio: int = 1
    num_blocks: tuple[int, ...] = (2,)
    large_size: tuple[int, ...] = (31,)
    small_size: tuple[int, ...] = (5,)
    dims: tuple[int, ...] = (64,)
    dw_dims: tuple[int, ...] = (64,)
    dropout: float = 0.2
    head_dropout: float = 0.2
    class_dropout: float = 0.1
    use_mask_channel: bool = True
    use_multi_scale: bool = False
    small_kernel_merged: bool = False
    revin: bool = False
    output_dim: int = 1

class ModernTCNClassifier(nn.Module):
    """Official ModernTCN classification backbone with a binary logit head."""

    def __init__(self, cfg: ModernTCNCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask = cfg.use_mask_channel
        self.n_vars = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)
        self.model = OfficialModernTCN(task_name='classification', patch_size=cfg.patch_size, patch_stride=cfg.patch_stride, stem_ratio=1, downsample_ratio=cfg.downsample_ratio, ffn_ratio=cfg.ffn_ratio, num_blocks=list(cfg.num_blocks), large_size=list(cfg.large_size), small_size=list(cfg.small_size), dims=list(cfg.dims), dw_dims=list(cfg.dw_dims), nvars=self.n_vars, small_kernel_merged=cfg.small_kernel_merged, backbone_dropout=cfg.dropout, head_dropout=cfg.head_dropout, use_multi_scale=cfg.use_multi_scale, revin=cfg.revin, affine=True, subtract_last=False, freq=None, seq_len=cfg.seq_len, c_in=(self.n_vars,), individual=False, target_window=1, class_drop=cfg.class_dropout, class_num=cfg.output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_mask:
            x = torch.cat([x, mask], dim=-1)
        x = x.permute(0, 2, 1)
        out = self.model(x)
        if self.cfg.output_dim == 1:
            out = out.squeeze(-1)
        return out
