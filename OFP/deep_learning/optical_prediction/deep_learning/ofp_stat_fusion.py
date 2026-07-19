from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class OFPStatFusionCfg:
    n_sensors: int
    stat_count: int = 10
    d_hidden: int = 96
    dropout: float = 0.2
    output_dim: int = 1


def window_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Keep high-order statistics out of fp16/bf16. Extreme normalized optical
    # readings can overflow z**4 under autocast and poison the whole model.
    x = x.float()
    mask = mask.float()
    valid = mask > 0
    count = mask.sum(dim=1).clamp(min=1.0)
    has_valid = valid.any(dim=1)

    mean = (x * mask).sum(dim=1) / count
    centered = (x - mean.unsqueeze(1)) * mask
    var = centered.square().sum(dim=1) / count
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
    z = z.clamp(min=-20.0, max=20.0)
    skew = (z.pow(3) * mask).sum(dim=1) / count
    kurt = (z.pow(4) * mask).sum(dim=1) / count

    stats = torch.stack(
        [mean, std, min_v, max_v, value_range, last_v, delta, slope, skew, kurt],
        dim=-1,
    )
    stats = torch.nan_to_num(stats, nan=0.0, posinf=50.0, neginf=-50.0)
    stats = stats.clamp(min=-50.0, max=50.0)
    return stats.flatten(start_dim=1)


def primary_logit(output) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim > 1 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    return output


class OFPStatFusionWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, cfg: OFPStatFusionCfg) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        stat_dim = cfg.n_sensors * cfg.stat_count
        self.output_dim = int(cfg.output_dim)
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(cfg.d_hidden + self.output_dim),
            nn.Linear(cfg.d_hidden + self.output_dim, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, self.output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base_logit = primary_logit(self.backbone(x, mask))
        if base_logit.ndim == 1:
            base_logit = base_logit.unsqueeze(-1)
        stat_emb = self.stat_proj(window_stats(x, mask))
        out = self.fusion_head(torch.cat([base_logit, stat_emb], dim=-1))
        if self.output_dim == 1:
            out = out.squeeze(-1)
        return out


def wrap_with_ofp_stat_fusion(backbone: nn.Module, n_sensors: int, d_hidden: int = 96, dropout: float = 0.2):
    output_dim = int(getattr(getattr(backbone, "cfg", None), "output_dim", 1))
    cfg = OFPStatFusionCfg(n_sensors=n_sensors, d_hidden=d_hidden, dropout=dropout, output_dim=output_dim)
    return OFPStatFusionWrapper(backbone, cfg), asdict(cfg)
