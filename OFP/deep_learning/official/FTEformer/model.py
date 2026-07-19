from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from OFP.deep_learning.optical_prediction.deep_learning.mf_transformer_v3 import MFTransformerV3, MFTransformerV3Cfg
from OFP.deep_learning.optical_prediction.deep_learning.ofp_stat_fusion import primary_logit, window_stats
from OFP.deep_learning.official.ofp_formal_protocol.protocol import DEFAULT_HORIZON_HOURS


class FTEformerOfficialModel(nn.Module):
    """FTEformer with OFP statistical fusion while preserving diagnostic outputs."""

    def __init__(
        self,
        backbone: MFTransformerV3,
        n_sensors: int,
        hidden: int = 96,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = int(output_dim)
        stat_dim = int(n_sensors) * 10
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden + 1),
            nn.Linear(hidden + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        outputs = self.backbone(x, mask)
        base = primary_logit(outputs).unsqueeze(-1)
        stats = self.stat_proj(window_stats(x, mask))
        fused_logit = self.fusion_head(torch.cat([base, stats], dim=-1))
        if self.output_dim == 1:
            fused_logit = fused_logit.squeeze(-1)
        return (fused_logit, *outputs[1:])


def build_model(seq_len: int, n_sensors: int) -> tuple[nn.Module, dict, bool]:
    cfg = MFTransformerV3Cfg(
        seq_len=int(seq_len),
        n_sensors=int(n_sensors),
        d_model=96,
        n_heads=4,
        e_layers=2,
        d_ff=192,
        n_mfp_layers=2,
        time_pool_size=max(1, int(seq_len) // 48),
        use_dynamic_sensor_mask=True,
        sensor_mask_mode="soft",
        sensor_mask_temperature=0.5,
        sensor_contrastive_lambda=0.3,
        sensor_contrastive_weight=0.03,
        dropout=0.2,
    )
    backbone = MFTransformerV3(cfg)
    output_dim = len(DEFAULT_HORIZON_HOURS) + 1
    model = FTEformerOfficialModel(backbone, n_sensors=n_sensors, hidden=96, dropout=0.2, output_dim=output_dim)
    model_cfg = {
        "backbone": asdict(cfg),
        "ofp_stat_fusion": {
            "n_sensors": int(n_sensors),
            "stat_count": 10,
            "d_hidden": 96,
            "dropout": 0.2,
            "output_dim": output_dim,
        },
        "sensor_contrastive_weight": float(cfg.sensor_contrastive_weight),
    }
    return model, model_cfg, True
