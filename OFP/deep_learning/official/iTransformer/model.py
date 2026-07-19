from __future__ import annotations

from dataclasses import asdict

import torch.nn as nn

from OFP.deep_learning.optical_prediction.deep_learning.models import ITransformerCfg, ITransformerClassifier
from OFP.deep_learning.optical_prediction.deep_learning.ofp_stat_fusion import wrap_with_ofp_stat_fusion
from OFP.deep_learning.official.ofp_formal_protocol.protocol import DEFAULT_HORIZON_HOURS


def build_model(seq_len: int, n_sensors: int) -> tuple[nn.Module, dict, bool]:
    cfg = ITransformerCfg(
        seq_len=int(seq_len),
        n_sensors=int(n_sensors),
        d_model=96,
        n_heads=4,
        e_layers=2,
        d_ff=192,
        dropout=0.2,
        output_dim=len(DEFAULT_HORIZON_HOURS) + 1,
    )
    backbone = ITransformerClassifier(cfg)
    model, fusion_cfg = wrap_with_ofp_stat_fusion(backbone, n_sensors=n_sensors, d_hidden=96, dropout=0.2)
    return model, {"backbone": asdict(cfg), "ofp_stat_fusion": fusion_cfg}, False
