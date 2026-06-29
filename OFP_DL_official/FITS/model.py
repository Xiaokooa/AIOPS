from __future__ import annotations

from dataclasses import asdict

import torch.nn as nn

from model.Optical_prediction_model.deep_learning.interpretable_fits import InterpFITSCfg, InterpFITSClassifier
from model.Optical_prediction_model.deep_learning.ofp_stat_fusion import wrap_with_ofp_stat_fusion
from OFP_DL_official.ofp_formal_protocol.protocol import DEFAULT_HORIZON_HOURS


def build_model(seq_len: int, n_sensors: int) -> tuple[nn.Module, dict, bool]:
    cfg = InterpFITSCfg(
        seq_len=int(seq_len),
        n_sensors=int(n_sensors),
        use_mask_channel=True,
        cut_freq=min(30, int(seq_len) // 2 + 1),
        d_hidden=96,
        dropout=0.2,
        individual=False,
        output_dim=len(DEFAULT_HORIZON_HOURS) + 1,
    )
    backbone = InterpFITSClassifier(cfg)
    model, fusion_cfg = wrap_with_ofp_stat_fusion(backbone, n_sensors=n_sensors, d_hidden=96, dropout=0.2)
    return model, {"backbone": asdict(cfg), "ofp_stat_fusion": fusion_cfg}, False
