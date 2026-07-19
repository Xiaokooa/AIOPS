from __future__ import annotations

from dataclasses import asdict

import torch.nn as nn

from OFP.deep_learning.optical_prediction.deep_learning.moderntcn_wrapper import ModernTCNCfg, ModernTCNClassifier
from OFP.deep_learning.optical_prediction.deep_learning.ofp_stat_fusion import wrap_with_ofp_stat_fusion
from OFP.deep_learning.official.ofp_formal_protocol.protocol import DEFAULT_HORIZON_HOURS


def build_model(seq_len: int, n_sensors: int) -> tuple[nn.Module, dict, bool]:
    cfg = ModernTCNCfg(
        seq_len=int(seq_len),
        n_sensors=int(n_sensors),
        patch_size=32,
        patch_stride=16,
        num_blocks=(2,),
        large_size=(31,),
        small_size=(5,),
        dims=(64,),
        dw_dims=(64,),
        dropout=0.2,
        head_dropout=0.2,
        class_dropout=0.1,
        use_mask_channel=True,
        output_dim=len(DEFAULT_HORIZON_HOURS) + 1,
    )
    backbone = ModernTCNClassifier(cfg)
    model, fusion_cfg = wrap_with_ofp_stat_fusion(backbone, n_sensors=n_sensors, d_hidden=96, dropout=0.2)
    return model, {"backbone": asdict(cfg), "ofp_stat_fusion": fusion_cfg}, False
