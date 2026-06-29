from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from OFP_DL_official.ofp_formal_protocol.protocol import DEFAULT_HORIZON_HOURS, PRIMARY_HORIZON_HOURS
from model.Optical_prediction_model.deep_learning.patchtst_wrapper import PatchTSTCfg, PatchTSTClassifier


class SegmentHorizonWrapper(nn.Module):
    def __init__(self, backbone: PatchTSTClassifier, pred_len: int, n_horizons: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pred_len = int(pred_len)
        self.n_horizons = int(n_horizons)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x, mask)
        return logits.reshape(logits.shape[0], self.pred_len, self.n_horizons)


def build_segment_model(input_len: int, pred_len: int, n_sensors: int) -> tuple[nn.Module, dict]:
    """PatchTST in sequence-forecasting shape with auxiliary horizon heads.

    The model consumes a historical segment and emits one logit per future
    target timestamp and horizon. OFP compatibility is handled by the runner,
    which stitches the primary horizon logits back into per-timestamp prediction
    files.
    """

    horizon_hours = tuple(DEFAULT_HORIZON_HOURS)
    cfg = PatchTSTCfg(
        seq_len=int(input_len),
        n_sensors=int(n_sensors),
        d_model=96,
        n_heads=4,
        e_layers=2,
        d_ff=192,
        dropout=0.2,
        output_dim=int(pred_len) * len(horizon_hours),
    )
    model = SegmentHorizonWrapper(PatchTSTClassifier(cfg), pred_len=int(pred_len), n_horizons=len(horizon_hours))
    return model, {
        "backbone": asdict(cfg),
        "head": "segment_forecasting_logits_multi_horizon",
        "horizon_hours": list(horizon_hours),
        "primary_horizon_hours": PRIMARY_HORIZON_HOURS,
    }
