from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OFP.deep_learning.official.FTEformer.model import build_model as build_fteformer
from OFP.deep_learning.official.iTransformer.model import build_model as build_itransformer
from OFP.deep_learning.official.PatchTST.model import build_model as build_patchtst
from OFP.deep_learning.official.common.losses import (
    FTEformerLossCfg,
    fteformer_total_loss,
    make_sensor_prior,
    primary_logit,
    split_alarm_aux_logits,
)
from OFP.deep_learning.official.ofp_formal_protocol.protocol import DEFAULT_HORIZON_HOURS, SENSORS


def main() -> None:
    builders = {
        "itransformer": build_itransformer,
        "patchtst": build_patchtst,
        "fteformer": build_fteformer,
    }
    x = torch.randn(2, 32, len(SENSORS))
    mask = torch.ones_like(x)
    y = torch.zeros(2, len(DEFAULT_HORIZON_HOURS))
    y[1, :] = 1.0
    type_targets = torch.zeros(2, 5)
    type_targets[1, 0] = 1.0
    type_masks = torch.tensor([0.0, 1.0])
    prior = make_sensor_prior(
        ["thermal_anomaly", "current_bias_anomaly", "tx_power_anomaly", "rx_power_anomaly", "lane_imbalance"],
        SENSORS,
    )

    for name, builder in builders.items():
        model, cfg, use_fte = builder(32, len(SENSORS))
        out = model(x, mask)
        logits = primary_logit(out)
        assert logits.shape == (2, len(DEFAULT_HORIZON_HOURS) + 1), (name, logits.shape)
        alarm_logits, aux_logits = split_alarm_aux_logits(logits, aux_dim=len(DEFAULT_HORIZON_HOURS))
        assert alarm_logits.shape == (2,), (name, alarm_logits.shape)
        assert aux_logits.shape == (2, len(DEFAULT_HORIZON_HOURS)), (name, aux_logits.shape)
        if use_fte:
            fault_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux_logits, y)
            loss, parts = fteformer_total_loss(
                out,
                y,
                fault_loss,
                type_targets,
                type_masks,
                prior,
                FTEformerLossCfg(sensor_contrastive_weight=float(cfg["sensor_contrastive_weight"])),
            )
            assert torch.isfinite(loss), parts
        print(f"[preflight] {name} ok")


if __name__ == "__main__":
    main()
