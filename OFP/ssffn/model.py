"""SSFFN: 12 sensor tokens + 3 statistical tokens + a learnable module token.

The numerical core preserves the initialization and checkpoint names of the
reported split3 experiments. Neither input branch has its own encoder stack.
"""
import torch
from torch import nn

from ._engine.backbone import SENSORS
from ._engine.data import CompatCfg, compat_feature_names
from ._engine.feature_schema import select_feature_names
from ._engine.token_groups import OrganizedSFB

VARIANTS = ('full', 'no_sensor', 'no_statistics', 'no_sit', 'no_hss')


def cached_statistic_names():
    names = compat_feature_names(CompatCfg(feature_mode='ofp', preserve_timepoints=True))
    excluded = {*SENSORS, 'DeltaSeconds', 'Ts'}
    return [name for name in select_feature_names(names, 'statistical,expert') if name not in excluded]


STATISTIC_NAMES = tuple(name for name in cached_statistic_names() if name.startswith('Fe'))


def build_model(variant='full', training_seed=42, normalization=None, **kwargs):
    """Build the exact split3 model used by training and archived checkpoints.

    Inputs are normalized raw windows [B,32,13], their masks, and the retained
    compatibility cache [B,160]. Only the final 12 sensor measurements and the
    80 STATISTIC_NAMES columns are used. Rule/threshold columns have no effect.
    """
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    args = dict(temporal_encoder='shared_sit', seq_len=32, n_raw_features=13,
                n_stat_features=160, stat_feature_names=cached_statistic_names(),
                shared_layout='typed', fusion_mode='shared_sit', latent_dim=64,
                stat_modality_dropout=.15, followup='a4_typed', module_variant=variant,
                training_seed=training_seed, sfb_layout='split3', relation_mode='original',
                normalization=normalization)
    args.update(kwargs)
    return OrganizedSFB(**args)


class SSFFN(nn.Module):
    """Convenient tensor API for the two active input views.

    current_sensors: [batch,12], in SENSORS order, training-standardized.
    statistics: [batch,80], in STATISTIC_NAMES order, training-standardized.
    Returns a logit per observation; apply sigmoid for probabilities. For CSV
    preprocessing and checkpoint inference use ``python -m OFP.ssffn predict``.
    """
    def __init__(self, variant='full'):
        super().__init__()
        self.network = build_model(variant)
        names = cached_statistic_names()
        self.register_buffer('active_indices', torch.tensor([names.index(n) for n in STATISTIC_NAMES]))

    def forward(self, current_sensors, statistics):
        if current_sensors.ndim != 2 or current_sensors.shape[1] != 12:
            raise ValueError('Expected current_sensors [batch,12]')
        if statistics.shape != (len(current_sensors), 80):
            raise ValueError('Expected statistics [batch,80]')
        raw = torch.cat((current_sensors, current_sensors.new_zeros(len(current_sensors), 1)), 1)[:, None]
        cached = statistics.new_zeros(len(statistics), 160)
        cached[:, self.active_indices] = statistics
        return self.network(raw, torch.ones_like(raw), cached)
