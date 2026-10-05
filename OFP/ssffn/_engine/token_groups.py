# Numerical core retained for compatibility with the archived experiments.
"""Statistical token organization and causal historical channel descriptors."""
from collections import OrderedDict
import torch
from torch import nn
from OFP.ssffn._engine.modules import StatisticalModuleSIT
from OFP.ssffn._engine.backbone import SENSORS

LAYOUTS = ('original', 'split2', 'split3')
RELATIONS = ('original', 'history8', 'history16')
SLOTS = ('Min', 'Diff', 'Max', 'Skew', 'Kurt', 'Std')
RELATION_NAMES = [f'Historical{direction}Gap{bound}{moment}'
                  for direction in ('Tx','Rx') for bound in ('Min','Max') for moment in ('Mean','Std')]
RELATION_EXTRA_NAMES = [f'Historical{direction}Gap{bound}Slope'
                        for direction in ('Tx','Rx') for bound in ('Min','Max')]
RELATION_EXTRA_NAMES += ['HistoricalTxLaneChangeCorrelation','HistoricalRxLaneChangeCorrelation',
                         'HistoricalTxLaneSpreadStd','HistoricalRxLaneSpreadStd']


def masked_moments(values, valid):
    weights = valid.to(values.dtype)
    count = weights.sum(1)
    clean = torch.where(valid, values, torch.zeros_like(values))
    mean = clean.sum(1) / count.clamp_min(1)
    delta = torch.where(valid, values - mean[:, None], torch.zeros_like(values))
    variance = delta.square().sum(1) / count.clamp_min(1)
    std = variance.clamp_min(0).sqrt() * (count >= 2)
    return mean, std, count


def masked_slope(values, valid):
    # Position slope across the observed input prefix, expressed per full window.
    pos = torch.linspace(0, 1, values.shape[1], device=values.device, dtype=values.dtype)
    pos = pos[None, :, None].expand_as(values)
    mean, _, count = masked_moments(values, valid)
    center, _, _ = masked_moments(pos, valid)
    dx = torch.where(valid, pos - center[:, None], torch.zeros_like(pos))
    dy = torch.where(valid, values - mean[:, None], torch.zeros_like(values))
    return (dx * dy).sum(1) / dx.square().sum(1).clamp_min(1e-6) * (count >= 3)


def history_relations(physical, valid, physical_scale):
    """8 gap moments, 4 gap slopes, 2 lane-change correlations, 2 spread SDs.

    Every value uses only this endpoint's 32-position input prefix. Missing
    readings are excluded, not carried into historical cross-channel features.
    Scales come only from training-fitted sensor IQRs in physical coordinates.
    """
    gaps, gap_masks, synchrony, spread_stds = [], [], [], []
    for aggregate, lanes in ((2, [8, 9, 10, 11]), (3, [4, 5, 6, 7])):
        lane_x, lane_ok = physical[:, :, lanes], valid[:, :, lanes]
        all_lanes = lane_ok.all(-1)
        good = valid[:, :, aggregate] & all_lanes
        # Clear invalid entries before arithmetic so sentinels cannot overflow.
        lane_x = torch.where(lane_ok, lane_x, torch.zeros_like(lane_x))
        agg = torch.where(valid[:, :, aggregate], physical[:, :, aggregate], 0.)
        scale = physical_scale[aggregate].clamp_min(1e-6)
        gaps.extend([(agg - lane_x.amin(-1)) / scale,
                     (agg - lane_x.amax(-1)) / scale])
        gap_masks.extend([good, good])
        spread = (lane_x.amax(-1) - lane_x.amin(-1)) / scale
        spread_stds.append(masked_moments(spread[..., None], all_lanes[..., None])[1][:, 0])
        dx = lane_x[:, 1:] - lane_x[:, :-1]
        dx_ok = lane_ok[:, 1:] & lane_ok[:, :-1]
        correlations, supports = [], []
        for first in range(4):
            for second in range(first + 1, 4):
                ok = dx_ok[:, :, first] & dx_ok[:, :, second]
                a, b = dx[:, :, first:first+1], dx[:, :, second:second+1]
                ma, sa, count = masked_moments(a, ok[..., None])
                mb, sb, _ = masked_moments(b, ok[..., None])
                product = torch.where(ok[..., None], (a-ma[:, None])*(b-mb[:, None]), 0.)
                covariance = product.sum(1) / count.clamp_min(1)
                supported = (count[:, 0] >= 3) & (sa[:, 0] > 1e-6) & (sb[:, 0] > 1e-6)
                correlation = (covariance / (sa * sb).clamp_min(1e-6))[:, 0].clamp(-1, 1)
                correlations.append(correlation * supported)
                supports.append(supported)
        synchrony.append(torch.stack(correlations, -1).sum(-1) /
                         torch.stack(supports, -1).sum(-1).clamp_min(1))
    gap = torch.stack(gaps, -1)
    good = torch.stack(gap_masks, -1)
    mean, std, _ = masked_moments(gap, good)
    moments = torch.stack([mean, std], -1).flatten(1)
    extra = torch.cat([masked_slope(gap, good), torch.stack(synchrony, -1),
                       torch.stack(spread_stds, -1)], -1)
    return torch.cat([moments, extra], -1).nan_to_num().clamp(-100, 100)


class GroupedInput(nn.Module):
    """Partition original projection columns; do not alter backbone initialization."""
    def __init__(self, source, groups, continuous_ids, n_cached, relation_ids):
        super().__init__()
        self.groups = groups
        self.projections = nn.ModuleList()
        old_ids = list(continuous_ids)
        identities = []
        # Restore RNG afterwards: token regrouping must not shift training RNG.
        rng = torch.random.get_rng_state()
        try:
            for j, (name, indices) in enumerate(groups.items()):
                self.register_buffer(f'indices_{j}', torch.tensor(indices, dtype=torch.long))
                layer = nn.Linear(len(indices), 64)
                old_token = 1 if name == 'relations' else 0
                with torch.no_grad():
                    for col, index in enumerate(indices):
                        # New 8 descriptor columns inherit corresponding old
                        # relation projection columns, preserving weight scale.
                        old_index = index if index < n_cached else relation_ids[index-n_cached]
                        layer.weight[:, col].copy_(source.projections[0].weight[:, old_ids.index(old_index)])
                    layer.bias.copy_(source.projections[old_token].bias)
                self.projections.append(layer)
                identities.append(source.identity[:, old_token:old_token+1].detach().clone())
            self.identity = nn.Parameter(torch.cat(identities, 1))
        finally:
            torch.random.set_rng_state(rng)

    def forward(self, features):
        bounded = torch.asinh(features).clamp(-10, 10) / 3
        return torch.stack([layer(bounded.index_select(1, getattr(self, f'indices_{j}')))
                            for j, layer in enumerate(self.projections)], 1) + self.identity


class OrganizedSFB(StatisticalModuleSIT):
    def __init__(self, *args, sfb_layout='split2', relation_mode='original', normalization=None, **kwargs):
        assert sfb_layout in LAYOUTS and relation_mode in RELATIONS
        super().__init__(*args, **kwargs)
        self.sfb_layout, self.relation_mode = sfb_layout, relation_mode
        names = list(kwargs['stat_feature_names'])
        self.n_cached = len(names)
        stats = [names.index(f'Fe{s}{op}') for s in SENSORS for op in SLOTS]
        relation = [i for i, name in enumerate(names) if name.startswith('Fe') and i not in stats]
        assert len(stats) == 72 and len(relation) == 8
        self.register_buffer('relation_positions', torch.tensor(relation, dtype=torch.long))
        for key in ('raw_mean', 'raw_std'):
            value = normalization[key] if normalization is not None else ([0.]*12 if key == 'raw_mean' else [1.]*12)
            self.register_buffer('history_'+key, torch.as_tensor(value, dtype=torch.float32))
        extra_ids = list(range(self.n_cached, self.n_cached+8)) if relation_mode == 'history16' else []
        semantic_names = list(names)
        if relation_mode != 'original':
            for index, descriptor in zip(relation, RELATION_NAMES):
                semantic_names[index] = descriptor
        semantic_names += RELATION_EXTRA_NAMES if extra_ids else []
        if self.module_variant != 'no_statistics':
            source = self.expert_input
            if sfb_layout == 'original':
                assert relation_mode == 'original'
                groups = source.groups
            else:
                if sfb_layout == 'split2':
                    groups = OrderedDict(cumulative=stats, relations=relation+extra_ids)
                else:
                    level = [names.index(f'Fe{s}{op}') for s in SENSORS for op in ('Min', 'Diff', 'Max')]
                    variation = [names.index(f'Fe{s}{op}') for s in SENSORS for op in ('Skew', 'Kurt', 'Std')]
                    groups = OrderedDict(level=level, variability=variation, relations=relation+extra_ids)
                continuous_ids = next(ids for ids in source.groups.values() if len(ids) == 80)
                self.expert_input = GroupedInput(source, groups, continuous_ids, self.n_cached, relation)
            self.temporal_encoder_cfg.update(
                expert_groups={k:[semantic_names[i] for i in ids] for k, ids in groups.items()},
                token_feature_dimensions=[len(ids) for ids in groups.values()],
                expert_tokens=len(groups), statistical_tokens=len(groups),
                active_statistic_features=80+len(extra_ids), active_expert_features=80+len(extra_ids),
                active_feature_names=[semantic_names[i] for ids in groups.values() for i in ids],
                constant_indicator_token=sfb_layout == 'original',
            )
        cfg = self.temporal_encoder_cfg
        cfg.update(sfb_layout=sfb_layout, relation_mode=relation_mode,
                   relation_descriptors=[names[i] for i in relation] if relation_mode == 'original' else
                       RELATION_NAMES+(RELATION_EXTRA_NAMES if extra_ids else []),
                   historical_relation_window=32 if relation_mode != 'original' else None,
                   historical_relation_scale='training raw std times training sensor IQR',
                   historical_relation_missing='exclude invalid and padding; unsupported statistics zero',
                   initialization='Unchanged base parameters; inherited projection columns, biases and identities; RNG restored')
        cfg['total_tokens'] = cfg['sensor_tokens']+cfg['statistical_tokens']+cfg['module_tokens']

    def statistical_content(self, raw_x, raw_mask, stat_x):
        if self.relation_mode == 'original':
            return stat_x
        valid = (raw_mask[:, :, :12] > 0) & torch.isfinite(raw_x[:, :, :12])
        physical = raw_x[:, :, :12]*self.history_raw_std + self.history_raw_mean
        scale = self.history_raw_std * self.sensor_input.sensor_scale
        descriptors = history_relations(physical, valid, scale)
        out = stat_x.clone()
        out[:, self.relation_positions] = descriptors[:, :8]
        if self.relation_mode == 'history16':
            out = torch.cat([out, descriptors[:, 8:]], -1)
        return out

    def input_tokens(self, raw_x, raw_mask, stat_x):
        if self.module_variant != 'no_statistics':
            stat_x = self.statistical_content(raw_x, raw_mask, stat_x)
        return super().input_tokens(raw_x, raw_mask, stat_x)
