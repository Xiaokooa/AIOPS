# Numerical core retained for compatibility with the archived experiments.
"""Statistical-input main model and four prespecified module removals."""
from collections import OrderedDict
import hashlib
import zlib
import numpy as np
import torch
from OFP.ssffn._engine.backbone import SENSORS
from OFP.ssffn._engine.input_ablation import DistributionOnly, UniformNormalSampler
from OFP.ssffn._engine.statistical_input import StatisticalAblationSIT, parameter_hash
from OFP.ssffn._engine.distribution import DistributionHSS, diverse_indices

VARIANTS = ['full', 'no_sensor', 'no_statistics', 'no_sit', 'no_hss']


def sensor_distribution_signal(features, rule_pred=None, feature_names=None, mode=None, temporal_fraction=None):
    """Training-standardized sensor deviation; ignores every engineered column."""
    names = list(feature_names or [])
    indices = [names.index(name) for name in SENSORS]
    raw = np.nan_to_num(features[:, indices].astype(np.float32), nan=0., posinf=0., neginf=0.)
    return np.mean(np.abs(np.arcsinh(raw)), axis=1).astype(np.float32)


def position_digest(dataset):
    h = hashlib.sha256()
    for name, indices in zip(dataset.file_names, dataset.selected_positions):
        h.update(name.encode())
        h.update(np.asarray(indices, dtype=np.int64).tobytes())
    return h.hexdigest()


def start_policy(dataset):
    return dict(bootstrap_signal='mean absolute asinh of training-standardized 12 sensor values',
        bootstrap_positions_sha256=position_digest(dataset),
        bootstrap_selection='hybrid half sensor deviation, half random; original class quotas',
        excluded_sampling_inputs='All engineered features, Ma/Ru, quality flags and rule predictions',
        rule_signal_used=False, normal_quota=8,
        training_modules=len(dataset.file_names))


class RuleFreeHSS(DistributionHSS):
    """Original risk/coverage/random HSS with rule-free pool initialization."""
    def __init__(self, dataset, raw_indices, seed):
        self.policy_metadata = start_policy(dataset)
        self.policy_metadata.update(strategy='distribution_risk', pool_size=64,
            initial_risk='mean absolute centered training-quantile coordinates',
            refresh='model risk after epochs 2,4,6,8,10,12,14')
        self.dataset, self.cache, self.seed = dataset, dataset.cache, int(seed)
        self.raw_indices = np.asarray(raw_indices[:12])
        self.normal = [i for i, name in enumerate(dataset.file_names)
            if self.cache.label_by_file[name] == 0 and len(dataset.selected_positions[i])]
        values = np.concatenate([self.cache.get(n)[1][p][:, self.raw_indices]
            for n, p in zip(dataset.file_names, dataset.selected_positions) if len(p)])
        if len(values) > 100000:
            values = values[np.random.default_rng(seed).choice(len(values), 100000, replace=False)]
        self.knots = np.quantile(values, np.linspace(0, 1, 17), axis=0).T
        self.pools, self.coords = {}, {}
        self.original_counts = {i: len(dataset.selected_positions[i]) for i in self.normal}
        for i in self.normal:
            name = dataset.file_names[i]
            _, features, valid, _, _, _, _ = self.cache.get(name)
            eligible = np.flatnonzero(valid)
            coordinates = self.coordinates(features[eligible][:, self.raw_indices])
            deviation = np.mean(np.abs(2*coordinates-1), axis=1)
            scan_ids = np.unique(np.linspace(0, len(eligible)-1, min(256, len(eligible)), dtype=int))
            coverage = eligible[scan_ids[diverse_indices(coordinates[scan_ids], min(24, len(scan_ids)))]]
            hard = eligible[np.argsort(-deviation, kind='stable')[:24]]
            pool = np.unique(np.r_[coverage, hard, dataset.selected_positions[i]])
            if len(pool) < min(64, len(eligible)):
                rest = np.setdiff1d(eligible, pool)
                extra = np.random.default_rng(seed+zlib.crc32(name.encode())).choice(
                    rest, min(64-len(pool), len(rest)), replace=False)
                pool = np.unique(np.r_[pool, extra])
            assert len(pool) <= 64
            self.pools[i] = pool
            self.coords[i] = self.coordinates(features[pool][:, self.raw_indices])
            risk = np.mean(np.abs(2*self.coords[i]-1), axis=1)
            self.select(i, risk, epoch=0)
        h = hashlib.sha256()
        for i, pool in self.pools.items():
            h.update(dataset.file_names[i].encode())
            h.update(pool.astype(np.int64).tobytes())
        self.policy_metadata['candidate_pools_sha256'] = h.hexdigest()
        self.policy_metadata['initial_selected_positions_sha256'] = position_digest(dataset)


class NoHSSSampler(UniformNormalSampler):
    def __init__(self, dataset, raw_indices, seed):
        self.policy_metadata = start_policy(dataset)
        self.policy_metadata.update(strategy='uniform_normal', pool_size=None,
            refresh='uniform resampling at the same seven epoch boundaries; no model scoring')
        super().__init__(dataset, raw_indices, seed)
        self.policy_metadata['initial_selected_positions_sha256'] = position_digest(dataset)


class StatisticalModuleSIT(StatisticalAblationSIT):
    def __init__(self, *args, module_variant='full', **kwargs):
        assert module_variant in VARIANTS
        super().__init__(*args, feature_variant='statistics80', **kwargs)
        reference_hash = parameter_hash(self)
        self.module_variant = module_variant
        if module_variant == 'no_sensor':
            self.sensor_input = DistributionOnly(self.sensor_input)
            self.type_embedding = torch.nn.Parameter(self.type_embedding[:, 1:].detach().clone())
        elif module_variant == 'no_statistics':
            del self.expert_input
            self.type_embedding = torch.nn.Parameter(self.type_embedding[:, :1].detach().clone())
        elif module_variant == 'no_sit':
            del self.backbone
            del self.module_token
        cfg = self.temporal_encoder_cfg
        cfg.update(module_variant=module_variant,
            main_reference_initialization_sha256=reference_hash,
            initialization_sha256=parameter_hash(self),
            sensor_tokens=0 if module_variant == 'no_sensor' else 12,
            expert_tokens=0 if module_variant == 'no_statistics' else 2,
            statistical_tokens=0 if module_variant == 'no_statistics' else 2,
            module_tokens=0 if module_variant == 'no_sit' else 1,
            active_expert_features=0 if module_variant == 'no_statistics' else 80,
            active_statistic_features=0 if module_variant == 'no_statistics' else 80,
            sit_layers=0 if module_variant == 'no_sit' else 3,
            depth=0 if module_variant == 'no_sit' else 3,
            sensor_preprocessing='Current DDM measurements with training-fitted quantile encoding; no history residual',
            hss_scope='Rule-free sensor bootstrap and quantile pool; model risk refresh; uniform replacement in no_hss',
            feature_fusion='Fixed branch-balanced mean pooling' if module_variant == 'no_sit' else 'Shared SIT CLS readout')
        if module_variant == 'no_statistics':
            cfg.update(expert_groups={}, active_feature_names=[], token_feature_dimensions=[], constant_indicator_token=False)
        cfg['total_tokens'] = cfg['sensor_tokens']+cfg['expert_tokens']+cfg['module_tokens']

    def input_tokens(self, raw_x, raw_mask, stat_x):
        parts = []
        if self.module_variant != 'no_sit':
            parts.append(self.module_token.expand(len(raw_x), -1, -1))
        if self.module_variant != 'no_sensor':
            parts.append(self.sensor_input(raw_x, raw_mask)+self.type_embedding[:, :1])
        if self.module_variant != 'no_statistics':
            index = 0 if self.module_variant == 'no_sensor' else 1
            tokens = self.expert_input(stat_x)+self.type_embedding[:, index:index+1]
            if self.training and self.stat_modality_dropout:
                keep = (torch.rand(len(stat_x), 1, 1, device=stat_x.device) >= self.stat_modality_dropout).to(tokens.dtype)
                tokens = tokens*keep/(1-self.stat_modality_dropout)
            parts.append(tokens)
        return torch.cat(parts, dim=1)

    def encode(self, raw_x, raw_mask, stat_x, return_explain=False, **unused):
        if self.module_variant != 'no_sit':
            return super().encode(raw_x, raw_mask, stat_x, return_explain=return_explain, **unused)
        assert not return_explain
        z = self.input_norm(self.input_tokens(raw_x, raw_mask, stat_x))
        return self.output_norm(.5*(z[:, :12].mean(dim=1)+z[:, 12:].mean(dim=1)))
