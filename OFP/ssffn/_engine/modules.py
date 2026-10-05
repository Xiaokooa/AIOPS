import hashlib
import zlib
import numpy as np
from OFP.ssffn._engine.embeddings import SENSORS
from OFP.ssffn._engine.distribution import DistributionHSS, diverse_indices

def sensor_distribution_signal(features, rule_pred=None, feature_names=None, mode=None, temporal_fraction=None):
    names = list(feature_names or [])
    indices = [names.index(name) for name in SENSORS]
    raw = np.nan_to_num(features[:, indices].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(np.abs(np.arcsinh(raw)), axis=1).astype(np.float32)

def position_digest(dataset):
    h = hashlib.sha256()
    for name, indices in zip(dataset.file_names, dataset.selected_positions):
        h.update(name.encode())
        h.update(np.asarray(indices, dtype=np.int64).tobytes())
    return h.hexdigest()

def start_policy(dataset):
    return dict(bootstrap_signal='mean absolute asinh of training-standardized 12 sensor values', bootstrap_positions_sha256=position_digest(dataset), bootstrap_selection='hybrid half sensor deviation, half random; original class quotas', rule_signal_used=False, normal_quota=8, training_modules=len(dataset.file_names))

class UniformNormalSampler:

    def __init__(self, dataset, raw_indices, seed):
        self.dataset = dataset
        self.seed = int(seed)
        self.normal = [i for i, name in enumerate(dataset.file_names) if dataset.cache.label_by_file[name] == 0 and len(dataset.selected_positions[i])]
        self.quota = {i: len(dataset.selected_positions[i]) for i in self.normal}
        fit = [dataset.cache.get(name)[1][positions][:, np.asarray(raw_indices[:12])] for name, positions in zip(dataset.file_names, dataset.selected_positions) if len(positions)]
        values = np.concatenate(fit)
        if len(values) > 100000:
            values = values[np.random.default_rng(seed).choice(len(values), 100000, replace=False)]
        self.knots = np.quantile(values, np.linspace(0, 1, 17), axis=0).T
        self.eligible = {i: np.flatnonzero(dataset.cache.get(dataset.file_names[i])[2]) for i in self.normal}
        self.select(0)

    def select(self, epoch):
        for i in self.normal:
            rng = np.random.default_rng(self.seed + i * 1009 + epoch * 100003)
            positions = np.sort(rng.choice(self.eligible[i], self.quota[i], replace=False))
            self.dataset.selected_positions[i] = positions
            self.dataset.selected_labels[i] = np.zeros(len(positions), dtype=np.int8)
            self.dataset.selected_weights[i] = np.ones(len(positions), dtype=np.float32)

    def refresh(self, model, cfg, raw_indices, stat_indices, stat_input_indices, temporal_summary_mode, score_function, epoch):
        self.select(epoch)
        return dict(epoch=epoch, normal_modules=len(self.normal), scored_windows=0, selected_windows=sum(self.quota.values()), strategy='uniform over all valid normal windows')

class RuleFreeHSS(DistributionHSS):

    def __init__(self, dataset, raw_indices, seed):
        self.policy_metadata = start_policy(dataset)
        self.policy_metadata.update(strategy='distribution_risk', pool_size=64, initial_risk='mean absolute centered training-quantile coordinates', refresh='model risk after epochs 2,4,6,8,10,12,14')
        self.dataset, self.cache, self.seed = (dataset, dataset.cache, int(seed))
        self.raw_indices = np.asarray(raw_indices[:12])
        self.normal = [i for i, name in enumerate(dataset.file_names) if self.cache.label_by_file[name] == 0 and len(dataset.selected_positions[i])]
        values = np.concatenate([self.cache.get(n)[1][p][:, self.raw_indices] for n, p in zip(dataset.file_names, dataset.selected_positions) if len(p)])
        if len(values) > 100000:
            values = values[np.random.default_rng(seed).choice(len(values), 100000, replace=False)]
        self.knots = np.quantile(values, np.linspace(0, 1, 17), axis=0).T
        self.pools, self.coords = ({}, {})
        self.original_counts = {i: len(dataset.selected_positions[i]) for i in self.normal}
        for i in self.normal:
            name = dataset.file_names[i]
            _, features, valid, _, _, _, _ = self.cache.get(name)
            eligible = np.flatnonzero(valid)
            coordinates = self.coordinates(features[eligible][:, self.raw_indices])
            deviation = np.mean(np.abs(2 * coordinates - 1), axis=1)
            scan_ids = np.unique(np.linspace(0, len(eligible) - 1, min(256, len(eligible)), dtype=int))
            coverage = eligible[scan_ids[diverse_indices(coordinates[scan_ids], min(24, len(scan_ids)))]]
            hard = eligible[np.argsort(-deviation, kind='stable')[:24]]
            pool = np.unique(np.r_[coverage, hard, dataset.selected_positions[i]])
            if len(pool) < min(64, len(eligible)):
                rest = np.setdiff1d(eligible, pool)
                extra = np.random.default_rng(seed + zlib.crc32(name.encode())).choice(rest, min(64 - len(pool), len(rest)), replace=False)
                pool = np.unique(np.r_[pool, extra])
            assert len(pool) <= 64
            self.pools[i] = pool
            self.coords[i] = self.coordinates(features[pool][:, self.raw_indices])
            risk = np.mean(np.abs(2 * self.coords[i] - 1), axis=1)
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
        self.policy_metadata.update(strategy='uniform_normal', pool_size=None, refresh='uniform resampling at the same seven epoch boundaries; no model scoring')
        super().__init__(dataset, raw_indices, seed)
        self.policy_metadata['initial_selected_positions_sha256'] = position_digest(dataset)
