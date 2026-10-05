# Numerical core retained for compatibility with the archived experiments.
"""SIT transfer, distribution-aware HSS, and the paper's normal Top-K loss.

SIT is adapted to history tokens; this is explicitly not the current-state-only
DASA-Net. Expert encoding and the existing HTSF fusion are retained by the runner.
"""
from dataclasses import dataclass
import copy
import zlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SITConfig:
    seq_len: int
    n_features: int
    width: int = 64
    depth: int = 3
    heads: int = 4
    feedforward: int = 192
    dropout: float = 0.15
    history_preserved: bool = True


class HistorySIT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # HTSF's raw view has 12 physical sensors and a DeltaSeconds channel.
        assert cfg.n_features == 13
        self.history_projection = nn.Sequential(
            nn.Linear(2 * cfg.seq_len, cfg.width), nn.LayerNorm(cfg.width), nn.GELU())
        self.time_projection = nn.Linear(2 * cfg.seq_len, cfg.width)
        self.sensor_identity = nn.Parameter(torch.empty(1, 12, cfg.width))
        self.module_token = nn.Parameter(torch.empty(1, 1, cfg.width))
        nn.init.normal_(self.sensor_identity, std=.02)
        nn.init.normal_(self.module_token, std=.02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            cfg.width, cfg.heads, cfg.feedforward, cfg.dropout, activation='gelu',
            batch_first=True, norm_first=True) for _ in range(cfg.depth)])
        # LastLinearInputHook in HTSF extracts this 64-dimensional representation.
        self.head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 1))

    def forward(self, x, mask):
        history = torch.cat([x[:, :, :12].transpose(1, 2),
                             mask[:, :, :12].transpose(1, 2)], dim=-1)
        tokens = self.history_projection(history) + self.sensor_identity
        time_context = self.time_projection(torch.cat([x[:, :, 12], mask[:, :, 12]], dim=1))
        tokens = tokens + time_context.unsqueeze(1)
        z = torch.cat([self.module_token.expand(len(x), -1, -1), tokens], dim=1)
        for layer in self.layers:
            z = layer(z)
        return self.head(z[:, 0]).squeeze(-1)


def build_sit(seq_len, n_features):
    cfg = SITConfig(seq_len=int(seq_len), n_features=int(n_features))
    return HistorySIT(cfg), cfg.__dict__, None


class HistoryDoseSIT(HistorySIT):
    """Current distribution state plus a learned history residual per sensor.

    Current-state DOSE retains DASA's bounded, sensor-specific representation.
    Historical values/masks and elapsed time remain trainable inputs. All
    medians, scales and knots are fitted using selected TRAINING endpoints.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        head = self.head
        del self.head
        self.state_projection = nn.Sequential(nn.Linear(16, cfg.width),
                                              nn.LayerNorm(cfg.width), nn.GELU())
        self.history_strength = nn.Parameter(torch.tensor(.1))
        self.time_strength = nn.Parameter(torch.tensor(.1))
        self.register_buffer('sensor_median', torch.zeros(12))
        self.register_buffer('sensor_scale', torch.ones(12))
        self.register_buffer('state_knots', torch.linspace(-3, 3, 17).expand(12,17).clone())
        # Keep the output Linear last for HTSF's representation hook.
        self.head = head

    def fit_distribution(self, values):
        values = np.asarray(values, dtype=np.float64)
        q = np.quantile(values, [.25, .5, .75], axis=0)
        # Inputs already have the original HTSF z-normalization. Tiny positive
        # IQRs remain meaningful here, so only zero IQRs fall back to unit scale.
        scale = np.where(q[2] > q[0], q[2]-q[0], 1.)
        transformed = np.arcsinh((values-q[1])/scale).clip(-10,10)
        knots = np.quantile(transformed, np.linspace(0,1,17), axis=0).T
        for name, value in [('sensor_median',q[1]), ('sensor_scale',scale), ('state_knots',knots)]:
            target = getattr(self,name)
            target.copy_(torch.as_tensor(value, dtype=target.dtype, device=target.device))
        return dict(rows=len(values), median=q[1].tolist(), iqr=scale.tolist(),
                    knots=knots.tolist(), source='selected training endpoints only')

    def forward(self, x, mask):
        state = torch.asinh((x[:,:,:12]-self.sensor_median)/self.sensor_scale).clamp(-10,10)
        span = (self.state_knots[:,1:]-self.state_knots[:,:-1]).clamp_min(.001)
        basis = ((state[:,-1,:,None]-self.state_knots[:,:-1])/span).clamp(0,1)
        history = torch.cat([(state/10).transpose(1,2), mask[:,:,:12].transpose(1,2)],dim=-1)
        time = torch.cat([torch.asinh(x[:,:,12]).clamp(-10,10)/10,mask[:,:,12]],dim=1)
        tokens = self.state_projection(basis) + self.sensor_identity
        tokens = tokens + self.history_strength.tanh()*self.history_projection(history)
        tokens = tokens + self.time_strength.tanh()*self.time_projection(time).unsqueeze(1)
        z = torch.cat([self.module_token.expand(len(x),-1,-1),tokens],dim=1)
        for layer in self.layers:
            z = layer(z)
        return self.head(z[:,0]).squeeze(-1)


def build_dosit(seq_len, n_features):
    cfg = SITConfig(seq_len=int(seq_len), n_features=int(n_features))
    return HistoryDoseSIT(cfg), dict(**cfg.__dict__, state_encoding='robust_dose',
                                   history_fusion='learned_residual'), None


def negative_topk(logits, module_is_faulty, k=4):
    """One complete module group per HTSF update; faulty groups have no penalty."""
    if bool(module_is_faulty) or logits.numel() == 0:
        return logits.sum() * 0.0
    return F.softplus(torch.topk(logits.reshape(-1), min(k, logits.numel())).values).mean()


def diverse_indices(coordinates, count, excluded=()):
    """Deterministic farthest-first coverage in training-quantile coordinates."""
    count = min(int(count), len(coordinates) - len(set(excluded)))
    if count <= 0:
        return []
    blocked = set(int(i) for i in excluded)
    chosen = []
    if blocked:
        distances = np.min(np.sum((coordinates[:, None] - coordinates[list(blocked)][None]) ** 2, axis=-1), axis=1)
    else:
        center = np.median(coordinates, axis=0)
        first = int(np.argmin(np.sum((coordinates-center) ** 2, axis=1)))
        chosen.append(first)
        blocked.add(first)
        distances = np.sum((coordinates-coordinates[first]) ** 2, axis=1)
    while len(chosen) < count:
        distances[list(blocked)] = -np.inf
        index = int(np.argmax(distances))
        chosen.append(index)
        blocked.add(index)
        distances = np.minimum(distances, np.sum((coordinates-coordinates[index]) ** 2, axis=1))
    return chosen


class DistributionHSS:
    """Keep quotas fixed: half hard windows, quarter coverage, quarter random.

    Positives and faulty-module selection remain exactly as in original HSS.
    Risk mining sees only a <=64-window pool from each NORMAL TRAINING module.
    No validation/test observation contributes to the knots or hard mining.
    """
    def __init__(self, dataset, raw_indices, seed):
        self.dataset = dataset
        self.cache = dataset.cache
        self.raw_indices = np.asarray(raw_indices[:12])
        self.seed = int(seed)
        self.normal = [i for i, name in enumerate(dataset.file_names)
                       if self.cache.label_by_file[name] == 0 and len(dataset.selected_positions[i])]
        fit = []
        for i, name in enumerate(dataset.file_names):
            positions = dataset.selected_positions[i]
            if len(positions):
                features = self.cache.get(name)[1]
                fit.append(features[positions][:, self.raw_indices])
        values = np.concatenate(fit)
        if len(values) > 100000:
            values = values[np.random.default_rng(seed).choice(len(values), 100000, replace=False)]
        self.knots = np.quantile(values, np.linspace(0, 1, 17), axis=0).T
        self.pools = {}
        self.coords = {}
        self.original_counts = {i: len(dataset.selected_positions[i]) for i in self.normal}
        from OFP.ssffn._engine.data import row_signal_scores
        for i in self.normal:
            name = dataset.file_names[i]
            _, features, valid, _, _, rule, _ = self.cache.get(name)
            eligible = np.flatnonzero(valid)
            signal = row_signal_scores(features, rule, dataset.feature_names,
                                        dataset.cfg.sample_signal_mode,
                                        dataset.cfg.sample_signal_temporal_fraction)
            scan = eligible[np.unique(np.linspace(0, len(eligible)-1, min(256, len(eligible)), dtype=int))]
            c = self.coordinates(features[scan][:, self.raw_indices])
            coverage = scan[diverse_indices(c, min(24, len(scan)))]
            hard = eligible[np.argsort(-signal[eligible], kind='stable')[:24]]
            old = dataset.selected_positions[i]
            pool = np.unique(np.r_[coverage, hard, old])
            if len(pool) < min(64, len(eligible)):
                remaining = np.setdiff1d(eligible, pool)
                random = np.random.default_rng(seed + zlib.crc32(name.encode())).choice(
                    remaining, min(64-len(pool), len(remaining)), replace=False)
                pool = np.unique(np.r_[pool, random])
            assert len(pool) <= 64
            self.pools[i] = pool
            self.coords[i] = self.coordinates(features[pool][:, self.raw_indices])
            self.select(i, signal[pool], epoch=0)

    def coordinates(self, values):
        return np.stack([np.searchsorted(self.knots[j, 1:-1], values[:, j], side='right') / 16.
                         for j in range(12)], axis=1).astype(np.float32)

    def select(self, index, risk, epoch):
        pool = self.pools[index]
        quota = self.original_counts[index]
        nhard = max(1, int(np.ceil(quota * .5)))
        ncover = int(np.floor(quota * .25))
        hard = np.argsort(-risk, kind='stable')[:nhard].tolist()
        cover = diverse_indices(self.coords[index], ncover, hard)
        remaining = np.setdiff1d(np.arange(len(pool)), hard+cover)
        rng = np.random.default_rng(self.seed + index * 1009 + epoch * 100003)
        random = rng.choice(remaining, quota-len(hard)-len(cover), replace=False).tolist()
        selected = np.sort(pool[hard+cover+random])
        assert len(selected) == quota and len(np.unique(selected)) == quota
        self.dataset.selected_positions[index] = selected
        self.dataset.selected_labels[index] = np.zeros(quota, dtype=np.int8)
        self.dataset.selected_weights[index] = np.ones(quota, dtype=np.float32)

    def refresh(self, model, cfg, raw_indices, stat_indices, stat_input_indices,
                temporal_summary_mode, score_function, epoch):
        probe = copy.copy(self.dataset)
        probe.file_names = [self.dataset.file_names[i] for i in self.normal]
        probe.selected_positions = [self.pools[i] for i in self.normal]
        prior_training = model.training
        scores = score_function(model, self.cache, probe, cfg, raw_indices,
                                stat_indices, stat_input_indices, temporal_summary_mode)
        for i in self.normal:
            self.select(i, scores[self.dataset.file_names[i]], epoch)
        model.train(prior_training)
        return dict(epoch=epoch, normal_modules=len(self.normal),
                    scored_windows=sum(len(p) for p in self.pools.values()),
                    selected_windows=sum(self.original_counts.values()))
