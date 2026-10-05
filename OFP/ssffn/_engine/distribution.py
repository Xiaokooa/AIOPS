import copy
import numpy as np
import torch

def diverse_indices(coordinates, count, excluded=()):
    count = min(int(count), len(coordinates) - len(set(excluded)))
    if count <= 0:
        return []
    blocked = set((int(i) for i in excluded))
    chosen = []
    if blocked:
        distances = np.min(np.sum((coordinates[:, None] - coordinates[list(blocked)][None]) ** 2, axis=-1), axis=1)
    else:
        center = np.median(coordinates, axis=0)
        first = int(np.argmin(np.sum((coordinates - center) ** 2, axis=1)))
        chosen.append(first)
        blocked.add(first)
        distances = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    while len(chosen) < count:
        distances[list(blocked)] = -np.inf
        index = int(np.argmax(distances))
        chosen.append(index)
        blocked.add(index)
        distances = np.minimum(distances, np.sum((coordinates - coordinates[index]) ** 2, axis=1))
    return chosen

class DistributionHSS:

    def coordinates(self, values):
        return np.stack([np.searchsorted(self.knots[j, 1:-1], values[:, j], side='right') / 16.0 for j in range(12)], axis=1).astype(np.float32)

    def select(self, index, risk, epoch):
        pool = self.pools[index]
        quota = self.original_counts[index]
        nhard = max(1, int(np.ceil(quota * 0.5)))
        ncover = int(np.floor(quota * 0.25))
        hard = np.argsort(-risk, kind='stable')[:nhard].tolist()
        cover = diverse_indices(self.coords[index], ncover, hard)
        remaining = np.setdiff1d(np.arange(len(pool)), hard + cover)
        rng = np.random.default_rng(self.seed + index * 1009 + epoch * 100003)
        random = rng.choice(remaining, quota - len(hard) - len(cover), replace=False).tolist()
        selected = np.sort(pool[hard + cover + random])
        assert len(selected) == quota and len(np.unique(selected)) == quota
        self.dataset.selected_positions[index] = selected
        self.dataset.selected_labels[index] = np.zeros(quota, dtype=np.int8)
        self.dataset.selected_weights[index] = np.ones(quota, dtype=np.float32)

    def refresh(self, model, cfg, raw_indices, stat_indices, stat_input_indices, temporal_summary_mode, score_function, epoch):
        probe = copy.copy(self.dataset)
        probe.file_names = [self.dataset.file_names[i] for i in self.normal]
        probe.selected_positions = [self.pools[i] for i in self.normal]
        prior_training = model.training
        scores = score_function(model, self.cache, probe, cfg, raw_indices, stat_indices, stat_input_indices, temporal_summary_mode)
        for i in self.normal:
            self.select(i, scores[self.dataset.file_names[i]], epoch)
        model.train(prior_training)
        return dict(epoch=epoch, normal_modules=len(self.normal), scored_windows=sum((len(p) for p in self.pools.values())), selected_windows=sum(self.original_counts.values()))

def fit_distribution(self, values):
    values = np.asarray(values, dtype=np.float64)
    q = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    scale = np.where(q[2] > q[0], q[2] - q[0], 1.0)
    transformed = np.arcsinh((values - q[1]) / scale).clip(-10, 10)
    knots = np.quantile(transformed, np.linspace(0, 1, 17), axis=0).T
    for name, value in [('sensor_median', q[1]), ('sensor_scale', scale), ('state_knots', knots)]:
        target = getattr(self, name)
        target.copy_(torch.as_tensor(value, dtype=target.dtype, device=target.device))
    return dict(rows=len(values), median=q[1].tolist(), iqr=scale.tolist(), knots=knots.tolist(), source='selected training endpoints only')
