# Numerical core retained for compatibility with the archived experiments.
"""Train-only prefix coverage, bounded sensor histories and module objectives."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from OFP.ssffn._engine.distribution import DistributionHSS, HistoryDoseSIT, SITConfig, diverse_indices


class PrefixDistributionHSS(DistributionHSS):
    """Eight final windows: four hard, two covering, one prefix, one random.

    Expand only NORMAL TRAINING candidate pools with the first 32 eligible
    observations. Fault windows and the final optimization sample budget stay fixed.
    """
    def __init__(self, dataset, raw_indices, seed):
        self.prefix_ready = False
        super().__init__(dataset, raw_indices, seed)
        self.prefix = {}
        from OFP.ssffn._engine.data import row_signal_scores
        risks = {}
        for i in self.normal:
            name = dataset.file_names[i]
            _, features, valid, _, _, rule, _ = self.cache.get(name)
            self.prefix[i] = np.flatnonzero(valid)[:32]
            self.pools[i] = np.union1d(self.pools[i],self.prefix[i])
            assert len(self.pools[i]) <= 96
            self.coords[i] = self.coordinates(features[self.pools[i]][:,self.raw_indices])
            risks[i] = row_signal_scores(features,rule,dataset.feature_names,
                dataset.cfg.sample_signal_mode,dataset.cfg.sample_signal_temporal_fraction)[self.pools[i]]
        self.prefix_ready = True
        for i in self.normal:
            self.select(i,risks[i],epoch=0)

    def select(self,index,risk,epoch):
        if not self.prefix_ready:
            return super().select(index,risk,epoch)
        pool, quota = self.pools[index], self.original_counts[index]
        nhard = max(1,int(np.ceil(quota*.5)))
        ncover = int(np.floor(quota*.25))
        npre = min(1,quota-nhard-ncover)
        hard = np.argsort(-risk,kind='stable')[:nhard].tolist()
        prefix = []
        if npre:
            landmarks = [0,1,3,7,15,31]
            position = min(landmarks[min(epoch//2,len(landmarks)-1)],len(self.prefix[index])-1)
            candidates = np.roll(self.prefix[index],-position)
            for candidate in candidates:
                local = int(np.searchsorted(pool,candidate))
                if local not in hard:
                    prefix = [local]; break
        cover = diverse_indices(self.coords[index],ncover,hard+prefix)
        remaining = np.setdiff1d(np.arange(len(pool)),hard+prefix+cover)
        rng = np.random.default_rng(self.seed+index*1009+epoch*100003)
        random = rng.choice(remaining,quota-len(hard)-len(prefix)-len(cover),replace=False).tolist()
        selected = np.sort(pool[hard+prefix+cover+random])
        assert len(selected) == quota and len(np.unique(selected)) == quota
        self.dataset.selected_positions[index] = selected
        self.dataset.selected_labels[index] = np.zeros(quota,dtype=np.int8)
        self.dataset.selected_weights[index] = np.ones(quota,dtype=np.float32)


class QuantileResidualSIT(HistoryDoseSIT):
    """Current DOSE + bounded quantile histories and current-relative changes."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.history_projection = nn.Sequential(nn.Linear(3*cfg.seq_len,cfg.width),
                                                nn.LayerNorm(cfg.width),nn.GELU())
        self.history_strength = nn.Parameter(torch.full((1,12,1),.1))

    def forward(self,x,mask):
        state = torch.asinh((x[:,:,:12]-self.sensor_median)/self.sensor_scale).clamp(-10,10)
        spans = (self.state_knots[:,1:]-self.state_knots[:,:-1]).clamp_min(.001)
        basis = ((state[:,:,:,None]-self.state_knots[:,:-1])/spans).clamp(0,1)
        quantile = basis.mean(-1)*2-1
        change = quantile-quantile[:,-1:,:]
        history = torch.cat([quantile.transpose(1,2),change.transpose(1,2),
                             mask[:,:,:12].transpose(1,2)],dim=-1)
        time = torch.cat([torch.asinh(x[:,:,12]).clamp(-10,10)/10,mask[:,:,12]],dim=1)
        tokens = self.state_projection(basis[:,-1]) + self.sensor_identity
        tokens = tokens+self.history_strength.tanh()*self.history_projection(history)
        tokens = tokens+self.time_strength.tanh()*self.time_projection(time).unsqueeze(1)
        z = torch.cat([self.module_token.expand(len(x),-1,-1),tokens],dim=1)
        for layer in self.layers:
            z = layer(z)
        return self.head(z[:,0]).squeeze(-1)


def build_qrsit(seq_len,n_features):
    cfg = SITConfig(seq_len=int(seq_len),n_features=int(n_features))
    return QuantileResidualSIT(cfg),dict(**cfg.__dict__,state_encoding='robust_dose',
        history_fusion='quantile_history_and_current_relative_residual',history_gate='per_sensor'),None


def module_class_weight(dataset):
    positive = sum(bool(np.any(y > 0)) for y in dataset.selected_labels if len(y))
    negative = sum(not bool(np.any(y > 0)) for y in dataset.selected_labels if len(y))
    return negative/max(positive,1)


def balanced_module_margin(logits,flags,margin=.5):
    """Class-balanced two-sided max-logit loss for training module bags."""
    losses = {0:[],1:[]}
    for group in range(int(flags[-1,0])+1):
        positions = torch.where(flags[:,0] == group)[0]
        label = int(flags[positions[0],1])
        peak = logits[positions.to(logits.device)].max()
        losses[label].append(F.softplus(margin-peak) if label else F.softplus(margin+peak))
    return torch.stack([torch.stack(values).mean() for values in losses.values() if values]).mean()
