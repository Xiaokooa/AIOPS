# Numerical core retained for compatibility with the archived experiments.
"""Two input tokenizers feeding exactly one joint SIT backbone and one head.

Kept outside model.py to avoid shadowing the existing AIOPS model package.
"""
from collections import OrderedDict
import re
import numpy as np
import torch
from torch import nn
from OFP.ssffn._engine.distribution import HistoryDoseSIT

SENSORS = ['Temp','Curr','TxP0','RxP0','RxP1','RxP2','RxP3','RxP4','TxP1','TxP2','TxP3','TxP4']


def expert_groups(names, layout):
    groups = OrderedDict()
    if layout == 'typed':
        groups.update(continuous=[], indicators=[])
        for i,name in enumerate(names):
            indicator = name.startswith('Ru') or name.endswith(('InvalidFlag','MissingFlag','GapFlag')) or 'MissingCount' in name
            groups['indicators' if indicator else 'continuous'].append(i)
    elif layout == 'sensor_grouped':
        groups.update((sensor,[]) for sensor in SENSORS)
        groups['cross_sensor_and_global'] = []
        for i,name in enumerate(names):
            found = set(re.findall(r'Temp|Curr|TxP[0-4]|RxP[0-4]',name))
            key = next(iter(found)) if len(found) == 1 and 'TxRx' not in name else 'cross_sensor_and_global'
            groups[key].append(i)
    else:
        raise ValueError(layout)
    groups = OrderedDict((k,v) for k,v in groups.items() if v)
    assert sorted(i for values in groups.values() for i in values) == list(range(len(names)))
    return groups


class SensorInput(nn.Module):
    """Deterministic transforms plus linear token projections; no encoder stack."""
    fit_distribution = HistoryDoseSIT.fit_distribution

    def __init__(self, seq_len=32, width=64):
        super().__init__()
        self.register_buffer('sensor_median',torch.zeros(12))
        self.register_buffer('sensor_scale',torch.ones(12))
        self.register_buffer('state_knots',torch.linspace(-3,3,17).expand(12,17).clone())
        self.current_projection = nn.Linear(16,width)
        self.history_projection = nn.Linear(2*seq_len,width)
        self.time_projection = nn.Linear(2*seq_len,width)
        self.identity = nn.Parameter(torch.empty(1,12,width))
        nn.init.normal_(self.identity,std=.02)
        self.history_strength = nn.Parameter(torch.tensor(.1))
        self.time_strength = nn.Parameter(torch.tensor(.1))

    def forward(self,x,mask):
        state = torch.asinh((x[:,:,:12]-self.sensor_median)/self.sensor_scale).clamp(-10,10)
        span = (self.state_knots[:,1:]-self.state_knots[:,:-1]).clamp_min(.001)
        current = ((state[:,-1,:,None]-self.state_knots[:,:-1])/span).clamp(0,1)
        history = torch.cat([(state/10).transpose(1,2),mask[:,:,:12].transpose(1,2)],dim=-1)
        time = torch.cat([torch.asinh(x[:,:,12]).clamp(-10,10)/10,mask[:,:,12]],dim=1)
        return (self.current_projection(current)+self.identity
                +self.history_strength.tanh()*self.history_projection(history)
                +self.time_strength.tanh()*self.time_projection(time).unsqueeze(1))


class ExpertInput(nn.Module):
    """Each disjoint feature group becomes one token through a single linear map."""
    def __init__(self,names,layout,width=64):
        super().__init__()
        self.groups = expert_groups(names,layout)
        self.projections = nn.ModuleList()
        for i,indices in enumerate(self.groups.values()):
            self.register_buffer(f'indices_{i}',torch.tensor(indices,dtype=torch.long))
            self.projections.append(nn.Linear(len(indices),width))
        self.identity = nn.Parameter(torch.empty(1,len(self.groups),width))
        nn.init.normal_(self.identity,std=.02)

    def forward(self,features):
        # OFP standardization is already fitted on training modules. This fixed
        # compression bounds outliers without fitting another network or scaler.
        values = torch.asinh(features).clamp(-10,10)/3
        return torch.stack([layer(values.index_select(1,getattr(self,f'indices_{i}')))
                            for i,layer in enumerate(self.projections)],dim=1)+self.identity


class SharedInputSIT(nn.Module):
    def __init__(self,temporal_encoder,seq_len,n_raw_features,n_stat_features,stat_feature_names,
                 shared_layout='typed',latent_dim=64,stat_modality_dropout=.15,**unused):
        super().__init__()
        assert temporal_encoder == 'shared_sit' and n_raw_features == 13
        assert n_stat_features == len(stat_feature_names) == 160 and latent_dim == 64
        assert unused.get('fusion_mode') == 'shared_sit'
        self.seq_len = seq_len
        self.latent_dim = 64
        self.fusion_mode = 'shared_sit'
        self.layout = shared_layout
        self.stat_modality_dropout = float(stat_modality_dropout)
        self.sensor_input = SensorInput(seq_len,64)
        self.expert_input = ExpertInput(stat_feature_names,shared_layout,64)
        self.type_embedding = nn.Parameter(torch.empty(1,2,64))
        self.module_token = nn.Parameter(torch.empty(1,1,64))
        nn.init.normal_(self.type_embedding,std=.02)
        nn.init.normal_(self.module_token,std=.02)
        self.input_norm = nn.LayerNorm(64)
        self.backbone = nn.ModuleList([nn.TransformerEncoderLayer(64,4,192,.15,
            activation='gelu',batch_first=True,norm_first=True) for _ in range(3)])
        self.output_norm = nn.LayerNorm(64)
        self.classifier = nn.Linear(64,1)
        self.temporal_encoder_cfg = dict(architecture='two input tokenizers, one shared SIT, one head',
            layout=shared_layout,seq_len=seq_len,width=64,depth=3,heads=4,feedforward=192,dropout=.15,
            sensor_tokens=12,expert_tokens=len(self.expert_input.groups),
            total_tokens=13+len(self.expert_input.groups),
            sensor_preprocessing='training-fitted DOSE + linear history/time projections',
            expert_preprocessing='OFP training normalization, fixed asinh / 3, groupwise linear projection',
            expert_groups={k:[stat_feature_names[i] for i in indices] for k,indices in self.expert_input.groups.items()},
            independent_branch_encoders=0,independent_branch_heads=0,late_fusion_networks=0)

    def input_tokens(self,raw_x,raw_mask,stat_x):
        history = self.sensor_input(raw_x,raw_mask)+self.type_embedding[:,0:1]
        expert = self.expert_input(stat_x)+self.type_embedding[:,1:2]
        if self.training and self.stat_modality_dropout:
            keep = (torch.rand(len(stat_x),1,1,device=stat_x.device) >= self.stat_modality_dropout).to(expert.dtype)
            expert = expert*keep/(1-self.stat_modality_dropout)
        return torch.cat([self.module_token.expand(len(raw_x),-1,-1),history,expert],dim=1)

    def encode(self,raw_x,raw_mask,stat_x,return_explain=False,**unused):
        assert not return_explain, 'No independent branch representations exist in this architecture.'
        z = self.input_norm(self.input_tokens(raw_x,raw_mask,stat_x))
        for block in self.backbone:
            z = block(z)
        return self.output_norm(z[:,0])

    def forward(self,raw_x,raw_mask,stat_x):
        return self.classifier(self.encode(raw_x,raw_mask,stat_x)).squeeze(-1)

    def close(self):
        pass
