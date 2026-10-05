"""Sensor measurements and grouped statistical features mapped to tokens."""
from collections import OrderedDict
import math
import torch
from torch import nn
from .distribution import fit_distribution

SENSORS = ['Temp','Curr','TxP0','RxP0','RxP1','RxP2','RxP3','RxP4','TxP1','TxP2','TxP3','TxP4']

class SensorDistribution(nn.Module):
    fit_distribution = fit_distribution
    def __init__(self):
        super().__init__()
        self.register_buffer('sensor_median',torch.zeros(12))
        self.register_buffer('sensor_scale',torch.ones(12))
        self.register_buffer('state_knots',torch.linspace(-3,3,17).expand(12,17).clone())

class SensorMeasurementEmbedding(SensorDistribution):
    def __init__(self,seq_len=32):
        super().__init__()
        self.current_projection=nn.Linear(16,64)
        # Keep the initialization draw order fixed across component ablations.
        for _ in range(2):
            nn.init.kaiming_uniform_(torch.empty(64,2*seq_len),a=math.sqrt(5))
            nn.init.uniform_(torch.empty(64),-1/math.sqrt(2*seq_len),1/math.sqrt(2*seq_len))
        self.identity=nn.Parameter(torch.empty(1,12,64))
        nn.init.normal_(self.identity,std=.02)
    def forward(self,values,mask):
        state=torch.asinh((values[:,:,:12]-self.sensor_median)/self.sensor_scale).clamp(-10,10)
        span=(self.state_knots[:,1:]-self.state_knots[:,:-1]).clamp_min(.001)
        basis=((state[:,-1,:,None]-self.state_knots[:,:-1])/span).clamp(0,1)
        return self.current_projection(basis)+self.identity

class ProjectionInitializer(nn.Module):
    def __init__(self,names):
        super().__init__()
        # Fixed initialization pools preserve the declared seed schedule.
        self.projections=nn.ModuleList([nn.Linear(118,64),nn.Linear(42,64)])
        self.identity=nn.Parameter(torch.empty(1,2,64))
        nn.init.normal_(self.identity,std=.02)


class StatisticalFeatureEmbedding(nn.Module):
    def __init__(self,names,initializer):
        super().__init__()
        level=[names.index(f'Fe{s}{op}') for s in SENSORS for op in ('Min','Diff','Max')]
        variation=[names.index(f'Fe{s}{op}') for s in SENSORS for op in ('Skew','Kurt','Std')]
        relations=[i for i,n in enumerate(names) if n.startswith('Fe') and i not in level+variation]
        self.groups=OrderedDict(level=level,variability=variation,relations=relations)
        self.projections=nn.ModuleList()
        identities=[]
        state=torch.random.get_rng_state()
        for j,(group,indices) in enumerate(self.groups.items()):
            self.register_buffer(f'indices_{j}',torch.tensor(indices,dtype=torch.long))
            layer=nn.Linear(len(indices),64)
            token=1 if group=='relations' else 0
            with torch.no_grad():
                columns=indices
                layer.weight.copy_(initializer.projections[0].weight[:,columns])
                layer.bias.copy_(initializer.projections[token].bias)
            self.projections.append(layer)
            identities.append(initializer.identity[:,token:token+1].detach().clone())
        self.identity=nn.Parameter(torch.cat(identities,1))
        torch.random.set_rng_state(state)
    def forward(self,features):
        values=torch.asinh(features).clamp(-10,10)/3
        return torch.stack([layer(values.index_select(1,getattr(self,f'indices_{i}')))
                            for i,layer in enumerate(self.projections)],1)+self.identity
