"""Sensor and Statistical Feature Fusion Network."""
import torch
from torch import nn
from ._engine.embeddings import (SENSORS, SensorDistribution, SensorMeasurementEmbedding,
                                 ProjectionInitializer, StatisticalFeatureEmbedding)

from ._engine.feature_schema import STATISTIC_NAMES as _STATISTIC_NAMES

VARIANTS=('full','no_sensor','no_statistics','no_sit','no_hss')

def cached_statistic_names():
    return list(_STATISTIC_NAMES)

STATISTIC_NAMES=tuple(_STATISTIC_NAMES)

class SSFFNModel(nn.Module):
    def __init__(self,module_variant='full',seq_len=32,stat_feature_names=None,stat_modality_dropout=.15,**unused):
        super().__init__()
        if module_variant not in VARIANTS: raise ValueError(module_variant)
        names=list(stat_feature_names or cached_statistic_names())
        self.module_variant,self.seq_len,self.latent_dim=module_variant,seq_len,64
        self.stat_modality_dropout=stat_modality_dropout
        self.sensor_input=SensorMeasurementEmbedding(seq_len)
        initializer=ProjectionInitializer(names)
        self.statistical_input=initializer
        self.type_embedding=nn.Parameter(torch.empty(1,2,64))
        self.module_token=nn.Parameter(torch.empty(1,1,64))
        nn.init.normal_(self.type_embedding,std=.02)
        nn.init.normal_(self.module_token,std=.02)
        self.input_norm=nn.LayerNorm(64)
        self.backbone=nn.ModuleList([nn.TransformerEncoderLayer(64,4,192,.15,
            activation='gelu',batch_first=True,norm_first=True) for _ in range(3)])
        self.output_norm=nn.LayerNorm(64)
        self.classifier=nn.Linear(64,1)
        if module_variant!='no_statistics':
            self.statistical_input=StatisticalFeatureEmbedding(names,initializer)
        if module_variant=='no_sensor':
            self.sensor_input=SensorDistribution()
            self.type_embedding=nn.Parameter(self.type_embedding[:,1:].detach().clone())
        elif module_variant=='no_statistics':
            del self.statistical_input
            self.type_embedding=nn.Parameter(self.type_embedding[:,:1].detach().clone())
        elif module_variant=='no_sit':
            del self.backbone
            del self.module_token
        groups=self.statistical_input.groups if module_variant!='no_statistics' else {}
        self.model_config=dict(architecture='SSFFN',width=64,
            depth=0 if module_variant=='no_sit' else 3,heads=4,feedforward=192,dropout=.15,
            sensor_tokens=0 if module_variant=='no_sensor' else 12,
            statistical_tokens=len(groups),module_tokens=0 if module_variant=='no_sit' else 1,
            active_statistic_features=sum(len(ids) for ids in groups.values()),
            statistical_groups={k:[names[i] for i in ids] for k,ids in groups.items()},
            token_feature_dimensions=[len(ids) for ids in groups.values()])
        cfg=self.model_config
        cfg['total_tokens']=cfg['sensor_tokens']+cfg['statistical_tokens']+cfg['module_tokens']

    def input_tokens(self,raw_x,raw_mask,stat_x):
        parts=[]
        if self.module_variant!='no_sit': parts.append(self.module_token.expand(len(raw_x),-1,-1))
        if self.module_variant!='no_sensor': parts.append(self.sensor_input(raw_x,raw_mask)+self.type_embedding[:,:1])
        if self.module_variant!='no_statistics':
            index=0 if self.module_variant=='no_sensor' else 1
            tokens=self.statistical_input(stat_x)+self.type_embedding[:,index:index+1]
            if self.training and self.stat_modality_dropout:
                keep=(torch.rand(len(stat_x),1,1,device=stat_x.device)>=self.stat_modality_dropout).to(tokens.dtype)
                tokens=tokens*keep/(1-self.stat_modality_dropout)
            parts.append(tokens)
        return torch.cat(parts,dim=1)

    def encode(self,raw_x,raw_mask,stat_x,**unused):
        values=self.input_norm(self.input_tokens(raw_x,raw_mask,stat_x))
        if self.module_variant=='no_sit': return self.output_norm(.5*(values[:,:12].mean(1)+values[:,12:].mean(1)))
        for layer in self.backbone: values=layer(values)
        return self.output_norm(values[:,0])

    def forward(self,raw_x,raw_mask,stat_x):
        return self.classifier(self.encode(raw_x,raw_mask,stat_x)).squeeze(-1)

    def close(self): pass

def build_model(variant='full',training_seed=42,normalization=None,**kwargs):
    kwargs.setdefault('module_variant',variant)
    return SSFFNModel(**kwargs)

class SSFFN(nn.Module):
    """Inputs: training-standardized sensors [B,12] and statistics [B,80].

    Columns follow SENSORS and STATISTIC_NAMES. Outputs are observation logits.
    Use the predict command for CSV preprocessing and checkpoint loading.
    """
    def __init__(self,variant='full'):
        super().__init__()
        self.network=build_model(variant)
    def forward(self,current_sensors,statistics):
        if current_sensors.ndim!=2 or current_sensors.shape[1]!=12: raise ValueError('Expected sensors [batch,12]')
        if statistics.shape!=(len(current_sensors),80): raise ValueError('Expected statistics [batch,80]')
        raw=torch.cat([current_sensors,current_sensors.new_zeros(len(current_sensors),1)],1)[:,None]
        return self.network(raw,torch.ones_like(raw),statistics)
