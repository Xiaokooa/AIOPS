# Numerical core retained for compatibility with the archived experiments.
"""A4-based expert-path study: all variants keep one shared SIT and BCE."""
from collections import OrderedDict
import hashlib
import numpy as np
import torch
from torch import nn
from OFP.ssffn._engine.input_ablation import AblatedSIT

CONFIGS={
    'a4_typed':dict(grouping='typed',expert=True,dropout=.15),
    'a4_no_expert':dict(grouping='none',expert=False,dropout=0.),
    'a4_no_dropout':dict(grouping='typed',expert=True,dropout=0.),
    'a4_semantic':dict(grouping='semantic80',expert=True,dropout=.15),
    'a4_random80':dict(grouping='random80',expert=True,dropout=.15),
}
RANDOM_GROUP_SEED=20260928

class ExpertFollowupSIT(AblatedSIT):
    def __init__(self,*args,followup='a4_typed',training_seed=42,**kwargs):
        info=CONFIGS[followup]
        assert float(kwargs.get('stat_modality_dropout',.15))==info['dropout']
        super().__init__(*args,ablation='no_history',**kwargs)
        self.followup=followup
        names=kwargs['stat_feature_names']
        rng=torch.random.get_rng_state()
        if not info['expert']:
            del self.expert_input
            self.type_embedding=nn.Parameter(self.type_embedding[:,:1].detach().clone())
        elif info['grouping']!='typed':
            expert=self.expert_input
            source_columns={i:layer.weight[:,j].detach().clone()
                for layer,indices in zip(expert.projections,expert.groups.values())
                for j,i in enumerate(indices)}
            source_bias=[layer.bias.detach().clone() for layer in expert.projections]
            if info['grouping']=='semantic80':
                groups=OrderedDict(statistical_context=[i for i,n in enumerate(names) if n.startswith('Fe')],
                    risk_and_quality=[i for i,n in enumerate(names) if not n.startswith('Fe')])
            else:
                order=np.random.default_rng(RANDOM_GROUP_SEED).permutation(len(names))
                groups=OrderedDict(random_a=sorted(order[:80].tolist()),random_b=sorted(order[80:].tolist()))
            assert [len(v) for v in groups.values()]==[80,80]
            assert sorted(i for g in groups.values() for i in g)==list(range(160))
            layers=[]
            for j,indices in enumerate(groups.values()):
                layer=nn.Linear(80,64)
                # Preserve each feature's original initial projection column.
                # Semantic and random grouping share this matching rule.
                with torch.no_grad():
                    layer.weight.copy_(torch.stack([source_columns[i] for i in indices],dim=1))
                    layer.bias.copy_(source_bias[j])
                layers.append(layer)
                setattr(expert,f'indices_{j}',torch.tensor(indices,dtype=torch.long))
            expert.groups=groups;expert.projections=nn.ModuleList(layers)
        torch.random.set_rng_state(rng)
        cfg=self.temporal_encoder_cfg
        cfg.update(followup=followup,training_seed=int(training_seed),data_seed=42,
            history_residual=False,expert_tokens=2 if info['expert'] else 0,
            total_tokens=15 if info['expert'] else 13,active_expert_features=160 if info['expert'] else 0,
            layout=info['grouping'],expert_modality_dropout=info['dropout'],
            expert_groups={k:[names[i] for i in v] for k,v in self.expert_input.groups.items()} if info['expert'] else {},
            initialization='Construct identical untrained A4; retain common weights and feature-wise projection columns; restore CPU RNG',
            random_group_seed=RANDOM_GROUP_SEED if info['grouping']=='random80' else None)
        h=hashlib.sha256()
        for name,p in self.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
        cfg['initialization_sha256']=h.hexdigest()

    def input_tokens(self,raw_x,raw_mask,stat_x):
        parts=[self.module_token.expand(len(raw_x),-1,-1),self.sensor_input(raw_x,raw_mask)+self.type_embedding[:,:1]]
        if CONFIGS[self.followup]['expert']:
            expert=self.expert_input(stat_x)+self.type_embedding[:,1:2]
            if self.training and self.stat_modality_dropout:
                keep=(torch.rand(len(stat_x),1,1,device=stat_x.device)>=self.stat_modality_dropout).to(expert.dtype)
                expert=expert*keep/(1-self.stat_modality_dropout)
            parts.append(expert)
        return torch.cat(parts,dim=1)
