# Numerical core retained for compatibility with the archived experiments.
"""Predeclared A1--A6 interventions, preserving the frozen shared-SIT source."""
import copy
import hashlib
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from OFP.ssffn._engine.backbone import SharedInputSIT, SensorInput

VARIANTS = ['no_expert','no_sensor','scalar_state','no_history','uniform_attention','uniform_sampling']

def state_basis(module,x):
    state=torch.asinh((x[:,:,:12]-module.sensor_median)/module.sensor_scale).clamp(-10,10)
    span=(module.state_knots[:,1:]-module.state_knots[:,:-1]).clamp_min(.001)
    basis=((state[:,-1,:,None]-module.state_knots[:,:-1])/span).clamp(0,1)
    return state,basis

class ScalarSensorInput(SensorInput):
    def forward(self,x,mask):
        state,_=state_basis(self,x)
        history=torch.cat([(state/10).transpose(1,2),mask[:,:,:12].transpose(1,2)],dim=-1)
        gap=torch.cat([torch.asinh(x[:,:,12]).clamp(-10,10)/10,mask[:,:,12]],dim=1)
        return self.current_projection(state[:,-1,:,None]/10)+self.identity+\
            self.history_strength.tanh()*self.history_projection(history)+\
            self.time_strength.tanh()*self.time_projection(gap).unsqueeze(1)

class CurrentSensorInput(SensorInput):
    def forward(self,x,mask):
        _,basis=state_basis(self,x)
        return self.current_projection(basis)+self.identity

class DistributionOnly(nn.Module):
    fit_distribution=SensorInput.fit_distribution
    def __init__(self,source):
        super().__init__()
        for name in ['sensor_median','sensor_scale','state_knots']:
            self.register_buffer(name,getattr(source,name).clone())

class UniformBlock(nn.Module):
    """Replace QK attention by uniform V aggregation, retaining attention dropout."""
    def __init__(self,source):
        super().__init__()
        self.value=nn.Linear(64,64)
        with torch.no_grad():
            self.value.weight.copy_(source.self_attn.in_proj_weight[128:])
            self.value.bias.copy_(source.self_attn.in_proj_bias[128:])
        self.output=copy.deepcopy(source.self_attn.out_proj)
        for name in ['norm1','norm2','linear1','linear2','dropout','dropout1','dropout2']:
            setattr(self,name,copy.deepcopy(getattr(source,name)))
        self.attention_dropout=source.self_attn.dropout
        self.activation=source.activation
    def forward(self,x):
        b,n,d=x.shape
        v=self.value(self.norm1(x)).reshape(b,n,4,16).transpose(1,2)
        attention=torch.full((b,4,n,n),1/n,dtype=v.dtype,device=v.device)
        attention=F.dropout(attention,p=self.attention_dropout,training=self.training)
        mixed=(attention@v).transpose(1,2).reshape(b,n,d)
        x=x+self.dropout1(self.output(mixed))
        z=self.norm2(x)
        return x+self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(z)))))

class AblatedSIT(SharedInputSIT):
    def __init__(self,*args,ablation='full',**kwargs):
        super().__init__(*args,**kwargs)
        assert ablation in ['full',*VARIANTS]
        self.ablation=ablation
        # Build the identical untrained Full first; unchanged parameters retain
        # exactly its initialization. Local replacement initialization must not
        # shift the RNG state used by later training dropout.
        rng=torch.random.get_rng_state()
        if ablation=='no_expert':
            del self.expert_input
            self.type_embedding=nn.Parameter(self.type_embedding[:,:1].detach().clone())
        elif ablation=='no_sensor':
            self.sensor_input=DistributionOnly(self.sensor_input)
            self.type_embedding=nn.Parameter(self.type_embedding[:,1:].detach().clone())
        elif ablation=='scalar_state':
            self.sensor_input.__class__=ScalarSensorInput
            self.sensor_input.current_projection=nn.Linear(1,64)
        elif ablation=='no_history':
            self.sensor_input.__class__=CurrentSensorInput
            for name in ['history_projection','time_projection','history_strength','time_strength']:
                delattr(self.sensor_input,name)
        elif ablation=='uniform_attention':
            self.backbone=nn.ModuleList([UniformBlock(block) for block in self.backbone])
        torch.random.set_rng_state(rng)
        cfg=self.temporal_encoder_cfg
        cfg.update(ablation=ablation,sensor_tokens=0 if ablation=='no_sensor' else 12,
                   expert_tokens=0 if ablation=='no_expert' else 2)
        cfg['total_tokens']=1+cfg['sensor_tokens']+cfg['expert_tokens']
        cfg['active_expert_features']=0 if ablation=='no_expert' else 160
        cfg['attention']='uniform V aggregation' if ablation=='uniform_attention' else 'learned QK attention'
        cfg['history_residual']=ablation not in ['no_history','no_sensor']
        cfg['current_state_basis']='scalar' if ablation=='scalar_state' else 'quantile'
        cfg['initialization']='Identical untrained Full first; unchanged weights preserved; RNG restored after replacements'
        digest=hashlib.sha256()
        for name,param in self.named_parameters():
            digest.update(name.encode());digest.update(param.detach().cpu().numpy().tobytes())
        cfg['initialization_sha256']=digest.hexdigest()

    def input_tokens(self,raw_x,raw_mask,stat_x):
        parts=[self.module_token.expand(len(raw_x),-1,-1)]
        if self.ablation!='no_sensor':
            parts.append(self.sensor_input(raw_x,raw_mask)+self.type_embedding[:,:1])
        if self.ablation!='no_expert':
            type_index=0 if self.ablation=='no_sensor' else 1
            expert=self.expert_input(stat_x)+self.type_embedding[:,type_index:type_index+1]
            if self.training and self.stat_modality_dropout:
                keep=(torch.rand(len(stat_x),1,1,device=stat_x.device)>=self.stat_modality_dropout).to(expert.dtype)
                expert=expert*keep/(1-self.stat_modality_dropout)
            parts.append(expert)
        return torch.cat(parts,dim=1)

class UniformNormalSampler:
    """A6 samples 8 windows from all valid positions; never scores a model."""
    def __init__(self,dataset,raw_indices,seed):
        self.dataset=dataset; self.seed=int(seed)
        self.normal=[i for i,name in enumerate(dataset.file_names)
                     if dataset.cache.label_by_file[name]==0 and len(dataset.selected_positions[i])]
        self.quota={i:len(dataset.selected_positions[i]) for i in self.normal}
        fit=[dataset.cache.get(name)[1][positions][:,np.asarray(raw_indices[:12])]
             for name,positions in zip(dataset.file_names,dataset.selected_positions) if len(positions)]
        values=np.concatenate(fit)
        if len(values)>100000:
            values=values[np.random.default_rng(seed).choice(len(values),100000,replace=False)]
        # Stored for a matched preprocessing audit; not used for selection.
        self.knots=np.quantile(values,np.linspace(0,1,17),axis=0).T
        self.eligible={i:np.flatnonzero(dataset.cache.get(dataset.file_names[i])[2]) for i in self.normal}
        self.select(0)
    def select(self,epoch):
        for i in self.normal:
            rng=np.random.default_rng(self.seed+i*1009+epoch*100003)
            positions=np.sort(rng.choice(self.eligible[i],self.quota[i],replace=False))
            self.dataset.selected_positions[i]=positions
            self.dataset.selected_labels[i]=np.zeros(len(positions),dtype=np.int8)
            self.dataset.selected_weights[i]=np.ones(len(positions),dtype=np.float32)
    def refresh(self,model,cfg,raw_indices,stat_indices,stat_input_indices,temporal_summary_mode,score_function,epoch):
        self.select(epoch)
        return dict(epoch=epoch,normal_modules=len(self.normal),scored_windows=0,
                    selected_windows=sum(self.quota.values()),strategy='uniform over all valid normal windows')
