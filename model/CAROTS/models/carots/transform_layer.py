import random
import numpy as np
import torch
import torch.nn as nn


class AddBias(nn.Module):
    def __init__(self, **kwargs):
        super(AddBias, self).__init__()
        self.bias_candidates = kwargs['bias_candidates']
        self.percent = kwargs['percent']

    def forward(self, ts, indices):
        n_steps = int((ts.shape[1] * self.percent))
        ts_bias = ts.clone().detach()
        
        steps = np.random.choice(ts.shape[1], n_steps, replace=False)
        bias = torch.tensor(random.choices(self.bias_candidates, k=len(ts)*n_steps*len(indices)), device=ts.device).view(len(ts), n_steps, len(indices))
        for i, step in enumerate(steps):
            ts_bias[:, step, indices] += bias[:, i, :]
        return ts_bias
