import random

import torch
import torch.nn as nn

from models.carots import transform_layer


class NegativeAugmentor(nn.Module):
    def __init__(self, cfg):
        super(NegativeAugmentor, self).__init__()
        self.cfg = cfg
        self.cfg_negative_augmentor = cfg.CAROTS.NEGATIVE_AUGMENTOR

    @torch.no_grad()
    def get_indices_to_disturb(self, causality_mtx):
        if self.cfg_negative_augmentor.DISTURB_ALL:
            indices = list(range(len(causality_mtx)))
            return indices
        
        def dfs(node, visited, subgraph):
            visited[node] = True
            subgraph.append(node)
            for neighbor, is_causal in enumerate(causality_mtx[node]):
                if is_causal and not visited[neighbor]:
                    if random.random() > self.cfg_negative_augmentor.CUTOFF_PROBABILITY:
                        dfs(neighbor, visited, subgraph)
                    else:
                        break
        
        visited = [False] * len(causality_mtx)
        subgraph = []
        
        start_node = random.choice(range(len(causality_mtx)))
        dfs(start_node, visited, subgraph)
        
        indices = subgraph
        return indices
    
    @torch.no_grad()
    def disturb(self, x, indices):
        transform_name = self.cfg_negative_augmentor.TRANSFORM_NAME
        transform_cfg = getattr(self.cfg.TRANSFORM, transform_name)
        transform_class = getattr(transform_layer, transform_name)
        transform_instance = transform_class(**transform_cfg)
        disturbed_x = transform_instance(x, indices)

        return disturbed_x

    def forward(self, x, causality_mtx):
        assert causality_mtx.dtype == torch.float, "causality_mtx elements must be of type float"
        causality_mtx = (causality_mtx > 0.5).int()
        indices_to_disturb = self.get_indices_to_disturb(causality_mtx)
        disturbed_x = self.disturb(x, indices_to_disturb)
        self.indices_to_disturb = indices_to_disturb

        assert not disturbed_x.requires_grad, "disturbed_x should not require gradients"
        return disturbed_x.clone().detach()