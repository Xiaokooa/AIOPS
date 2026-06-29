import copy

import torch
import torch.nn.functional as F


class Scorer():
    def __init__(self, cfg, model):
        super(Scorer, self).__init__()
        self.cfg = cfg
        self.model = model

        self.centroid_wo_norm, self.centroid = self.init_centroid()
    
    @torch.no_grad()
    def init_centroid(self):
        from datasets.loader import construct_loader
        from trainer import prepare_inputs

        is_training = copy.deepcopy(self.model.training)
        self.model.eval()
        
        loader = construct_loader(self.cfg, split="train")
        outputs = []
        for inputs in loader:
            inputs, _ = prepare_inputs(inputs)
            if self.cfg.CAROTS.POSITIVE_AUGMENTOR.ENABLE:
                output = self.model(inputs, positive_augment=True, negative_augment=True)
            else:
                output = self.model(inputs, positive_augment=False, negative_augment=True)
            outputs.append(output[:(len(output) // 2)])
            
        outputs = torch.concat(outputs, dim=0)
        centroid_wo_norm = torch.mean(outputs, dim=0, keepdim=True)
        
        outputs = F.normalize(outputs, p=2, dim=1)
        centroid = torch.mean(outputs, dim=0, keepdim=True)

        if is_training:
            self.model.train()

        return centroid_wo_norm, centroid
    
    @torch.no_grad()
    def get_anomaly_scores(self, x):
        is_training = copy.deepcopy(self.model.training)
        self.model.eval()

        if self.cfg.SCORER.TYPE == "causal_discoverer":            
            x, y = x[:, :self.cfg.CUTS_PLUS.INPUT_STEP], x[:, self.cfg.CUTS_PLUS.INPUT_STEP:]
            Graph = (self.model.causal_discoverer.causality_mtx > 0.5).float()
            Graph = Graph[None].expand(x.size(0), -1, -1)
            y_pred = self.model.causal_discoverer(x, Graph)
            y_pred = y_pred.transpose(1, 2)
            assert y.shape == y_pred.shape  #  (B, T, N)
            score = F.mse_loss(y_pred, y, reduction='none').mean(dim=(1, 2))
            return score
        
        x = self.model(x, positive_augment=False, negative_augment=False)

        if self.cfg.SCORER.TYPE == "l2":
            score = torch.cdist(x, self.centroid_wo_norm).squeeze()
        elif self.cfg.SCORER.TYPE == "cos":
            centroid_normalized = F.normalize(self.centroid, dim=1)
            x_normalized = F.normalize(x, dim=1)
            score = -torch.matmul(x_normalized, centroid_normalized.t()).squeeze() * 0.5 + 0.5
        else:
            raise ValueError("Unsupported SCORER.TYPE")

        if is_training:
            self.model.train()

        return score
