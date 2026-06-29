import copy

import torch
import torch.nn.functional as F


class Scorer():
    def __init__(self, cfg, model):
        super(Scorer, self).__init__()
        self.cfg = cfg
        self.model = model
    
    @torch.no_grad()
    def get_anomaly_scores(self, x):
        is_training = copy.deepcopy(self.model.training)
        self.model.eval()

        # Forward pass through the model
        reconstructed_x = self.model(x)
        
        # Calculate reconstruction loss
        loss = F.mse_loss(reconstructed_x, x, reduction='none')
        
        # Sum the loss over the feature dimensions
        score = loss.sum(dim=tuple(range(1, loss.dim())))
        
        # Restore the original training state of the model
        self.model.train(is_training)
        
        return score