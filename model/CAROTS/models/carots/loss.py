import torch
import torch.nn.functional as F


def loss_fn(x, cfg):
    n = x.size(0) // 2
    
    # Normalize the input tensor
    x = F.normalize(x, p=2, dim=1)

    # Compute the similarity matrix
    sim_mtx = torch.matmul(x, x.T)
    
    sim_mtx_pos = sim_mtx[:n, :n]
    indices = torch.where(sim_mtx_pos < cfg.CAROTS.SIM_THRESHOLD)
    # Add n_orig to the column indices
    indices_neg = (indices[0], indices[1] + n)
    
    # Make the values -inf for elements corresponding to indices
    sim_mtx[indices] = float('-inf')
    sim_mtx[indices_neg] = float('-inf')
    
    sim_mtx = torch.exp(sim_mtx / cfg.CAROTS.TEMPERATURE)
    sim_mtx = sim_mtx - torch.diag_embed(torch.diagonal(sim_mtx))

    # Normalize the similarity matrix
    denominator = sim_mtx + sim_mtx[:, n:].sum(dim=1, keepdim=True)
    sim_mtx = sim_mtx / denominator

    # Apply log transformation
    eps = 1e-12
    sim_mtx = -torch.log(sim_mtx + eps)
    sim_mtx.fill_diagonal_(0)

    # Compute the loss
    loss = torch.sum(sim_mtx[torch.where(sim_mtx_pos >= cfg.CAROTS.SIM_THRESHOLD)]) / (len(torch.where(sim_mtx_pos >= cfg.CAROTS.SIM_THRESHOLD)[0]) - n)

    return loss
