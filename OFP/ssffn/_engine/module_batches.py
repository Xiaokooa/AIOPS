import math
import torch
from torch.nn import functional as F

class ModuleGroupedLoader:

    def __init__(self, loader, modules_per_step=32):
        self.loader = loader
        self.modules_per_step = int(modules_per_step)

    def __len__(self):
        return math.ceil(len(self.loader) / self.modules_per_step)

    @staticmethod
    def pack(items):
        flags = []
        for group, item in enumerate(items):
            count = len(item[3])
            flags.append(torch.tensor([group, int(item[5].item())], dtype=torch.long).expand(count, 2))
        return (*[torch.cat([item[i] for item in items], dim=0) for i in range(5)], torch.cat(flags, dim=0))

    def __iter__(self):
        items = []
        for item in self.loader:
            items.append(item)
            if len(items) == self.modules_per_step:
                yield self.pack(items)
                items = []
        if items:
            yield self.pack(items)

def module_mean_loss(per_row_loss, weights, flags):
    ids = flags[:, 0].to(per_row_loss.device)
    groups = int(flags[-1, 0]) + 1
    numerator = torch.zeros(groups, device=per_row_loss.device, dtype=per_row_loss.dtype)
    denominator = torch.zeros_like(numerator)
    numerator.scatter_add_(0, ids, per_row_loss * weights)
    denominator.scatter_add_(0, ids, weights)
    return (numerator / denominator.clamp_min(1.0)).mean()

def module_topk_loss(logits, flags, k=4):
    normal_losses = []
    for group in range(int(flags[-1, 0]) + 1):
        positions = torch.where(flags[:, 0] == group)[0]
        if int(flags[positions[0], 1]) == 0:
            z = logits[positions.to(logits.device)]
            normal_losses.append(F.softplus(z.topk(min(k, len(z))).values).mean())
    return torch.stack(normal_losses).mean() if normal_losses else logits.sum() * 0.0
