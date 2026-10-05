# Numerical core retained for compatibility with the archived experiments.
from __future__ import annotations


import torch


def primary_logit(output: torch.Tensor | tuple) -> torch.Tensor:
    logits = output[0] if isinstance(output, tuple) else output
    if logits.ndim > 1 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    return logits


def primary_horizon_tensor(values: torch.Tensor, primary_index: int = -1) -> torch.Tensor:
    if values.ndim > 1:
        return values[:, int(primary_index)]
    return values


def split_alarm_aux_logits(logits: torch.Tensor, aux_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split model outputs into an event alarm logit and auxiliary horizon logits.

    First-warning models emit `[alarm, horizon...]`.  In the current
    OFP-compatible 24h-lookback setting this is `[alarm, h1]`.  Older
    checkpoints emitted only the horizon logits; for those, the primary horizon
    remains the alarm proxy for backward compatibility.
    """
    aux_dim = int(aux_dim)
    if logits.ndim > 1 and logits.shape[-1] == aux_dim + 1:
        return logits[..., 0], logits[..., 1:]
    if logits.ndim > 1 and logits.shape[-1] == aux_dim:
        return primary_horizon_tensor(logits), logits
    if logits.ndim > 1 and logits.shape[-1] == 1:
        squeezed = logits.squeeze(-1)
        return squeezed, squeezed.unsqueeze(-1)
    return logits, logits.unsqueeze(-1) if logits.ndim == 1 else logits
