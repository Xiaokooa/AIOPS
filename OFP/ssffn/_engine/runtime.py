from __future__ import annotations
import os
import numpy as np
import torch

def resolve_runtime_device(requested: str) -> str:
    requested = str(requested or 'cuda').strip()
    if requested.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but torch.cuda.is_available() is False. torch={torch.__version__} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        device = torch.device(requested)
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        _ = torch.empty(1, device=device)
        return str(device)
    return str(torch.device(requested))

def runtime_device_summary(device: str) -> str:
    pieces = [f'device={device}', f'torch={torch.__version__}', f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"]
    if str(device).startswith('cuda') and torch.cuda.is_available():
        dev = torch.device(device)
        idx = torch.cuda.current_device() if dev.index is None else int(dev.index)
        props = torch.cuda.get_device_properties(idx)
        pieces.extend([f'cuda_count={torch.cuda.device_count()}', f'cuda_index={idx}', f'cuda_name={props.name}', f'cuda_total_gb={props.total_memory / 1024 ** 3:.2f}'])
    return ' '.join(pieces)

def format_metric_summary(metrics: dict[str, float]) -> str:
    keys = ['final_score', 'f1_score', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'fn', 'tn', 'all_hit_cnt', 'all_predict_pos_cnt', 'all_true_pos_cnt', 'avg_lead_score', 'avg_lead_hour', 'min_lead_score', 'min_lead_hour', 'lead_pread_cnt', 'evaluated_module_cnt']
    parts: list[str] = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, (int, np.integer)) or key in {'tp', 'fp', 'fn', 'tn', 'all_hit_cnt', 'all_predict_pos_cnt', 'all_true_pos_cnt', 'lead_pread_cnt', 'evaluated_module_cnt'}:
            parts.append(f'{key}={int(float(value))}')
        else:
            parts.append(f'{key}={float(value):.6g}')
    return ' '.join(parts)
