"""Load self-contained checkpoints and score complete module CSV files."""
import json
from pathlib import Path

import numpy as np
import torch

from ._engine import training as engine
from ._engine.data import TrainingConfig, FeatureNormStats, FeatureCache
from .model import build_model


def load_checkpoint(path, device='cpu'):
    # Release checkpoints contain tensors and primitive containers only.
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact.get('release_model') != 'SSFFN':
        raise ValueError('Use a checkpoint exported by the SSFFN release trainer')
    cfg = TrainingConfig(**artifact['cfg'])
    cfg.device = engine.resolve_runtime_device(device)
    cfg.module_cache_dir = ''  # Never reuse a training-data cache for incoming CSVs.
    cfg.max_cached_files = 32
    stats = FeatureNormStats(**artifact['normalization'])
    names = engine.get_feature_names(cfg)
    if stats.feature_names != names:
        raise ValueError('Checkpoint feature schema does not match the 80-feature SSFFN release')
    raw = [names.index(n) for n in artifact['raw_features']]
    stat = [names.index(n) for n in artifact['base_stat_features']]
    mean, std = stats.arrays()
    model = build_model(artifact['module_variant'], artifact['training_seed'],
                        normalization=dict(raw_mean=mean[raw[:12]], raw_std=std[raw[:12]]))
    model.load_state_dict(artifact['state_dict'], strict=True)
    model.to(cfg.device).eval()
    return model, cfg, stats, raw, stat, artifact['stat_input_indices']


def predict(checkpoint, data_dir, file_names, output_dir, threshold, device='cpu'):
    from .splits import validate_index
    import pandas as pd
    validate_index(pd.DataFrame(dict(file_name=file_names, Label=0, folder_index=1)))
    if not file_names:
        raise ValueError('No CSV files were selected')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must lie in [0,1]')
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Prediction output must be empty: {output_dir}')
    model, cfg, stats, raw, stat, indices = load_checkpoint(checkpoint, device)
    mean, std = stats.arrays()
    cache = FeatureCache(Path(data_dir), mean, std, cfg, {})
    estimator = engine.frozen_encoder_head(model)
    # Scoring uses every row. Target labels and validity-for-training do not
    # select inference positions and are never passed as neural inputs.
    frames = engine.score_fused_files_to_memory(estimator, model, cache, list(file_names),
             cfg, raw, stat, indices, 'none', 0, 'predict')
    engine.write_threshold_predictions(frames, output_dir, threshold)
    return frames
