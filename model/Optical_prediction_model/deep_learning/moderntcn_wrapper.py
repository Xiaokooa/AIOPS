"""ModernTCN binary classifier wrapper for the optical R4 task.

This reuses the official ModernTCN classification implementation under
``model/ModernTCN-main`` and adapts it to the local training loop:

Input : (B, L, N) values
Mask  : (B, L, N) observed mask
Output: (B,) binary logits or (B, output_dim) multi-horizon logits
"""
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

MODERNTCN_ROOT = (
    Path(__file__).resolve().parents[3]
    / "model"
    / "ModernTCN-main"
    / "ModernTCN-classification"
)


def _load_module_from_path(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_model_class():
    """Load official ModernTCN while avoiding package-name collisions."""
    models_dir = MODERNTCN_ROOT / "models"
    layers_dir = MODERNTCN_ROOT / "layers"

    saved = {
        key: sys.modules.get(key)
        for key in ["models", "models.ModernTCN_Layer", "layers", "layers.RevIN"]
    }

    try:
        layer_mod = _load_module_from_path(
            "moderntcn_layer_module",
            models_dir / "ModernTCN_Layer.py",
        )
        revin_mod = _load_module_from_path(
            "moderntcn_revin_module",
            layers_dir / "RevIN.py",
        )

        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [str(models_dir)]
        layers_pkg = types.ModuleType("layers")
        layers_pkg.__path__ = [str(layers_dir)]
        sys.modules["models"] = models_pkg
        sys.modules["models.ModernTCN_Layer"] = layer_mod
        sys.modules["layers"] = layers_pkg
        sys.modules["layers.RevIN"] = revin_mod

        model_mod = _load_module_from_path(
            "moderntcn_classification_model",
            models_dir / "ModernTCN.py",
        )
        return model_mod.ModernTCN
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


OfficialModernTCN = _load_official_model_class()


@dataclass
class ModernTCNCfg:
    seq_len: int = 288
    n_sensors: int = 12
    patch_size: int = 16
    patch_stride: int = 8
    downsample_ratio: int = 2
    ffn_ratio: int = 1
    num_blocks: tuple[int, ...] = (2,)
    large_size: tuple[int, ...] = (31,)
    small_size: tuple[int, ...] = (5,)
    dims: tuple[int, ...] = (64,)
    dw_dims: tuple[int, ...] = (64,)
    dropout: float = 0.2
    head_dropout: float = 0.2
    class_dropout: float = 0.1
    use_mask_channel: bool = True
    use_multi_scale: bool = False
    small_kernel_merged: bool = False
    revin: bool = False
    output_dim: int = 1


class ModernTCNClassifier(nn.Module):
    """Official ModernTCN classification backbone with a binary logit head."""

    def __init__(self, cfg: ModernTCNCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask = cfg.use_mask_channel
        self.n_vars = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)

        self.model = OfficialModernTCN(
            task_name="classification",
            patch_size=cfg.patch_size,
            patch_stride=cfg.patch_stride,
            stem_ratio=1,
            downsample_ratio=cfg.downsample_ratio,
            ffn_ratio=cfg.ffn_ratio,
            num_blocks=list(cfg.num_blocks),
            large_size=list(cfg.large_size),
            small_size=list(cfg.small_size),
            dims=list(cfg.dims),
            dw_dims=list(cfg.dw_dims),
            nvars=self.n_vars,
            small_kernel_merged=cfg.small_kernel_merged,
            backbone_dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
            use_multi_scale=cfg.use_multi_scale,
            revin=cfg.revin,
            affine=True,
            subtract_last=False,
            freq=None,
            seq_len=cfg.seq_len,
            c_in=(self.n_vars,),
            individual=False,
            target_window=1,
            class_drop=cfg.class_dropout,
            class_num=cfg.output_dim,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x/mask: (B, L, N). ModernTCN expects (B, M, L).
        if self.use_mask:
            x = torch.cat([x, mask], dim=-1)
        x = x.permute(0, 2, 1)
        out = self.model(x)
        if self.cfg.output_dim == 1:
            out = out.squeeze(-1)
        return out
