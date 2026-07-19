"""PatchTST encoder + binary classification head.

PatchTST (Nie et al., ICLR 2023) is channel-independent: each variable is
patched and embedded separately, then a shared Transformer processes patches.
We re-use its backbone and replace the forecasting head with a pooled
classification head.

Input :  (B, L, N)  -- raw normalized time series
Mask  :  (B, L, N)  -- 1 = valid (used only for masked-aware normalization)
Output:  logits (B,) or (B, output_dim)
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PATCHTST_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "PatchTST" / "PatchTST-main" / "PatchTST_supervised"


def _load_module_from_path(name: str, file_path: Path):
    """Load a single .py file as a uniquely-named module to avoid `layers` clashing
    with iTransformer's `layers` package (different code, same import name)."""
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# PatchTST_backbone imports from `layers.PatchTST_layers` and `layers.RevIN`
# via simple `from layers.X import Y` style. We need these on sys.path with the
# *PatchTST* version, but iTransformer also installed a `layers` package. The
# safest fix: load all PatchTST `layers/*.py` as a synthetic `patchtst_layers`
# package, then patch sys.modules so `from layers.X import Y` resolves to
# PatchTST's files when executing PatchTST_backbone.

_patchtst_layers_dir = PATCHTST_ROOT / "layers"
# Pre-load each submodule.
_layers_mods = {}
for _name in ["PatchTST_layers", "RevIN", "PatchTST_backbone"]:
    pass  # will load below in correct order

# Order matters: backbone depends on PatchTST_layers + RevIN.
_pl = _load_module_from_path("patchtst_layers_module", _patchtst_layers_dir / "PatchTST_layers.py")
_rv = _load_module_from_path("patchtst_revin_module", _patchtst_layers_dir / "RevIN.py")
# Re-inject under `layers` namespace so `from layers.PatchTST_layers import ...`
# inside PatchTST_backbone.py resolves correctly. We rebuild the `layers`
# package to point to PatchTST's directory temporarily.
import types as _types
_layers_pkg = _types.ModuleType("layers")
_layers_pkg.__path__ = [str(_patchtst_layers_dir)]  # make it a package
sys.modules["layers"] = _layers_pkg
sys.modules["layers.PatchTST_layers"] = _pl
sys.modules["layers.RevIN"] = _rv
_bb = _load_module_from_path("patchtst_backbone_module", _patchtst_layers_dir / "PatchTST_backbone.py")
# Note: after this, `layers` in sys.modules now points to PatchTST. If anything
# else later imports iTransformer `layers`, it must come BEFORE this file in
# the import order. (In our pipeline, models.py is imported earlier, so iT's
# `layers` is captured by that point as concrete class references, not as a
# module name lookup -- safe.)

TSTiEncoder = _bb.TSTiEncoder
RevIN = _rv.RevIN


@dataclass
class PatchTSTCfg:
    seq_len: int = 288
    n_sensors: int = 12
    patch_len: int = 16
    stride: int = 8
    use_mask_channel: bool = True
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    attn_dropout: float = 0.0
    revin: bool = True
    output_dim: int = 1


class PatchTSTClassifier(nn.Module):
    """Channel-independent patched-Transformer for binary failure prediction."""

    def __init__(self, cfg: PatchTSTCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_mask = cfg.use_mask_channel
        # If using mask channel, treat it as N extra channels (each channel
        # independently patched and embedded).
        self.n_vars = cfg.n_sensors * (2 if cfg.use_mask_channel else 1)

        # Patch geometry
        patch_num = int((cfg.seq_len - cfg.patch_len) / cfg.stride + 1)
        # End padding the same way PatchTST does so info at the tail isn't lost.
        self.padding_patch = "end"
        patch_num += 1
        self.patch_num = patch_num

        self.pad = nn.ReplicationPad1d((0, cfg.stride))

        # RevIN on the value channels only (mask channels remain {0,1}).
        self.revin = cfg.revin
        if self.revin:
            self.revin_layer = RevIN(cfg.n_sensors, affine=True, subtract_last=False)

        # Channel-independent encoder.
        self.backbone = TSTiEncoder(
            c_in=self.n_vars,
            patch_num=patch_num,
            patch_len=cfg.patch_len,
            n_layers=cfg.e_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            attn_dropout=cfg.attn_dropout,
            dropout=cfg.dropout,
        )

        head_nf = self.n_vars * cfg.d_model * patch_num
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(head_nf),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_nf, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.output_dim),
        )

    def _patchify(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, nvars, L) -> padded -> unfold -> (B, nvars, patch_num, patch_len)
        z = self.pad(z)
        z = z.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)
        return z.permute(0, 1, 3, 2)  # (B, nvars, patch_len, patch_num)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N), mask: (B, L, N)
        # RevIN on values
        if self.revin:
            x = self.revin_layer(x, "norm")
        # Concatenate mask channels.
        if self.use_mask:
            inp = torch.cat([x, mask], dim=-1)        # (B, L, 2N)
        else:
            inp = x                                   # (B, L, N)
        inp = inp.permute(0, 2, 1)                    # (B, nvars, L)
        z = self._patchify(inp)                       # (B, nvars, patch_len, patch_num)
        z = self.backbone(z)                          # (B, nvars, d_model, patch_num)
        logit = self.head(z)
        if self.cfg.output_dim == 1:
            logit = logit.squeeze(-1)                 # (B,)
        return logit
