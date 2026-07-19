"""Stage A representation learning and frozen representation extraction."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import HTSFConfig
from .data import (
    B2_FEATURES,
    Batch,
    Standardizer,
    iter_manifest_batches,
)
from .model import HTSFEncoder, model_manifest
from .variants import VariantSpec


def resolve_torch_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested for HTSF but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if cuda_available else "cpu")
    raise ValueError(f"unknown torch device={requested!r}")


def seed_everything(seed: int) -> None:
    # CuBLAS requires this environment choice before its first handle is
    # created; without it, warn_only determinism emits one warning per layer
    # and does not actually guarantee repeatability on CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def _batch_tensors(batch: Batch, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(batch.raw).to(device),
        torch.from_numpy(batch.raw_mask).to(device),
        torch.from_numpy(batch.engineered).to(device),
        torch.from_numpy(batch.engineered_mask).to(device),
        torch.from_numpy(batch.target).to(device),
        torch.from_numpy(batch.weight).to(device),
    )


@torch.inference_mode()
def score_auxiliary(
    model: HTSFEncoder,
    data_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    device: torch.device,
    tensor_cache_dir: Path | None = None,
) -> pd.DataFrame:
    model.eval()
    rows: list[pd.DataFrame] = []
    for batch in iter_manifest_batches(
        data_dir,
        manifest,
        config,
        raw_standardizer,
        engineered_standardizer,
        shuffle=False,
        seed=config.seed,
        tensor_cache_dir=tensor_cache_dir,
    ):
        raw, raw_mask, engineered, engineered_mask, _target, _weight = _batch_tensors(
            batch, device
        )
        output = model(raw, raw_mask, engineered, engineered_mask)
        score = torch.sigmoid(output.auxiliary_logit).cpu().numpy()
        rows.append(
            pd.DataFrame(
                {
                    "file_name": batch.file_names,
                    "row_index": batch.row_indices,
                    "auxiliary_score": score,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def apply_adaptive_negative_weights(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    max_extra: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if max_extra <= 0:
        return manifest.copy(), {"updated_negative_rows": 0.0, "score_min": 0.0, "score_max": 0.0}
    result = manifest.merge(
        scores,
        on=["file_name", "row_index"],
        how="left",
        validate="one_to_one",
    )
    if result["auxiliary_score"].isna().any():
        raise ValueError("ANW scoring failed to cover every sampled endpoint")
    negative = result["target"].to_numpy(dtype=np.int8) <= 0
    negative_scores = result.loc[negative, "auxiliary_score"].to_numpy(dtype=np.float32)
    if len(negative_scores) == 0:
        return manifest.copy(), {"updated_negative_rows": 0.0, "score_min": 0.0, "score_max": 0.0}
    low = float(np.min(negative_scores))
    high = float(np.max(negative_scores))
    scaled = np.zeros(len(result), dtype=np.float32)
    if high > low:
        scaled[negative] = np.clip(
            (result.loc[negative, "auxiliary_score"].to_numpy(dtype=np.float32) - low)
            / (high - low),
            0.0,
            1.0,
        )
    weights = result["sample_weight"].to_numpy(dtype=np.float32)
    weights[negative] = 1.0 + float(max_extra) * scaled[negative]
    result["sample_weight"] = weights
    result = result.drop(columns=["auxiliary_score"])
    return result, {
        "updated_negative_rows": float(negative.sum()),
        "score_min": low,
        "score_max": high,
    }


def train_encoder(
    data_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
    variant: VariantSpec,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    output_dir: Path,
    tensor_cache_dir: Path | None = None,
) -> tuple[HTSFEncoder, pd.DataFrame, dict[str, object], torch.device]:
    if not variant.needs_encoder:
        raise ValueError("sampled_b2_xgb does not train a representation encoder")
    seed_everything(config.seed)
    device = resolve_torch_device(config.training.device)
    model = HTSFEncoder(config, variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    working_manifest = manifest.copy()
    history: list[dict[str, float]] = []
    adaptive_meta: dict[str, float] | None = None
    for epoch in range(1, int(config.training.epochs) + 1):
        model.train()
        total_loss_numerator = 0.0
        total_loss_weight = 0.0
        rows_seen = 0
        for batch in iter_manifest_batches(
            data_dir,
            working_manifest,
            config,
            raw_standardizer,
            engineered_standardizer,
            shuffle=True,
            seed=config.seed + epoch,
            tensor_cache_dir=tensor_cache_dir,
        ):
            raw, raw_mask, engineered, engineered_mask, target, weight = _batch_tensors(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(raw, raw_mask, engineered, engineered_mask)
            per_row = loss_function(output.auxiliary_logit, target.float())
            loss_numerator = torch.sum(per_row * weight)
            loss_weight = torch.clamp(weight.sum(), min=1.0)
            loss = loss_numerator / loss_weight
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.training.grad_clip))
            optimizer.step()
            count = int(len(batch.target))
            total_loss_numerator += float(loss_numerator.detach().cpu())
            total_loss_weight += float(loss_weight.detach().cpu())
            rows_seen += count
        if rows_seen == 0:
            raise ValueError("representation training saw no endpoint rows")
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(total_loss_numerator / max(total_loss_weight, 1.0)),
                "rows_seen": float(rows_seen),
            }
        )
        print(
            f"[representation] variant={variant.name} epoch={epoch}/"
            f"{config.training.epochs} loss="
            f"{total_loss_numerator / max(total_loss_weight, 1.0):.6f} rows={rows_seen}",
            flush=True,
        )
        if (
            config.weighting.mode in {"anw", "tpw_anw"}
            and adaptive_meta is None
            and epoch == int(config.weighting.adaptive_warmup_epoch)
        ):
            scores = score_auxiliary(
                model,
                data_dir,
                working_manifest,
                config,
                raw_standardizer,
                engineered_standardizer,
                device,
                tensor_cache_dir,
            )
            working_manifest, adaptive_meta = apply_adaptive_negative_weights(
                working_manifest,
                scores,
                config.weighting.adaptive_negative_max_extra,
            )
            adaptive_meta["applied_after_epoch"] = float(epoch)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = model_manifest(model, config, variant)
    manifest_payload.update(
        {
            "torch_device": str(device),
            "torch_version": torch.__version__,
            "history": history,
            "adaptive_negative_weight": adaptive_meta,
            "gradient_flow": "end_to_end; no hook or detach in either branch",
        }
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "variant": variant.name,
        },
        output_dir / "representation.pt",
    )
    (output_dir / "representation_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )
    working_manifest.to_csv(output_dir / "weighted_train_endpoints.csv", index=False)
    return model, working_manifest, manifest_payload, device


def decision_feature_names(variant: VariantSpec, latent_dim: int) -> list[str]:
    names: list[str] = []
    for component in variant.decision_inputs:
        if component == "b2":
            names.extend(B2_FEATURES)
        elif component in {"temporal", "engineered", "fused"}:
            names.extend(f"{component}_{index:03d}" for index in range(int(latent_dim)))
        else:
            raise ValueError(f"unknown decision component={component!r}")
    if len(names) != len(set(names)):
        raise RuntimeError("decision feature names are not unique")
    return names


def _decision_matrix_for_batch(
    batch: Batch,
    variant: VariantSpec,
    model: HTSFEncoder | None,
    device: torch.device,
) -> np.ndarray:
    representations: dict[str, np.ndarray] = {}
    if model is not None:
        raw, raw_mask, engineered, engineered_mask, _target, _weight = _batch_tensors(
            batch, device
        )
        with torch.inference_mode():
            encoded, _diagnostics = model.encode(raw, raw_mask, engineered, engineered_mask)
        representations = {
            name: value.detach().cpu().numpy().astype(np.float32)
            for name, value in encoded.items()
        }
    parts: list[np.ndarray] = []
    for component in variant.decision_inputs:
        if component == "b2":
            parts.append(batch.b2.astype(np.float32))
        else:
            parts.append(representations[component])
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def extract_training_table(
    data_dir: Path,
    manifest: pd.DataFrame,
    config: HTSFConfig,
    variant: VariantSpec,
    raw_standardizer: Standardizer,
    engineered_standardizer: Standardizer,
    model: HTSFEncoder | None,
    device: torch.device,
    tensor_cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    if model is not None:
        model.eval()
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    identities: list[pd.DataFrame] = []
    for batch in iter_manifest_batches(
        data_dir,
        manifest,
        config,
        raw_standardizer,
        engineered_standardizer,
        shuffle=False,
        seed=config.seed,
        tensor_cache_dir=tensor_cache_dir,
    ):
        matrices.append(_decision_matrix_for_batch(batch, variant, model, device))
        targets.append(batch.target.astype(np.float32))
        weights.append(batch.weight.astype(np.float32))
        identities.append(
            pd.DataFrame(
                {
                    "file_name": batch.file_names,
                    "row_index": batch.row_indices,
                    "timestamp": batch.timestamps,
                }
            )
        )
    if not matrices:
        raise ValueError("no rows were extracted for the XGBoost stage")
    names = decision_feature_names(variant, config.representation.latent_dim)
    matrix = np.concatenate(matrices, axis=0)
    if matrix.shape[1] != len(names):
        raise RuntimeError("decision matrix width does not match its feature manifest")
    return (
        matrix,
        np.concatenate(targets),
        np.concatenate(weights),
        pd.concat(identities, ignore_index=True),
        names,
    )


def decision_matrix_for_batch(
    batch: Batch,
    variant: VariantSpec,
    model: HTSFEncoder | None,
    device: torch.device,
) -> np.ndarray:
    return _decision_matrix_for_batch(batch, variant, model, device)
