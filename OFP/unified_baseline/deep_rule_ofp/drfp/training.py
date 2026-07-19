"""Deterministic training loop for DRFP-Net."""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .config import DRFPConfig
from .data import FailurePredictionDataset, ModuleUniformEndpointSampler
from .losses import DRFPLoss


@dataclass
class TrainingResult:
    model: nn.Module
    history: pd.DataFrame
    positive_weights: np.ndarray
    best_epoch: int
    best_validation_loss: float
    seconds_total: float


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device='cuda' but CUDA is unavailable")
    return torch.device(requested)


def positive_weights_from_targets(
    targets: np.ndarray,
    maximum: float = 20.0,
) -> np.ndarray:
    labels = np.asarray(targets, dtype=np.float64)
    if labels.ndim != 2 or not len(labels):
        raise ValueError("targets must be a non-empty [sample, horizon] matrix")
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weights = np.divide(
        negatives,
        np.maximum(positives, 1.0),
        out=np.ones_like(negatives, dtype=np.float64),
    )
    weights[positives <= 0] = 1.0
    return np.clip(weights, 1.0, float(maximum)).astype(np.float32)


def _targets_array(dataset: Dataset) -> np.ndarray:
    method = getattr(dataset, "targets_array", None)
    if callable(method):
        return np.asarray(method(), dtype=np.float32)
    value = getattr(dataset, "target_matrix", None)
    if value is not None:
        return np.asarray(value, dtype=np.float32)
    raise TypeError("dataset must expose targets_array() or target_matrix")


def _tensor(batch: Mapping[str, Any], *names: str) -> Tensor:
    for name in names:
        value = batch.get(name)
        if isinstance(value, Tensor):
            return value
    raise KeyError(f"batch has none of the required tensor fields: {names}")


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> tuple[dict[str, Tensor], Tensor]:
    inputs = {
        "raw": _tensor(batch, "raw").to(device=device, dtype=torch.float32, non_blocking=True),
        "raw_mask": _tensor(batch, "raw_mask").to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "delta_hours": _tensor(batch, "delta_hours").to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "stats": _tensor(batch, "statistics", "stats").to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "stats_mask": _tensor(batch, "stat_mask", "stats_mask").to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "rule": _tensor(batch, "rule").to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "rule_mask": _tensor(batch, "rule_mask").to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
    }
    targets = _tensor(batch, "targets", "target").to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return inputs, targets


def _make_loader(
    dataset: Dataset,
    config: DRFPConfig,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = None
    if shuffle:
        if not isinstance(dataset, FailurePredictionDataset):
            raise TypeError("module-uniform training requires FailurePredictionDataset")
        module_count = len({ref.module_index for ref in dataset.endpoints})
        sampler = ModuleUniformEndpointSampler(
            dataset,
            num_samples=module_count * config.sampling.endpoints_per_module,
            seed=seed,
        )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=config.training.num_workers > 0,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DRFPLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_examples = 0
    totals: dict[str, float] = {}
    # v1 deliberately trains in float32.  Cumulative products followed by
    # probability-space BCE can saturate in float16; correctness is more
    # important than an unverified mixed-precision speedup.
    use_amp = False
    for batch in loader:
        inputs, targets = batch_to_device(batch, device)
        batch_size = int(targets.shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**inputs)
                components = criterion(
                    outputs,
                    targets,
                    return_components=True,
                )
                if not isinstance(components, dict):
                    raise RuntimeError("DRFPLoss did not return components")
                loss = components["total"]
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        total_examples += batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
    if total_examples == 0:
        raise ValueError("empty data loader")
    averages = {name: value / total_examples for name, value in totals.items()}
    return averages["total"], averages


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: DRFPConfig,
) -> TrainingResult:
    """Train with module-uniform endpoints and validation-loss early stopping."""

    seed_everything(config.seed)
    device = resolve_device(config.training.device)
    model = model.to(device)
    positive_weights = positive_weights_from_targets(
        _targets_array(train_dataset),
        maximum=config.training.max_positive_weight,
    )
    criterion = DRFPLoss(
        horizon_count=len(config.task.horizons_hours),
        pos_weight=positive_weights,
        fused_weight=config.training.fused_loss_weight,
        deep_aux_weight=config.training.deep_auxiliary_weight,
        rule_aux_weight=config.training.rule_auxiliary_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = _make_loader(train_dataset, config, shuffle=True, seed=config.seed)
    validation_loader = _make_loader(
        validation_dataset, config, shuffle=False, seed=config.seed + 1
    )

    started = time.perf_counter()
    rows: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, config.training.epochs + 1):
        if isinstance(train_loader.sampler, ModuleUniformEndpointSampler):
            train_loader.sampler.set_epoch(epoch - 1)
        train_loss, train_components = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            config.training.grad_clip,
        )
        with torch.no_grad():
            validation_loss, validation_components = _run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                optimizer=None,
                grad_clip=0.0,
            )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        row.update({f"train_{key}": value for key, value in train_components.items()})
        row.update({f"validation_{key}": value for key, value in validation_components.items()})
        rows.append(row)
        print(
            f"[epoch {epoch:02d}] train={train_loss:.6f} validation={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = float(validation_loss)
            best_epoch = int(epoch)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        patience = config.training.early_stopping_patience
        if patience > 0 and epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(copy.deepcopy(best_state))
    model.to(device)
    return TrainingResult(
        model=model,
        history=pd.DataFrame(rows),
        positive_weights=positive_weights,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        seconds_total=float(time.perf_counter() - started),
    )
