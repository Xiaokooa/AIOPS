from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from OFP_DL_official.ofp_formal_protocol.protocol import (
    SENSORS,
    choose_threshold_by_final_score,
    files_for_fold,
    read_manifest,
)
from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder

from OFP_DL_official.common.formal_data import (
    HORIZON_HOURS,
    PRIMARY_HORIZON_INDEX,
    TYPE_NAMES,
    BatchedModuleWindowDataset,
    FormalModuleCache,
    FullRowWindowDataset,
    ModuleBalancedBatchedModuleWindowDataset,
    SampledBatchedModuleWindowDataset,
    assert_prediction_coverage,
    choose_alarm_strategy_by_final_score,
    collect_module_scores,
    compute_norm_stats_all_rows,
    compute_type_reference_all_train_negatives,
    make_window,
    official_collate,
    primary_labels,
    set_seed,
    write_predictions,
)
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.common.ofp_feature_sequence import (
    OFPFeatureModuleCache,
    compute_ofp_feature_norm_stats,
    save_feature_norm_stats,
)
from OFP_DL_official.common.losses import (
    FTEformerLossCfg,
    first_warning_event_loss,
    fteformer_total_loss,
    make_sensor_prior,
    primary_logit,
    split_alarm_aux_logits,
)

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional for headless runs.
    _tqdm = None


ModelBuilder = Callable[[int, int], tuple[nn.Module, dict, bool]]


@dataclass
class OfficialDeepCfg:
    seq_len: int = 288
    epochs: int = 8
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-2
    patience: int = 3
    grad_clip: float = 1.0
    threshold_grid_size: int = 99
    max_cached_files: int = 512
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_stride: int = 1
    num_workers: int = 0
    type_z_threshold: float = 3.0
    type_fallback_z_threshold: float = 2.0
    fixed_threshold: float = 0.5
    timestamp_mode: str = "legacy_float32"
    save_checkpoint: bool = True
    shuffle_train_rows: bool = False
    log_batches: int = 1000
    batched_windows: bool = True
    amp: bool = False
    allow_tf32: bool = True
    module_cache_dir: str = ""
    input_mode: str = "model2_features"
    train_sampling: str = "module_balanced"
    negative_ratio: float = 10.0
    pos_weight_cap: float = 20.0
    aux_bce_weight: float = 0.3
    event_loss_weight: float = 1.0
    event_hit_mode: str = "primary_window"
    event_modules_per_epoch: int = 256
    event_max_windows_per_module: int = 256
    event_module_batch_size: int = 4
    event_positive_fraction: float = 0.5
    module_positive_windows_per_module: int = 32
    module_negative_windows_per_faulty_module: int = 8
    module_normal_windows_per_module: int = 8
    index_val_fraction: float = 0.1
    selection_metric: str = "f1_score"
    max_train_files: int = 0
    max_val_files: int = 0
    max_test_files: int = 0
    alarm_smoothing: str = "ema"
    alarm_smooth_window: int = 1
    alarm_consecutive_k: int = 1
    alarm_search_smooth_windows: tuple[int, ...] = (1, 3, 6, 12)
    alarm_search_consecutive_ks: tuple[int, ...] = (1, 2, 3)
    first_alarm_only: bool = True
    use_tqdm: bool = True
    force_tqdm: bool = False


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def _paper_ready_guard(cfg: OfficialDeepCfg) -> None:
    if int(cfg.train_stride) != 1:
        raise ValueError("Official paper-grade runs require train_stride == 1")
    if int(cfg.epochs) <= 0:
        raise ValueError("Official paper-grade runs require epochs > 0")
    if bool(cfg.batched_windows) and bool(cfg.shuffle_train_rows):
        raise ValueError("Batched official windows stream by module and do not support row shuffling")
    if cfg.train_sampling not in {"full", "pos_all_neg_ratio", "module_balanced"}:
        raise ValueError(f"unknown train_sampling: {cfg.train_sampling}")
    if cfg.input_mode not in {"raw", "model2_features"}:
        raise ValueError(f"unknown input_mode: {cfg.input_mode}")
    if cfg.train_sampling != "full" and not bool(cfg.batched_windows):
        raise ValueError("Sampled DL protocol requires batched_windows=True")
    if cfg.alarm_smoothing not in {"none", "ema"}:
        raise ValueError(f"unknown alarm_smoothing: {cfg.alarm_smoothing}")
    if int(cfg.alarm_smooth_window) <= 0:
        raise ValueError("alarm_smooth_window must be positive")
    if int(cfg.alarm_consecutive_k) <= 0:
        raise ValueError("alarm_consecutive_k must be positive")
    if float(cfg.aux_bce_weight) < 0:
        raise ValueError("aux_bce_weight must be non-negative")
    if float(cfg.event_loss_weight) < 0:
        raise ValueError("event_loss_weight must be non-negative")
    if cfg.event_hit_mode not in {"primary_window", "pre_fault"}:
        raise ValueError(f"unknown event_hit_mode: {cfg.event_hit_mode}")
    if int(cfg.event_modules_per_epoch) < 0:
        raise ValueError("event_modules_per_epoch must be non-negative")
    if int(cfg.event_max_windows_per_module) <= 0:
        raise ValueError("event_max_windows_per_module must be positive")
    if int(cfg.event_module_batch_size) <= 0:
        raise ValueError("event_module_batch_size must be positive")
    if not 0 <= float(cfg.event_positive_fraction) <= 1:
        raise ValueError("event_positive_fraction must be in [0, 1]")
    if int(cfg.module_positive_windows_per_module) == 0:
        raise ValueError("module_positive_windows_per_module must be non-zero")
    if int(cfg.module_negative_windows_per_faulty_module) < 0:
        raise ValueError("module_negative_windows_per_faulty_module must be non-negative")
    if int(cfg.module_normal_windows_per_module) < 0:
        raise ValueError("module_normal_windows_per_module must be non-negative")
    if not 0.0 <= float(cfg.index_val_fraction) < 0.5:
        raise ValueError("index_val_fraction must be in [0, 0.5)")
    if int(cfg.max_train_files) < 0 or int(cfg.max_val_files) < 0 or int(cfg.max_test_files) < 0:
        raise ValueError("max_*_files debug limits must be non-negative")


def _configure_torch_runtime(cfg: OfficialDeepCfg) -> None:
    if str(cfg.device).startswith("cuda") and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def resolve_runtime_device(requested: str) -> str:
    """Resolve and validate the requested PyTorch device.

    A formal CUDA run should not silently fall back to CPU. If a script asks for
    CUDA and the active environment cannot provide it, fail early with the
    relevant environment details.
    """
    requested = str(requested or "cuda").strip()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                f"torch={torch.__version__} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
        device = torch.device(requested)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        _ = torch.empty(1, device=device)
        return str(device)
    return str(torch.device(requested))


def runtime_device_summary(device: str) -> str:
    pieces = [
        f"device={device}",
        f"torch={torch.__version__}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    ]
    if str(device).startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
        idx = torch.cuda.current_device() if dev.index is None else int(dev.index)
        props = torch.cuda.get_device_properties(idx)
        pieces.extend(
            [
                f"cuda_count={torch.cuda.device_count()}",
                f"cuda_index={idx}",
                f"cuda_name={props.name}",
                f"cuda_total_gb={props.total_memory / (1024 ** 3):.2f}",
            ]
        )
    return " ".join(pieces)


def format_metric_summary(metrics: dict[str, float]) -> str:
    keys = [
        "final_score",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "all_hit_cnt",
        "all_predict_pos_cnt",
        "all_true_pos_cnt",
        "avg_lead_score",
        "avg_lead_hour",
        "min_lead_score",
        "min_lead_hour",
        "lead_pread_cnt",
        "evaluated_module_cnt",
    ]
    parts: list[str] = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, (int, np.integer)) or key in {
            "tp",
            "fp",
            "fn",
            "tn",
            "all_hit_cnt",
            "all_predict_pos_cnt",
            "all_true_pos_cnt",
            "lead_pread_cnt",
            "evaluated_module_cnt",
        }:
            parts.append(f"{key}={int(float(value))}")
        else:
            parts.append(f"{key}={float(value):.6g}")
    return " ".join(parts)


def cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=bool(enabled))
    return torch.cuda.amp.autocast(enabled=bool(enabled))


def cuda_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=bool(enabled))
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def should_use_tqdm(enabled: bool, force: bool = False) -> bool:
    return bool(enabled) and _tqdm is not None and (bool(force) or sys.stdout.isatty())


def build_train_loader(
    file_names: list[str],
    cache,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    type_reference,
    cfg: OfficialDeepCfg,
) -> tuple[object, DataLoader]:
    if bool(cfg.batched_windows):
        if cfg.train_sampling == "module_balanced":
            dataset = ModuleBalancedBatchedModuleWindowDataset(
                file_names,
                cache=cache,
                seq_len=seq_len,
                mean=mean,
                std=std,
                type_reference=type_reference,
                batch_rows=cfg.batch_size,
                positive_windows_per_module=cfg.module_positive_windows_per_module,
                negative_windows_per_faulty_module=cfg.module_negative_windows_per_faulty_module,
                normal_windows_per_module=cfg.module_normal_windows_per_module,
                seed=cfg.seed,
            )
        elif cfg.train_sampling == "pos_all_neg_ratio":
            dataset = SampledBatchedModuleWindowDataset(
                file_names,
                cache=cache,
                seq_len=seq_len,
                mean=mean,
                std=std,
                type_reference=type_reference,
                batch_rows=cfg.batch_size,
                negative_ratio=cfg.negative_ratio,
                seed=cfg.seed,
            )
        else:
            dataset = BatchedModuleWindowDataset(
                file_names,
                cache=cache,
                seq_len=seq_len,
                mean=mean,
                std=std,
                type_reference=type_reference,
                row_stride=cfg.train_stride,
                batch_rows=cfg.batch_size,
            )
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=str(cfg.device).startswith("cuda"),
            persistent_workers=bool(cfg.num_workers > 0),
        )
        return dataset, loader

    dataset = FullRowWindowDataset(
        file_names,
        cache=cache,
        seq_len=seq_len,
        mean=mean,
        std=std,
        type_reference=type_reference,
        row_stride=cfg.train_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=bool(cfg.shuffle_train_rows),
        num_workers=cfg.num_workers,
        collate_fn=official_collate,
        pin_memory=str(cfg.device).startswith("cuda"),
        persistent_workers=bool(cfg.num_workers > 0),
    )
    return dataset, loader


def dataset_row_count(dataset: object) -> int:
    return int(getattr(dataset, "total_rows", len(dataset)))


def source_dataset_row_count(dataset: object) -> int:
    return int(getattr(dataset, "source_total_rows", dataset_row_count(dataset)))


def source_dataset_pos_count(dataset: object) -> int:
    return int(getattr(dataset, "source_pos_rows", getattr(dataset, "pos_rows", 0)))


def source_dataset_neg_count(dataset: object) -> int:
    return int(getattr(dataset, "source_neg_rows", getattr(dataset, "neg_rows", 0)))


def protocol_name(prefix: str, cfg: OfficialDeepCfg) -> str:
    if cfg.train_sampling == "pos_all_neg_ratio":
        ratio = f"{float(cfg.negative_ratio):g}".replace(".", "p")
        sampling = f"sampled_all_pos_neg{ratio}x"
    elif cfg.train_sampling == "module_balanced":
        sampling = (
            f"module_balanced"
            f"_p{int(cfg.module_positive_windows_per_module)}"
            f"_fn{int(cfg.module_negative_windows_per_faulty_module)}"
            f"_nn{int(cfg.module_normal_windows_per_module)}"
        )
    else:
        sampling = "full_train"
    alarm = (
        f"alarm_{cfg.alarm_smoothing}"
        f"w{int(cfg.alarm_smooth_window)}"
        f"k{int(cfg.alarm_consecutive_k)}"
        f"{'_once' if cfg.first_alarm_only else '_interval'}"
    )
    loss = (
        f"fw_event{float(cfg.event_loss_weight):g}"
        f"_aux{float(cfg.aux_bce_weight):g}"
        f"_m{int(cfg.event_modules_per_epoch)}"
        f"_{cfg.event_hit_mode}"
    ).replace(".", "p")
    selection = f"sel_{cfg.selection_metric}_val{float(cfg.index_val_fraction):g}".replace(".", "p")
    return f"{prefix}_{cfg.input_mode}_{sampling}_{loss}_{selection}_{alarm}_{cfg.timestamp_mode}"


def effective_pos_weight(dataset: object, cfg: OfficialDeepCfg) -> np.ndarray:
    pos = np.asarray(getattr(dataset, "pos_rows_by_horizon", [getattr(dataset, "pos_rows", 0)]), dtype=np.float64)
    neg = np.asarray(getattr(dataset, "neg_rows_by_horizon", [getattr(dataset, "neg_rows", 0)]), dtype=np.float64)
    value = np.maximum(1.0, neg / np.maximum(pos, 1.0))
    if float(cfg.pos_weight_cap) > 0:
        value = np.minimum(value, float(cfg.pos_weight_cap))
    return value.astype(np.float32)


def build_event_module_pools(
    file_names: list[str],
    cache,
    cfg: OfficialDeepCfg,
) -> tuple[list[str], list[str], int]:
    positive: list[str] = []
    normal: list[str] = []
    skipped = 0
    for name in file_names:
        _timestamps, _values, anomaly, labels, valid_mask = cache.get(name)
        positions = np.flatnonzero(np.asarray(valid_mask, dtype=bool)).astype(np.int64)
        if len(positions) <= 0:
            skipped += 1
            continue
        if np.any(np.asarray(anomaly) > 0):
            primary = primary_labels(labels)
            if cfg.event_hit_mode == "pre_fault" or np.any(primary[positions] > 0):
                positive.append(name)
            else:
                skipped += 1
        else:
            normal.append(name)
    return positive, normal, skipped


def split_index_train_validation(
    index_df: pd.DataFrame,
    fold: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    train_pool = index_df.loc[index_df["folder_index"] != int(fold)].copy()
    test_files = index_df.loc[index_df["folder_index"] == int(fold), "file_name"].tolist()
    if train_pool.empty or not test_files:
        raise ValueError(f"fold {fold} has empty train/test split")
    if float(val_fraction) <= 0:
        return train_pool["file_name"].tolist(), [], test_files

    rng = np.random.default_rng(int(seed) + int(fold) * 1_000_003 + 4049)
    val_indices: list[int] = []
    for _label, group in train_pool.groupby("Label", sort=True):
        n_group = len(group)
        if n_group <= 1:
            continue
        n_val = int(round(float(n_group) * float(val_fraction)))
        n_val = max(1, min(n_val, n_group - 1))
        chosen = rng.choice(group.index.to_numpy(), size=n_val, replace=False)
        val_indices.extend(int(x) for x in chosen)
    val_set = set(val_indices)
    train_files = train_pool.loc[~train_pool.index.isin(val_set), "file_name"].tolist()
    val_files = train_pool.loc[train_pool.index.isin(val_set), "file_name"].tolist()
    if not train_files or not val_files:
        raise ValueError(
            f"fold {fold} validation split failed: train={len(train_files)} val={len(val_files)}"
        )
    return train_files, val_files, test_files


def sample_event_files(
    positive_files: list[str],
    normal_files: list[str],
    cfg: OfficialDeepCfg,
    epoch: int,
) -> list[str]:
    total_available = len(positive_files) + len(normal_files)
    if total_available <= 0 or float(cfg.event_loss_weight) <= 0:
        return []
    target = int(cfg.event_modules_per_epoch)
    if target <= 0 or target >= total_available:
        return list(positive_files) + list(normal_files)
    rng = np.random.default_rng(int(cfg.seed) + int(epoch) * 1_000_003 + 917)
    target_pos = min(len(positive_files), int(round(target * float(cfg.event_positive_fraction))))
    target_neg = min(len(normal_files), target - target_pos)
    remaining = target - target_pos - target_neg
    if remaining > 0 and target_pos < len(positive_files):
        add = min(remaining, len(positive_files) - target_pos)
        target_pos += add
        remaining -= add
    if remaining > 0 and target_neg < len(normal_files):
        target_neg += min(remaining, len(normal_files) - target_neg)
    chosen: list[str] = []
    if target_pos > 0:
        chosen.extend(rng.choice(positive_files, size=target_pos, replace=False).astype(str).tolist())
    if target_neg > 0:
        chosen.extend(rng.choice(normal_files, size=target_neg, replace=False).astype(str).tolist())
    rng.shuffle(chosen)
    return chosen


def event_module_logits(
    model: nn.Module,
    cache,
    file_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: OfficialDeepCfg,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    _timestamps, values, anomaly, labels, valid_mask = cache.get(file_name)
    positions = np.flatnonzero(np.asarray(valid_mask, dtype=bool)).astype(np.int64)
    if len(positions) <= 0:
        return None
    primary = primary_labels(labels)
    hit_positions = (primary[positions] > 0) if cfg.event_hit_mode == "primary_window" else np.ones(len(positions), dtype=bool)
    target_value = float(np.any(np.asarray(anomaly) > 0) and np.any(hit_positions))
    if np.any(np.asarray(anomaly) > 0) and target_value <= 0.0:
        return None
    max_windows = int(cfg.event_max_windows_per_module)
    if len(positions) > max_windows:
        if target_value > 0.5 and cfg.event_hit_mode == "primary_window":
            hit_all = positions[primary[positions] > 0].astype(np.int64)
            outside_all = positions[primary[positions] <= 0].astype(np.int64)
            hit_quota = max(1, min(len(hit_all), max_windows // 2))
            outside_quota = max_windows - hit_quota
            chosen_hit = rng.choice(hit_all, size=hit_quota, replace=False) if len(hit_all) > hit_quota else hit_all
            if len(outside_all) > outside_quota:
                chosen_outside = rng.choice(outside_all, size=outside_quota, replace=False)
            else:
                chosen_outside = outside_all
            positions = np.sort(np.concatenate([chosen_hit, chosen_outside])).astype(np.int64)
        else:
            positions = np.sort(rng.choice(positions, size=max_windows, replace=False)).astype(np.int64)
    hit_positions = (primary[positions] > 0) if cfg.event_hit_mode == "primary_window" else np.ones(len(positions), dtype=bool)

    logits: list[torch.Tensor] = []
    batch_size = max(1, int(cfg.batch_size))
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
    for start in range(0, len(positions), batch_size):
        batch_pos = positions[start : start + batch_size]
        xs, masks = [], []
        for row_idx in batch_pos:
            x, mask = make_window(values, int(row_idx), cfg.seq_len, mean, std)
            xs.append(x)
            masks.append(mask)
        xb = torch.from_numpy(np.stack(xs).astype(np.float32)).to(cfg.device)
        mb = torch.from_numpy(np.stack(masks).astype(np.float32)).to(cfg.device)
        with cuda_autocast(amp_enabled):
            outputs = model(xb, mb)
        logits_all = primary_logit(outputs)
        alarm_logits, _aux_logits = split_alarm_aux_logits(logits_all, aux_dim=len(HORIZON_HOURS))
        logits.append(alarm_logits.float())

    alarm = torch.cat(logits, dim=0)
    target = torch.tensor([target_value], dtype=torch.float32, device=cfg.device)
    hit_mask = torch.from_numpy(np.asarray(hit_positions, dtype=bool)).view(1, -1).to(cfg.device)
    return alarm.unsqueeze(0), hit_mask, target


def train_first_warning_event_epoch(
    model: nn.Module,
    cache,
    event_files: list[str],
    optimizer: torch.optim.Optimizer,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: OfficialDeepCfg,
    epoch: int,
) -> dict[str, float]:
    if not event_files or float(cfg.event_loss_weight) <= 0:
        return {
            "first_warning_loss": 0.0,
            "weighted_first_warning_loss": 0.0,
            "first_warning_modules": 0.0,
            "first_warning_positive_modules": 0.0,
            "first_warning_normal_modules": 0.0,
        }
    model.train()
    rng = np.random.default_rng(int(cfg.seed) + int(epoch) * 2_000_003 + 31)
    losses: list[torch.Tensor] = []
    total_raw = 0.0
    total_weighted = 0.0
    n_modules = 0
    n_pos = 0
    n_normal = 0

    def flush() -> None:
        nonlocal total_raw, total_weighted, losses
        if not losses:
            return
        raw_loss = torch.stack(losses).mean()
        weighted = float(cfg.event_loss_weight) * raw_loss
        optimizer.zero_grad(set_to_none=True)
        weighted.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()
        batch_n = len(losses)
        total_raw += float(raw_loss.detach().cpu()) * batch_n
        total_weighted += float(weighted.detach().cpu()) * batch_n
        losses = []

    for name in event_files:
        item = event_module_logits(model, cache, name, mean, std, cfg, rng)
        if item is None:
            continue
        alarm_logits, hit_mask, target = item
        loss = first_warning_event_loss(alarm_logits, hit_mask, target)
        losses.append(loss)
        n_modules += 1
        if float(target.item()) > 0.5:
            n_pos += 1
        else:
            n_normal += 1
        if len(losses) >= int(cfg.event_module_batch_size):
            flush()
    flush()

    denom = max(n_modules, 1)
    return {
        "first_warning_loss": total_raw / denom,
        "weighted_first_warning_loss": total_weighted / denom,
        "first_warning_modules": float(n_modules),
        "first_warning_positive_modules": float(n_pos),
        "first_warning_normal_modules": float(n_normal),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    fault_loss_fn: nn.Module,
    device: str,
    grad_clip: float,
    use_fteformer_losses: bool,
    sensor_prior: torch.Tensor,
    fte_loss_cfg: FTEformerLossCfg,
    log_prefix: str = "",
    log_batches: int = 1000,
    use_amp: bool = False,
    max_batches: int = 0,
    use_tqdm: bool = True,
    force_tqdm: bool = False,
    fault_loss_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0}
    n_seen = 0
    total_batches = len(loader)
    display_batches = min(total_batches, int(max_batches)) if int(max_batches) > 0 else total_batches
    total_rows = dataset_row_count(loader.dataset) if hasattr(loader, "dataset") else 0
    started = time.time()
    amp_enabled = bool(use_amp) and str(device).startswith("cuda")
    scaler = cuda_grad_scaler(amp_enabled)
    batch_idx = 0
    iterator = enumerate(loader, 1)
    progress_bar = None
    tqdm_active = should_use_tqdm(use_tqdm, force_tqdm)
    if tqdm_active:
        progress_bar = _tqdm(
            iterator,
            total=display_batches,
            desc=log_prefix.strip() or "train",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
            file=sys.stdout,
            ascii=True,
        )
        iterator = progress_bar
    for batch_idx, (xs, masks, ys, type_targets, type_masks) in iterator:
        xs = xs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True).float()
        type_targets = type_targets.to(device, non_blocking=True)
        type_masks = type_masks.to(device, non_blocking=True)
        if not torch.isfinite(xs).all():
            raise FloatingPointError(f"{log_prefix} non-finite xs at batch {batch_idx}")
        if not torch.isfinite(masks).all():
            raise FloatingPointError(f"{log_prefix} non-finite masks at batch {batch_idx}")
        if not torch.isfinite(ys).all():
            raise FloatingPointError(f"{log_prefix} non-finite labels at batch {batch_idx}")

        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(amp_enabled):
            outputs = model(xs, masks)
            logits_all = primary_logit(outputs)
            _alarm_logits, aux_logits = split_alarm_aux_logits(logits_all, aux_dim=ys.shape[-1] if ys.ndim > 1 else 1)
            if ys.ndim > 1 and (aux_logits.ndim <= 1 or aux_logits.shape[-1] != ys.shape[-1]):
                raise ValueError(
                    f"{log_prefix} multi-horizon auxiliary labels require logits shape (batch,{ys.shape[-1]}), "
                    f"got {tuple(aux_logits.shape)} from raw logits {tuple(logits_all.shape)}"
                )
            if not torch.isfinite(aux_logits).all():
                raise FloatingPointError(
                    f"{log_prefix} non-finite logits at batch {batch_idx}; "
                    f"logit_min={float(torch.nan_to_num(aux_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0).min().cpu()):.6g} "
                    f"logit_max={float(torch.nan_to_num(aux_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0).max().cpu()):.6g}"
                )
            fault_loss_raw = fault_loss_fn(aux_logits, ys)
            fault_loss = float(fault_loss_weight) * fault_loss_raw
            if use_fteformer_losses:
                if not isinstance(outputs, tuple) or len(outputs) < 7:
                    raise ValueError("FTEformer official loss requires tuple outputs with diagnostic branches")
                loss, parts = fteformer_total_loss(
                    outputs,
                    ys,
                    fault_loss,
                    type_targets,
                    type_masks,
                    sensor_prior,
                    fte_loss_cfg,
                )
            else:
                loss = fault_loss
                parts = {
                    "aux_bce_loss": float(fault_loss_raw.detach().cpu()),
                    "weighted_aux_bce_loss": float(fault_loss.detach().cpu()),
                }
            if use_fteformer_losses:
                parts["aux_bce_loss"] = float(fault_loss_raw.detach().cpu())
                parts["weighted_aux_bce_loss"] = float(fault_loss.detach().cpu())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{log_prefix} non-finite loss at batch {batch_idx}; "
                    f"fault_loss={float(torch.nan_to_num(fault_loss.detach(), nan=-1.0, posinf=-2.0, neginf=-3.0).cpu()):.6g}"
                )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"{log_prefix} non-finite gradient in {name} at batch {batch_idx}")
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"{log_prefix} non-finite gradient in {name} at batch {batch_idx}")
            optimizer.step()

        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError(f"{log_prefix} non-finite parameter {name} after batch {batch_idx}")

        batch_n = int(xs.size(0))
        n_seen += batch_n
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch_n
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_n

        if int(log_batches) > 0 and (batch_idx % int(log_batches) == 0 or batch_idx == display_batches):
            avg_loss = totals["loss"] / max(n_seen, 1)
            elapsed_min = (time.time() - started) / 60.0
            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    rows=f"{n_seen}/{total_rows}",
                    min=f"{elapsed_min:.1f}",
                )
            else:
                prefix = f"{log_prefix} " if log_prefix else ""
                print(
                    f"{prefix}batch={batch_idx}/{display_batches} rows={n_seen}/{total_rows} "
                    f"avg_loss={avg_loss:.4f} elapsed_min={elapsed_min:.1f}",
                    flush=True,
                )
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
    if progress_bar is not None:
        progress_bar.close()

    elapsed_seconds = time.time() - started
    denom = max(n_seen, 1)
    averages = {key: value / denom for key, value in totals.items()}
    batches_seen = float(batch_idx if n_seen else 0)
    rows_per_second = float(n_seen) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    batches_per_second = batches_seen / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return averages | {
        "rows_seen": float(n_seen),
        "batches_seen": batches_seen,
        "elapsed_seconds": float(elapsed_seconds),
        "rows_per_second": rows_per_second,
        "batches_per_second": batches_per_second,
    }


def run_deep_fold(
    model_name: str,
    model_builder: ModelBuilder,
    fold: int,
    data_dir: Path,
    manifest_df: pd.DataFrame,
    out_root: Path,
    cfg: OfficialDeepCfg,
) -> dict:
    _paper_ready_guard(cfg)
    t0 = time.time()
    set_seed(cfg.seed + int(fold))

    train_files, val_files, test_files = files_for_fold(manifest_df, int(fold))
    run_dir = Path(out_root) / model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = FormalModuleCache(
        data_dir,
        max_cached_files=cfg.max_cached_files,
        timestamp_mode=cfg.timestamp_mode,
        module_cache_dir=Path(cfg.module_cache_dir) if cfg.module_cache_dir else None,
    )
    cfg.device = resolve_runtime_device(cfg.device)
    _configure_torch_runtime(cfg)

    model, model_cfg, use_fteformer_losses = model_builder(cfg.seq_len, len(SENSORS))
    print(
        f"[official-init] model={model_name} fold={fold} protocol=ofp_formal_first_event "
        f"train_modules={len(train_files)} val_modules={len(val_files)} test_modules={len(test_files)} "
        f"seq_len={cfg.seq_len} epochs={cfg.epochs} batch_size={cfg.batch_size} "
        f"sampling={cfg.train_sampling} negative_ratio={cfg.negative_ratio} "
        f"pos_weight_cap={cfg.pos_weight_cap} aux_bce_weight={cfg.aux_bce_weight} "
        f"event_loss_weight={cfg.event_loss_weight} event_modules_per_epoch={cfg.event_modules_per_epoch} "
        f"horizons={list(HORIZON_HOURS)} "
        f"lookback_hours=24 primary_horizon={HORIZON_HOURS[PRIMARY_HORIZON_INDEX]} "
        f"threshold_mode=validation_grid first_alarm_only={cfg.first_alarm_only} "
        f"timestamp_mode={cfg.timestamp_mode} amp={cfg.amp} tf32={cfg.allow_tf32}",
        flush=True,
    )
    print(f"[official-init] {runtime_device_summary(cfg.device)}", flush=True)

    print(f"[official {model_name} fold={fold}] computing train-only normalization")
    norm = compute_norm_stats_all_rows(data_dir, train_files, timestamp_mode=cfg.timestamp_mode)
    mean, std = norm.arrays()
    (run_dir / "norm_stats.json").write_text(json.dumps(asdict(norm), indent=2), encoding="utf-8")

    if use_fteformer_losses:
        print(f"[official {model_name} fold={fold}] computing train-only weak type reference")
        type_ref = compute_type_reference_all_train_negatives(
            data_dir,
            train_files,
            z_threshold=cfg.type_z_threshold,
            fallback_z_threshold=cfg.type_fallback_z_threshold,
            timestamp_mode=cfg.timestamp_mode,
        )
        type_ref.to_json(run_dir / "weak_type_reference.json")
    else:
        type_ref = None
        print(f"[official {model_name} fold={fold}] skip weak type reference (unused by this model)")

    print(f"[official {model_name} fold={fold}] indexing full training rows")
    train_dataset, loader = build_train_loader(
        train_files,
        cache=cache,
        seq_len=cfg.seq_len,
        mean=mean,
        std=std,
        type_reference=type_ref,
        cfg=cfg,
    )
    if train_dataset.pos_rows <= 0 or train_dataset.neg_rows <= 0:
        raise ValueError(
            f"{model_name} fold {fold} requires both positive and negative rows; "
            f"pos={train_dataset.pos_rows} neg={train_dataset.neg_rows}"
        )
    print(
        f"[official {model_name} fold={fold}] train loader rows={dataset_row_count(train_dataset)} "
        f"batches={len(loader)} batch_size={cfg.batch_size} "
        f"batched_windows={cfg.batched_windows} shuffle_rows={cfg.shuffle_train_rows}",
        flush=True,
    )
    event_positive_files, event_normal_files, event_skipped = build_event_module_pools(train_files, cache, cfg)
    print(
        f"[official {model_name} fold={fold}] first-warning event modules "
        f"positive={len(event_positive_files)} normal={len(event_normal_files)} skipped={event_skipped}",
        flush=True,
    )

    model = model.to(cfg.device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    pos_weight = effective_pos_weight(train_dataset, cfg)
    fault_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sensor_prior = make_sensor_prior(TYPE_NAMES, SENSORS).to(cfg.device)
    fte_loss_cfg = FTEformerLossCfg(sensor_contrastive_weight=float(model_cfg.get("sensor_contrastive_weight", 0.03)))

    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    best_alarm_strategy = {
        "alarm_smoothing": cfg.alarm_smoothing,
        "alarm_smooth_window": int(cfg.alarm_smooth_window),
        "alarm_consecutive_k": int(cfg.alarm_consecutive_k),
    }
    best_val_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        ep0 = time.time()
        train_parts = train_one_epoch(
            model,
            loader,
            optimizer,
            fault_loss_fn,
            cfg.device,
            cfg.grad_clip,
            use_fteformer_losses,
            sensor_prior,
            fte_loss_cfg,
            log_prefix=f"[official {model_name} fold={fold} ep={epoch:02d}]",
            log_batches=cfg.log_batches,
            use_amp=cfg.amp,
            use_tqdm=cfg.use_tqdm,
            force_tqdm=cfg.force_tqdm,
            fault_loss_weight=cfg.aux_bce_weight,
        )
        event_files = sample_event_files(event_positive_files, event_normal_files, cfg, epoch)
        event_parts = train_first_warning_event_epoch(
            model,
            cache,
            event_files,
            optimizer,
            mean,
            std,
            cfg,
            epoch,
        )
        train_parts = {
            **train_parts,
            **{f"event_{k}": v for k, v in event_parts.items()},
            "combined_train_loss": train_parts["loss"] + event_parts["weighted_first_warning_loss"],
        }

        print(f"[official {model_name} fold={fold}] scoring validation after epoch {epoch}")
        val_rows = collect_module_scores(
            model,
            cache,
            val_files,
            cfg.seq_len,
            mean,
            std,
            cfg.batch_size,
            cfg.device,
            use_amp=cfg.amp,
            use_tqdm=cfg.use_tqdm,
            force_tqdm=cfg.force_tqdm,
            progress_desc=f"[official {model_name} fold={fold} val ep={epoch:02d}]",
        )
        threshold, alarm_strategy, val_metrics = choose_alarm_strategy_by_final_score(
            val_rows,
            grid_size=cfg.threshold_grid_size,
            smoothing=cfg.alarm_smoothing,
            smooth_windows=cfg.alarm_search_smooth_windows,
            consecutive_ks=cfg.alarm_search_consecutive_ks,
            selection_metric=cfg.selection_metric,
        )
        key = (
            float(val_metrics.get(cfg.selection_metric, val_metrics["final_score"])),
            float(val_metrics["f1_score"]),
            float(val_metrics["recall"]),
            float(val_metrics["precision"]),
            float(val_metrics["final_score"]),
        )
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_threshold = float(threshold)
            best_alarm_strategy = dict(alarm_strategy)
            best_val_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "seconds": time.time() - ep0,
            "threshold": float(threshold),
            **{f"alarm_{k}": v for k, v in alarm_strategy.items()},
            **{f"train_{k}": v for k, v in train_parts.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[official {model_name} fold={fold}] ep={epoch:02d} "
            f"aux_loss={train_parts['loss']:.4f} "
            f"event_loss={train_parts['event_weighted_first_warning_loss']:.4f} "
            f"val_final={val_metrics['final_score']:.4f} "
            f"val_f1={val_metrics['f1_score']:.4f} thr={threshold:.3f} "
            f"alarm={alarm_strategy} "
            f"train_min={train_parts.get('elapsed_seconds', 0.0) / 60.0:.2f} "
            f"epoch_min={row['seconds'] / 60.0:.2f} "
            f"rows_per_sec={train_parts.get('rows_per_second', 0.0):.2f} "
            f"batches_per_sec={train_parts.get('batches_per_second', 0.0):.3f}"
        )
        if stale >= int(cfg.patience):
            print(f"[official {model_name} fold={fold}] early stop after {stale} stale epochs")
            break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError(f"{model_name} fold {fold} did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    print(f"[official {model_name} fold={fold}] scoring full test modules={len(test_files)}")
    test_rows = collect_module_scores(
        model,
        cache,
        test_files,
        cfg.seq_len,
        mean,
        std,
        cfg.batch_size,
        cfg.device,
        use_amp=cfg.amp,
        use_tqdm=cfg.use_tqdm,
        force_tqdm=cfg.force_tqdm,
        progress_desc=f"[official {model_name} fold={fold} test]",
    )
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(
        test_rows,
        pred_dir,
        best_threshold,
        alarm_smoothing=str(best_alarm_strategy.get("alarm_smoothing", cfg.alarm_smoothing)),
        alarm_smooth_window=int(best_alarm_strategy.get("alarm_smooth_window", cfg.alarm_smooth_window)),
        alarm_consecutive_k=int(best_alarm_strategy.get("alarm_consecutive_k", cfg.alarm_consecutive_k)),
        first_alarm_only=cfg.first_alarm_only,
    )
    assert_prediction_coverage(pred_dir, test_files)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    metrics = metrics_to_dict(summary_df)
    if int(metrics.get("evaluated_module_cnt", -1)) != len(test_files):
        raise AssertionError(
            f"evaluated module count mismatch: {metrics.get('evaluated_module_cnt')} vs {len(test_files)}"
        )
    print(
        f"[official-final] model={model_name} fold={fold} threshold={best_threshold:.6g} "
        f"best_epoch={best_epoch} elapsed_min={(time.time() - t0) / 60.0:.2f} "
        f"{format_metric_summary(metrics)}",
        flush=True,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)

    result = {
        "model": model_name,
        "fold": int(fold),
        "protocol": protocol_name("ofp_formal_lookback24h_ahead1h_first_warning", cfg),
        "paper_ready": True,
        "input_mode": cfg.input_mode,
        "train_sampling": cfg.train_sampling,
        "negative_ratio": float(cfg.negative_ratio),
        "aux_bce_weight": float(cfg.aux_bce_weight),
        "event_loss_weight": float(cfg.event_loss_weight),
        "event_hit_mode": cfg.event_hit_mode,
        "event_modules_per_epoch": int(cfg.event_modules_per_epoch),
        "threshold": float(best_threshold),
        "alarm_strategy": best_alarm_strategy,
        "first_alarm_only": bool(cfg.first_alarm_only),
        "best_epoch": int(best_epoch),
        "train_modules": len(train_files),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "source_train_rows": source_dataset_row_count(train_dataset),
        "source_train_pos_rows": source_dataset_pos_count(train_dataset),
        "source_train_neg_rows": source_dataset_neg_count(train_dataset),
        "train_rows": dataset_row_count(train_dataset),
        "train_pos_rows": int(train_dataset.pos_rows),
        "train_neg_rows": int(train_dataset.neg_rows),
        "train_normal_modules": int(getattr(train_dataset, "normal_modules", 0)),
        "train_trainable_faulty_modules": int(getattr(train_dataset, "trainable_faulty_modules", 0)),
        "train_true_transition_modules": int(getattr(train_dataset, "true_transition_modules", 0)),
        "train_start_in_window_modules": int(getattr(train_dataset, "start_in_window_modules", 0)),
        "train_left_censored_modules": int(getattr(train_dataset, "left_censored_modules", 0)),
        "train_faulty_no_primary_window_modules": int(getattr(train_dataset, "faulty_no_primary_window_modules", 0)),
        "n_params": n_params,
        "input_dim": int(getattr(model_cfg.get("ofp_stat_fusion", {}), "get", lambda *_: 0)("n_sensors", 0) or 0),
        "seconds": time.time() - t0,
        "pos_weight": np.asarray(pos_weight, dtype=float).tolist(),
        "pos_weight_cap": float(cfg.pos_weight_cap),
        "horizon_hours": list(HORIZON_HOURS),
        "primary_horizon_index": int(PRIMARY_HORIZON_INDEX),
        "model_cfg": model_cfg,
        "run_cfg": asdict(cfg),
        "loss_cfg": asdict(fte_loss_cfg) if use_fteformer_losses else None,
        "val_module_metrics": best_val_metrics,
        "metrics": metrics,
        "history": history,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if cfg.save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "run_cfg": asdict(cfg),
                "threshold": best_threshold,
                "alarm_strategy": best_alarm_strategy,
                "first_alarm_only": bool(cfg.first_alarm_only),
                "val_module_metrics": best_val_metrics,
            },
            run_dir / "model.pt",
        )
    return result


def run_deep_fold_index(
    model_name: str,
    model_builder: ModelBuilder,
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    cfg: OfficialDeepCfg,
) -> dict:
    """Run a deep model under the OFP README-style three-fold protocol.

    Split is exactly:
      train = folder_index != fold
      test  = folder_index == fold

    There is no internal validation split in this OFP-aligned mode. The model
    uses all non-test modules for training, saves the final epoch checkpoint,
    and evaluates with a fixed threshold, matching the no-validation spirit of
    the original OFP model2 reproduction.
    """
    _paper_ready_guard(cfg)
    t0 = time.time()
    set_seed(cfg.seed + int(fold))

    train_files_all, test_files = files_for_index_fold(index_df, int(fold))
    train_files, val_files, test_files = split_index_train_validation(
        index_df,
        int(fold),
        float(cfg.index_val_fraction),
        int(cfg.seed),
    )
    if int(cfg.max_train_files) > 0:
        train_files = train_files[: int(cfg.max_train_files)]
    if int(cfg.max_val_files) > 0:
        val_files = val_files[: int(cfg.max_val_files)]
    if int(cfg.max_test_files) > 0:
        test_files = test_files[: int(cfg.max_test_files)]
    run_dir = Path(out_root) / model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.device = resolve_runtime_device(cfg.device)
    _configure_torch_runtime(cfg)

    if cfg.input_mode == "model2_features":
        cache = OFPFeatureModuleCache(
            data_dir,
            max_cached_files=cfg.max_cached_files,
            timestamp_mode=cfg.timestamp_mode,
            module_cache_dir=Path(cfg.module_cache_dir) if cfg.module_cache_dir else None,
        )
        norm = compute_ofp_feature_norm_stats(cache, train_files)
        input_dim = len(norm.feature_names)
    else:
        cache = FormalModuleCache(
            data_dir,
            max_cached_files=cfg.max_cached_files,
            timestamp_mode=cfg.timestamp_mode,
            module_cache_dir=Path(cfg.module_cache_dir) if cfg.module_cache_dir else None,
        )
        norm = compute_norm_stats_all_rows(data_dir, train_files, timestamp_mode=cfg.timestamp_mode)
        input_dim = len(SENSORS)

    model, model_cfg, use_fteformer_losses = model_builder(cfg.seq_len, input_dim)
    if cfg.input_mode == "model2_features" and use_fteformer_losses:
        print(
            "[ofp-index-init] input_mode=model2_features disables FTEformer concept-specific losses; "
            "using BCE/event losses over engineered feature channels.",
            flush=True,
        )
        use_fteformer_losses = False
    print(
        f"[ofp-index-init] model={model_name} fold={fold} protocol=ofp_index_lookback24h_ahead1h_first_warning "
        f"train_modules={len(train_files)} val_modules={len(val_files)} "
        f"test_modules={len(test_files)} train_pool={len(train_files_all)} "
        f"input_mode={cfg.input_mode} input_dim={input_dim} "
        f"seq_len={cfg.seq_len} epochs={cfg.epochs} batch_size={cfg.batch_size} "
        f"sampling={cfg.train_sampling} negative_ratio={cfg.negative_ratio} "
        f"pos_weight_cap={cfg.pos_weight_cap} aux_bce_weight={cfg.aux_bce_weight} "
        f"event_loss_weight={cfg.event_loss_weight} event_modules_per_epoch={cfg.event_modules_per_epoch} "
        f"event_hit_mode={cfg.event_hit_mode} selection_metric={cfg.selection_metric} "
        f"index_val_fraction={cfg.index_val_fraction} "
        f"horizons={list(HORIZON_HOURS)} "
        f"lookback_hours=24 primary_horizon={HORIZON_HOURS[PRIMARY_HORIZON_INDEX]} "
        f"fixed_threshold={cfg.fixed_threshold} threshold_grid={bool(val_files)} "
        f"first_alarm_only={cfg.first_alarm_only} timestamp_mode={cfg.timestamp_mode} amp={cfg.amp} tf32={cfg.allow_tf32}",
        flush=True,
    )
    print(f"[ofp-index-init] {runtime_device_summary(cfg.device)}", flush=True)

    print(f"[ofp-index {model_name} fold={fold}] computing train-only normalization")
    mean, std = norm.arrays()
    if cfg.input_mode == "model2_features":
        save_feature_norm_stats(norm, run_dir / "norm_stats.json")
    else:
        (run_dir / "norm_stats.json").write_text(json.dumps(asdict(norm), indent=2), encoding="utf-8")

    if use_fteformer_losses and cfg.input_mode == "raw":
        print(f"[ofp-index {model_name} fold={fold}] computing train-only weak type reference")
        type_ref = compute_type_reference_all_train_negatives(
            data_dir,
            train_files,
            z_threshold=cfg.type_z_threshold,
            fallback_z_threshold=cfg.type_fallback_z_threshold,
            timestamp_mode=cfg.timestamp_mode,
        )
        type_ref.to_json(run_dir / "weak_type_reference.json")
    else:
        type_ref = None
        print(f"[ofp-index {model_name} fold={fold}] skip weak type reference (unused by this model)")

    print(f"[ofp-index {model_name} fold={fold}] indexing full training rows")
    train_dataset, loader = build_train_loader(
        train_files,
        cache=cache,
        seq_len=cfg.seq_len,
        mean=mean,
        std=std,
        type_reference=type_ref,
        cfg=cfg,
    )
    if train_dataset.pos_rows <= 0 or train_dataset.neg_rows <= 0:
        raise ValueError(
            f"{model_name} fold {fold} requires both positive and negative rows; "
            f"pos={train_dataset.pos_rows} neg={train_dataset.neg_rows}"
        )
    print(
        f"[ofp-index {model_name} fold={fold}] train loader rows={dataset_row_count(train_dataset)} "
        f"batches={len(loader)} batch_size={cfg.batch_size} "
        f"batched_windows={cfg.batched_windows} shuffle_rows={cfg.shuffle_train_rows}",
        flush=True,
    )
    event_positive_files, event_normal_files, event_skipped = build_event_module_pools(train_files, cache, cfg)
    print(
        f"[ofp-index {model_name} fold={fold}] first-warning event modules "
        f"positive={len(event_positive_files)} normal={len(event_normal_files)} skipped={event_skipped}",
        flush=True,
    )

    model = model.to(cfg.device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    pos_weight = effective_pos_weight(train_dataset, cfg)
    fault_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sensor_prior = make_sensor_prior(TYPE_NAMES, SENSORS).to(cfg.device)
    fte_loss_cfg = FTEformerLossCfg(sensor_contrastive_weight=float(model_cfg.get("sensor_contrastive_weight", 0.03)))

    best_state = None
    best_epoch = 0
    best_threshold = float(cfg.fixed_threshold)
    best_alarm_strategy = {
        "alarm_smoothing": cfg.alarm_smoothing,
        "alarm_smooth_window": int(cfg.alarm_smooth_window),
        "alarm_consecutive_k": int(cfg.alarm_consecutive_k),
    }
    best_val_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    stale = 0
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        ep0 = time.time()
        train_parts = train_one_epoch(
            model,
            loader,
            optimizer,
            fault_loss_fn,
            cfg.device,
            cfg.grad_clip,
            use_fteformer_losses,
            sensor_prior,
            fte_loss_cfg,
            log_prefix=f"[ofp-index {model_name} fold={fold} ep={epoch:02d}]",
            log_batches=cfg.log_batches,
            use_amp=cfg.amp,
            use_tqdm=cfg.use_tqdm,
            force_tqdm=cfg.force_tqdm,
            fault_loss_weight=cfg.aux_bce_weight,
        )
        event_files = sample_event_files(event_positive_files, event_normal_files, cfg, epoch)
        event_parts = train_first_warning_event_epoch(
            model,
            cache,
            event_files,
            optimizer,
            mean,
            std,
            cfg,
            epoch,
        )
        train_parts = {
            **train_parts,
            **{f"event_{k}": v for k, v in event_parts.items()},
            "combined_train_loss": train_parts["loss"] + event_parts["weighted_first_warning_loss"],
        }
        threshold = float(cfg.fixed_threshold)
        alarm_strategy = dict(best_alarm_strategy)
        val_metrics: dict[str, float] | None = None
        if val_files:
            print(f"[ofp-index {model_name} fold={fold}] scoring validation after epoch {epoch}")
            val_rows = collect_module_scores(
                model,
                cache,
                val_files,
                cfg.seq_len,
                mean,
                std,
                cfg.batch_size,
                cfg.device,
                use_amp=cfg.amp,
                use_tqdm=cfg.use_tqdm,
                force_tqdm=cfg.force_tqdm,
                progress_desc=f"[ofp-index {model_name} fold={fold} val ep={epoch:02d}]",
            )
            threshold, alarm_strategy, val_metrics = choose_alarm_strategy_by_final_score(
                val_rows,
                grid_size=cfg.threshold_grid_size,
                smoothing=cfg.alarm_smoothing,
                smooth_windows=cfg.alarm_search_smooth_windows,
                consecutive_ks=cfg.alarm_search_consecutive_ks,
                selection_metric=cfg.selection_metric,
            )
            key = (
                float(val_metrics.get(cfg.selection_metric, val_metrics["final_score"])),
                float(val_metrics["f1_score"]),
                float(val_metrics["recall"]),
                float(val_metrics["precision"]),
                float(val_metrics["final_score"]),
            )
            improved = best_key is None or key > best_key
            if improved:
                best_key = key
                best_epoch = int(epoch)
                best_threshold = float(threshold)
                best_alarm_strategy = dict(alarm_strategy)
                best_val_metrics = dict(val_metrics)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        row = {
            "epoch": epoch,
            "seconds": time.time() - ep0,
            "threshold": float(threshold),
            **{f"alarm_{k}": v for k, v in alarm_strategy.items()},
            **{f"train_{k}": v for k, v in train_parts.items()},
        }
        if val_metrics is not None:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        val_text = ""
        if val_metrics is not None:
            val_text = (
                f"val_{cfg.selection_metric}={val_metrics.get(cfg.selection_metric, 0.0):.4f} "
                f"val_f1={val_metrics['f1_score']:.4f} "
                f"val_recall={val_metrics['recall']:.4f} "
                f"thr={threshold:.3f} alarm={alarm_strategy} "
            )
        print(
            f"[ofp-index {model_name} fold={fold}] ep={epoch:02d} "
            f"aux_loss={train_parts['loss']:.4f} "
            f"event_loss={train_parts['event_weighted_first_warning_loss']:.4f} "
            f"{val_text}"
            f"threshold={threshold:.3f} "
            f"train_min={train_parts.get('elapsed_seconds', 0.0) / 60.0:.2f} "
            f"rows_per_sec={train_parts.get('rows_per_second', 0.0):.2f} "
            f"batches_per_sec={train_parts.get('batches_per_second', 0.0):.3f}"
        )
        if val_files and stale >= int(cfg.patience):
            print(f"[ofp-index {model_name} fold={fold}] early stop after {stale} stale epochs")
            break

    if val_files:
        if best_state is None:
            raise RuntimeError(f"{model_name} fold {fold} did not produce a validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = int(cfg.epochs)
        best_threshold = float(cfg.fixed_threshold)
        best_alarm_strategy = {
            "alarm_smoothing": cfg.alarm_smoothing,
            "alarm_smooth_window": int(cfg.alarm_smooth_window),
            "alarm_consecutive_k": int(cfg.alarm_consecutive_k),
        }

    print(f"[ofp-index {model_name} fold={fold}] scoring full test modules={len(test_files)}")
    test_rows = collect_module_scores(
        model,
        cache,
        test_files,
        cfg.seq_len,
        mean,
        std,
        cfg.batch_size,
        cfg.device,
        use_amp=cfg.amp,
        use_tqdm=cfg.use_tqdm,
        force_tqdm=cfg.force_tqdm,
        progress_desc=f"[ofp-index {model_name} fold={fold} test]",
    )
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(
        test_rows,
        pred_dir,
        best_threshold,
        alarm_smoothing=str(best_alarm_strategy.get("alarm_smoothing", cfg.alarm_smoothing)),
        alarm_smooth_window=int(best_alarm_strategy.get("alarm_smooth_window", cfg.alarm_smooth_window)),
        alarm_consecutive_k=int(best_alarm_strategy.get("alarm_consecutive_k", cfg.alarm_consecutive_k)),
        first_alarm_only=cfg.first_alarm_only,
    )
    assert_prediction_coverage(pred_dir, test_files)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    metrics = metrics_to_dict(summary_df)
    if int(metrics.get("evaluated_module_cnt", -1)) != len(test_files):
        raise AssertionError(
            f"evaluated module count mismatch: {metrics.get('evaluated_module_cnt')} vs {len(test_files)}"
        )
    print(
        f"[ofp-index-final] model={model_name} fold={fold} threshold={best_threshold:.6g} "
        f"best_epoch={best_epoch} elapsed_min={(time.time() - t0) / 60.0:.2f} "
        f"{format_metric_summary(metrics)}",
        flush=True,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)

    result = {
        "model": model_name,
        "fold": int(fold),
        "protocol": protocol_name("ofp_index_lookback24h_ahead1h_first_warning", cfg),
        "paper_ready": True,
        "input_mode": cfg.input_mode,
        "train_sampling": cfg.train_sampling,
        "negative_ratio": float(cfg.negative_ratio),
        "aux_bce_weight": float(cfg.aux_bce_weight),
        "event_loss_weight": float(cfg.event_loss_weight),
        "event_hit_mode": cfg.event_hit_mode,
        "event_modules_per_epoch": int(cfg.event_modules_per_epoch),
        "threshold": float(best_threshold),
        "alarm_strategy": best_alarm_strategy,
        "first_alarm_only": bool(cfg.first_alarm_only),
        "best_epoch": int(best_epoch),
        "train_modules": len(train_files),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "source_train_rows": source_dataset_row_count(train_dataset),
        "source_train_pos_rows": source_dataset_pos_count(train_dataset),
        "source_train_neg_rows": source_dataset_neg_count(train_dataset),
        "train_rows": dataset_row_count(train_dataset),
        "train_pos_rows": int(train_dataset.pos_rows),
        "train_neg_rows": int(train_dataset.neg_rows),
        "train_normal_modules": int(getattr(train_dataset, "normal_modules", 0)),
        "train_trainable_faulty_modules": int(getattr(train_dataset, "trainable_faulty_modules", 0)),
        "train_true_transition_modules": int(getattr(train_dataset, "true_transition_modules", 0)),
        "train_start_in_window_modules": int(getattr(train_dataset, "start_in_window_modules", 0)),
        "train_left_censored_modules": int(getattr(train_dataset, "left_censored_modules", 0)),
        "train_faulty_no_primary_window_modules": int(getattr(train_dataset, "faulty_no_primary_window_modules", 0)),
        "n_params": n_params,
        "input_dim": int(input_dim),
        "seconds": time.time() - t0,
        "pos_weight": np.asarray(pos_weight, dtype=float).tolist(),
        "pos_weight_cap": float(cfg.pos_weight_cap),
        "horizon_hours": list(HORIZON_HOURS),
        "primary_horizon_index": int(PRIMARY_HORIZON_INDEX),
        "model_cfg": model_cfg,
        "run_cfg": asdict(cfg),
        "loss_cfg": asdict(fte_loss_cfg) if use_fteformer_losses else None,
        "val_module_metrics": best_val_metrics,
        "metrics": metrics,
        "history": history,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if cfg.save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "run_cfg": asdict(cfg),
                "threshold": best_threshold,
                "alarm_strategy": best_alarm_strategy,
                "first_alarm_only": bool(cfg.first_alarm_only),
                "val_module_metrics": best_val_metrics,
            },
            run_dir / "model.pt",
        )
    return result


def aggregate_results(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "paper_ready": item["paper_ready"],
            "protocol": item.get("protocol", ""),
            "input_mode": item.get("input_mode", ""),
            "train_sampling": item.get("train_sampling", "full"),
            "negative_ratio": item.get("negative_ratio", 0.0),
            "aux_bce_weight": item.get("aux_bce_weight", 1.0),
            "event_loss_weight": item.get("event_loss_weight", 0.0),
            "event_hit_mode": item.get("event_hit_mode", ""),
            "event_modules_per_epoch": item.get("event_modules_per_epoch", 0),
            "threshold": item["threshold"],
            "alarm_smoothing": item.get("alarm_strategy", {}).get("alarm_smoothing", ""),
            "alarm_smooth_window": item.get("alarm_strategy", {}).get("alarm_smooth_window", 1),
            "alarm_consecutive_k": item.get("alarm_strategy", {}).get("alarm_consecutive_k", 1),
            "first_alarm_only": item.get("first_alarm_only", False),
            "best_epoch": item["best_epoch"],
            "train_modules": item["train_modules"],
            "val_modules": item["val_modules"],
            "test_modules": item["test_modules"],
            "source_train_rows": item.get("source_train_rows", item["train_rows"]),
            "source_train_pos_rows": item.get("source_train_pos_rows", item["train_pos_rows"]),
            "source_train_neg_rows": item.get("source_train_neg_rows", item["train_neg_rows"]),
            "train_rows": item["train_rows"],
            "train_pos_rows": item["train_pos_rows"],
            "train_neg_rows": item["train_neg_rows"],
            "module_positive_windows_per_module": item.get("run_cfg", {}).get("module_positive_windows_per_module", ""),
            "module_negative_windows_per_faulty_module": item.get("run_cfg", {}).get("module_negative_windows_per_faulty_module", ""),
            "module_normal_windows_per_module": item.get("run_cfg", {}).get("module_normal_windows_per_module", ""),
            "index_val_fraction": item.get("run_cfg", {}).get("index_val_fraction", ""),
            "selection_metric": item.get("run_cfg", {}).get("selection_metric", ""),
            "train_normal_modules": item.get("train_normal_modules", ""),
            "train_trainable_faulty_modules": item.get("train_trainable_faulty_modules", ""),
            "train_true_transition_modules": item.get("train_true_transition_modules", ""),
            "train_start_in_window_modules": item.get("train_start_in_window_modules", ""),
            "train_left_censored_modules": item.get("train_left_censored_modules", ""),
            "train_faulty_no_primary_window_modules": item.get("train_faulty_no_primary_window_modules", ""),
            "n_params": item["n_params"],
            "seconds": item["seconds"],
        }
        row.update(item["metrics"])
        rows.append(row)
    df_new = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "fold_metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        df_new = pd.concat([old, df_new], ignore_index=True)
        df_new.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df_new.sort_values(["model", "fold"], inplace=True)
    df_new.to_csv(metrics_path, index=False)
    numeric = [
        col
        for col in df_new.columns
        if col
        not in {"model", "paper_ready", "protocol", "input_mode", "train_sampling", "event_hit_mode", "selection_metric"}
        and pd.api.types.is_numeric_dtype(df_new[col])
    ]
    summary = df_new.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def add_deep_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_official_results/lookback24h_ahead1h_legacy_benchmark"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument(
        "--timestamp_mode",
        choices=["legacy_float32", "strict_int64"],
        default="legacy_float32",
    )
    parser.add_argument("--no_save_checkpoint", action="store_true")
    parser.add_argument("--shuffle_train_rows", action="store_true")
    parser.add_argument("--log_batches", type=int, default=1000)
    parser.add_argument("--no_batched_windows", action="store_true")
    parser.add_argument("--amp", action="store_true", help="enable CUDA autocast mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="disable CUDA autocast mixed precision")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--module_cache_dir", default="")
    parser.add_argument("--input_mode", choices=["raw", "model2_features"], default="model2_features")
    parser.add_argument("--train_sampling", choices=["full", "pos_all_neg_ratio", "module_balanced"], default="module_balanced")
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--aux_bce_weight", type=float, default=0.3)
    parser.add_argument("--event_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_hit_mode", choices=["primary_window", "pre_fault"], default="primary_window")
    parser.add_argument("--event_modules_per_epoch", type=int, default=256)
    parser.add_argument("--event_max_windows_per_module", type=int, default=256)
    parser.add_argument("--event_module_batch_size", type=int, default=4)
    parser.add_argument("--event_positive_fraction", type=float, default=0.5)
    parser.add_argument("--module_positive_windows_per_module", type=int, default=32)
    parser.add_argument("--module_negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--module_normal_windows_per_module", type=int, default=8)
    parser.add_argument("--index_val_fraction", type=float, default=0.1)
    parser.add_argument("--selection_metric", choices=["final_score", "f1_score", "precision", "recall", "accuracy"], default="f1_score")
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_val_files", type=int, default=0)
    parser.add_argument("--max_test_files", type=int, default=0)
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--alarm_search_smooth_windows", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--alarm_search_consecutive_ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--force_tqdm", action="store_true")
    return parser


def cfg_from_args(args: argparse.Namespace) -> OfficialDeepCfg:
    return OfficialDeepCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        threshold_grid_size=args.threshold_grid_size,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        fixed_threshold=args.fixed_threshold,
        timestamp_mode=args.timestamp_mode,
        save_checkpoint=not args.no_save_checkpoint,
        shuffle_train_rows=args.shuffle_train_rows,
        log_batches=args.log_batches,
        batched_windows=not args.no_batched_windows,
        amp=bool(args.amp) and not bool(args.no_amp),
        allow_tf32=not args.no_tf32,
        module_cache_dir=args.module_cache_dir,
        input_mode=args.input_mode,
        train_sampling=args.train_sampling,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        aux_bce_weight=args.aux_bce_weight,
        event_loss_weight=args.event_loss_weight,
        event_hit_mode=args.event_hit_mode,
        event_modules_per_epoch=args.event_modules_per_epoch,
        event_max_windows_per_module=args.event_max_windows_per_module,
        event_module_batch_size=args.event_module_batch_size,
        event_positive_fraction=args.event_positive_fraction,
        module_positive_windows_per_module=args.module_positive_windows_per_module,
        module_negative_windows_per_faulty_module=args.module_negative_windows_per_faulty_module,
        module_normal_windows_per_module=args.module_normal_windows_per_module,
        index_val_fraction=args.index_val_fraction,
        selection_metric=args.selection_metric,
        max_train_files=args.max_train_files,
        max_val_files=args.max_val_files,
        max_test_files=args.max_test_files,
        alarm_smoothing=args.alarm_smoothing,
        alarm_smooth_window=args.alarm_smooth_window,
        alarm_consecutive_k=args.alarm_consecutive_k,
        alarm_search_smooth_windows=tuple(int(x) for x in args.alarm_search_smooth_windows),
        alarm_search_consecutive_ks=tuple(int(x) for x in args.alarm_search_consecutive_ks),
        first_alarm_only=not args.no_first_alarm_only,
        use_tqdm=not args.no_tqdm,
        force_tqdm=args.force_tqdm,
    )


def run_single_model_cli(model_name: str, model_builder: ModelBuilder) -> None:
    parser = argparse.ArgumentParser(description=f"Run official OFP formal benchmark for {model_name}.")
    add_deep_args(parser)
    args = parser.parse_args()
    index_df = read_index(args.index_path)
    cfg = cfg_from_args(args)
    results = []
    for fold in args.folds:
        result = run_deep_fold_index(
            model_name,
            model_builder,
            int(fold),
            args.data_dir,
            index_df,
            args.out_root,
            cfg,
        )
        results.append(result)
        aggregate_results([result], args.out_root)
        print(
            f"[official done] model={model_name} fold={fold} "
            f"{format_metric_summary(result['metrics'])}"
        )
    aggregate_results(results, args.out_root)

