from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

REDLAMP_ROOT = Path("D:/RedLamp")
if str(REDLAMP_ROOT) not in sys.path:
    sys.path.insert(0, str(REDLAMP_ROOT))

from models.meta import ConvAEC  # type: ignore  # noqa: E402

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_deep_models import make_window, split_train_val, write_predictions
from OFP_DL_official.ofp_protocol.run_ofp_baselines import SENSORS, read_index
from OFP_DL_official.legacy_protocol.run_deep_models_module_level import (
    choose_ofp_threshold,
    compute_feature_norm_stats,
    metrics_to_dict,
)


SEC_IN_HOUR = 3600.0
ANOMALY_TYPES = ["normal", "spike", "noise", "scale", "cutoff"]


@dataclass
class RedLampOFPCfg:
    seq_len: int = 64
    epochs: int = 3
    batch_size: int = 256
    score_batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    windows_per_module: int = 8
    full_train_windows: bool = False
    include_pre_event_normals: bool = True
    val_fraction: float = 0.1
    threshold_grid_size: int = 99
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_metric: str = "f1"
    embedding_dim: int = 96
    c_loss_ratio: float = 0.1
    cls_score_weight: float = 0.5
    norm_max_files: int = 512
    max_cached_files: int = 1024
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModuleArrayCache:
    def __init__(self, data_dir: Path, max_cached_files: int = 1024) -> None:
        self.data_dir = Path(data_dir)
        self.max_cached_files = int(max_cached_files)
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.order: list[str] = []

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            if file_name in self.order:
                self.order.remove(file_name)
            self.order.append(file_name)
            return self.cache[file_name]

        df = pd.read_csv(self.data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        values = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        item = (timestamps, values, anomaly)

        self.cache[file_name] = item
        self.order.append(file_name)
        while len(self.order) > self.max_cached_files:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def redlamp_params(cfg: RedLampOFPCfg, n_features: int) -> SimpleNamespace:
    return SimpleNamespace(
        name="ConvAEC",
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        epoch=cfg.epochs,
        max_grad_norm=cfg.grad_clip,
        seed=cfg.seed,
        n_features=n_features,
        n_time=cfg.seq_len,
        num_filters=[64, 64, 128, 128],
        embedding_dim=cfg.embedding_dim,
        kernel_size=4,
        dropout=0.2,
        normalization="batch",
        stride=2,
        padding=2,
        anomaly_types=ANOMALY_TYPES,
        classes=len(ANOMALY_TYPES),
        classifier_dim=32,
        c_loss_ratio=cfg.c_loss_ratio,
        apply_anomaly_mask=True,
        label_smoothing=True,
        alpha=0.1,
        beta=0.01,
    )


def one_hot(index: int, classes: int) -> np.ndarray:
    out = np.zeros(classes, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def inject_pseudo_anomaly(x: np.ndarray, anomaly_type: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = np.array(x, copy=True)
    mask = np.ones_like(y, dtype=np.float32)
    seq_len, n_features = y.shape
    n_feat = int(rng.integers(1, min(3, n_features) + 1))
    feat_idx = rng.choice(np.arange(n_features), size=n_feat, replace=False)
    length = int(rng.integers(max(2, seq_len // 16), max(3, seq_len // 4) + 1))
    start = int(rng.integers(0, max(1, seq_len - length + 1)))
    end = min(seq_len, start + length)

    if anomaly_type == "spike":
        times = rng.integers(start, end, size=n_feat)
        for f, t in zip(feat_idx, times):
            y[int(t), int(f)] += float(rng.normal(0.0, 4.0))
            mask[int(t), int(f)] = 0.0
    elif anomaly_type == "noise":
        y[start:end, feat_idx] += rng.normal(0.0, 1.5, size=(end - start, n_feat)).astype(np.float32)
        mask[start:end, feat_idx] = 0.0
    elif anomaly_type == "scale":
        factor = float(rng.choice([0.3, 0.5, 1.8, 2.5]))
        y[start:end, feat_idx] *= factor
        mask[start:end, feat_idx] = 0.0
    elif anomaly_type == "cutoff":
        y[start:end, feat_idx] = 0.0
        mask[start:end, feat_idx] = 0.0
    else:
        pass
    return y.astype(np.float32), mask.astype(np.float32)


class RedLampOFPDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        cache: ModuleArrayCache,
        seq_len: int,
        mean: np.ndarray,
        std: np.ndarray,
        seed: int,
    ) -> None:
        self.samples = samples
        self.cache = cache
        self.seq_len = int(seq_len)
        self.mean = mean
        self.std = std
        self.seed = int(seed)
        self.classes = len(ANOMALY_TYPES)

    def __len__(self) -> int:
        return len(self.samples) * 2

    def __getitem__(self, idx: int):
        base_idx = idx // 2
        is_pseudo = idx % 2 == 1
        file_name, row_idx = self.samples[base_idx]
        _timestamps, values, _anomaly = self.cache.get(file_name)
        x, _obs_mask = make_window(values, row_idx, self.seq_len, self.mean, self.std)
        rng = np.random.default_rng(self.seed + idx * 1009)

        if is_pseudo:
            type_idx = 1 + int(rng.integers(0, self.classes - 1))
            x, anomaly_mask = inject_pseudo_anomaly(x, ANOMALY_TYPES[type_idx], rng)
            label = one_hot(type_idx, self.classes)
        else:
            anomaly_mask = np.ones_like(x, dtype=np.float32)
            label = one_hot(0, self.classes)

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(anomaly_mask.astype(np.float32)),
            torch.from_numpy(label),
        )


def collate_redlamp(batch):
    x = torch.stack([item[0] for item in batch])
    anomaly_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return x, anomaly_mask, labels


def sample_healthy_windows(
    train_files: list[str],
    healthy_files: set[str],
    cache: ModuleArrayCache,
    windows_per_module: int,
    seed: int,
) -> list[tuple[str, int]]:
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, int]] = []
    for name in train_files:
        if name not in healthy_files:
            continue
        timestamps, _values, _anomaly = cache.get(name)
        n_rows = len(timestamps)
        if n_rows == 0:
            continue
        count = min(int(windows_per_module), n_rows)
        replace = n_rows < count
        chosen = rng.choice(np.arange(n_rows), size=count, replace=replace)
        samples.extend((name, int(i)) for i in chosen)
    rng.shuffle(samples)
    return samples


def build_full_window_plan(
    train_files: list[str],
    healthy_files: set[str],
    cache: ModuleArrayCache,
    include_pre_event_normals: bool,
) -> list[tuple[str, int]]:
    plan: list[tuple[str, int]] = []
    for name in train_files:
        timestamps, _values, anomaly = cache.get(name)
        n_rows = len(timestamps)
        if n_rows == 0:
            continue
        if name in healthy_files:
            count = n_rows
        elif include_pre_event_normals:
            anomaly_idx = np.where(anomaly > 0)[0]
            count = int(anomaly_idx[0]) if len(anomaly_idx) else 0
        else:
            count = 0
        if count > 0:
            plan.append((name, count))
    return plan


def normalized_sliding_windows(
    values: np.ndarray,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    count: int | None = None,
) -> np.ndarray:
    valid = np.isfinite(values).astype(np.float32)
    x = np.where(valid > 0, values, 0.0).astype(np.float32)
    x = (x - mean) / std
    x = x * valid
    if count is not None:
        x = x[: int(count)]
    if len(x) == 0:
        return np.zeros((0, seq_len, values.shape[1]), dtype=np.float32)
    pad_x = np.zeros((seq_len - 1, values.shape[1]), dtype=np.float32)
    x_pad = np.concatenate([pad_x, x], axis=0)
    return np.lib.stride_tricks.sliding_window_view(x_pad, window_shape=seq_len, axis=0).transpose(0, 2, 1)


def make_redlamp_batch(
    normal_windows: np.ndarray,
    cfg: RedLampOFPCfg,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normals = np.array(normal_windows, copy=True).astype(np.float32)
    pseudo = np.empty_like(normals, dtype=np.float32)
    pseudo_mask = np.empty_like(normals, dtype=np.float32)
    pseudo_labels = []
    for i in range(len(normals)):
        type_idx = 1 + int(rng.integers(0, len(ANOMALY_TYPES) - 1))
        pseudo[i], pseudo_mask[i] = inject_pseudo_anomaly(normals[i], ANOMALY_TYPES[type_idx], rng)
        pseudo_labels.append(one_hot(type_idx, len(ANOMALY_TYPES)))
    normal_mask = np.ones_like(normals, dtype=np.float32)
    normal_labels = np.repeat(one_hot(0, len(ANOMALY_TYPES))[None, :], len(normals), axis=0)
    x = np.concatenate([normals, pseudo], axis=0)
    anomaly_mask = np.concatenate([normal_mask, pseudo_mask], axis=0)
    labels = np.concatenate([normal_labels, np.stack(pseudo_labels).astype(np.float32)], axis=0)
    return torch.from_numpy(x), torch.from_numpy(anomaly_mask), torch.from_numpy(labels)


def train_epoch(
    model: ConvAEC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: RedLampOFPCfg,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ae = 0.0
    total_cls = 0.0
    n = 0
    for x, anomaly_mask, label in loader:
        if x.shape[0] <= 1:
            continue
        x = x.to(cfg.device)
        anomaly_mask = anomaly_mask.to(cfg.device)
        label = label.to(cfg.device)
        optimizer.zero_grad()
        predicted, pred_label, _pred_enc = model(x)
        loss, loss_ae, loss_cls = model.calculate_loss(x, predicted, label, pred_label, anomaly_mask, epoch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        bsz = int(x.shape[0])
        total_loss += float(loss.item()) * bsz
        total_ae += float(loss_ae.item()) * bsz
        total_cls += float(loss_cls.item()) * bsz
        n += bsz
    denom = max(n, 1)
    return {"loss": total_loss / denom, "loss_ae": total_ae / denom, "loss_cls": total_cls / denom, "samples": float(n)}


def train_epoch_full_windows(
    model: ConvAEC,
    train_plan: list[tuple[str, int]],
    cache: ModuleArrayCache,
    mean: np.ndarray,
    std: np.ndarray,
    optimizer: torch.optim.Optimizer,
    cfg: RedLampOFPCfg,
    epoch: int,
) -> dict[str, float]:
    model.train()
    rng = np.random.default_rng(cfg.seed + epoch * 1_000_003)
    order = np.arange(len(train_plan))
    rng.shuffle(order)
    total_loss = 0.0
    total_ae = 0.0
    total_cls = 0.0
    n = 0
    batch_normal = max(1, cfg.batch_size // 2)
    for plan_idx in order:
        name, count = train_plan[int(plan_idx)]
        _timestamps, values, _anomaly = cache.get(name)
        windows_view = normalized_sliding_windows(values, cfg.seq_len, mean, std, count=count)
        if len(windows_view) == 0:
            continue
        starts = np.arange(0, len(windows_view), batch_normal)
        rng.shuffle(starts)
        for start in starts:
            end = min(len(windows_view), int(start) + batch_normal)
            x, anomaly_mask, label = make_redlamp_batch(windows_view[int(start):end], cfg, rng)
            if x.shape[0] <= 1:
                continue
            x = x.to(cfg.device)
            anomaly_mask = anomaly_mask.to(cfg.device)
            label = label.to(cfg.device)
            optimizer.zero_grad()
            predicted, pred_label, _pred_enc = model(x)
            loss, loss_ae, loss_cls = model.calculate_loss(x, predicted, label, pred_label, anomaly_mask, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            bsz = int(x.shape[0])
            total_loss += float(loss.item()) * bsz
            total_ae += float(loss_ae.item()) * bsz
            total_cls += float(loss_cls.item()) * bsz
            n += bsz
    denom = max(n, 1)
    return {"loss": total_loss / denom, "loss_ae": total_ae / denom, "loss_cls": total_cls / denom, "samples": float(n)}


@torch.no_grad()
def score_windows_raw(model: ConvAEC, x: torch.Tensor, cfg: RedLampOFPCfg) -> np.ndarray:
    model.eval()
    x = x.to(cfg.device)
    pred, pred_label, _enc = model(x)
    recon = ((x - pred) ** 2).mean(dim=(1, 2))
    nonnormal = 1.0 - pred_label[:, 0]
    raw = recon + float(cfg.cls_score_weight) * nonnormal
    return raw.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def fit_score_calibration(
    model: ConvAEC,
    dataset: RedLampOFPDataset,
    cfg: RedLampOFPCfg,
    max_batches: int = 128,
) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_redlamp)
    scores: list[np.ndarray] = []
    for step, (x, _mask, label) in enumerate(loader):
        normal = label[:, 0] > 0.5
        if normal.any():
            scores.append(score_windows_raw(model, x[normal], cfg))
        if step + 1 >= int(max_batches):
            break
    all_scores = np.concatenate(scores) if scores else np.zeros(1, dtype=np.float32)
    return float(all_scores.mean()), float(all_scores.std() + 1e-6)


@torch.no_grad()
def fit_score_calibration_full(
    model: ConvAEC,
    train_plan: list[tuple[str, int]],
    cache: ModuleArrayCache,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: RedLampOFPCfg,
    max_batches: int = 128,
) -> tuple[float, float]:
    scores: list[np.ndarray] = []
    batch_normal = max(1, cfg.batch_size)
    seen_batches = 0
    for name, count in train_plan:
        _timestamps, values, _anomaly = cache.get(name)
        windows_view = normalized_sliding_windows(values, cfg.seq_len, mean, std, count=count)
        for start in range(0, len(windows_view), batch_normal):
            end = min(len(windows_view), start + batch_normal)
            xb = torch.from_numpy(np.array(windows_view[start:end], copy=True).astype(np.float32))
            scores.append(score_windows_raw(model, xb, cfg))
            seen_batches += 1
            if seen_batches >= int(max_batches):
                all_scores = np.concatenate(scores) if scores else np.zeros(1, dtype=np.float32)
                return float(all_scores.mean()), float(all_scores.std() + 1e-6)
    all_scores = np.concatenate(scores) if scores else np.zeros(1, dtype=np.float32)
    return float(all_scores.mean()), float(all_scores.std() + 1e-6)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


@torch.no_grad()
def score_one_module(
    model: ConvAEC,
    values: np.ndarray,
    seq_len: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    cfg: RedLampOFPCfg,
    score_mean: float,
    score_std: float,
) -> np.ndarray:
    valid = np.isfinite(values).astype(np.float32)
    x = np.where(valid > 0, values, 0.0).astype(np.float32)
    x = (x - mean) / std
    x = x * valid
    pad_x = np.zeros((seq_len - 1, values.shape[1]), dtype=np.float32)
    x_pad = np.concatenate([pad_x, x], axis=0)
    x_windows = np.lib.stride_tricks.sliding_window_view(x_pad, window_shape=seq_len, axis=0).transpose(0, 2, 1)
    scores: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        end = min(len(values), start + batch_size)
        xb = torch.from_numpy(np.array(x_windows[start:end], copy=True))
        raw = score_windows_raw(model, xb, cfg)
        scores.append(sigmoid((raw - score_mean) / score_std).astype(np.float32))
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def collect_module_scores(
    model: ConvAEC,
    data_dir: Path,
    file_names: list[str],
    cache: ModuleArrayCache,
    cfg: RedLampOFPCfg,
    mean: np.ndarray,
    std: np.ndarray,
    score_mean: float,
    score_std: float,
) -> list[dict]:
    rows = []
    for name in file_names:
        timestamps, values, anomaly = cache.get(name)
        scores = score_one_module(
            model=model,
            values=values,
            seq_len=cfg.seq_len,
            mean=mean,
            std=std,
            batch_size=cfg.score_batch_size,
            cfg=cfg,
            score_mean=score_mean,
            score_std=score_std,
        )
        true_pos = anomaly > 0
        true_label = int(true_pos.any())
        true_ts = float(timestamps[np.argmax(true_pos)]) if true_label else None
        rows.append(
            {
                "file_name": name,
                "timestamps": timestamps,
                "scores": scores,
                "true_label": true_label,
                "true_ts": true_ts,
            }
        )
    return rows


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    cfg: RedLampOFPCfg,
    max_train_files: int | None,
    max_val_files: int | None,
    max_test_files: int | None,
) -> dict:
    t0 = time.time()
    set_seed(cfg.seed + fold)
    train_files, val_files, test_files = split_train_val(index_df, fold, cfg.val_fraction, cfg.seed)
    if max_train_files is not None:
        train_files = train_files[: int(max_train_files)]
    if max_val_files is not None:
        val_files = val_files[: int(max_val_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]

    label_by_file = dict(zip(index_df["file_name"], index_df["Label"]))
    healthy_files = {name for name, label in label_by_file.items() if int(label) == 0}
    cache = ModuleArrayCache(data_dir, max_cached_files=cfg.max_cached_files)
    mean, std = compute_feature_norm_stats(
        data_dir=data_dir,
        train_files=train_files,
        feature_mode="raw",
        rolling_windows=(12, 36),
        max_files=cfg.norm_max_files,
    )
    samples: list[tuple[str, int]] = []
    train_plan: list[tuple[str, int]] = []
    if cfg.full_train_windows:
        train_plan = build_full_window_plan(
            train_files=train_files,
            healthy_files=healthy_files,
            cache=cache,
            include_pre_event_normals=cfg.include_pre_event_normals,
        )
        normal_train_windows = int(sum(count for _name, count in train_plan))
        if normal_train_windows <= 0:
            raise RuntimeError("No full-window training samples were available for REDLAMP-OFP.")
        dataset = None
        loader = None
    else:
        samples = sample_healthy_windows(
            train_files=train_files,
            healthy_files=healthy_files,
            cache=cache,
            windows_per_module=cfg.windows_per_module,
            seed=cfg.seed + fold,
        )
        if not samples:
            raise RuntimeError("No healthy training windows were available for REDLAMP-OFP.")
        normal_train_windows = len(samples)
        dataset = RedLampOFPDataset(samples=samples, cache=cache, seq_len=cfg.seq_len, mean=mean, std=std, seed=cfg.seed + fold)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_redlamp)
    params = redlamp_params(cfg, n_features=len(SENSORS))
    model = ConvAEC(params).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        ep0 = time.time()
        if cfg.full_train_windows:
            metrics = train_epoch_full_windows(model, train_plan, cache, mean, std, optimizer, cfg, epoch)
        else:
            assert loader is not None
            metrics = train_epoch(model, loader, optimizer, cfg, epoch)
        metrics["epoch"] = epoch
        metrics["seconds"] = time.time() - ep0
        history.append(metrics)
        print(
            f"[redlamp_ofp fold={fold}] ep{epoch:02d} "
            f"loss={metrics['loss']:.4f} ae={metrics['loss_ae']:.4f} cls={metrics['loss_cls']:.4f} "
            f"time={metrics['seconds']:.1f}s"
        )

    if cfg.full_train_windows:
        score_mean, score_std = fit_score_calibration_full(model, train_plan, cache, mean, std, cfg)
    else:
        assert dataset is not None
        score_mean, score_std = fit_score_calibration(model, dataset, cfg)
    print(f"[redlamp_ofp fold={fold}] score calibration mean={score_mean:.6f} std={score_std:.6f}")

    print(f"[redlamp_ofp fold={fold}] scoring val modules={len(val_files)}")
    val_rows = collect_module_scores(model, data_dir, val_files, cache, cfg, mean, std, score_mean, score_std)
    threshold, val_module_metrics = choose_ofp_threshold(val_rows, cfg)
    print(
        f"[redlamp_ofp fold={fold}] val final={val_module_metrics['final_score']:.4f} "
        f"F1={val_module_metrics['f1_score']:.4f} thr={threshold:.3f}"
    )

    print(f"[redlamp_ofp fold={fold}] scoring test modules={len(test_files)}")
    test_rows = collect_module_scores(model, data_dir, test_files, cache, cfg, mean, std, score_mean, score_std)
    run_dir = out_root / "redlamp_ofp" / f"fold_{fold}"
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(test_rows, pred_dir, threshold)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    test_metrics = metrics_to_dict(summary_df)

    result = {
        "model": "redlamp_ofp",
        "fold": int(fold),
        "training_mode": "full_normal_window_redlamp" if cfg.full_train_windows else "healthy_pseudo_anomaly_redlamp",
        "threshold_selection": f"validation_ofp_{cfg.threshold_metric}",
        "threshold": float(threshold),
        "train_modules": len(train_files),
        "healthy_train_windows": int(normal_train_windows),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "score_mean": score_mean,
        "score_std": score_std,
        "seconds": time.time() - t0,
        "run_cfg": asdict(cfg),
        "val_module_metrics": val_module_metrics,
        "metrics": test_metrics,
        "history": history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "run_cfg": asdict(cfg)}, run_dir / "model.pt")
    return result


def aggregate(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "training_mode": item["training_mode"],
            "threshold": item["threshold"],
            "train_modules": item["train_modules"],
            "healthy_train_windows": item["healthy_train_windows"],
            "val_modules": item["val_modules"],
            "test_modules": item["test_modules"],
            "seconds": item["seconds"],
        }
        row.update(item["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "fold_metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df.sort_values(["model", "fold"], inplace=True)
    df.to_csv(metrics_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run REDLAMP under OFP first-warning evaluation.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_legacy_protocol/redlamp_ofp"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--score_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--windows_per_module", type=int, default=8)
    parser.add_argument("--full_train_windows", action="store_true")
    parser.add_argument("--exclude_pre_event_normals", action="store_false", dest="include_pre_event_normals")
    parser.set_defaults(include_pre_event_normals=True)
    parser.add_argument("--embedding_dim", type=int, default=96)
    parser.add_argument("--c_loss_ratio", type=float, default=0.1)
    parser.add_argument("--cls_score_weight", type=float, default=0.5)
    parser.add_argument("--threshold_metric", choices=["final", "f1"], default="f1")
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--norm_max_files", type=int, default=512)
    parser.add_argument("--max_cached_files", type=int, default=1024)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    cfg = RedLampOFPCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        score_batch_size=args.score_batch_size,
        lr=args.lr,
        windows_per_module=args.windows_per_module,
        full_train_windows=args.full_train_windows,
        include_pre_event_normals=args.include_pre_event_normals,
        val_fraction=args.val_fraction,
        threshold_grid_size=args.threshold_grid_size,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_metric=args.threshold_metric,
        embedding_dim=args.embedding_dim,
        c_loss_ratio=args.c_loss_ratio,
        cls_score_weight=args.cls_score_weight,
        norm_max_files=args.norm_max_files,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        device=args.device,
    )
    results = []
    for fold in args.folds:
        print(f"[run-redlamp-ofp] fold={fold} device={cfg.device}")
        result = run_fold(
            fold=fold,
            data_dir=args.data_dir,
            index_df=index_df,
            out_root=args.out_root,
            cfg=cfg,
            max_train_files=args.max_train_files,
            max_val_files=args.max_val_files,
            max_test_files=args.max_test_files,
        )
        results.append(result)
        aggregate([result], args.out_root)
        print(
            f"[done-redlamp-ofp] fold={fold} "
            f"Final={result['metrics'].get('final_score', 0):.4f} "
            f"F1={result['metrics'].get('f1_score', 0):.4f} "
            f"P={result['metrics'].get('precision', 0):.4f} "
            f"R={result['metrics'].get('recall', 0):.4f} "
            f"time={result['seconds']:.1f}s"
        )
    aggregate(results, args.out_root)


if __name__ == "__main__":
    main()

