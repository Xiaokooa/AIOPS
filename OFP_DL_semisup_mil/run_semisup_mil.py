from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True.*",
    category=UserWarning,
)

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.common.formal_data import (
    HORIZON_HOURS,
    PRIMARY_HORIZON_INDEX,
    apply_timestamp_mode,
    choose_alarm_strategy_by_final_score,
    compute_norm_stats_all_rows,
    make_window,
    write_predictions,
)
from OFP_DL_official.common.trainer import format_metric_summary, resolve_runtime_device, runtime_device_summary
from OFP_DL_official.common.losses import first_warning_event_loss, split_alarm_aux_logits
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.ofp_formal_protocol.protocol import (
    SENSORS,
    ahead_labels_and_valid_mask_multi_first_event,
    first_anomaly_timestamp,
)
from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder


@dataclass
class SemiSupMilCfg:
    seq_len: int = 288
    d_model: int = 96
    n_heads: int = 4
    layers: int = 2
    dropout: float = 0.2
    pretrain_epochs: int = 2
    finetune_epochs: int = 3
    pretrain_batch_size: int = 128
    module_batch_size: int = 8
    score_batch_size: int = 512
    pretrain_samples_per_file: int = 16
    windows_per_module: int = 32
    positive_windows_per_faulty: int = 16
    negative_ratio: float = 1.0
    aux_bce_weight: float = 0.3
    event_loss_weight: float = 1.0
    mask_ratio: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    threshold_grid_size: int = 99
    selection_metric: str = "final_score"
    alarm_smoothing: str = "ema"
    alarm_smooth_window: int = 1
    alarm_consecutive_k: int = 1
    alarm_search_smooth_windows: tuple[int, ...] = (1, 3, 6, 12)
    alarm_search_consecutive_ks: tuple[int, ...] = (1, 2, 3)
    first_alarm_only: bool = True
    max_cached_files: int = 512
    timestamp_mode: str = "legacy_float32"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModuleCache:
    def __init__(self, data_dir: Path, mean: np.ndarray, std: np.ndarray, cfg: SemiSupMilCfg) -> None:
        self.data_dir = Path(data_dir)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.cfg = cfg
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            item = self.cache.pop(file_name)
            self.cache[file_name] = item
            return item
        df = pd.read_csv(self.data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        timestamps = apply_timestamp_mode(timestamps, self.cfg.timestamp_mode)
        values = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        labels, valid_mask = ahead_labels_and_valid_mask_multi_first_event(
            timestamps.astype(float),
            anomaly,
            HORIZON_HOURS,
        )
        item = (timestamps, values, anomaly, labels.astype(np.int8), valid_mask.astype(bool))
        self.cache[file_name] = item
        while len(self.cache) > int(self.cfg.max_cached_files):
            self.cache.popitem(last=False)
        return item

    def normalized_window(self, values: np.ndarray, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        return make_window(values, int(row_idx), self.cfg.seq_len, self.mean, self.std)


class NormalWindowDataset(Dataset):
    def __init__(self, files: list[str], cache: ModuleCache, samples_per_file: int, seed: int) -> None:
        self.cache = cache
        self.samples: list[tuple[str, int]] = []
        rng = np.random.default_rng(seed)
        for name in files:
            _ts, _values, anomaly, _labels, valid_mask = cache.get(name)
            if np.any(anomaly > 0):
                continue
            positions = np.flatnonzero(valid_mask)
            if len(positions) == 0:
                continue
            take = min(int(samples_per_file), len(positions))
            chosen = rng.choice(positions, size=take, replace=False)
            self.samples.extend((name, int(i)) for i in chosen)
        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, row_idx = self.samples[idx]
        _ts, values, _anomaly, _labels, _valid_mask = self.cache.get(name)
        x, mask = self.cache.normalized_window(values, row_idx)
        return torch.from_numpy(x), torch.from_numpy(mask)


class ModuleBagDataset(Dataset):
    def __init__(self, files: list[str], cache: ModuleCache, cfg: SemiSupMilCfg, seed: int) -> None:
        self.cache = cache
        self.cfg = cfg
        self.seed = int(seed)
        self.epoch = 0
        self.files = [name for name in files if np.any(cache.get(name)[4])]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _sample(rng: np.random.Generator, arr: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(arr) == 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(arr, size=int(count), replace=len(arr) < int(count)).astype(np.int64)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        timestamps, values, anomaly, labels, valid_mask = self.cache.get(name)
        rng = np.random.default_rng(self.seed + idx + self.epoch * 1_000_003)
        valid_positions = np.flatnonzero(valid_mask).astype(np.int64)
        fault = int(np.any(anomaly > 0))
        primary = labels[:, PRIMARY_HORIZON_INDEX] if labels.ndim > 1 else labels
        pos = valid_positions[primary[valid_positions] > 0]
        neg = valid_positions[primary[valid_positions] <= 0]
        n_pos = min(int(self.cfg.positive_windows_per_faulty), int(self.cfg.windows_per_module)) if fault else 0
        pos_chosen = self._sample(rng, pos, n_pos)
        neg_chosen = self._sample(rng, neg if len(neg) else valid_positions, int(self.cfg.windows_per_module) - len(pos_chosen))
        chosen = np.concatenate([pos_chosen, neg_chosen])
        if len(chosen) < int(self.cfg.windows_per_module):
            fill = self._sample(rng, valid_positions, int(self.cfg.windows_per_module) - len(chosen))
            chosen = np.concatenate([chosen, fill])
        rng.shuffle(chosen)
        chosen = chosen[: int(self.cfg.windows_per_module)]

        xs, masks = [], []
        for row_idx in chosen:
            x, mask = self.cache.normalized_window(values, int(row_idx))
            xs.append(x)
            masks.append(mask)
        point_targets = np.asarray(labels[chosen], dtype=np.float32)
        event_mask = np.full(len(chosen), bool(fault), dtype=np.bool_)
        return (
            torch.from_numpy(np.stack(xs).astype(np.float32)),
            torch.from_numpy(np.stack(masks).astype(np.float32)),
            torch.from_numpy(point_targets),
            torch.tensor(float(fault), dtype=torch.float32),
            torch.from_numpy(event_mask),
        )


def collate_bags(batch):
    xs = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    point_targets = torch.stack([item[2] for item in batch])
    module_targets = torch.stack([item[3] for item in batch])
    event_masks = torch.stack([item[4] for item in batch])
    return xs, masks, point_targets, module_targets, event_masks


class SemiSupTemporalMIL(nn.Module):
    def __init__(self, n_sensors: int, cfg: SemiSupMilCfg) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = int(n_sensors) * 2
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)
        self.recon_head = nn.Linear(cfg.d_model, n_sensors)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, len(HORIZON_HOURS) + 1),
        )
        nn.init.constant_(self.risk_head[-1].bias, -2.0)

    def encode_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, mask], dim=-1)
        tokens = self.input_proj(inp) + self.pos_embedding[:, : x.size(1), :]
        return self.encoder(tokens)

    def pooled(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        row_mask = (mask.sum(dim=-1) > 0).float()
        denom = row_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (tokens * row_mask.unsqueeze(-1)).sum(dim=1) / denom

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.recon_head(self.encode_tokens(x, mask))

    def risk_logits(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(x, mask)
        return self.risk_head(self.pooled(tokens, mask))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain_epoch(model: SemiSupTemporalMIL, loader: DataLoader, optimizer, cfg: SemiSupMilCfg) -> float:
    model.train()
    total, denom = 0.0, 0.0
    for x, mask in loader:
        x = x.to(cfg.device)
        mask = mask.to(cfg.device)
        observed = mask > 0
        corrupt = (torch.rand_like(x) < float(cfg.mask_ratio)) & observed
        if not corrupt.any():
            corrupt = observed
        x_in = x.masked_fill(corrupt, 0.0)
        mask_in = mask.masked_fill(corrupt, 0.0)
        recon = model.reconstruct(x_in, mask_in)
        loss = ((recon - x).pow(2) * corrupt.float()).sum() / corrupt.float().sum().clamp(min=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu()) * float(corrupt.sum().detach().cpu())
        denom += float(corrupt.sum().detach().cpu())
    return total / max(denom, 1.0)


def mil_loss(
    logits: torch.Tensor,
    point_targets: torch.Tensor,
    module_targets: torch.Tensor,
    event_masks: torch.Tensor,
    cfg: SemiSupMilCfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    alarm_logits, aux_logits = split_alarm_aux_logits(logits, aux_dim=point_targets.shape[-1])
    aux_loss = nn.functional.binary_cross_entropy_with_logits(aux_logits, point_targets)
    event_loss = first_warning_event_loss(alarm_logits, event_masks, module_targets)
    total = float(cfg.aux_bce_weight) * aux_loss + float(cfg.event_loss_weight) * event_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "aux_bce_loss": float(aux_loss.detach().cpu()),
        "event_first_warning_loss": float(event_loss.detach().cpu()),
    }


def finetune_epoch(model: SemiSupTemporalMIL, dataset: ModuleBagDataset, loader: DataLoader, optimizer, cfg: SemiSupMilCfg, epoch: int) -> dict[str, float]:
    model.train()
    dataset.set_epoch(epoch)
    totals: dict[str, float] = {"loss": 0.0, "aux_bce_loss": 0.0, "event_first_warning_loss": 0.0}
    n_modules = 0
    for xs, masks, point_targets, module_targets, event_masks in loader:
        bsz, n_windows, seq_len, n_sensors = xs.shape
        xs = xs.to(cfg.device).reshape(bsz * n_windows, seq_len, n_sensors)
        masks = masks.to(cfg.device).reshape(bsz * n_windows, seq_len, n_sensors)
        point_targets = point_targets.to(cfg.device)
        module_targets = module_targets.to(cfg.device)
        event_masks = event_masks.to(cfg.device)
        logits = model.risk_logits(xs, masks).reshape(bsz, n_windows, len(HORIZON_HOURS) + 1)
        loss, parts = mil_loss(logits, point_targets, module_targets, event_masks, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value) * bsz
        n_modules += bsz
    denom = max(n_modules, 1)
    return {key: value / denom for key, value in totals.items()} | {"modules": float(n_modules)}


@torch.no_grad()
def score_module(model: SemiSupTemporalMIL, cache: ModuleCache, file_name: str, cfg: SemiSupMilCfg) -> dict:
    timestamps, values, anomaly, _labels, valid_mask = cache.get(file_name)
    scores: list[np.ndarray] = []
    for start in range(0, len(values), int(cfg.score_batch_size)):
        end = min(len(values), start + int(cfg.score_batch_size))
        xs, masks = [], []
        for row_idx in range(start, end):
            x, mask = cache.normalized_window(values, row_idx)
            xs.append(x)
            masks.append(mask)
        xb = torch.from_numpy(np.stack(xs).astype(np.float32)).to(cfg.device)
        mb = torch.from_numpy(np.stack(masks).astype(np.float32)).to(cfg.device)
        logits = model.risk_logits(xb, mb)
        alarm_logits, _aux_logits = split_alarm_aux_logits(logits, aux_dim=len(HORIZON_HOURS))
        scores.append(torch.sigmoid(alarm_logits).detach().cpu().numpy())
    true_ts = first_anomaly_timestamp(timestamps.astype(float), anomaly)
    return {
        "file_name": file_name,
        "timestamps": timestamps,
        "scores": np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, dtype=np.float32),
        "true_label": int(true_ts is not None),
        "true_ts": true_ts,
        "valid_mask": valid_mask,
    }


def score_modules(model: SemiSupTemporalMIL, cache: ModuleCache, files: list[str], cfg: SemiSupMilCfg) -> list[dict]:
    model.eval()
    return [score_module(model, cache, name, cfg) for name in files]


def limit_files(files: list[str], max_files: int | None) -> list[str]:
    return files[: int(max_files)] if max_files is not None else files


def run_fold(args: argparse.Namespace, cfg: SemiSupMilCfg) -> dict:
    set_seed(cfg.seed + int(args.fold))
    cfg.device = resolve_runtime_device(cfg.device)
    print(f"[semisup-init] {runtime_device_summary(cfg.device)}", flush=True)
    if str(cfg.device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    index_df = read_index(args.index_path)
    train_all, test_files = files_for_index_fold(index_df, int(args.fold))
    label_map = dict(zip(index_df["file_name"].astype(str), index_df["Label"].astype(int)))
    train_labels = [int(label_map.get(name, 0)) for name in train_all]
    train_files, val_files = train_test_split(
        train_all,
        test_size=float(args.val_fraction),
        random_state=cfg.seed + int(args.fold),
        stratify=train_labels if len(set(train_labels)) > 1 else None,
    )
    train_files = limit_files(train_files, args.max_train_files)
    val_files = limit_files(val_files, args.max_val_files)
    test_files = limit_files(test_files, args.max_test_files)
    normal_pretrain_files = [name for name in train_files if int(label_map.get(name, 0)) == 0]
    normal_pretrain_files = limit_files(normal_pretrain_files, args.max_pretrain_files)

    out_dir = Path(args.out_root) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_norm_stats_all_rows(args.data_dir, train_files, timestamp_mode=cfg.timestamp_mode)
    mean, std = stats.arrays()
    cache = ModuleCache(args.data_dir, mean, std, cfg)
    model = SemiSupTemporalMIL(len(SENSORS), cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict] = []
    if int(cfg.pretrain_epochs) > 0 and normal_pretrain_files:
        pre_ds = NormalWindowDataset(normal_pretrain_files, cache, cfg.pretrain_samples_per_file, cfg.seed)
        pre_loader = DataLoader(pre_ds, batch_size=cfg.pretrain_batch_size, shuffle=True, drop_last=False)
        for epoch in range(1, int(cfg.pretrain_epochs) + 1):
            loss = pretrain_epoch(model, pre_loader, optimizer, cfg)
            history.append({"stage": "pretrain", "epoch": epoch, "loss": loss, "samples": len(pre_ds)})
            print(f"[pretrain] epoch={epoch} loss={loss:.6f} samples={len(pre_ds)}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        print("[finetune-init] reset optimizer after normal-module pretraining")

    ft_ds = ModuleBagDataset(train_files, cache, cfg, cfg.seed + int(args.fold))
    ft_loader = DataLoader(ft_ds, batch_size=cfg.module_batch_size, shuffle=True, collate_fn=collate_bags, drop_last=False)
    for epoch in range(1, int(cfg.finetune_epochs) + 1):
        parts = finetune_epoch(model, ft_ds, ft_loader, optimizer, cfg, epoch)
        history.append({"stage": "finetune", "epoch": epoch, **parts, "dataset_modules": len(ft_ds)})
        print(
            f"[finetune] epoch={epoch} loss={parts['loss']:.6f} "
            f"event={parts['event_first_warning_loss']:.6f} aux={parts['aux_bce_loss']:.6f} "
            f"modules={int(parts['modules'])}"
        )

    val_rows = score_modules(model, cache, val_files, cfg)
    threshold, alarm_strategy, val_metrics = choose_alarm_strategy_by_final_score(
        val_rows,
        grid_size=cfg.threshold_grid_size,
        smoothing=cfg.alarm_smoothing,
        smooth_windows=cfg.alarm_search_smooth_windows,
        consecutive_ks=cfg.alarm_search_consecutive_ks,
        selection_metric=cfg.selection_metric,
    )
    print(f"[val-select] metric={cfg.selection_metric} threshold={threshold:.4f} alarm={alarm_strategy}")
    print(f"[val-select] {format_metric_summary(val_metrics)}")
    test_rows = score_modules(model, cache, test_files, cfg)
    pred_dir = out_dir / "predictions"
    eval_dir = out_dir / "evaluation"
    write_predictions(
        test_rows,
        pred_dir,
        threshold,
        alarm_smoothing=str(alarm_strategy.get("alarm_smoothing", cfg.alarm_smoothing)),
        alarm_smooth_window=int(alarm_strategy.get("alarm_smooth_window", cfg.alarm_smooth_window)),
        alarm_consecutive_k=int(alarm_strategy.get("alarm_consecutive_k", cfg.alarm_consecutive_k)),
        first_alarm_only=cfg.first_alarm_only,
    )
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, args.data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}
    print(f"[test-metrics] {format_metric_summary(metrics)}")
    result = {
        "fold": int(args.fold),
        "protocol": "semisup_normal_pretrain_first_warning_event_aux_multi_horizon",
        "train_modules": len(train_files),
        "normal_pretrain_modules": len(normal_pretrain_files),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "threshold": float(threshold),
        "selection_metric": str(cfg.selection_metric),
        "alarm_strategy": alarm_strategy,
        "first_alarm_only": bool(cfg.first_alarm_only),
        "val_metrics": val_metrics,
        "metrics": metrics,
        "cfg": asdict(cfg),
        "norm_stats": asdict(stats),
        "history": history,
    }
    (out_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": asdict(cfg),
            "threshold": threshold,
            "alarm_strategy": alarm_strategy,
            "first_alarm_only": bool(cfg.first_alarm_only),
        },
        out_dir / "model.pt",
    )
    print(
        f"[done] fold={args.fold} threshold={threshold:.4f} alarm={alarm_strategy} "
        f"final={metrics.get('final_score', 0.0):.4f} f1={metrics.get('f1_score', 0.0):.4f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-supervised normal pretraining + first-event 120h MIL fine-tuning.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_semisup_mil_results"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_pretrain_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrain_epochs", type=int, default=2)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--pretrain_batch_size", type=int, default=128)
    parser.add_argument("--module_batch_size", type=int, default=8)
    parser.add_argument("--score_batch_size", type=int, default=512)
    parser.add_argument("--pretrain_samples_per_file", type=int, default=16)
    parser.add_argument("--windows_per_module", type=int, default=32)
    parser.add_argument("--positive_windows_per_faulty", type=int, default=16)
    parser.add_argument("--aux_bce_weight", type=float, default=0.3)
    parser.add_argument("--event_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument(
        "--selection_metric",
        choices=["final_score", "f1_score", "precision", "recall", "accuracy"],
        default="final_score",
    )
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--alarm_search_smooth_windows", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--alarm_search_consecutive_ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--timestamp_mode", default="legacy_float32", choices=["legacy_float32", "ofp_legacy", "strict_int64", "strict"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SemiSupMilCfg(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        layers=args.layers,
        dropout=args.dropout,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        module_batch_size=args.module_batch_size,
        score_batch_size=args.score_batch_size,
        pretrain_samples_per_file=args.pretrain_samples_per_file,
        windows_per_module=args.windows_per_module,
        positive_windows_per_faulty=args.positive_windows_per_faulty,
        aux_bce_weight=args.aux_bce_weight,
        event_loss_weight=args.event_loss_weight,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold_grid_size=args.threshold_grid_size,
        selection_metric=args.selection_metric,
        alarm_smoothing=args.alarm_smoothing,
        alarm_smooth_window=args.alarm_smooth_window,
        alarm_consecutive_k=args.alarm_consecutive_k,
        alarm_search_smooth_windows=tuple(int(x) for x in args.alarm_search_smooth_windows),
        alarm_search_consecutive_ks=tuple(int(x) for x in args.alarm_search_consecutive_ks),
        first_alarm_only=not args.no_first_alarm_only,
        max_cached_files=args.max_cached_files,
        timestamp_mode=args.timestamp_mode,
        seed=args.seed,
        device=args.device,
    )
    t0 = time.time()
    result = run_fold(args, cfg)
    print(f"[elapsed] seconds={time.time() - t0:.1f} final={result['metrics'].get('final_score', 0.0):.4f}")


if __name__ == "__main__":
    main()
