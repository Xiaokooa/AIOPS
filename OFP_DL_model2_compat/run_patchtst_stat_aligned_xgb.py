from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_model2_compat.run_model2_compat_deep import (
    CompatBatchedDataset,
    CompatCfg,
    Model2FeatureCache,
    cap_files_stratified,
    compat_alarm_logit,
    compat_feature_names,
    compute_feature_norm_stats,
    cuda_autocast,
    effective_pos_weight,
    file_label_map,
    format_duration,
    format_epoch_status,
    format_rate,
    log_run_header,
    progress_bar,
    split_train_val_files,
)
from OFP_DL_model2_compat.run_model2_compat_tabular_fusion import (
    LastLinearInputHook,
    evaluate_prediction_output,
    log_stage_done,
    log_stage_start,
    positive_scores,
    select_threshold_for_scores,
    write_threshold_predictions,
)
from OFP_DL_official.PatchTST.model import build_model as build_patchtst
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.common.trainer import format_metric_summary, resolve_runtime_device


RUN_NAME = "patchtst_stat_aligned_xgb"
DEFAULT_THRESHOLD_GRID = (
    "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,"
    "0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99"
)
RAW_SEQUENCE_FEATURES = [
    "Temp",
    "Curr",
    "TxP0",
    "RxP0",
    "RxP1",
    "RxP2",
    "RxP3",
    "RxP4",
    "TxP1",
    "TxP2",
    "TxP3",
    "TxP4",
]


def configure_runtime(cfg: CompatCfg, seed: int, fold: int) -> None:
    torch.manual_seed(int(seed) + int(fold))
    np.random.seed(int(seed) + int(fold))
    cfg.device = resolve_runtime_device(cfg.device)
    if str(cfg.device).startswith("cuda") and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def feature_indices(feature_names: list[str], stat_feature_mode: str) -> tuple[list[int], list[str], list[int], list[str]]:
    raw_names = [name for name in RAW_SEQUENCE_FEATURES if name in feature_names]
    raw_indices = [feature_names.index(name) for name in raw_names]
    if not raw_indices:
        raise ValueError("No raw sequence features were found in compat feature names")

    raw_set = set(raw_names) | {"Ts"}
    mode = str(stat_feature_mode).lower()
    if mode == "all_engineered":
        stat_names = [name for name in feature_names if name not in raw_set]
    elif mode == "model2_expert":
        keep = {"FeCo", "FeTxP0", "FeRxP0", "TsDelta"}
        stat_names = [name for name in feature_names if any(name.startswith(prefix) for prefix in keep)]
    elif mode == "statistics":
        keys = (
            "_mean",
            "_std",
            "_delta",
            "_exp_range",
            "Range",
            "Std",
            "Count",
            "Storm",
            "Rate",
            "Flag",
            "Elapsed",
            "DeltaSeconds",
        )
        stat_names = [name for name in feature_names if name not in raw_set and any(key in name for key in keys)]
    elif mode == "all":
        stat_names = list(feature_names)
    else:
        raise ValueError("Unknown stat_feature_mode; use all_engineered, model2_expert, statistics, or all")
    if not stat_names:
        stat_names = [name for name in feature_names if name not in raw_set]
    stat_indices = [feature_names.index(name) for name in stat_names]
    return raw_indices, raw_names, stat_indices, stat_names


class SelectedWindowStatDataset(IterableDataset):
    def __init__(
        self,
        base: CompatBatchedDataset,
        cache: Model2FeatureCache,
        cfg: CompatCfg,
        raw_indices: list[int],
        stat_indices: list[int],
    ) -> None:
        self.base = base
        self.cache = cache
        self.cfg = cfg
        self.raw_indices = np.asarray(raw_indices, dtype=np.int64)
        self.stat_indices = np.asarray(stat_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def file_names(self) -> list[str]:
        return self.base.file_names

    @property
    def selected_positions(self) -> list[np.ndarray]:
        return self.base.selected_positions

    def __iter__(self):
        seq_len = int(self.cfg.seq_len)
        pairs = list(zip(self.base.file_names, self.base.selected_positions))
        worker = get_worker_info()
        if worker is not None:
            pairs = pairs[worker.id :: worker.num_workers]
        weight_by_name = {name: weights for name, weights in zip(self.base.file_names, self.base.selected_weights)}
        for name, selected in pairs:
            if len(selected) <= 0:
                continue
            _ts, features, _valid_mask, labels, _anomaly_labels, _rule_pred, _extra = self.cache.get(name)
            raw = features[:, self.raw_indices].astype(np.float32)
            stat = features[:, self.stat_indices].astype(np.float32)
            pad_x = np.zeros((seq_len - 1, raw.shape[1]), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, raw], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, np.ones_like(raw, dtype=np.float32)], axis=0))
            x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
            ys = torch.from_numpy(labels[selected].astype(np.float32))
            ws = torch.from_numpy(weight_by_name.get(name, np.ones(len(selected), dtype=np.float32)).astype(np.float32))
            stat_tensor = torch.from_numpy(stat[selected].astype(np.float32))
            for start in range(0, len(selected), int(self.cfg.batch_size)):
                end = start + int(self.cfg.batch_size)
                idx = torch.from_numpy(np.asarray(selected[start:end], dtype=np.int64))
                yield (
                    x_windows.index_select(0, idx).contiguous(),
                    m_windows.index_select(0, idx).contiguous(),
                    stat_tensor[start:end].contiguous(),
                    ys[start:end].contiguous(),
                    ws[start:end].contiguous(),
                )


class PatchTSTStatAligner(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_raw_features: int,
        n_stat_features: int,
        embedding_dim: int = 256,
        latent_dim: int = 128,
        stat_hidden: int = 256,
        attn_heads: int = 4,
        dropout: float = 0.2,
        fusion_mode: str = "gated_attn",
    ) -> None:
        super().__init__()
        self.fusion_mode = str(fusion_mode)
        self.patchtst, self.patchtst_cfg, _ = build_patchtst(int(seq_len), int(n_raw_features))
        self.patch_hook = LastLinearInputHook(self.patchtst)
        self.seq_embedding = nn.LazyLinear(int(embedding_dim))
        self.seq_projection = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(embedding_dim), int(latent_dim)),
        )
        self.stat_mlp = nn.Sequential(
            nn.Linear(int(n_stat_features), int(stat_hidden)),
            nn.LayerNorm(int(stat_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(stat_hidden), int(latent_dim)),
        )
        self.feature_alignment = nn.LayerNorm(int(latent_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(latent_dim),
            num_heads=int(attn_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.modality_gate = nn.Sequential(
            nn.LayerNorm(int(latent_dim) * 2),
            nn.Linear(int(latent_dim) * 2, int(latent_dim)),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(int(latent_dim) * 2),
            nn.Linear(int(latent_dim) * 2, int(latent_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(latent_dim), 1)

    def close(self) -> None:
        self.patch_hook.close()

    def encode(self, raw_x: torch.Tensor, raw_mask: torch.Tensor, stat_x: torch.Tensor) -> torch.Tensor:
        self.patch_hook.clear()
        patch_out = self.patchtst(raw_x, raw_mask)
        emb = self.patch_hook.value
        if emb is None:
            emb = compat_alarm_logit(patch_out).reshape(raw_x.shape[0], -1)
        if emb.ndim > 2:
            emb = emb.flatten(start_dim=1)
        if emb.ndim == 1:
            emb = emb.unsqueeze(-1)
        seq_emb = self.seq_embedding(emb)
        seq_latent = self.seq_projection(seq_emb)
        stat_latent = self.stat_mlp(stat_x)
        tokens = torch.stack([seq_latent, stat_latent], dim=1)
        aligned = self.feature_alignment(tokens)
        attended, _weights = self.cross_attention(aligned, aligned, aligned, need_weights=False)
        if self.fusion_mode == "attn_mean":
            pooled = torch.cat([tokens.mean(dim=1), attended.mean(dim=1)], dim=1)
        elif self.fusion_mode == "gated_attn":
            seq_context = 0.5 * (seq_latent + attended[:, 0, :])
            stat_context = 0.5 * (stat_latent + attended[:, 1, :])
            gate = self.modality_gate(torch.cat([seq_latent, stat_latent], dim=1))
            gated = gate * seq_context + (1.0 - gate) * stat_context
            pooled = torch.cat([gated, attended.mean(dim=1)], dim=1)
        else:
            raise ValueError(f"Unknown fusion_mode={self.fusion_mode!r}")
        return self.fusion(pooled)

    def forward(self, raw_x: torch.Tensor, raw_mask: torch.Tensor, stat_x: torch.Tensor) -> torch.Tensor:
        fused = self.encode(raw_x, raw_mask, stat_x)
        return self.classifier(fused).squeeze(-1)


def initialize_lazy_layers(model: PatchTSTStatAligner, dataset: SelectedWindowStatDataset, cfg: CompatCfg) -> None:
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    try:
        raw_x, raw_m, stat_x, _ys, _ws = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Cannot initialize fusion model because the dataset is empty") from exc
    model.eval()
    with torch.no_grad():
        model(
            raw_x.to(cfg.device),
            raw_m.to(cfg.device),
            stat_x.to(cfg.device),
        )


def train_fusion_encoder(
    train_files: list[str],
    cache: Model2FeatureCache,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    args: argparse.Namespace,
    fold: int,
) -> tuple[PatchTSTStatAligner, CompatBatchedDataset, dict[str, Any]]:
    started = log_stage_start(
        "build_fusion_dataset",
        RUN_NAME,
        fold,
        files=len(train_files),
        sample_selection=cfg.sample_selection,
        sampling_mode=cfg.sampling_mode,
    )
    base_dataset = CompatBatchedDataset(train_files, cache, cfg)
    fusion_dataset = SelectedWindowStatDataset(base_dataset, cache, cfg, raw_indices, stat_indices)
    log_stage_done(
        "build_fusion_dataset",
        started,
        RUN_NAME,
        fold,
        rows=int(base_dataset.total_rows),
        pos=int(base_dataset.pos_rows),
        neg=int(base_dataset.neg_rows),
    )
    model = PatchTSTStatAligner(
        seq_len=cfg.seq_len,
        n_raw_features=len(raw_indices),
        n_stat_features=len(stat_indices),
        embedding_dim=args.seq_embedding_dim,
        latent_dim=args.latent_dim,
        stat_hidden=args.stat_hidden,
        attn_heads=args.attn_heads,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
    ).to(cfg.device)
    initialize_lazy_layers(model, fusion_dataset, cfg)
    loader = DataLoader(fusion_dataset, batch_size=None, shuffle=False, num_workers=int(cfg.num_workers))
    pos_weight = effective_pos_weight(base_dataset, cfg)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=cfg.device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history: list[dict[str, float]] = []
    adaptive_meta: dict[str, float] = {}
    adaptive_applied = False
    train_started = time.time()
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        epoch_started = time.time()
        for batch_idx, (raw_x, raw_m, stat_x, ys, ws) in enumerate(loader, 1):
            raw_x = raw_x.to(cfg.device)
            raw_m = raw_m.to(cfg.device)
            stat_x = stat_x.to(cfg.device)
            ys = ys.to(cfg.device).float()
            ws = ws.to(cfg.device).float()
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(bool(cfg.amp) and str(cfg.device).startswith("cuda")):
                logits = model(raw_x, raw_m, stat_x)
                loss_raw = loss_fn(logits, ys)
                loss = (loss_raw * ws).sum() / torch.clamp(ws.sum(), min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()
            batch_n = int(raw_x.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_n
            n_seen += batch_n
            if int(cfg.log_batches) > 0 and batch_idx % int(cfg.log_batches) == 0:
                elapsed = time.time() - epoch_started
                print(
                    f"[batch] ep={epoch:02d} batch={batch_idx:>5}/{len(loader):<5} "
                    f"{progress_bar(batch_idx, len(loader), width=16)} "
                    f"loss={total_loss / max(n_seen, 1):.5f} rows={n_seen} "
                    f"rate={format_rate(n_seen, elapsed)} elapsed={format_duration(elapsed)}",
                    flush=True,
                )
        epoch_seconds = time.time() - epoch_started
        elapsed_total = time.time() - train_started
        eta = (elapsed_total / max(epoch, 1)) * max(int(cfg.epochs) - epoch, 0)
        avg_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": float(epoch), "loss": float(avg_loss), "rows_seen": float(n_seen), "elapsed_seconds": epoch_seconds})
        print(
            format_epoch_status(
                "align-train",
                RUN_NAME,
                fold,
                epoch,
                int(cfg.epochs),
                float(avg_loss),
                int(n_seen),
                float(epoch_seconds),
                elapsed_total,
                eta,
            ),
            flush=True,
        )
        if (
            float(cfg.adaptive_negative_weight) > 0.0
            and not adaptive_applied
            and epoch >= int(cfg.adaptive_warmup_epochs)
            and epoch < int(cfg.epochs)
        ):
            score_started = log_stage_start("adaptive_negative_weight", RUN_NAME, fold, after_epoch=epoch)
            scores = score_selected_positions(model, cache, base_dataset, cfg, raw_indices, stat_indices)
            adaptive_meta = base_dataset.apply_adaptive_negative_weights(scores, float(cfg.adaptive_negative_weight))
            adaptive_meta["applied_after_epoch"] = float(epoch)
            adaptive_applied = True
            print(
                f"[adaptive-neg] rows={int(adaptive_meta.get('updated_negative_rows', 0))} "
                f"score_min={adaptive_meta.get('min_score', 0.0):.5f} "
                f"score_max={adaptive_meta.get('max_score', 0.0):.5f} "
                f"max_extra={cfg.adaptive_negative_weight}",
                flush=True,
            )
            log_stage_done("adaptive_negative_weight", score_started, RUN_NAME, fold)
    meta = {
        "train_rows": int(base_dataset.total_rows),
        "train_pos_rows": int(base_dataset.pos_rows),
        "train_neg_rows": int(base_dataset.neg_rows),
        "pos_weight": float(pos_weight),
        "history": history,
        "adaptive_negative_weight_meta": adaptive_meta,
    }
    return model, base_dataset, meta


@torch.no_grad()
def encode_file_positions(
    model: PatchTSTStatAligner,
    cache: Model2FeatureCache,
    file_name: str,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    positions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    timestamps, features, _valid_mask, labels, _anomaly_labels, rule_pred, extra = cache.get(file_name)
    n_rows = int(len(features))
    if positions is None:
        positions = np.arange(n_rows, dtype=np.int64)
    else:
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[(positions >= 0) & (positions < n_rows)]
    if n_rows == 0 or len(positions) == 0:
        return {
            "timestamps": np.zeros(0, dtype=np.int64),
            "fused": np.zeros((0, 0), dtype=np.float32),
            "logits": np.zeros(0, dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int8),
            "weights": np.ones(0, dtype=np.float32),
            "rule_pred": np.zeros(0, dtype=np.int8),
            "extra": extra,
        }
    raw = features[:, np.asarray(raw_indices, dtype=np.int64)].astype(np.float32)
    stat = features[:, np.asarray(stat_indices, dtype=np.int64)].astype(np.float32)
    seq_len = int(cfg.seq_len)
    pad_x = np.zeros((seq_len - 1, raw.shape[1]), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = torch.from_numpy(np.concatenate([pad_x, raw], axis=0))
    m_pad = torch.from_numpy(np.concatenate([pad_m, np.ones_like(raw, dtype=np.float32)], axis=0))
    x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
    m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
    fused_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    for start in range(0, len(positions), int(cfg.batch_size)):
        batch_pos = positions[start : start + int(cfg.batch_size)]
        idx = torch.from_numpy(batch_pos.astype(np.int64))
        raw_x = x_windows.index_select(0, idx).contiguous().to(cfg.device)
        raw_m = m_windows.index_select(0, idx).contiguous().to(cfg.device)
        stat_x = torch.from_numpy(stat[batch_pos].astype(np.float32)).to(cfg.device)
        with cuda_autocast(bool(cfg.amp) and str(cfg.device).startswith("cuda")):
            fused = model.encode(raw_x, raw_m, stat_x)
            logits = model.classifier(fused).squeeze(-1)
        fused_parts.append(fused.float().detach().cpu().numpy().astype(np.float32))
        logit_parts.append(logits.float().detach().cpu().numpy().astype(np.float32))
    return {
        "timestamps": timestamps[positions].astype(np.int64),
        "fused": np.concatenate(fused_parts, axis=0).astype(np.float32),
        "logits": np.concatenate(logit_parts, axis=0).astype(np.float32),
        "labels": labels[positions].astype(np.int8),
        "weights": np.ones(len(positions), dtype=np.float32),
        "rule_pred": rule_pred[positions].astype(np.int8),
        "extra": extra,
    }


def score_selected_positions(
    model: PatchTSTStatAligner,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for file_name, positions in zip(dataset.file_names, dataset.selected_positions):
        block = encode_file_positions(model, cache, file_name, cfg, raw_indices, stat_indices, positions)
        out[file_name] = (1.0 / (1.0 + np.exp(-block["logits"]))).astype(np.float32)
    return out


def collect_fused_training_table(
    model: PatchTSTStatAligner,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    started = log_stage_start("collect_fused_train_table", RUN_NAME, fold, files=len(dataset.file_names))
    weight_by_name = {name: weights for name, weights in zip(dataset.file_names, dataset.selected_weights)}
    for idx, (file_name, positions) in enumerate(zip(dataset.file_names, dataset.selected_positions), 1):
        if len(positions) == 0:
            continue
        block = encode_file_positions(model, cache, file_name, cfg, raw_indices, stat_indices, positions)
        x_parts.append(block["fused"])
        y_parts.append(block["labels"])
        w_parts.append(weight_by_name.get(file_name, np.ones(len(positions), dtype=np.float32)).astype(np.float32))
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold} stage=collect_fused_train_table "
                f"{progress_bar(idx, len(dataset.file_names), width=18)} files={idx}/{len(dataset.file_names)} "
                f"rows={sum(len(part) for part in y_parts)} rate={format_rate(idx, elapsed)} "
                f"elapsed={format_duration(elapsed)}",
                flush=True,
            )
    if not x_parts:
        raise ValueError("No fused rows collected for XGBoost")
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int8)
    w = np.concatenate(w_parts, axis=0).astype(np.float32)
    log_stage_done(
        "collect_fused_train_table",
        started,
        RUN_NAME,
        fold,
        rows=len(y),
        features=x.shape[1],
        positives=int(np.sum(y > 0)),
        negatives=int(np.sum(y <= 0)),
    )
    return x, y, w, [f"aligned_latent_{idx:03d}" for idx in range(x.shape[1])]


def build_aligned_xgb_model(args: argparse.Namespace, y_train: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    classes = np.unique(y_train)
    if len(classes) < 2:
        from sklearn.dummy import DummyClassifier

        constant = int(classes[0]) if len(classes) else 0
        return DummyClassifier(strategy="constant", constant=constant), {
            "xgb_balance_mode": "dummy",
            "xgb_scale_pos_weight": 1.0,
        }

    from xgboost import XGBClassifier

    pos = float(np.sum(y_train > 0))
    neg = float(np.sum(y_train <= 0))
    raw_scale = neg / max(pos, 1.0)
    balance_mode = str(args.xgb_balance_mode).lower()
    if balance_mode == "auto":
        scale_pos_weight = raw_scale
    elif balance_mode == "sqrt":
        scale_pos_weight = float(np.sqrt(raw_scale))
    elif balance_mode == "none":
        scale_pos_weight = 1.0
    else:
        raise ValueError("Unknown xgb_balance_mode; use auto, sqrt, or none")

    estimator = XGBClassifier(
        n_estimators=int(args.ml_n_estimators),
        max_depth=int(args.xgb_max_depth),
        learning_rate=float(args.xgb_lr),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample_bytree),
        eval_metric="logloss",
        tree_method=str(args.xgb_tree_method),
        random_state=int(seed),
        n_jobs=int(args.n_jobs),
        scale_pos_weight=float(scale_pos_weight),
    )
    meta: dict[str, Any] = {
        "xgb_balance_mode": balance_mode,
        "xgb_scale_pos_weight": float(scale_pos_weight),
        "xgb_raw_scale_pos_weight": float(raw_scale),
    }
    return estimator, meta


def score_fused_files_to_memory(
    estimator: Any,
    model: PatchTSTStatAligner,
    cache: Model2FeatureCache,
    file_names: list[str],
    cfg: CompatCfg,
    raw_indices: list[int],
    stat_indices: list[int],
    fold: int,
    stage_name: str,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    started = log_stage_start(stage_name, RUN_NAME, fold, files=len(file_names))
    for idx, file_name in enumerate(file_names, 1):
        block = encode_file_positions(model, cache, file_name, cfg, raw_indices, stat_indices)
        score = positive_scores(estimator, block["fused"])
        parts = [
            pd.DataFrame(
                {
                    "timestamp": block["timestamps"].astype(np.int64),
                    "score": score.astype(np.float32),
                    "rule_predict": block["rule_pred"].astype(int),
                    "source": "aligned_latent_xgb",
                }
            )
        ]
        extra = block["extra"]
        if len(extra):
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": extra[:, 0].astype(np.int64),
                        "score": np.nan,
                        "rule_predict": extra[:, 1].astype(int),
                        "source": "model2_extra_rule",
                    }
                )
            )
        frames[file_name] = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        if idx % 200 == 0:
            elapsed = time.time() - started
            print(
                f"[stage-progress] model={RUN_NAME} fold={fold} stage={stage_name} "
                f"{progress_bar(idx, len(file_names), width=18)} files={idx}/{len(file_names)} "
                f"rate={format_rate(idx, elapsed)} elapsed={format_duration(elapsed)}",
                flush=True,
            )
    log_stage_done(stage_name, started, RUN_NAME, fold)
    return frames


def run_fold(fold: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = CompatCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        fixed_threshold=args.fixed_threshold,
        target_mode=args.target_mode,
        feature_mode=args.feature_mode,
        sampling_mode=args.sampling_mode,
        positive_windows_per_module=args.positive_windows_per_module,
        negative_windows_per_faulty_module=args.negative_windows_per_faulty_module,
        normal_windows_per_module=args.normal_windows_per_module,
        val_fraction=args.val_fraction,
        threshold_search=bool(args.threshold_search),
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        rule_mode=args.rule_mode,
        sample_selection=args.sample_selection,
        sample_topk_fraction=args.sample_topk_fraction,
        temporal_positive_weight=args.temporal_positive_weight,
        temporal_weight_horizon_hours=args.temporal_weight_horizon_hours,
        adaptive_negative_weight=args.adaptive_negative_weight,
        adaptive_warmup_epochs=args.adaptive_warmup_epochs,
        max_cached_files=args.max_cached_files,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        amp=bool(args.amp),
        allow_tf32=not args.no_tf32,
        log_batches=args.log_batches,
        max_train_files=args.max_train_files,
        max_test_files=args.max_test_files,
        min_hit_lead_hours=args.min_hit_lead_hours,
    )
    configure_runtime(cfg, int(args.seed), int(fold))
    fold_started = time.time()
    print(f"[aligned-init] fold={fold} device={cfg.device} torch={torch.__version__}", flush=True)
    index_df = read_index(args.index_path)
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    label_by_file = file_label_map(index_df)
    if int(args.max_train_files) > 0:
        train_files = cap_files_stratified(train_files, label_by_file, int(args.max_train_files), int(args.seed) + int(fold))
    if int(args.max_test_files) > 0:
        test_files = cap_files_stratified(test_files, label_by_file, int(args.max_test_files), int(args.seed) + 7919 + int(fold))
    train_files, val_files = split_train_val_files(train_files, label_by_file, cfg, int(fold))
    run_dir = Path(args.out_root) / RUN_NAME / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_run_header(
        "PATCHTST STAT-ALIGNED CROSS-ATTENTION XGB",
        {
            "fold": fold,
            "train/val/test": f"{len(train_files)}/{len(val_files)}/{len(test_files)} modules",
            "target": cfg.target_mode,
            "feature mode": cfg.feature_mode,
            "stat features": args.stat_feature_mode,
            "fusion mode": args.fusion_mode,
            "rule mode": cfg.rule_mode,
            "sample selection": cfg.sample_selection,
            "xgb balance": args.xgb_balance_mode,
            "xgb sample weight": bool(args.xgb_use_sample_weight),
            "seq_len": cfg.seq_len,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "device": cfg.device,
            "out_dir": run_dir,
        },
    )
    stats_started = log_stage_start("feature_norm_stats", RUN_NAME, fold, files=len(train_files), data_dir=args.data_dir)
    stats = compute_feature_norm_stats(args.data_dir, train_files, cfg, label_by_file)
    log_stage_done("feature_norm_stats", stats_started, RUN_NAME, fold)
    mean, std = stats.arrays()
    cache = Model2FeatureCache(args.data_dir, mean, std, cfg, label_by_file)
    names = compat_feature_names(cfg)
    raw_indices, raw_names, stat_indices, stat_names = feature_indices(names, args.stat_feature_mode)
    (run_dir / "feature_groups.json").write_text(
        json.dumps(
            {
                "raw_sequence_features": raw_names,
                "statistic_features": stat_names,
                "all_compat_features": names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model, dataset, train_meta = train_fusion_encoder(train_files, cache, cfg, raw_indices, stat_indices, args, int(fold))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": asdict(cfg),
            "patchtst_cfg": model.patchtst_cfg,
            "raw_features": raw_names,
            "stat_features": stat_names,
            "fusion_mode": args.fusion_mode,
            "train_meta": train_meta,
        },
        run_dir / "aligned_encoder.pt",
    )
    x_train, y_train, sample_weight, latent_names = collect_fused_training_table(
        model, cache, dataset, cfg, raw_indices, stat_indices, int(fold)
    )
    estimator, xgb_meta = build_aligned_xgb_model(args, y_train, int(args.seed) + int(fold))
    ml_started = log_stage_start(
        "xgb_train_on_fused_latent",
        RUN_NAME,
        fold,
        rows=len(y_train),
        features=x_train.shape[1],
        balance=args.xgb_balance_mode,
        scale_pos_weight=f"{float(xgb_meta.get('xgb_scale_pos_weight', 1.0)):.4f}",
        sample_weight=bool(args.xgb_use_sample_weight),
    )
    if bool(args.xgb_use_sample_weight):
        estimator.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        estimator.fit(x_train, y_train)
    log_stage_done("xgb_train_on_fused_latent", ml_started, RUN_NAME, fold)
    with (run_dir / "xgb_fused_latent.pkl").open("wb") as fh:
        pickle.dump({"model": estimator, "feature_names": latent_names, "xgb_meta": xgb_meta}, fh)
    val_scores = score_fused_files_to_memory(estimator, model, cache, val_files, cfg, raw_indices, stat_indices, int(fold), "score_val_files")
    test_scores = score_fused_files_to_memory(estimator, model, cache, test_files, cfg, raw_indices, stat_indices, int(fold), "score_test_files")
    threshold, val_metrics = select_threshold_for_scores(val_scores, args.data_dir, run_dir, "aligned_latent_xgb", args)
    pred_dir = run_dir / "predictions" / "aligned_latent_xgb"
    eval_dir = run_dir / "evaluation" / "aligned_latent_xgb"
    test_rows = write_threshold_predictions(test_scores, pred_dir, threshold)
    metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
    print(f"[aligned-done] fold={fold} threshold={threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}", flush=True)
    result = {
        "deep_model": RUN_NAME,
        "fold": int(fold),
        "mode": "aligned_latent_xgb",
        "ml_model": "xgb",
        "ml_feature_set": "aligned_latent",
        "selector": "none",
        "selected_feature_count": int(x_train.shape[1]),
        "fusion_mode": args.fusion_mode,
        "xgb_balance_mode": args.xgb_balance_mode,
        "xgb_use_sample_weight": bool(args.xgb_use_sample_weight),
        "xgb_scale_pos_weight": float(xgb_meta.get("xgb_scale_pos_weight", 1.0)),
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_rows": int(test_rows),
        "metrics": metrics,
    }
    print(f"[fold-done] run={RUN_NAME} fold={fold} elapsed={format_duration(time.time() - fold_started)}", flush=True)
    model.close()
    del model, cache, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [result]


def aggregate_results(results: list[dict[str, Any]], out_root: Path, args: argparse.Namespace) -> None:
    rows = []
    for item in results:
        row = {
            "deep_model": item["deep_model"],
            "fold": int(item["fold"]),
            "mode": item["mode"],
            "ml_model": item.get("ml_model", ""),
            "ml_feature_set": item.get("ml_feature_set", ""),
            "selector": item.get("selector", ""),
            "selected_feature_count": int(item.get("selected_feature_count", 0)),
            "fusion_mode": item.get("fusion_mode", args.fusion_mode),
            "xgb_balance_mode": item.get("xgb_balance_mode", args.xgb_balance_mode),
            "xgb_use_sample_weight": bool(item.get("xgb_use_sample_weight", args.xgb_use_sample_weight)),
            "xgb_scale_pos_weight": float(item.get("xgb_scale_pos_weight", 1.0)),
            "threshold": float(item.get("threshold", 0.0)),
            "target_mode": args.target_mode,
            "feature_mode": args.feature_mode,
            "stat_feature_mode": args.stat_feature_mode,
            "sampling_mode": args.sampling_mode,
            "rule_mode": args.rule_mode,
            "min_hit_lead_hours": float(args.min_hit_lead_hours),
            "test_rows": int(item.get("test_rows", 0)),
        }
        row.update(item.get("metrics", {}))
        rows.append(row)
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        frame.drop_duplicates(
            subset=[
                "deep_model",
                "fold",
                "mode",
                "stat_feature_mode",
                "fusion_mode",
                "xgb_balance_mode",
                "xgb_use_sample_weight",
                "rule_mode",
            ],
            keep="last",
            inplace=True,
        )
    frame.sort_values(["deep_model", "mode", "fold"], inplace=True)
    frame.to_csv(path, index=False)
    group_cols = [
        "deep_model",
        "mode",
        "target_mode",
        "feature_mode",
        "stat_feature_mode",
        "fusion_mode",
        "sampling_mode",
        "rule_mode",
        "xgb_balance_mode",
        "xgb_use_sample_weight",
        "min_hit_lead_hours",
    ]
    non_metric_cols = set(group_cols) | {"fold", "ml_model", "ml_feature_set", "selector"}
    numeric = [col for col in frame.columns if col not in non_metric_cols and pd.api.types.is_numeric_dtype(frame[col])]
    summary = frame.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST + statistic MLP feature alignment + cross-attention + XGBoost.")
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_model2_compat_results/patchtst_stat_aligned_xgb"))
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.3)
    parser.add_argument("--target_mode", choices=["ahead120", "anomaly", "module_fault"], default="module_fault")
    parser.add_argument("--feature_mode", choices=["model2", "model2_plus"], default="model2_plus")
    parser.add_argument("--stat_feature_mode", choices=["all_engineered", "model2_expert", "statistics", "all"], default="all_engineered")
    parser.add_argument("--sampling_mode", choices=["row", "module_balanced"], default="module_balanced")
    parser.add_argument("--positive_windows_per_module", type=int, default=32)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--normal_windows_per_module", type=int, default=8)
    parser.add_argument("--rule_mode", choices=["none", "temp", "model2_simple"], default="model2_simple")
    parser.add_argument("--sample_selection", choices=["random", "signal_topk", "hybrid"], default="hybrid")
    parser.add_argument("--sample_topk_fraction", type=float, default=0.5)
    parser.add_argument("--temporal_positive_weight", type=float, default=2.0)
    parser.add_argument("--temporal_weight_horizon_hours", type=float, default=120.0)
    parser.add_argument("--adaptive_negative_weight", type=float, default=1.0)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_grid", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--no_threshold_search", dest="threshold_search", action="store_false")
    parser.set_defaults(threshold_search=True)
    parser.add_argument("--min_hit_lead_hours", type=float, default=0.0)
    parser.add_argument("--max_cached_files", type=int, default=128)
    parser.add_argument("--max_train_files", type=int, default=2500)
    parser.add_argument("--max_test_files", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_batches", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--seq_embedding_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--stat_hidden", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fusion_mode", choices=["gated_attn", "attn_mean"], default="gated_attn")
    parser.add_argument("--xgb_balance_mode", choices=["auto", "sqrt", "none"], default="auto")
    parser.add_argument("--xgb_use_sample_weight", dest="xgb_use_sample_weight", action="store_true")
    parser.add_argument("--no_xgb_sample_weight", dest="xgb_use_sample_weight", action="store_false")
    parser.set_defaults(xgb_use_sample_weight=True)
    parser.add_argument("--ml_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--rf_max_depth", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.gpu_id).strip():
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip()
    results: list[dict[str, Any]] = []
    for fold in args.folds:
        results.extend(run_fold(int(fold), args))
        aggregate_results(results, Path(args.out_root), args)
    aggregate_results(results, Path(args.out_root), args)


if __name__ == "__main__":
    main()
