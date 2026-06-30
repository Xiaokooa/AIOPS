from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_model2_compat.run_model2_compat_deep import (
    BUILDERS,
    CompatBatchedDataset,
    CompatCfg,
    Model2FeatureCache,
    apply_threshold,
    cap_files_stratified,
    compat_alarm_logit,
    compat_feature_names,
    compute_feature_norm_stats,
    cuda_autocast,
    effective_pos_weight,
    evaluate_prediction_dir_compat,
    evaluate_prediction_frames,
    file_label_map,
    format_epoch_status,
    format_duration,
    format_rate,
    log_run_header,
    parse_threshold_grid,
    progress_bar,
    score_dataset_selected_positions,
    score_files_to_memory,
    split_train_val_files,
    train_one_epoch,
)
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.common.trainer import format_metric_summary, resolve_runtime_device, runtime_device_summary


DEFAULT_DEEP_MODELS = ["patchtst", "itransformer", "fteformer", "moderntcn"]
DEFAULT_THRESHOLD_GRID = "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50"


@dataclass
class IndexSelector:
    indices: list[int]
    feature_names: list[str]

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.indices:
            return x[:, :0]
        return x[:, np.asarray(self.indices, dtype=np.int64)]

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names)


class LastLinearInputHook:
    """Capture the vector entering the final Linear layer as a generic embedding."""

    def __init__(self, model: nn.Module) -> None:
        linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if not linears:
            raise ValueError("Cannot extract embeddings because the model has no nn.Linear layers")
        self.value: torch.Tensor | None = None
        self.handle = linears[-1].register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        value = inputs[0] if inputs and torch.is_tensor(inputs[0]) else output
        if torch.is_tensor(value):
            self.value = value.detach()

    def clear(self) -> None:
        self.value = None

    def close(self) -> None:
        self.handle.remove()


def parse_csv_words(text: str) -> list[str]:
    out: list[str] = []
    for part in str(text).replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


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


def log_stage_start(stage: str, deep_model_name: str, fold: int | None, **fields: object) -> float:
    parts = [f"model={deep_model_name}"]
    if fold is not None:
        parts.append(f"fold={fold}")
    parts.append(f"stage={stage}")
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print("[stage-start] " + " ".join(parts), flush=True)
    return time.time()


def log_stage_done(stage: str, started: float, deep_model_name: str, fold: int | None, **fields: object) -> None:
    parts = [f"model={deep_model_name}"]
    if fold is not None:
        parts.append(f"fold={fold}")
    parts.append(f"stage={stage}")
    parts.append(f"elapsed={format_duration(time.time() - started)}")
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print("[stage-done] " + " ".join(parts), flush=True)


def train_deep_model(
    model_name: str,
    train_files: list[str],
    cache: Model2FeatureCache,
    cfg: CompatCfg,
    fold: int | None = None,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], CompatBatchedDataset]:
    dataset_started = log_stage_start(
        "build_deep_dataset",
        model_name,
        fold,
        files=len(train_files),
        sample_selection=cfg.sample_selection,
        sampling_mode=cfg.sampling_mode,
    )
    dataset = CompatBatchedDataset(train_files, cache, cfg)
    log_stage_done(
        "build_deep_dataset",
        dataset_started,
        model_name,
        fold,
        rows=int(dataset.total_rows),
        source_rows=int(dataset.source_total_rows),
        pos=int(dataset.pos_rows),
        neg=int(dataset.neg_rows),
    )
    loader_started = log_stage_start(
        "deep_train",
        model_name,
        fold,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        rows=int(dataset.total_rows),
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=int(cfg.num_workers))
    pos_weight = effective_pos_weight(dataset, cfg)
    print(
        f"[deep-train-ready] model={model_name} fold={fold if fold is not None else '-'} "
        f"batches={len(loader)} pos_weight={pos_weight:.4f} device={cfg.device}",
        flush=True,
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=cfg.device),
        reduction="none",
    )
    model, model_cfg, _use_special_losses = BUILDERS[model_name](int(cfg.seq_len), len(compat_feature_names(cfg)))
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    history: list[dict[str, float]] = []
    adaptive_weight_meta: dict[str, float] = {}
    adaptive_applied = False
    total_epochs = int(cfg.epochs)
    train_started = time.time()
    for epoch in range(1, total_epochs + 1):
        parts = train_one_epoch(model, loader, optimizer, loss_fn, cfg, epoch)
        history.append({"epoch": float(epoch), **parts})
        elapsed_total = time.time() - train_started
        eta = (elapsed_total / max(epoch, 1)) * max(total_epochs - epoch, 0)
        print(
            format_epoch_status(
                "deep-train",
                model_name,
                fold,
                epoch,
                total_epochs,
                float(parts["loss"]),
                int(parts["rows_seen"]),
                float(parts["elapsed_seconds"]),
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
            adaptive_started = log_stage_start("adaptive_negative_weight", model_name, fold, after_epoch=epoch)
            scores_by_file = score_dataset_selected_positions(model, dataset, cache, cfg)
            adaptive_weight_meta = dataset.apply_adaptive_negative_weights(scores_by_file, float(cfg.adaptive_negative_weight))
            adaptive_weight_meta["applied_after_epoch"] = float(epoch)
            adaptive_applied = True
            print(
                f"[adaptive-neg] rows={int(adaptive_weight_meta.get('updated_negative_rows', 0))} "
                f"score_min={adaptive_weight_meta.get('min_score', 0.0):.5f} "
                f"score_max={adaptive_weight_meta.get('max_score', 0.0):.5f} "
                f"max_extra={cfg.adaptive_negative_weight}",
                flush=True,
            )
            log_stage_done("adaptive_negative_weight", adaptive_started, model_name, fold)
    train_meta = {
        "train_rows": int(dataset.total_rows),
        "train_source_rows": int(dataset.source_total_rows),
        "train_pos_rows": int(dataset.pos_rows),
        "train_neg_rows": int(dataset.neg_rows),
        "pos_weight": float(pos_weight),
        "adaptive_negative_weight_meta": adaptive_weight_meta,
        "history": history,
    }
    del optimizer, loader
    log_stage_done("deep_train", loader_started, model_name, fold)
    return model, model_cfg, train_meta, dataset


@torch.no_grad()
def extract_file_block(
    model: nn.Module,
    cache: Model2FeatureCache,
    file_name: str,
    cfg: CompatCfg,
    hook: LastLinearInputHook,
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
            "model2": np.zeros((0, len(compat_feature_names(cfg))), dtype=np.float32),
            "embedding": np.zeros((0, 0), dtype=np.float32),
            "deep_score": np.zeros(0, dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int8),
            "rule_pred": np.zeros(0, dtype=np.int8),
            "extra": extra,
        }

    seq_len = int(cfg.seq_len)
    n_features = len(compat_feature_names(cfg))
    pad_x = np.zeros((seq_len - 1, n_features), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = torch.from_numpy(np.concatenate([pad_x, features.astype(np.float32)], axis=0))
    m_pad = torch.from_numpy(np.concatenate([pad_m, np.ones_like(features, dtype=np.float32)], axis=0))
    x_windows = x_pad.unfold(0, seq_len, 1).permute(0, 2, 1)
    m_windows = m_pad.unfold(0, seq_len, 1).permute(0, 2, 1)

    score_parts: list[np.ndarray] = []
    embedding_parts: list[np.ndarray] = []
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
    batch_size = max(1, int(cfg.batch_size))
    for start in range(0, len(positions), batch_size):
        batch_pos = positions[start : start + batch_size]
        idx = torch.from_numpy(batch_pos)
        xb = x_windows.index_select(0, idx).contiguous().to(cfg.device)
        mb = m_windows.index_select(0, idx).contiguous().to(cfg.device)
        hook.clear()
        with cuda_autocast(amp_enabled):
            outputs = model(xb, mb)
            alarm_logits = compat_alarm_logit(outputs)
        scores = torch.sigmoid(alarm_logits).float().detach().cpu().numpy().reshape(-1)
        emb = hook.value
        if emb is None:
            emb = alarm_logits.detach().reshape(alarm_logits.shape[0], -1)
        if emb.ndim > 2:
            emb = emb.flatten(start_dim=1)
        if emb.ndim == 1:
            emb = emb.unsqueeze(-1)
        if int(emb.shape[0]) != int(len(batch_pos)):
            emb = alarm_logits.detach().reshape(alarm_logits.shape[0], -1)
        score_parts.append(scores.astype(np.float32))
        embedding_parts.append(emb.float().detach().cpu().numpy().astype(np.float32))

    return {
        "timestamps": timestamps[positions].astype(np.int64),
        "model2": features[positions].astype(np.float32),
        "embedding": np.concatenate(embedding_parts, axis=0).astype(np.float32),
        "deep_score": np.concatenate(score_parts, axis=0).astype(np.float32),
        "labels": labels[positions].astype(np.int8),
        "rule_pred": rule_pred[positions].astype(np.int8),
        "extra": extra,
    }


def make_feature_matrix(
    block: dict[str, np.ndarray],
    base_feature_names: list[str],
    deep_model_name: str,
    feature_set: str,
    deep_feature_parts: set[str],
) -> tuple[np.ndarray, list[str]]:
    pieces: list[np.ndarray] = []
    names: list[str] = []
    mode = str(feature_set).lower()
    if mode in {"model2", "fusion"}:
        pieces.append(block["model2"])
        names.extend(base_feature_names)
    if mode in {"embedding", "fusion"} and "embedding" in deep_feature_parts:
        emb = block["embedding"]
        pieces.append(emb)
        names.extend([f"{deep_model_name}_emb_{idx:03d}" for idx in range(emb.shape[1])])
    if mode in {"embedding", "fusion"} and "score" in deep_feature_parts:
        pieces.append(block["deep_score"].reshape(-1, 1).astype(np.float32))
        names.append(f"{deep_model_name}_deep_score")
    if not pieces:
        return np.zeros((len(block["labels"]), 0), dtype=np.float32), []
    x = np.concatenate(pieces, axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names


def collect_training_table(
    model: nn.Module,
    cache: Model2FeatureCache,
    dataset: CompatBatchedDataset,
    cfg: CompatCfg,
    deep_model_name: str,
    feature_set: str,
    deep_feature_parts: set[str],
    fold: int | None = None,
    stage_name: str = "collect_training_table",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    base_names = compat_feature_names(cfg)
    hook = LastLinearInputHook(model)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    started = log_stage_start(
        stage_name,
        deep_model_name,
        fold,
        files=len(dataset.file_names),
        feature_set=feature_set,
        deep_parts=",".join(sorted(deep_feature_parts)),
    )
    try:
        for idx, (file_name, positions) in enumerate(zip(dataset.file_names, dataset.selected_positions), 1):
            if len(positions) == 0:
                continue
            block = extract_file_block(model, cache, file_name, cfg, hook, positions=positions)
            x_file, names = make_feature_matrix(block, base_names, deep_model_name, feature_set, deep_feature_parts)
            if not feature_names:
                feature_names = names
            x_parts.append(x_file)
            y_parts.append(block["labels"].astype(np.int8))
            if idx % 200 == 0:
                elapsed = time.time() - started
                print(
                    f"[stage-progress] model={deep_model_name} fold={fold if fold is not None else '-'} "
                    f"stage={stage_name} {progress_bar(idx, len(dataset.file_names), width=18)} "
                    f"files={idx}/{len(dataset.file_names)} rows={sum(len(part) for part in y_parts)} "
                    f"rate={format_rate(idx, elapsed)} elapsed={format_duration(elapsed)}",
                    flush=True,
                )
    finally:
        hook.close()
    if not x_parts:
        raise ValueError("No rows collected for tabular ML training")
    x_out = np.concatenate(x_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)
    log_stage_done(
        stage_name,
        started,
        deep_model_name,
        fold,
        rows=len(y_out),
        features=x_out.shape[1],
        positives=int(np.sum(y_out > 0)),
        negatives=int(np.sum(y_out <= 0)),
    )
    return x_out, y_out, feature_names


def fit_feature_selector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    selector_name: str,
    select_k: int,
    seed: int,
    n_jobs: int,
    deep_model_name: str = "",
    fold: int | None = None,
) -> IndexSelector:
    n_features = int(x_train.shape[1])
    if n_features == 0:
        return IndexSelector([], [])
    k = int(select_k)
    if k <= 0 or k >= n_features or str(selector_name).lower() == "none":
        return IndexSelector(list(range(n_features)), list(feature_names))

    name = str(selector_name).lower()
    started = None
    if deep_model_name:
        started = log_stage_start(
            "feature_select",
            deep_model_name,
            fold,
            selector=name,
            input_features=n_features,
            select_k=k,
            rows=len(y_train),
        )
    if name == "variance":
        scores = np.nanvar(x_train, axis=0)
    elif name == "f_classif":
        from sklearn.feature_selection import f_classif

        scores, _p = f_classif(x_train, y_train)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    elif name == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif

        scores = mutual_info_classif(x_train, y_train, random_state=int(seed))
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    elif name in {"extra_trees", "model_importance"}:
        from sklearn.ensemble import ExtraTreesClassifier

        estimator = ExtraTreesClassifier(
            n_estimators=160,
            random_state=int(seed),
            class_weight="balanced",
            n_jobs=int(n_jobs),
            max_features="sqrt",
        )
        estimator.fit(x_train, y_train)
        scores = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        raise ValueError(f"Unknown selector={selector_name!r}")

    order = np.argsort(scores)[::-1]
    indices = sorted(int(i) for i in order[: min(k, n_features)])
    selected_names = [feature_names[i] for i in indices]
    if started is not None:
        log_stage_done("feature_select", started, deep_model_name, fold, selected=len(indices))
    return IndexSelector(indices, selected_names)


def build_ml_model(name: str, args: argparse.Namespace, y_train: np.ndarray, seed: int):
    model_name = str(name).lower()
    classes = np.unique(y_train)
    if len(classes) < 2:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="constant", constant=int(classes[0]) if len(classes) else 0)

    pos = float(np.sum(y_train > 0))
    neg = float(np.sum(y_train <= 0))
    scale_pos_weight = neg / max(pos, 1.0)
    if model_name in {"rf", "random_forest", "tree"}:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(args.ml_n_estimators),
            max_depth=None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
            min_samples_leaf=int(args.rf_min_samples_leaf),
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=int(args.n_jobs),
            verbose=0,
        )
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
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
    if model_name in {"xgbrf", "xgb_rf"}:
        from xgboost import XGBRFClassifier

        return XGBRFClassifier(
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
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(args.ml_n_estimators),
            learning_rate=float(args.xgb_lr),
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=int(args.n_jobs),
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=int(args.ml_n_estimators),
            learning_rate=float(args.xgb_lr),
            depth=int(args.xgb_max_depth),
            random_seed=int(seed),
            verbose=False,
            auto_class_weights="Balanced",
        )
    raise ValueError(f"Unknown ML model={name!r}")


def positive_scores(estimator: Any, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float32)
        if proba.ndim == 2 and proba.shape[1] == 1:
            classes = np.asarray(getattr(estimator, "classes_", []))
            if len(classes) == 1 and int(classes[0]) == 1:
                return proba[:, 0].astype(np.float32)
            return np.zeros(proba.shape[0], dtype=np.float32)
        return proba.reshape(-1).astype(np.float32)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(x), dtype=float).reshape(-1)
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    return np.asarray(estimator.predict(x), dtype=np.float32).reshape(-1)


def train_ml_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_names: list[str],
    args: argparse.Namespace,
    seed: int,
    deep_model_name: str = "",
    fold: int | None = None,
    stage_name: str = "ml_train",
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    stage_started = None
    if deep_model_name:
        stage_started = log_stage_start(
            stage_name,
            deep_model_name,
            fold,
            models=",".join(model_names),
            rows=len(y_train),
            features=x_train.shape[1],
        )
    for name in model_names:
        tag = str(name).lower()
        try:
            estimator = build_ml_model(tag, args, y_train, seed)
        except ImportError as exc:
            if bool(getattr(args, "require_all_ml", False)):
                raise
            print(f"[ml-skip] model={tag} reason=missing optional dependency: {exc}")
            continue
        started = time.time()
        print(
            f"[ml-train] deep_model={deep_model_name or '-'} fold={fold if fold is not None else '-'} "
            f"model={tag} rows={len(y_train)} features={x_train.shape[1]}",
            flush=True,
        )
        estimator.fit(x_train, y_train)
        print(
            f"[ml-train] deep_model={deep_model_name or '-'} fold={fold if fold is not None else '-'} "
            f"model={tag} done elapsed={format_duration(time.time() - started)}",
            flush=True,
        )
        models[tag] = estimator
    if not models:
        raise ValueError("No tabular ML models were trained; install optional dependencies or change --ml_models")
    if stage_started is not None:
        log_stage_done(stage_name, stage_started, deep_model_name, fold, trained=",".join(sorted(models)))
    return models


def score_ml_models_to_memory(
    estimators: dict[str, Any],
    selector: IndexSelector,
    model: nn.Module,
    cache: Model2FeatureCache,
    file_names: list[str],
    cfg: CompatCfg,
    deep_model_name: str,
    feature_set: str,
    deep_feature_parts: set[str],
    fold: int | None = None,
    stage_name: str = "ml_score",
) -> dict[str, dict[str, pd.DataFrame]]:
    base_names = compat_feature_names(cfg)
    frames_by_model: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in estimators}
    hook = LastLinearInputHook(model)
    started = log_stage_start(
        stage_name,
        deep_model_name,
        fold,
        files=len(file_names),
        models=",".join(sorted(estimators)),
        feature_set=feature_set,
    )
    try:
        for idx, file_name in enumerate(file_names, 1):
            block = extract_file_block(model, cache, file_name, cfg, hook)
            x_raw, _names = make_feature_matrix(block, base_names, deep_model_name, feature_set, deep_feature_parts)
            x = selector.transform(x_raw)
            extra = block["extra"]
            for model_tag, estimator in estimators.items():
                score = positive_scores(estimator, x)
                frames: list[pd.DataFrame] = [
                    pd.DataFrame(
                        {
                            "timestamp": block["timestamps"].astype(np.int64),
                            "score": score.astype(np.float32),
                            "rule_predict": block["rule_pred"].astype(int),
                            "source": f"{model_tag}_{feature_set}",
                        }
                    )
                ]
                if len(extra):
                    frames.append(
                        pd.DataFrame(
                            {
                                "timestamp": extra[:, 0].astype(np.int64),
                                "score": np.nan,
                                "rule_predict": extra[:, 1].astype(int),
                                "source": "model2_extra_rule",
                            }
                        )
                    )
                frames_by_model[model_tag][file_name] = pd.concat(frames, ignore_index=True).sort_values("timestamp")
            if idx % 200 == 0:
                elapsed = time.time() - started
                print(
                    f"[stage-progress] model={deep_model_name} fold={fold if fold is not None else '-'} "
                    f"stage={stage_name} {progress_bar(idx, len(file_names), width=18)} "
                    f"files={idx}/{len(file_names)} rate={format_rate(idx, elapsed)} "
                    f"elapsed={format_duration(elapsed)}",
                    flush=True,
                )
    finally:
        hook.close()
    log_stage_done(stage_name, started, deep_model_name, fold)
    return frames_by_model


def select_threshold_for_scores(
    score_frames: dict[str, pd.DataFrame],
    label_dir: Path,
    run_dir: Path,
    tag: str,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    candidates = parse_threshold_grid(args.threshold_grid)
    if not bool(args.threshold_search) or not score_frames or not candidates:
        return float(args.fixed_threshold), {}

    search_rows: list[dict[str, float]] = []
    best_threshold = float(args.fixed_threshold)
    best_metrics: dict[str, float] = {}
    best_detail: pd.DataFrame | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None
    metric_key = str(args.threshold_metric)
    for threshold in candidates:
        pred_frames = {name: apply_threshold(frame, float(threshold)) for name, frame in score_frames.items()}
        summary_df, detail_df = evaluate_prediction_frames(
            pred_frames,
            label_dir,
            min_hit_lead_hours=float(args.min_hit_lead_hours),
        )
        metrics = {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}
        value = float(metrics.get(metric_key, metrics.get("f1_score", 0.0)))
        row = {"threshold": float(threshold)}
        row.update(metrics)
        search_rows.append(row)
        key = (
            value,
            float(metrics.get("final_score", 0.0)),
            float(metrics.get("precision", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            float(metrics.get("recall", 0.0)),
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
            best_detail = detail_df

    val_dir = Path(run_dir) / "validation" / tag
    val_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search_rows).to_csv(val_dir / "threshold_search.csv", index=False)
    pd.DataFrame({"Item": list(best_metrics.keys()), "Value": list(best_metrics.values())}).to_csv(
        val_dir / "best_evaluate_result.csv",
        index=False,
    )
    if best_detail is not None:
        best_detail.to_csv(val_dir / "best_module_decisions.csv", index=False)
    print(
        f"[val-select] tag={tag} threshold={best_threshold:.4f} "
        f"{metric_key}={best_metrics.get(metric_key, 0.0):.5f} {format_metric_summary(best_metrics)}"
    )
    return best_threshold, best_metrics


def write_threshold_predictions(score_frames: dict[str, pd.DataFrame], out_dir: Path, threshold: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for name, frame in sorted(score_frames.items()):
        pred = apply_threshold(frame, float(threshold))
        pred.to_csv(out_dir / name, index=False)
        total_rows += int(len(pred))
    return total_rows


def evaluate_prediction_output(out_dir: Path, label_dir: Path, eval_dir: Path, min_hit_lead_hours: float) -> tuple[dict[str, float], pd.DataFrame]:
    summary_df, detail_df = evaluate_prediction_dir_compat(
        out_dir,
        label_dir,
        min_hit_lead_hours=float(min_hit_lead_hours),
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}
    return metrics, detail_df


def make_parallel_or_frames(
    left_scores: dict[str, pd.DataFrame],
    left_threshold: float,
    right_scores: dict[str, pd.DataFrame],
    right_threshold: float,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in sorted(set(left_scores) | set(right_scores)):
        left_raw = left_scores.get(name, pd.DataFrame())
        right_raw = right_scores.get(name, pd.DataFrame())
        left = apply_threshold(left_raw, left_threshold) if not left_raw.empty else pd.DataFrame(columns=["timestamp", "predict"])
        right = apply_threshold(right_raw, right_threshold) if not right_raw.empty else pd.DataFrame(columns=["timestamp", "predict"])
        left_small = (
            left[["timestamp", "predict"]].groupby("timestamp", as_index=False)["predict"].max()
            if not left.empty
            else pd.DataFrame(columns=["timestamp", "predict"])
        )
        right_small = (
            right[["timestamp", "predict"]].groupby("timestamp", as_index=False)["predict"].max()
            if not right.empty
            else pd.DataFrame(columns=["timestamp", "predict"])
        )
        merged = left_small.merge(right_small, on="timestamp", how="outer", suffixes=("_left", "_right")).fillna(0)
        if merged.empty:
            out[name] = pd.DataFrame(columns=["timestamp", "predict", "score", "rule_predict", "source"])
            continue
        pred_left = pd.to_numeric(merged["predict_left"], errors="coerce").fillna(0).astype(int)
        pred_right = pd.to_numeric(merged["predict_right"], errors="coerce").fillna(0).astype(int)
        out[name] = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(merged["timestamp"], errors="coerce").astype(np.int64),
                "predict": ((pred_left > 0) | (pred_right > 0)).astype(int),
                "score": np.nan,
                "rule_predict": 0,
                "source": "parallel_or",
            }
        ).sort_values("timestamp")
    return out


def write_prediction_frames(pred_frames: dict[str, pd.DataFrame], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for name, frame in sorted(pred_frames.items()):
        frame.to_csv(out_dir / name, index=False)
        total_rows += int(len(frame))
    return total_rows


def save_selected_features(path: Path, selector: IndexSelector) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "rank": np.arange(1, len(selector.feature_names) + 1, dtype=int),
            "feature_index": selector.indices,
            "feature_name": selector.feature_names,
        }
    ).to_csv(path, index=False)


def run_fold(deep_model_name: str, fold: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    deep_parts = set(parse_csv_words(args.deep_feature_parts))
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
    configure_runtime(cfg, args.seed, fold)
    fold_started = time.time()
    print(f"[tabular-init] deep_model={deep_model_name} fold={fold} {runtime_device_summary(cfg.device)}", flush=True)
    index_started = log_stage_start("read_index_split", deep_model_name, fold, index_path=args.index_path)
    index_df = read_index(args.index_path)
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    label_by_file = file_label_map(index_df)
    if int(args.max_train_files) > 0:
        train_files = cap_files_stratified(train_files, label_by_file, int(args.max_train_files), int(args.seed) + int(fold))
    if int(args.max_test_files) > 0:
        test_files = cap_files_stratified(test_files, label_by_file, int(args.max_test_files), int(args.seed) + 7919 + int(fold))
    train_files, val_files = split_train_val_files(train_files, label_by_file, cfg, int(fold))
    log_stage_done(
        "read_index_split",
        index_started,
        deep_model_name,
        fold,
        train=len(train_files),
        val=len(val_files),
        test=len(test_files),
    )

    run_dir = Path(args.out_root) / deep_model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_run_header(
        "OFP MODEL2-COMPAT HYBRID FUSION",
        {
            "deep model": deep_model_name,
            "fold": fold,
            "train/val/test": f"{len(train_files)}/{len(val_files)}/{len(test_files)} modules",
            "target": cfg.target_mode,
            "features": cfg.feature_mode,
            "ml feature set": args.ml_feature_set,
            "ml models": " ".join(args.ml_models),
            "selector": f"{args.selector} top_k={args.select_k}",
            "sample selection": cfg.sample_selection,
            "rule mode": cfg.rule_mode,
            "min hit lead": f"{args.min_hit_lead_hours} h",
            "seq_len": cfg.seq_len,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "device": cfg.device,
            "out_dir": run_dir,
        },
    )

    stats_started = log_stage_start("feature_norm_stats", deep_model_name, fold, files=len(train_files), data_dir=args.data_dir)
    stats = compute_feature_norm_stats(args.data_dir, train_files, cfg, label_by_file)
    log_stage_done("feature_norm_stats", stats_started, deep_model_name, fold)
    mean, std = stats.arrays()
    cache_started = log_stage_start("feature_cache_init", deep_model_name, fold, data_dir=args.data_dir)
    cache = Model2FeatureCache(args.data_dir, mean, std, cfg, label_by_file)
    log_stage_done("feature_cache_init", cache_started, deep_model_name, fold)
    model, model_cfg, train_meta, dataset = train_deep_model(deep_model_name, train_files, cache, cfg, fold=fold)
    save_started = log_stage_start("save_deep_checkpoint", deep_model_name, fold, path=run_dir / "deep_model.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "model_cfg": model_cfg}, run_dir / "deep_model.pt")
    log_stage_done("save_deep_checkpoint", save_started, deep_model_name, fold)

    x_raw, y_train, raw_feature_names = collect_training_table(
        model,
        cache,
        dataset,
        cfg,
        deep_model_name,
        args.ml_feature_set,
        deep_parts,
        fold=fold,
        stage_name="collect_ml_train_table",
    )
    selector = fit_feature_selector(
        x_raw,
        y_train,
        raw_feature_names,
        args.selector,
        args.select_k,
        int(args.seed) + int(fold),
        args.n_jobs,
        deep_model_name=deep_model_name,
        fold=fold,
    )
    x_train = selector.transform(x_raw)
    save_selected_features(run_dir / "tabular" / "selected_features.csv", selector)
    ml_names = [name.lower() for name in args.ml_models]
    ml_estimators = train_ml_models(
        x_train,
        y_train,
        ml_names,
        args,
        int(args.seed) + int(fold),
        deep_model_name=deep_model_name,
        fold=fold,
    )
    trained_ml_names = list(ml_estimators.keys())
    print(
        f"[ml-ready] deep_model={deep_model_name} fold={fold} requested={','.join(ml_names)} "
        f"trained={','.join(trained_ml_names)}",
        flush=True,
    )
    pickle_started = log_stage_start("save_ml_models", deep_model_name, fold, path=run_dir / "tabular" / "ml_models.pkl")
    with (run_dir / "tabular" / "ml_models.pkl").open("wb") as fh:
        pickle.dump({"selector": selector, "models": ml_estimators}, fh)
    log_stage_done("save_ml_models", pickle_started, deep_model_name, fold)

    val_score_frames = score_ml_models_to_memory(
        ml_estimators,
        selector,
        model,
        cache,
        val_files,
        cfg,
        deep_model_name,
        args.ml_feature_set,
        deep_parts,
        fold=fold,
        stage_name="score_val_files",
    )
    test_score_frames = score_ml_models_to_memory(
        ml_estimators,
        selector,
        model,
        cache,
        test_files,
        cfg,
        deep_model_name,
        args.ml_feature_set,
        deep_parts,
        fold=fold,
        stage_name="score_test_files",
    )

    results: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    eval_started = log_stage_start("threshold_eval", deep_model_name, fold, models=",".join(trained_ml_names))
    for ml_name in trained_ml_names:
        tag = f"ml_{ml_name}_{args.ml_feature_set}"
        tag_started = log_stage_start("threshold_eval_one", deep_model_name, fold, tag=tag)
        threshold, val_metrics = select_threshold_for_scores(val_score_frames[ml_name], args.data_dir, run_dir, tag, args)
        thresholds[tag] = threshold
        pred_dir = run_dir / "predictions" / tag
        eval_dir = run_dir / "evaluation" / tag
        test_rows = write_threshold_predictions(test_score_frames[ml_name], pred_dir, threshold)
        metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
        result = {
            "deep_model": deep_model_name,
            "fold": int(fold),
            "mode": tag,
            "ml_model": ml_name,
            "ml_feature_set": args.ml_feature_set,
            "deep_feature_parts": sorted(deep_parts),
            "selector": args.selector,
            "selected_feature_count": len(selector.indices),
            "threshold": float(threshold),
            "val_metrics": val_metrics,
            "test_rows": int(test_rows),
            "metrics": metrics,
        }
        print(f"[tabular-done] fold={fold} {tag} threshold={threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}", flush=True)
        log_stage_done("threshold_eval_one", tag_started, deep_model_name, fold, tag=tag)
        results.append(result)
    log_stage_done("threshold_eval", eval_started, deep_model_name, fold)

    if bool(args.write_deep_predictions) or bool(args.enable_parallel_fusion):
        deep_score_started = log_stage_start("score_deep_val_files", deep_model_name, fold, files=len(val_files))
        deep_val_scores = score_files_to_memory(model, cache, val_files, cfg)
        log_stage_done("score_deep_val_files", deep_score_started, deep_model_name, fold)
        deep_threshold, deep_val_metrics = select_threshold_for_scores(deep_val_scores, args.data_dir, run_dir, "deep_only", args)
        deep_test_started = log_stage_start("score_deep_test_files", deep_model_name, fold, files=len(test_files))
        deep_test_scores = score_files_to_memory(model, cache, test_files, cfg)
        log_stage_done("score_deep_test_files", deep_test_started, deep_model_name, fold)
        thresholds["deep_only"] = deep_threshold
        if bool(args.write_deep_predictions):
            pred_dir = run_dir / "predictions" / "deep_only"
            eval_dir = run_dir / "evaluation" / "deep_only"
            test_rows = write_threshold_predictions(deep_test_scores, pred_dir, deep_threshold)
            metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
            results.append(
                {
                    "deep_model": deep_model_name,
                    "fold": int(fold),
                    "mode": "deep_only",
                    "ml_model": "",
                    "ml_feature_set": "",
                    "deep_feature_parts": [],
                    "selector": "",
                    "selected_feature_count": 0,
                    "threshold": float(deep_threshold),
                    "val_metrics": deep_val_metrics,
                    "test_rows": int(test_rows),
                    "metrics": metrics,
                }
            )
            print(f"[deep-only-done] fold={fold} threshold={deep_threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}", flush=True)

        if bool(args.enable_parallel_fusion):
            parallel_ml_name = str(args.parallel_ml_model).lower()
            if parallel_ml_name in ml_estimators and args.parallel_feature_set == args.ml_feature_set:
                parallel_selector = selector
                parallel_estimators = {parallel_ml_name: ml_estimators[parallel_ml_name]}
                parallel_val_scores = {parallel_ml_name: val_score_frames[parallel_ml_name]}
                parallel_test_scores = {parallel_ml_name: test_score_frames[parallel_ml_name]}
            else:
                x_parallel_raw, y_parallel, parallel_feature_names = collect_training_table(
                    model,
                    cache,
                    dataset,
                    cfg,
                    deep_model_name,
                    args.parallel_feature_set,
                    deep_parts,
                    fold=fold,
                    stage_name="collect_parallel_train_table",
                )
                parallel_selector = fit_feature_selector(
                    x_parallel_raw,
                    y_parallel,
                    parallel_feature_names,
                    args.parallel_selector,
                    args.parallel_select_k,
                    int(args.seed) + int(fold) + 7919,
                    args.n_jobs,
                    deep_model_name=deep_model_name,
                    fold=fold,
                )
                save_selected_features(run_dir / "parallel_xgb" / "selected_features.csv", parallel_selector)
                x_parallel = parallel_selector.transform(x_parallel_raw)
                parallel_estimators = train_ml_models(
                    x_parallel,
                    y_parallel,
                    [parallel_ml_name],
                    args,
                    int(args.seed) + int(fold) + 7919,
                    deep_model_name=deep_model_name,
                    fold=fold,
                    stage_name="parallel_ml_train",
                )
                parallel_val_scores = score_ml_models_to_memory(
                    parallel_estimators,
                    parallel_selector,
                    model,
                    cache,
                    val_files,
                    cfg,
                    deep_model_name,
                    args.parallel_feature_set,
                    deep_parts,
                    fold=fold,
                    stage_name="score_parallel_val_files",
                )
                parallel_test_scores = score_ml_models_to_memory(
                    parallel_estimators,
                    parallel_selector,
                    model,
                    cache,
                    test_files,
                    cfg,
                    deep_model_name,
                    args.parallel_feature_set,
                    deep_parts,
                    fold=fold,
                    stage_name="score_parallel_test_files",
                )
            parallel_tag = f"parallel_{parallel_ml_name}_{args.parallel_feature_set}_or_deep"
            parallel_threshold, parallel_val_metrics = select_threshold_for_scores(
                parallel_val_scores[parallel_ml_name],
                args.data_dir,
                run_dir,
                f"{parallel_tag}_ml",
                args,
            )
            parallel_test_pred = make_parallel_or_frames(
                parallel_test_scores[parallel_ml_name],
                parallel_threshold,
                deep_test_scores,
                deep_threshold,
            )
            pred_dir = run_dir / "predictions" / parallel_tag
            eval_dir = run_dir / "evaluation" / parallel_tag
            test_rows = write_prediction_frames(parallel_test_pred, pred_dir)
            metrics, _detail = evaluate_prediction_output(pred_dir, args.data_dir, eval_dir, args.min_hit_lead_hours)
            results.append(
                {
                    "deep_model": deep_model_name,
                    "fold": int(fold),
                    "mode": parallel_tag,
                    "ml_model": parallel_ml_name,
                    "ml_feature_set": args.parallel_feature_set,
                    "deep_feature_parts": ["parallel_or"],
                    "selector": args.parallel_selector,
                    "selected_feature_count": len(parallel_selector.indices),
                    "threshold": float(parallel_threshold),
                    "deep_threshold": float(deep_threshold),
                    "val_metrics": parallel_val_metrics,
                    "test_rows": int(test_rows),
                    "metrics": metrics,
                }
            )
            print(f"[parallel-done] fold={fold} {parallel_tag} threshold={parallel_threshold:.4f} rows={test_rows} {format_metric_summary(metrics)}", flush=True)

    fold_summary = {
        "deep_model": deep_model_name,
        "fold": int(fold),
        "seconds": time.time() - started,
        "cfg": asdict(cfg),
        "args": vars(args),
        "feature_norm_stats": asdict(stats),
        "deep_model_cfg": model_cfg,
        "deep_train": train_meta,
        "raw_feature_count": len(raw_feature_names),
        "selected_features": selector.get_feature_names_out(),
        "thresholds": thresholds,
        "results": results,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2, default=str), encoding="utf-8")
    print(
        f"[fold-done] deep_model={deep_model_name} fold={fold} "
        f"elapsed={format_duration(time.time() - fold_started)} results={len(results)} out={run_dir}",
        flush=True,
    )

    del model, dataset, cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def aggregate_results(results: list[dict[str, Any]], out_root: Path, args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for item in results:
        row = {
            "deep_model": item["deep_model"],
            "fold": int(item["fold"]),
            "mode": item["mode"],
            "ml_model": item.get("ml_model", ""),
            "ml_feature_set": item.get("ml_feature_set", ""),
            "selector": item.get("selector", ""),
            "selected_feature_count": int(item.get("selected_feature_count", 0)),
            "threshold": float(item.get("threshold", 0.0)),
            "deep_threshold": float(item.get("deep_threshold", np.nan)),
            "target_mode": args.target_mode,
            "feature_mode": args.feature_mode,
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
    new = pd.DataFrame(rows)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new.drop_duplicates(
            subset=["deep_model", "fold", "mode", "target_mode", "feature_mode", "sampling_mode", "rule_mode", "min_hit_lead_hours"],
            keep="last",
            inplace=True,
        )
    new.sort_values(["deep_model", "mode", "min_hit_lead_hours", "fold"], inplace=True)
    new.to_csv(path, index=False)
    group_cols = ["deep_model", "mode", "target_mode", "feature_mode", "sampling_mode", "rule_mode", "min_hit_lead_hours"]
    non_metric_cols = set(group_cols) | {"fold", "ml_model", "ml_feature_set", "selector"}
    numeric = [
        col
        for col in new.columns
        if col not in non_metric_cols and pd.api.types.is_numeric_dtype(new[col])
    ]
    summary = new.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model2-compatible deep embedding + tabular ML fusion experiments.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_DEEP_MODELS, choices=sorted(BUILDERS))
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_model2_compat_results/tabular_fusion"))
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
    parser.add_argument("--sampling_mode", choices=["row_ratio", "module_balanced"], default="module_balanced")
    parser.add_argument("--positive_windows_per_module", type=int, default=32)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--normal_windows_per_module", type=int, default=8)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_grid", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--no_threshold_search", dest="threshold_search", action="store_false")
    parser.set_defaults(threshold_search=True)
    parser.add_argument("--rule_mode", choices=["none", "temp", "model2_simple"], default="model2_simple")
    parser.add_argument("--sample_selection", choices=["random", "signal_topk", "hybrid"], default="random")
    parser.add_argument("--sample_topk_fraction", type=float, default=0.5)
    parser.add_argument("--temporal_positive_weight", type=float, default=0.0)
    parser.add_argument("--temporal_weight_horizon_hours", type=float, default=120.0)
    parser.add_argument("--adaptive_negative_weight", type=float, default=0.0)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=1)
    parser.add_argument("--min_hit_lead_hours", type=float, default=0.0)
    parser.add_argument("--ml_models", nargs="+", default=["rf", "xgb", "lgbm", "catboost"])
    parser.add_argument("--require_all_ml", action="store_true")
    parser.add_argument("--ml_feature_set", choices=["model2", "embedding", "fusion"], default="fusion")
    parser.add_argument("--deep_feature_parts", default="embedding,score")
    parser.add_argument("--selector", choices=["none", "variance", "f_classif", "mutual_info", "extra_trees", "model_importance"], default="extra_trees")
    parser.add_argument("--select_k", type=int, default=96)
    parser.add_argument("--ml_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=0)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--enable_parallel_fusion", action="store_true")
    parser.add_argument("--parallel_ml_model", default="xgb")
    parser.add_argument("--parallel_feature_set", choices=["model2", "embedding", "fusion"], default="model2")
    parser.add_argument("--parallel_selector", choices=["none", "variance", "f_classif", "mutual_info", "extra_trees", "model_importance"], default="extra_trees")
    parser.add_argument("--parallel_select_k", type=int, default=64)
    parser.add_argument("--write_deep_predictions", action="store_true")
    parser.add_argument("--max_cached_files", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--log_batches", type=int, default=0)
    parser.add_argument("--max_train_files", type=int, default=2500)
    parser.add_argument("--max_test_files", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: list[dict[str, Any]] = []
    for model_name in args.models:
        for fold in args.folds:
            results = run_fold(str(model_name), int(fold), args)
            all_results.extend(results)
            aggregate_results(results, args.out_root, args)
    aggregate_results(all_results, args.out_root, args)


if __name__ == "__main__":
    main()
