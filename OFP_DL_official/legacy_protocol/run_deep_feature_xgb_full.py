from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_deep_models import compute_norm_stats, make_model, split_train_val
from OFP_DL_official.common.formal_data import PRIMARY_HORIZON_INDEX
from OFP_DL_official.ofp_protocol.run_ofp_baselines import (
    SENSORS,
    ahead120_label,
    first_event_valid_mask,
    read_index,
)
from OFP_DL_official.ofp_protocol.run_ofp_model2_strict import (
    DEFAULT_FEATURE_LIST,
    TEMP_OUTLIER,
    extract_for_file,
    positive_probability,
)


SEC_IN_HOUR = 3600.0


@dataclass
class HybridCfg:
    seq_len: int
    batch_size: int
    embed_features: int
    projection_seed: int
    label_mode: str
    threshold_grid_size: int
    norm_max_files: int
    include_ofp_features: bool
    include_deep_features: bool
    include_extra_rule: bool
    device: str


@dataclass
class FoldSummary:
    model: str
    fold: int
    encoder_model: str
    encoder_checkpoint: str
    label_mode: str
    threshold: float
    threshold_source: str
    train_modules: int
    val_modules: int
    threshold_modules: int
    test_modules: int
    train_rows: int
    val_rows: int
    test_rows: int
    feature_dim: int
    seconds: float
    val_metrics: dict[str, float]
    metrics: dict[str, float]


def read_raw_module(data_dir: Path, file_name: str) -> pd.DataFrame:
    cols = {"timestamp", "anomaly", *SENSORS}
    df = pd.read_csv(data_dir / file_name, usecols=lambda c: c in cols)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    for col in SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "anomaly" in df.columns:
        df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).astype(np.int8)
    return df


def make_windows(values: np.ndarray, seq_len: int, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values).astype(np.float32)
    x = np.where(valid > 0, values, 0.0).astype(np.float32)
    x = (x - mean) / std
    x = x * valid
    pad_x = np.zeros((seq_len - 1, values.shape[1]), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = np.concatenate([pad_x, x], axis=0)
    m_pad = np.concatenate([pad_m, valid], axis=0)
    x_windows = np.lib.stride_tricks.sliding_window_view(
        x_pad, window_shape=seq_len, axis=0
    ).transpose(0, 2, 1)
    m_windows = np.lib.stride_tricks.sliding_window_view(
        m_pad, window_shape=seq_len, axis=0
    ).transpose(0, 2, 1)
    return x_windows, m_windows


def itransformer_pooled_and_logit(model: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if model.use_mask_channel:
        inp = torch.cat([x, mask], dim=-1)
    else:
        inp = x
    if model.use_norm:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * mask).sum(dim=1, keepdim=True) / denom
        var = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var + 1e-5)
        x_norm = (x - mean) / std * mask
        if model.use_mask_channel:
            inp = torch.cat([x_norm, mask], dim=-1)
        else:
            inp = x_norm
    enc_out = model.enc_embedding(inp, None)
    enc_out, _ = model.encoder(enc_out, attn_mask=None)
    pooled = enc_out[:, : model.cfg.n_sensors, :].mean(dim=1)
    logit = model.head(pooled)
    if model.cfg.output_dim == 1:
        logit = logit.squeeze(-1)
    return pooled, logit


def select_primary_logit(logit: torch.Tensor) -> torch.Tensor:
    if logit.ndim == 2 and logit.shape[-1] > 1:
        return logit[:, PRIMARY_HORIZON_INDEX]
    if logit.ndim > 1:
        return logit.squeeze(-1)
    return logit


def model_features(
    model: nn.Module,
    encoder_model: str,
    x: torch.Tensor,
    mask: torch.Tensor,
    projection: torch.Tensor | None,
) -> torch.Tensor:
    if encoder_model == "itransformer" and hasattr(model, "enc_embedding"):
        pooled, logit = itransformer_pooled_and_logit(model, x, mask)
        logit = select_primary_logit(logit)
        cols = [logit.unsqueeze(-1), torch.sigmoid(logit).unsqueeze(-1)]
        if projection is not None and projection.numel() > 0:
            cols.append(pooled @ projection)
        return torch.cat(cols, dim=-1)

    logit = model(x, mask)
    if isinstance(logit, tuple):
        logit = logit[0]
    logit = select_primary_logit(logit)
    return torch.stack([logit, torch.sigmoid(logit)], dim=-1)


def load_encoder(
    checkpoint_path: Path,
    encoder_model: str,
    seq_len: int,
    n_sensors: int,
    device: str,
) -> nn.Module:
    model, _model_cfg = make_model(encoder_model, seq_len, n_sensors)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def make_projection(model: nn.Module, encoder_model: str, embed_features: int, seed: int, device: str) -> torch.Tensor | None:
    if embed_features <= 0:
        return None
    if encoder_model == "itransformer" and hasattr(model, "cfg"):
        dim = int(model.cfg.d_model)
    else:
        return None
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(dim, embed_features)).astype(np.float32)
    return torch.from_numpy(proj).to(device)


def deep_feature_frame(
    data_dir: Path,
    file_name: str,
    model: nn.Module,
    encoder_model: str,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: HybridCfg,
    projection: torch.Tensor | None,
) -> pd.DataFrame:
    raw = read_raw_module(data_dir, file_name)
    timestamps = raw["timestamp"].to_numpy(dtype=np.int64)
    values = raw[list(SENSORS)].to_numpy(dtype=np.float32)
    x_windows, m_windows = make_windows(values, cfg.seq_len, mean, std)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), cfg.batch_size):
            end = min(len(values), start + cfg.batch_size)
            xb = torch.from_numpy(np.array(x_windows[start:end], copy=True)).to(cfg.device)
            mb = torch.from_numpy(np.array(m_windows[start:end], copy=True)).to(cfg.device)
            feat = model_features(model, encoder_model, xb, mb, projection)
            parts.append(feat.detach().cpu().numpy().astype(np.float32))
    arr = np.concatenate(parts, axis=0) if parts else np.empty((0, 2), dtype=np.float32)
    cols = ["deep_logit", "deep_prob"] + [f"deep_rp_{i}" for i in range(max(arr.shape[1] - 2, 0))]
    out = pd.DataFrame(arr, columns=cols)
    out.insert(0, "Ts", timestamps)
    return out


def labels_for_normal_rows(data_dir: Path, file_name: str, normal: pd.DataFrame, label_mode: str) -> np.ndarray:
    if label_mode in {"anomaly", "ahead120"}:
        raw = read_raw_module(data_dir, file_name)
        labels = ahead120_label(raw)
        label_df = pd.DataFrame({"Ts": raw["timestamp"].to_numpy(dtype=np.int64), "label": labels.astype(np.int8)})
        label_df = label_df.drop_duplicates("Ts", keep="first").set_index("Ts")
        aligned = label_df.reindex(normal["Ts"].to_numpy(dtype=np.int64))
        return aligned["label"].fillna(0).to_numpy(dtype=np.int8)
    raise ValueError(f"Unknown label_mode: {label_mode}")


def valid_mask_for_normal_rows(data_dir: Path, file_name: str, normal: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly"})
    valid = first_event_valid_mask(raw)
    valid_df = pd.DataFrame({"Ts": pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=np.int64), "valid": valid})
    valid_df = valid_df.drop_duplicates("Ts", keep="first").set_index("Ts")
    aligned = valid_df.reindex(normal["Ts"].to_numpy(dtype=np.int64))
    return aligned["valid"].fillna(False).to_numpy(dtype=bool)


def hybrid_features_for_file(
    data_dir: Path,
    file_name: str,
    model: nn.Module,
    encoder_model: str,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: HybridCfg,
    projection: torch.Tensor | None,
    with_label: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, pd.DataFrame]:
    normal, extra = extract_for_file(data_dir, file_name, with_label=True)
    if len(normal) == 0:
        return np.empty((0, 0), dtype=np.float32), None, np.empty((0,), dtype=np.int64), extra
    valid_mask = valid_mask_for_normal_rows(data_dir, file_name, normal)
    if not np.any(valid_mask):
        return np.empty((0, 0), dtype=np.float32), None, np.empty((0,), dtype=np.int64), extra
    normal_valid = normal.loc[valid_mask]

    parts: list[np.ndarray] = []
    if cfg.include_ofp_features:
        parts.append(normal_valid[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32))
    if cfg.include_deep_features:
        deep_df = deep_feature_frame(data_dir, file_name, model, encoder_model, mean, std, cfg, projection)
        deep_df = deep_df.drop_duplicates("Ts", keep="first").set_index("Ts")
        deep_cols = [c for c in deep_df.columns if c.startswith("deep_")]
        aligned = deep_df.reindex(normal_valid["Ts"].to_numpy(dtype=np.int64))
        deep_arr = aligned[deep_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        parts.append(deep_arr)
    X = np.concatenate(parts, axis=1) if parts else np.empty((len(normal_valid), 0), dtype=np.float32)
    y = labels_for_normal_rows(data_dir, file_name, normal, cfg.label_mode)[valid_mask] if with_label else None
    timestamps = normal_valid["Ts"].to_numpy(dtype=np.int64)
    return X, y, timestamps, extra


def load_train_arrays(
    data_dir: Path,
    file_names: list[str],
    model: nn.Module,
    encoder_model: str,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: HybridCfg,
    projection: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        X, y, _ts, _extra = hybrid_features_for_file(
            data_dir, name, model, encoder_model, mean, std, cfg, projection, with_label=True
        )
        if len(X):
            xs.append(X)
            ys.append(y)
            n_rows += len(X)
        if idx % 250 == 0:
            print(f"  [hybrid-load] {idx}/{len(file_names)} files, rows={n_rows}")
    X_train = np.concatenate(xs, axis=0) if xs else np.empty((0, 0), dtype=np.float32)
    y_train = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int8)
    return X_train, y_train, n_rows


def first_anomaly_ts(data_dir: Path, file_name: str) -> float | None:
    raw = read_raw_module(data_dir, file_name)
    anomaly = raw["anomaly"].to_numpy(dtype=np.int8)
    if not np.any(anomaly > 0):
        return None
    timestamps = raw["timestamp"].to_numpy(dtype=np.int64)
    return float(timestamps[int(np.argmax(anomaly > 0))])


def collect_hybrid_scores(
    classifier,
    data_dir: Path,
    file_names: list[str],
    model: nn.Module,
    encoder_model: str,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: HybridCfg,
    projection: torch.Tensor | None,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    total_rows = 0
    for idx, name in enumerate(file_names, 1):
        X, _y, timestamps, extra = hybrid_features_for_file(
            data_dir, name, model, encoder_model, mean, std, cfg, projection, with_label=False
        )
        frames = []
        if len(X):
            scores = positive_probability(classifier, X).astype(np.float32)
            frames.append(pd.DataFrame({"timestamp": timestamps, "score": scores}))
        if cfg.include_extra_rule and len(extra):
            extra_ts = pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64)
            extra_score = (pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER).astype(np.float32)
            frames.append(pd.DataFrame({"timestamp": extra_ts, "score": extra_score}))
        pred = (
            pd.concat(frames, ignore_index=True).sort_values("timestamp")
            if frames
            else pd.DataFrame(columns=["timestamp", "score"])
        )
        true_ts = first_anomaly_ts(data_dir, name)
        pred_ts = pd.to_numeric(pred["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(pred_ts)
        if true_ts is not None:
            valid_mask &= pred_ts < float(true_ts)
        rows.append(
            {
                "file_name": name,
                "timestamps": pred["timestamp"].to_numpy(dtype=np.int64),
                "scores": pred["score"].to_numpy(dtype=np.float32),
                "true_label": int(true_ts is not None),
                "true_ts": true_ts,
                "valid_mask": valid_mask,
            }
        )
        total_rows += int(np.sum(valid_mask))
        if idx % 250 == 0:
            print(f"  [hybrid-score] {idx}/{len(file_names)} files, rows={total_rows}")
    return rows, total_rows


def evaluate_scores(module_rows: list[dict], threshold: float) -> dict[str, float]:
    true_pos, pred_pos, hit = set(), set(), set()
    lead_hours: list[float] = []
    evaluated = 0
    for row in module_rows:
        evaluated += 1
        name = row["file_name"]
        if row["true_label"]:
            true_pos.add(name)
        scores = row["scores"]
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            true_ts = row.get("true_ts")
            valid_mask = np.ones(len(scores), dtype=bool) if true_ts is None else row["timestamps"] < float(true_ts)
        idx = np.where((scores >= threshold) & valid_mask)[0]
        if len(idx) == 0:
            continue
        pred_ts = float(row["timestamps"][int(idx[0])])
        if row["true_label"] == 0:
            pred_pos.add(name)
        elif row["true_ts"] is not None and pred_ts < row["true_ts"]:
            pred_pos.add(name)
            hit.add(name)
            lead_hours.append(abs(row["true_ts"] - pred_ts) / SEC_IN_HOUR)

    tp = len(hit)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = evaluated - tp - fp - fn
    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    avg_lead_hour = float(np.mean(lead_hours)) if lead_hours else 0.0
    min_lead_hour = float(np.min(lead_hours)) if lead_hours else 0.0
    avg_lead_score = float(np.tanh(avg_lead_hour))
    min_lead_score = float(np.tanh(min_lead_hour))
    final_score = f1 + avg_lead_score + min_lead_score + accuracy
    return {
        "final_score": final_score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "all_hit_cnt": float(tp),
        "all_predict_pos_cnt": float(len(pred_pos)),
        "all_true_pos_cnt": float(len(true_pos)),
        "avg_lead_score": avg_lead_score,
        "avg_lead_hour": avg_lead_hour,
        "min_lead_score": min_lead_score,
        "min_lead_hour": min_lead_hour,
        "lead_pread_cnt": float(len(lead_hours)),
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "evaluated_module_cnt": float(evaluated),
    }


def choose_threshold(module_rows: list[dict], grid_size: int, metric: str) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in np.linspace(0.01, 0.99, int(grid_size)):
        metrics = evaluate_scores(module_rows, float(threshold))
        if best_metrics is None:
            best_threshold, best_metrics = float(threshold), metrics
            continue
        if metric == "f1":
            key = (
                metrics["f1_score"],
                metrics["precision"],
                metrics["recall"],
                metrics["accuracy"],
                -metrics["all_predict_pos_cnt"],
            )
            best_key = (
                best_metrics["f1_score"],
                best_metrics["precision"],
                best_metrics["recall"],
                best_metrics["accuracy"],
                -best_metrics["all_predict_pos_cnt"],
            )
        elif metric == "precision":
            key = (
                metrics["precision"],
                metrics["f1_score"],
                metrics["recall"],
                metrics["accuracy"],
                -metrics["all_predict_pos_cnt"],
            )
            best_key = (
                best_metrics["precision"],
                best_metrics["f1_score"],
                best_metrics["recall"],
                best_metrics["accuracy"],
                -best_metrics["all_predict_pos_cnt"],
            )
        elif metric == "final":
            key = (
                metrics["final_score"],
                metrics["f1_score"],
                metrics["precision"],
                metrics["recall"],
                -metrics["all_predict_pos_cnt"],
            )
            best_key = (
                best_metrics["final_score"],
                best_metrics["f1_score"],
                best_metrics["precision"],
                best_metrics["recall"],
                -best_metrics["all_predict_pos_cnt"],
            )
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")
        if key > best_key:
            best_threshold, best_metrics = float(threshold), metrics
    assert best_metrics is not None
    best_metrics = dict(best_metrics)
    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def write_predictions(module_rows: list[dict], out_dir: Path, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in module_rows:
        scores = row["scores"]
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            true_ts = row.get("true_ts")
            valid_mask = np.ones(len(scores), dtype=bool) if true_ts is None else row["timestamps"] < float(true_ts)
        pred = ((scores >= threshold) & valid_mask).astype(int)
        pd.DataFrame(
            {
                "timestamp": row["timestamps"],
                "predict": pred,
                "score": scores,
                "valid_for_eval": valid_mask.astype(int),
            }
        ).to_csv(out_dir / row["file_name"], index=False)


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(r["Item"]): float(r["Value"]) for _, r in summary_df.iterrows()}


def checkpoint_for_fold(root: Path, encoder_model: str, fold: int) -> Path:
    return root / encoder_model / f"fold_{fold}" / "model.pt"


def limit_files_stratified(file_names: list[str], index_df: pd.DataFrame, limit: int | None, seed: int) -> list[str]:
    if limit is None or len(file_names) <= int(limit):
        return file_names
    name_col = "file_name"
    label_map = dict(zip(index_df[name_col].astype(str), index_df["Label"].astype(int)))
    positives = [name for name in file_names if int(label_map.get(str(name), 0)) == 1]
    negatives = [name for name in file_names if int(label_map.get(str(name), 0)) == 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if not positives or not negatives:
        selected = file_names[: int(limit)]
    else:
        pos_take = min(len(positives), max(1, int(limit) // 2))
        neg_take = min(len(negatives), int(limit) - pos_take)
        if neg_take == 0:
            neg_take = min(len(negatives), 1)
            pos_take = min(len(positives), int(limit) - neg_take)
        selected = positives[:pos_take] + negatives[:neg_take]
        if len(selected) < int(limit):
            used = set(selected)
            rest = [name for name in positives[pos_take:] + negatives[neg_take:] if name not in used]
            selected.extend(rest[: int(limit) - len(selected)])
    rng.shuffle(selected)
    return selected[: int(limit)]


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    encoder_root: Path,
    out_root: Path,
    encoder_model: str,
    cfg: HybridCfg,
    xgb_estimators: int,
    n_jobs: int,
    max_train_files: int | None,
    max_val_files: int | None,
    max_test_files: int | None,
    train_with_val: bool,
    threshold_source: str,
    threshold_metric: str,
    fixed_threshold: float,
) -> FoldSummary:
    t0 = time.time()
    train_files, val_files, test_files = split_train_val(index_df, fold, 0.1, 42)
    train_files = limit_files_stratified(train_files, index_df, max_train_files, 10_000 + fold)
    val_files = limit_files_stratified(val_files, index_df, max_val_files, 20_000 + fold)
    test_files = limit_files_stratified(test_files, index_df, max_test_files, 30_000 + fold)
    fit_files = [*train_files, *val_files] if train_with_val else train_files

    checkpoint_path = checkpoint_for_fold(encoder_root, encoder_model, fold)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {checkpoint_path}")

    print(
        f"[hybrid] fold={fold} fit={len(fit_files)} "
        f"train_split={len(train_files)} val_split={len(val_files)} test={len(test_files)}"
    )
    print(f"[hybrid] load encoder {checkpoint_path}")
    model = load_encoder(checkpoint_path, encoder_model, cfg.seq_len, len(SENSORS), cfg.device)
    projection = make_projection(model, encoder_model, cfg.embed_features, cfg.projection_seed + fold, cfg.device)

    print(f"[hybrid] compute norm stats max_files={cfg.norm_max_files}")
    mean, std = compute_norm_stats(data_dir, train_files, max_files=cfg.norm_max_files)

    print(f"[hybrid] load full training rows label={cfg.label_mode}")
    X_train, y_train, train_rows = load_train_arrays(
        data_dir, fit_files, model, encoder_model, mean, std, cfg, projection
    )
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Training labels contain one class only: {np.unique(y_train)}")
    feature_dim = int(X_train.shape[1])
    print(
        f"[hybrid] X_train={X_train.shape} pos={int(y_train.sum())} "
        f"neg={int(len(y_train) - y_train.sum())}"
    )

    classifier = XGBClassifier(
        n_estimators=int(xgb_estimators),
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=2024,
        n_jobs=int(n_jobs),
    )
    classifier.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()

    val_row_count = 0
    threshold_modules = 0
    if threshold_source == "fixed":
        threshold = float(fixed_threshold)
        val_metrics = {"threshold": threshold, "threshold_source": "fixed"}
        print(f"[hybrid] use fixed threshold={threshold:.3f}")
    else:
        threshold_files = val_files if threshold_source == "val" else fit_files
        threshold_modules = len(threshold_files)
        print(f"[hybrid] score threshold modules={threshold_modules} source={threshold_source}")
        val_rows, val_row_count = collect_hybrid_scores(
            classifier, data_dir, threshold_files, model, encoder_model, mean, std, cfg, projection
        )
        threshold, val_metrics = choose_threshold(val_rows, cfg.threshold_grid_size, threshold_metric)
        val_metrics["threshold_source"] = threshold_source
        val_metrics["threshold_metric"] = threshold_metric
        print(
            f"[hybrid] fold={fold} threshold final={val_metrics['final_score']:.4f} "
            f"F1={val_metrics['f1_score']:.4f} threshold={threshold:.3f}"
        )

    print(f"[hybrid] score test modules={len(test_files)}")
    test_rows, test_row_count = collect_hybrid_scores(
        classifier, data_dir, test_files, model, encoder_model, mean, std, cfg, projection
    )
    run_dir = out_root / f"{encoder_model}_xgb_{cfg.label_mode}" / f"fold_{fold}"
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(test_rows, pred_dir, threshold)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)

    summary = FoldSummary(
        model=f"{encoder_model}_xgb_{cfg.label_mode}",
        fold=int(fold),
        encoder_model=encoder_model,
        encoder_checkpoint=str(checkpoint_path),
        label_mode=cfg.label_mode,
        threshold=float(threshold),
        threshold_source=threshold_source,
        train_modules=len(fit_files),
        val_modules=len(val_files),
        threshold_modules=int(threshold_modules),
        test_modules=len(test_files),
        train_rows=int(train_rows),
        val_rows=int(val_row_count),
        test_rows=int(test_row_count),
        feature_dim=feature_dim,
        seconds=time.time() - t0,
        val_metrics=val_metrics,
        metrics=metrics,
    )
    (run_dir / "fold_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def aggregate(results: list[FoldSummary], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item.model,
            "fold": item.fold,
            "encoder_model": item.encoder_model,
            "label_mode": item.label_mode,
            "threshold": item.threshold,
            "threshold_source": item.threshold_source,
            "train_modules": item.train_modules,
            "val_modules": item.val_modules,
            "threshold_modules": item.threshold_modules,
            "test_modules": item.test_modules,
            "train_rows": item.train_rows,
            "val_rows": item.val_rows,
            "test_rows": item.test_rows,
            "feature_dim": item.feature_dim,
            "seconds": item.seconds,
        }
        row.update(item.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df.sort_values(["model", "fold"], inplace=True)
    df.to_csv(path, index=False)
    numeric = [c for c in df.columns if c not in {"model", "fold", "encoder_model", "label_mode", "threshold_source"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-row hybrid deep-feature + XGBoost OFP experiment.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--encoder_root", type=Path, default=Path("output/ofp_legacy_protocol/deep_module_level_3060_e3"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_legacy_protocol/deep_feature_xgb_full"))
    parser.add_argument("--encoder_model", default="itransformer", choices=["itransformer", "fteformer", "patchtst", "moderntcn", "fits"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--embed_features", type=int, default=8)
    parser.add_argument("--projection_seed", type=int, default=2026)
    parser.add_argument("--label_mode", choices=["anomaly", "ahead120"], default="anomaly")
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--norm_max_files", type=int, default=512)
    parser.add_argument("--xgb_estimators", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--train_with_val", action="store_true")
    parser.add_argument("--threshold_source", choices=["val", "train", "fixed"], default="val")
    parser.add_argument("--threshold_metric", choices=["final", "f1", "precision"], default="final")
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--no_ofp_features", action="store_true")
    parser.add_argument("--no_deep_features", action="store_true")
    parser.add_argument("--no_extra_rule", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    cfg = HybridCfg(
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        embed_features=int(args.embed_features),
        projection_seed=int(args.projection_seed),
        label_mode=args.label_mode,
        threshold_grid_size=int(args.threshold_grid_size),
        norm_max_files=int(args.norm_max_files),
        include_ofp_features=not args.no_ofp_features,
        include_deep_features=not args.no_deep_features,
        include_extra_rule=not args.no_extra_rule,
        device=args.device,
    )
    if not cfg.include_ofp_features and not cfg.include_deep_features:
        raise ValueError("At least one feature source must be enabled.")
    all_results: list[FoldSummary] = []
    for fold in args.folds:
        result = run_fold(
            fold=fold,
            data_dir=args.data_dir,
            index_df=index_df,
            encoder_root=args.encoder_root,
            out_root=args.out_root,
            encoder_model=args.encoder_model,
            cfg=cfg,
            xgb_estimators=args.xgb_estimators,
            n_jobs=args.n_jobs,
            max_train_files=args.max_train_files,
            max_val_files=args.max_val_files,
            max_test_files=args.max_test_files,
            train_with_val=args.train_with_val,
            threshold_source=args.threshold_source,
            threshold_metric=args.threshold_metric,
            fixed_threshold=args.fixed_threshold,
        )
        all_results.append(result)
        aggregate([result], args.out_root)
        print(
            f"[done-hybrid] {result.model} fold={result.fold} "
            f"Final={result.metrics.get('final_score', 0):.4f} "
            f"F1={result.metrics.get('f1_score', 0):.4f} "
            f"P={result.metrics.get('precision', 0):.4f} "
            f"R={result.metrics.get('recall', 0):.4f} "
            f"train_rows={result.train_rows} test_rows={result.test_rows}"
        )
    aggregate(all_results, args.out_root)


if __name__ == "__main__":
    main()

