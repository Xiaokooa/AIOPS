from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP.deep_learning.official.ofp_protocol.run_deep_models import split_train_val
from OFP.deep_learning.official.ofp_protocol.run_ofp_baselines import ahead120_label, read_index
from OFP.deep_learning.official.ofp_protocol.run_ofp_model2_strict import (
    DEFAULT_FEATURE_LIST,
    NA_DEFAULT,
    TEMP_OUTLIER,
    extract_for_file,
    positive_probability,
)


SEC_IN_HOUR = 3600.0


@dataclass
class NeuralCfg:
    label_mode: str = "anomaly"
    pre_event_hours: float = 1.0
    teacher_mode: str = "none"
    teacher_threshold: float = 0.3
    teacher_estimators: int = 100
    teacher_n_jobs: int = 1
    model_type: str = "mlp"
    num_bins: int = 16
    bin_temperature: float = 0.1
    bin_include_raw: bool = True
    prior_mode: str = "none"
    neural_prior_cap: float = 1.0
    legacy_timestamp_float32: bool = False
    epochs: int = 3
    batch_size: int = 8192
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.1
    norm_type: str = "batch"
    clip_value: float = 20.0
    pos_weight: float = 1.0
    auto_pos_weight: bool = False
    balanced_epoch: bool = False
    negative_ratio: int = 5
    threshold_min: float = 0.01
    threshold_max: float = 0.99
    threshold_grid_size: int = 99
    threshold_metric: str = "f1"
    include_extra_rule: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FoldSummary:
    model: str
    fold: int
    label_mode: str
    threshold: float
    threshold_metric: str
    train_modules: int
    val_modules: int
    test_modules: int
    train_rows: int
    val_rows: int
    test_rows: int
    feature_dim: int
    seconds: float
    val_metrics: dict[str, float]
    metrics: dict[str, float]
    train_history: list[dict[str, float]]


def make_norm(hidden_dim: int, norm_type: str) -> nn.Module | None:
    if norm_type == "batch":
        return nn.BatchNorm1d(hidden_dim)
    if norm_type == "layer":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "none":
        return None
    raise ValueError(f"Unknown norm_type: {norm_type}")


class MLPWarningModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float, norm_type: str = "batch") -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, int(layers))):
            blocks.append(nn.Linear(dim, hidden_dim))
            norm = make_norm(hidden_dim, norm_type)
            if norm is not None:
                blocks.append(norm)
            blocks.extend([nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SoftBinMLPWarningModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        thresholds: np.ndarray,
        temperature: float,
        include_raw: bool,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        thr = torch.as_tensor(thresholds, dtype=torch.float32)
        self.register_buffer("thresholds", thr)
        self.temperature = float(max(temperature, 1e-4))
        self.include_raw = bool(include_raw)
        first_dim = int(thr.numel()) + (input_dim if include_raw else 0)
        blocks: list[nn.Module] = []
        dim = first_dim
        for _ in range(max(1, int(layers))):
            blocks.append(nn.Linear(dim, hidden_dim))
            norm = make_norm(hidden_dim, norm_type)
            if norm is not None:
                blocks.append(norm)
            blocks.extend([nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid((x.unsqueeze(-1) - self.thresholds.unsqueeze(0)) / self.temperature)
        z = z.flatten(start_dim=1)
        if self.include_raw:
            z = torch.cat([x, z], dim=1)
        return self.net(z).squeeze(-1)


def compute_bin_thresholds(X: np.ndarray, mean: np.ndarray, std: np.ndarray, cfg: NeuralCfg) -> np.ndarray:
    qs = np.linspace(0.05, 0.95, int(cfg.num_bins), dtype=np.float32)
    thresholds = np.empty((X.shape[1], len(qs)), dtype=np.float32)
    for col in range(X.shape[1]):
        thresholds[col] = np.quantile(X[:, col], qs).astype(np.float32)
    thresholds = (thresholds - mean[:, None]) / std[:, None]
    return np.nan_to_num(thresholds, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_model(input_dim: int, cfg: NeuralCfg, thresholds: np.ndarray | None) -> nn.Module:
    if cfg.model_type == "mlp":
        return MLPWarningModel(input_dim, cfg.hidden_dim, cfg.layers, cfg.dropout, cfg.norm_type)
    if cfg.model_type == "softbin_mlp":
        if thresholds is None:
            raise ValueError("softbin_mlp requires bin thresholds")
        return SoftBinMLPWarningModel(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            layers=cfg.layers,
            dropout=cfg.dropout,
            thresholds=thresholds,
            temperature=cfg.bin_temperature,
            include_raw=cfg.bin_include_raw,
            norm_type=cfg.norm_type,
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_files_stratified(file_names: list[str], index_df: pd.DataFrame, limit: int | None, seed: int) -> list[str]:
    if limit is None or len(file_names) <= int(limit):
        return file_names
    label_map = dict(zip(index_df["file_name"].astype(str), index_df["Label"].astype(int)))
    pos = [name for name in file_names if int(label_map.get(str(name), 0)) == 1]
    neg = [name for name in file_names if int(label_map.get(str(name), 0)) == 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_take = min(len(pos), max(1, int(limit) // 2))
    neg_take = min(len(neg), int(limit) - pos_take)
    selected = pos[:pos_take] + neg[:neg_take]
    if len(selected) < int(limit):
        used = set(selected)
        rest = [name for name in pos[pos_take:] + neg[neg_take:] if name not in used]
        selected.extend(rest[: int(limit) - len(selected)])
    rng.shuffle(selected)
    return selected[: int(limit)]


def ahead_labels_for_file(data_dir: Path, file_name: str, normal: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly"})
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce").astype("int64")
    labels = ahead120_label(raw)
    label_df = pd.DataFrame({"Ts": raw["timestamp"].to_numpy(dtype=np.int64), "label": labels.astype(np.int8)})
    label_df = label_df.drop_duplicates("Ts", keep="first").set_index("Ts")
    aligned = label_df.reindex(normal["Ts"].to_numpy(dtype=np.int64))
    return aligned["label"].fillna(0).to_numpy(dtype=np.int8)


def pre_event_labels_for_file(data_dir: Path, file_name: str, normal: pd.DataFrame, hours: float) -> np.ndarray:
    raw = pd.read_csv(data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly"})
    anomaly = pd.to_numeric(raw["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    if not np.any(anomaly > 0):
        return np.zeros(len(normal), dtype=np.int8)
    ts = pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    first_ts = float(ts[int(np.argmax(anomaly > 0))])
    normal_ts = normal["Ts"].to_numpy(dtype=np.float64)
    delta = first_ts - normal_ts
    return ((delta > 0.0) & (delta <= float(hours) * SEC_IN_HOUR)).astype(np.int8)


def labels_for_file(data_dir: Path, file_name: str, normal: pd.DataFrame, label_mode: str) -> np.ndarray:
    if label_mode in {"anomaly", "ahead120", "pre_event"}:
        return ahead_labels_for_file(data_dir, file_name, normal)
    raise ValueError(f"Unknown label_mode: {label_mode}")


def valid_mask_for_normal_rows(data_dir: Path, file_name: str, normal: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly"})
    anomaly = pd.to_numeric(raw["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    normal_ts = pd.to_numeric(normal["Ts"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(normal_ts)
    if np.any(anomaly > 0):
        ts = pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        first_ts = float(ts[int(np.argmax(anomaly > 0))])
        valid &= normal_ts < first_ts
    return valid


def load_arrays(
    data_dir: Path,
    file_names: list[str],
    label_mode: str,
    pre_event_hours: float,
    with_label: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[tuple[str, int, int]], int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        normal, _extra = extract_for_file(data_dir, name, with_label=True)
        if len(normal):
            valid_mask = valid_mask_for_normal_rows(data_dir, name, normal)
            if not np.any(valid_mask):
                continue
            normal_valid = normal.loc[valid_mask]
            x = normal_valid[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32)
            xs.append(x)
            if with_label:
                ys.append(labels_for_file(data_dir, name, normal, label_mode)[valid_mask])
            n_rows += int(np.sum(valid_mask))
        if idx % 500 == 0:
            print(f"  [load] {idx}/{len(file_names)} files, rows={n_rows}")
    X = np.concatenate(xs, axis=0) if xs else np.empty((0, len(DEFAULT_FEATURE_LIST)), dtype=np.float32)
    y = np.concatenate(ys, axis=0) if ys else None
    return X, y, np.empty((0,), dtype=np.int64), [], n_rows


def normalize_inplace(X: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_value: float | None = None) -> np.ndarray:
    X -= mean
    X /= std
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_value is not None and float(clip_value) > 0:
        np.clip(X, -float(clip_value), float(clip_value), out=X)
    return X


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: NeuralCfg,
    thresholds: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = build_model(X.shape[1], cfg, thresholds).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.auto_pos_weight:
        pos = float(np.sum(y > 0))
        neg = float(len(y) - pos)
        pos_weight = neg / max(pos, 1.0)
    else:
        pos_weight = float(cfg.pos_weight)
    pos_weight = float(np.clip(pos_weight, 1.0, 100.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=cfg.device))
    print(f"[train-mlp] pos_weight={pos_weight:.4f}")
    rng = np.random.default_rng(cfg.seed)
    n = len(y)
    pos_idx_all = np.flatnonzero(y > 0)
    neg_idx_all = np.flatnonzero(y <= 0)
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        if cfg.balanced_epoch and len(pos_idx_all) and len(neg_idx_all):
            neg_take = min(len(neg_idx_all), len(pos_idx_all) * max(1, int(cfg.negative_ratio)))
            neg_idx = rng.choice(neg_idx_all, size=neg_take, replace=False)
            order = np.concatenate([pos_idx_all, neg_idx])
            rng.shuffle(order)
        else:
            order = rng.permutation(n)
        total_loss = 0.0
        total = 0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = torch.from_numpy(X[idx]).to(cfg.device, non_blocking=True)
            yb = torch.from_numpy(y[idx].astype(np.float32)).to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            total += len(idx)
        item = {"epoch": float(epoch), "loss": total_loss / max(total, 1), "seconds": time.time() - t0}
        history.append(item)
        print(f"[train-mlp] epoch={epoch} loss={item['loss']:.6f} time={item['seconds']:.1f}s")
    return model, history


def build_teacher_targets(X: np.ndarray, y: np.ndarray, cfg: NeuralCfg) -> tuple[np.ndarray, dict[str, float]]:
    if cfg.teacher_mode == "none":
        return y.astype(np.float32), {}
    if cfg.teacher_mode not in {"rf_hard", "rf_soft"}:
        raise ValueError(f"Unknown teacher_mode: {cfg.teacher_mode}")
    print(
        "[teacher-rf] train RandomForestClassifier("
        f"n_estimators={cfg.teacher_estimators}, n_jobs={cfg.teacher_n_jobs})"
    )
    teacher = RandomForestClassifier(
        n_estimators=int(cfg.teacher_estimators),
        random_state=2024,
        verbose=0,
        n_jobs=int(cfg.teacher_n_jobs),
    )
    t0 = time.time()
    teacher.fit(X, y.astype(np.int8))
    proba = positive_probability(teacher, X).astype(np.float32)
    if cfg.teacher_mode == "rf_hard":
        target = (proba >= float(cfg.teacher_threshold)).astype(np.float32)
    else:
        target = proba
    stats = {
        "teacher_seconds": float(time.time() - t0),
        "teacher_target_mean": float(np.mean(target)) if len(target) else 0.0,
        "teacher_target_pos": float(np.sum(target > 0.0)),
        "teacher_threshold": float(cfg.teacher_threshold),
    }
    print(
        f"[teacher-rf] mode={cfg.teacher_mode} seconds={stats['teacher_seconds']:.1f} "
        f"target_mean={stats['teacher_target_mean']:.6f} target_pos={stats['teacher_target_pos']:.0f}"
    )
    return target, stats


@torch.no_grad()
def score_array(model: nn.Module, X: np.ndarray, cfg: NeuralCfg) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(X), cfg.batch_size):
        xb = torch.from_numpy(X[start : start + cfg.batch_size]).to(cfg.device)
        logits = model(xb)
        scores.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


def true_first_ts(data_dir: Path, file_name: str) -> float | None:
    raw = pd.read_csv(data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly"})
    anomaly = pd.to_numeric(raw["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    if not np.any(anomaly > 0):
        return None
    ts = pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    return float(ts[int(np.argmax(anomaly > 0))])


def legacy_timestamp_array(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32).astype(np.int64)


def expanding_feature(frame: pd.DataFrame, col: str) -> dict[str, pd.Series]:
    series = pd.to_numeric(frame[col], errors="coerce")
    exp = series.expanding()
    f_min = exp.min()
    f_max = exp.max()
    f_std = exp.std()
    if len(f_std):
        f_std.iloc[0] = 0.0
    f_kurt = exp.kurt()
    return {
        "Min": f_min,
        "Diff": f_max - f_min,
        "Std": f_std,
        "Kurt": f_kurt,
    }


def rule_prior_scores(normal: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if len(normal):
        f = normal.copy()
        stats = {
            col: expanding_feature(f, col)
            for col in [
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
        }
        rules = np.zeros(len(f), dtype=bool)
        rules |= stats["Temp"]["Min"].lt(0).to_numpy()
        rules |= stats["Curr"]["Min"].lt(5000).to_numpy()
        rules |= stats["Temp"]["Diff"].gt(100).to_numpy()
        rules |= stats["Curr"]["Diff"].gt(6000).to_numpy()
        rules |= stats["Temp"]["Std"].gt(10).to_numpy()
        rules |= stats["Curr"]["Std"].gt(1500).to_numpy()
        rules |= stats["Temp"]["Kurt"].fillna(NA_DEFAULT).gt(500).to_numpy()

        for col in ["TxP0", "RxP0"]:
            rules |= stats[col]["Min"].lt(0).to_numpy()
            rules |= stats[col]["Diff"].gt(1000).to_numpy()
            rules |= stats[col]["Std"].gt(500).to_numpy()
        rules |= (
            stats["TxP0"]["Kurt"].fillna(NA_DEFAULT).gt(500)
            & stats["RxP0"]["Kurt"].fillna(NA_DEFAULT).gt(500)
        ).to_numpy()

        for col in ["RxP1", "RxP2", "RxP3", "RxP4"]:
            rules |= stats[col]["Min"].lt(0).to_numpy()
            rules |= stats[col]["Diff"].gt(1000).to_numpy()
            rules |= stats[col]["Std"].gt(200).to_numpy()

        for col in ["TxP1", "TxP2", "TxP3", "TxP4"]:
            rules |= stats[col]["Min"].lt(0).to_numpy()
            rules |= stats[col]["Diff"].gt(1000).to_numpy()
            rules |= stats[col]["Std"].gt(100).to_numpy()

        parts.append(
            pd.DataFrame(
                {
                    "timestamp": normal["Ts"].to_numpy(dtype=np.int64),
                    "prior_score": rules.astype(np.float32),
                }
            )
        )
    if len(extra):
        parts.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                    "prior_score": (
                        pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER
                    ).astype(np.float32),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["timestamp", "prior_score"])
    return pd.concat(parts, ignore_index=True).sort_values("timestamp")


def collect_scores(
    model: nn.Module,
    data_dir: Path,
    file_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    cfg: NeuralCfg,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    n_rows = 0
    for idx, name in enumerate(file_names, 1):
        normal, extra = extract_for_file(data_dir, name, with_label=True)
        frames = []
        if len(normal):
            X = normal[DEFAULT_FEATURE_LIST].to_numpy(dtype=np.float32)
            normalize_inplace(X, mean, std, cfg.clip_value)
            scores = score_array(model, X, cfg)
            frames.append(pd.DataFrame({"timestamp": normal["Ts"].to_numpy(dtype=np.int64), "score": scores}))
        if cfg.include_extra_rule and len(extra):
            extra_ts = pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64)
            extra_score = (pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER).astype(np.float32)
            frames.append(pd.DataFrame({"timestamp": extra_ts, "score": extra_score}))
        pred = (
            pd.concat(frames, ignore_index=True).sort_values("timestamp")
            if frames
            else pd.DataFrame(columns=["timestamp", "score"])
        )
        if cfg.prior_mode in {"rule_or", "rule_only"}:
            prior = rule_prior_scores(normal, extra)
            pred = pred.merge(prior, on="timestamp", how="outer")
            pred["score"] = pd.to_numeric(pred["score"], errors="coerce").fillna(0.0)
            pred["prior_score"] = pd.to_numeric(pred["prior_score"], errors="coerce").fillna(0.0)
            if cfg.prior_mode == "rule_only":
                pred["score"] = pred["prior_score"]
            else:
                neural = np.minimum(pred["score"].to_numpy(dtype=np.float32), float(cfg.neural_prior_cap))
                pred["score"] = np.maximum(neural, pred["prior_score"].to_numpy(dtype=np.float32))
            pred = pred[["timestamp", "score"]].sort_values("timestamp")
        if cfg.legacy_timestamp_float32 and len(pred):
            pred["timestamp"] = legacy_timestamp_array(pred["timestamp"])
            pred = pred.sort_values("timestamp")
        true_ts = true_first_ts(data_dir, name)
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
        n_rows += int(np.sum(valid_mask))
        if idx % 500 == 0:
            print(f"  [score] {idx}/{len(file_names)} files, rows={n_rows}")
    return rows, n_rows


def evaluate_scores(module_rows: list[dict], threshold: float) -> dict[str, float]:
    true_pos, pred_pos, hit = set(), set(), set()
    leads: list[float] = []
    n = 0
    for row in module_rows:
        n += 1
        name = row["file_name"]
        if row["true_label"]:
            true_pos.add(name)
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            timestamps = row["timestamps"]
            true_ts = row.get("true_ts")
            valid_mask = np.ones(len(timestamps), dtype=bool) if true_ts is None else timestamps < float(true_ts)
        idx = np.where((row["scores"] >= threshold) & valid_mask)[0]
        if len(idx) == 0:
            continue
        pred_ts = float(row["timestamps"][int(idx[0])])
        if row["true_label"] == 0:
            pred_pos.add(name)
        elif row["true_ts"] is not None and pred_ts < row["true_ts"]:
            pred_pos.add(name)
            hit.add(name)
            leads.append(abs(row["true_ts"] - pred_ts) / SEC_IN_HOUR)
    tp = len(hit)
    fp = len(pred_pos) - tp
    fn = len(true_pos) - tp
    tn = n - tp - fp - fn
    precision = tp / len(pred_pos) if pred_pos else 0.0
    recall = tp / len(true_pos) if true_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    avg_lead = float(np.mean(leads)) if leads else 0.0
    min_lead = float(np.min(leads)) if leads else 0.0
    return {
        "final_score": f1 + float(np.tanh(avg_lead)) + float(np.tanh(min_lead)) + accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "all_hit_cnt": float(tp),
        "all_predict_pos_cnt": float(len(pred_pos)),
        "all_true_pos_cnt": float(len(true_pos)),
        "avg_lead_hour": avg_lead,
        "min_lead_hour": min_lead,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "evaluated_module_cnt": float(n),
    }


def choose_threshold(module_rows: list[dict], cfg: NeuralCfg) -> tuple[float, dict[str, float]]:
    best_thr = 0.5
    best: dict[str, float] | None = None
    for thr in np.linspace(float(cfg.threshold_min), float(cfg.threshold_max), cfg.threshold_grid_size):
        metrics = evaluate_scores(module_rows, float(thr))
        if cfg.threshold_metric == "f1":
            key = (metrics["f1_score"], metrics["precision"], metrics["recall"], -metrics["all_predict_pos_cnt"])
            best_key = (-1.0, -1.0, -1.0, 0.0) if best is None else (
                best["f1_score"], best["precision"], best["recall"], -best["all_predict_pos_cnt"]
            )
        elif cfg.threshold_metric == "precision":
            key = (metrics["precision"], metrics["f1_score"], metrics["recall"], -metrics["all_predict_pos_cnt"])
            best_key = (-1.0, -1.0, -1.0, 0.0) if best is None else (
                best["precision"], best["f1_score"], best["recall"], -best["all_predict_pos_cnt"]
            )
        else:
            key = (metrics["final_score"], metrics["f1_score"], metrics["precision"], metrics["recall"])
            best_key = (-1.0, -1.0, -1.0, -1.0) if best is None else (
                best["final_score"], best["f1_score"], best["precision"], best["recall"]
            )
        if best is None or key > best_key:
            best_thr, best = float(thr), metrics
    assert best is not None
    best = dict(best)
    best["threshold"] = best_thr
    return best_thr, best


def write_predictions(module_rows: list[dict], out_dir: Path, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in module_rows:
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            true_ts = row.get("true_ts")
            valid_mask = np.ones(len(row["timestamps"]), dtype=bool) if true_ts is None else row["timestamps"] < float(true_ts)
        pred = ((row["scores"] >= threshold) & valid_mask).astype(int)
        pd.DataFrame(
            {
                "timestamp": row["timestamps"],
                "predict": pred,
                "score": row["scores"],
                "valid_for_eval": valid_mask.astype(int),
            }
        ).to_csv(
            out_dir / row["file_name"], index=False
        )


def metrics_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {str(r["Item"]): float(r["Value"]) for _, r in summary_df.iterrows()}


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    cfg: NeuralCfg,
    max_train_files: int | None,
    max_val_files: int | None,
    max_test_files: int | None,
) -> FoldSummary:
    t0 = time.time()
    set_seed(cfg.seed + fold)
    train_files, val_files, test_files = split_train_val(index_df, fold, 0.1, cfg.seed)
    train_files = limit_files_stratified(train_files, index_df, max_train_files, 10_000 + fold)
    val_files = limit_files_stratified(val_files, index_df, max_val_files, 20_000 + fold)
    test_files = limit_files_stratified(test_files, index_df, max_test_files, 30_000 + fold)

    print(f"[neural-model2] fold={fold} train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    X_train, y_train, _ts, _meta, train_rows = load_arrays(
        data_dir,
        train_files,
        cfg.label_mode,
        cfg.pre_event_hours,
        with_label=True,
    )
    y_target, teacher_stats = build_teacher_targets(X_train, y_train, cfg)
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    thresholds = None
    if cfg.model_type == "softbin_mlp":
        print(f"[softbin] compute quantile thresholds bins={cfg.num_bins}")
        thresholds = compute_bin_thresholds(X_train, mean, std, cfg)
    normalize_inplace(X_train, mean, std, cfg.clip_value)
    print(
        f"[neural-model2] X_train={X_train.shape} pos={int(y_train.sum())} "
        f"neg={int(len(y_train) - y_train.sum())}"
    )
    model, history = train_model(X_train, y_target, cfg, thresholds)
    del X_train, y_train, y_target
    gc.collect()

    print(f"[neural-model2] score val modules={len(val_files)}")
    val_rows, val_count = collect_scores(model, data_dir, val_files, mean, std, cfg)
    threshold, val_metrics = choose_threshold(val_rows, cfg)
    print(
        f"[neural-model2] val F1={val_metrics['f1_score']:.4f} "
        f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} thr={threshold:.3f}"
    )

    print(f"[neural-model2] score test modules={len(test_files)}")
    test_rows, test_count = collect_scores(model, data_dir, test_files, mean, std, cfg)
    if cfg.teacher_mode != "none":
        model_name = f"neural_model2_{cfg.model_type}_{cfg.teacher_mode}_thr{cfg.teacher_threshold:g}"
    elif cfg.label_mode == "pre_event":
        model_name = f"neural_model2_{cfg.model_type}_pre_event_{cfg.pre_event_hours:g}h"
    else:
        model_name = f"neural_model2_{cfg.model_type}_{cfg.label_mode}"
    if cfg.prior_mode != "none":
        model_name = f"{model_name}_{cfg.prior_mode}"
    run_dir = out_root / model_name / f"fold_{fold}"
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(test_rows, pred_dir, threshold)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "cfg": asdict(cfg),
            "teacher_stats": teacher_stats,
            "feature_list": DEFAULT_FEATURE_LIST,
            "thresholds": thresholds,
        },
        run_dir / "model.pt",
    )
    summary = FoldSummary(
        model=model_name,
        fold=int(fold),
        label_mode=cfg.label_mode,
        threshold=float(threshold),
        threshold_metric=cfg.threshold_metric,
        train_modules=len(train_files),
        val_modules=len(val_files),
        test_modules=len(test_files),
        train_rows=int(train_rows),
        val_rows=int(val_count),
        test_rows=int(test_count),
        feature_dim=len(DEFAULT_FEATURE_LIST),
        seconds=time.time() - t0,
        val_metrics=val_metrics,
        metrics=metrics,
        train_history=[*history, teacher_stats] if teacher_stats else history,
    )
    (run_dir / "fold_summary.json").write_text(json.dumps(asdict(summary), indent=2, default=str), encoding="utf-8")
    return summary


def aggregate(results: list[FoldSummary], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item.model,
            "fold": item.fold,
            "label_mode": item.label_mode,
            "threshold": item.threshold,
            "threshold_metric": item.threshold_metric,
            "train_modules": item.train_modules,
            "val_modules": item.val_modules,
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
    numeric = [c for c in df.columns if c not in {"model", "fold", "label_mode", "threshold_metric"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural model2-style warning model on full OFP rows.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_legacy_protocol/neural_model2_full"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--label_mode", choices=["anomaly", "ahead120", "pre_event"], default="anomaly")
    parser.add_argument("--pre_event_hours", type=float, default=1.0)
    parser.add_argument("--teacher_mode", choices=["none", "rf_hard", "rf_soft"], default="none")
    parser.add_argument("--teacher_threshold", type=float, default=0.3)
    parser.add_argument("--teacher_estimators", type=int, default=100)
    parser.add_argument("--teacher_n_jobs", type=int, default=1)
    parser.add_argument("--model_type", choices=["mlp", "softbin_mlp"], default="mlp")
    parser.add_argument("--num_bins", type=int, default=16)
    parser.add_argument("--bin_temperature", type=float, default=0.1)
    parser.add_argument("--no_bin_raw", action="store_true")
    parser.add_argument("--prior_mode", choices=["none", "rule_or", "rule_only"], default="none")
    parser.add_argument("--neural_prior_cap", type=float, default=1.0)
    parser.add_argument("--legacy_timestamp_float32", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm_type", choices=["batch", "layer", "none"], default="batch")
    parser.add_argument("--clip_value", type=float, default=20.0)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--auto_pos_weight", action="store_true")
    parser.add_argument("--balanced_epoch", action="store_true")
    parser.add_argument("--negative_ratio", type=int, default=5)
    parser.add_argument("--threshold_metric", choices=["f1", "precision", "final"], default="f1")
    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--no_extra_rule", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    cfg = NeuralCfg(
        label_mode=args.label_mode,
        pre_event_hours=args.pre_event_hours,
        teacher_mode=args.teacher_mode,
        teacher_threshold=args.teacher_threshold,
        teacher_estimators=args.teacher_estimators,
        teacher_n_jobs=args.teacher_n_jobs,
        model_type=args.model_type,
        num_bins=args.num_bins,
        bin_temperature=args.bin_temperature,
        bin_include_raw=not args.no_bin_raw,
        prior_mode=args.prior_mode,
        neural_prior_cap=args.neural_prior_cap,
        legacy_timestamp_float32=args.legacy_timestamp_float32,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        norm_type=args.norm_type,
        clip_value=args.clip_value,
        pos_weight=args.pos_weight,
        auto_pos_weight=args.auto_pos_weight,
        balanced_epoch=args.balanced_epoch,
        negative_ratio=args.negative_ratio,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_metric=args.threshold_metric,
        threshold_grid_size=args.threshold_grid_size,
        include_extra_rule=not args.no_extra_rule,
        seed=args.seed,
        device=args.device,
    )
    results = []
    for fold in args.folds:
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
            f"[done-neural-model2] fold={fold} Final={result.metrics.get('final_score', 0):.4f} "
            f"F1={result.metrics.get('f1_score', 0):.4f} "
            f"P={result.metrics.get('precision', 0):.4f} "
            f"R={result.metrics.get('recall', 0):.4f}"
        )
    aggregate(results, args.out_root)


if __name__ == "__main__":
    main()
