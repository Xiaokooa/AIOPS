"""DL training loop reusing R4 frames; same threshold + event-level eval as XGB."""
from __future__ import annotations

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.common import base_utils as base
from model.Optical_prediction_model.common import task_utils as week1
from model.Optical_prediction_model.ml import baselines as ml_baselines
from model.Optical_prediction_model.deep_learning.data import (
    FailureWindowDataset,
    NormStats,
    WindowSliceCache,
    collate,
    compute_norm_stats,
)
from model.Optical_prediction_model.deep_learning import ofp_task
from model.Optical_prediction_model.deep_learning.models import (
    ITransformerCfg,
    ITransformerClassifier,
)


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    patience: int = 6
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    threshold_metric: str = "window_f1"  # "window_f1", "ofp_final_score", or "ofp_f1_score"
    early_stop_metric: str = "window_f1"  # "window_f1" or "threshold_metric"
    ofp_threshold_grid: str = "coarse"  # "coarse" generalizes better in laptop sweeps; "fine" is diagnostic.
    ofp_score_postprocess: str = "raw"
    ofp_postprocess_window: int = 1
    export_ofp_predictions: bool = False
    use_amp: bool = False
    use_multi_horizon_aux: bool = True
    primary_horizon_hours: int = 120
    monotonic_loss_weight: float = 0.1
    aux_loss_scale: float = 1.0
    # FTEformer dynamic sensor mask tuning
    sensor_mask_mode: str = "hard"
    sensor_contrastive_weight: float = 0.15
    sensor_mask_temperature: float = 0.3


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _r4_task_cfg() -> week1.Week1TaskConfig:
    return week1.Week1TaskConfig(
        obs_minutes=1440,
        lead_minutes=60,
        pred_minutes=60,
        step_minutes=60,
        feature_profile="fast",
        train_scope="prefirst",
        eval_scope="prefirst",
        train_label_mode="first_in_future",
        eval_label_mode="first_in_future",
        train_faulty_max_windows=96,
        train_healthy_max_windows=6,
        train_faulty_build_stride_minutes=30,
        train_healthy_build_stride_minutes=360,
    )


def load_task_frames(
    task_cfg: week1.Week1TaskConfig | None = None,
    tps_lead_minutes: int = 60,
    tps_step_minutes: int = 15,
    tps_tolerance_minutes: int = 5,
    use_tps: bool = True,
    dummy_out: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build/cache train+val+test frames for an arbitrary Week1TaskConfig.

    Applies progressive TPS to train only (val/test untouched).
    """
    task_cfg = task_cfg or _r4_task_cfg()
    dummy_out = dummy_out or (PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments" / "_dl_tmp")
    cfg = ml_baselines.ExperimentConfig(task=task_cfg, output_dir=dummy_out)
    frames = ml_baselines.load_frames(cfg, force_rebuild=False)
    if not use_tps:
        return frames
    targets = week1.progressive_tps_targets(tps_lead_minutes, tps_step_minutes)
    frames["train"] = week1.apply_leadtime_tps_frame(
        frames["train"], target_minutes=targets, tolerance_minutes=tps_tolerance_minutes,
    )
    return frames


def load_r4_frames(dummy_out: Path | None = None) -> dict[str, pd.DataFrame]:
    """Backwards-compat alias for R4 task config."""
    return load_task_frames(task_cfg=_r4_task_cfg(), dummy_out=dummy_out)


def event_metrics_from_preds(
    test_meta_frame: pd.DataFrame,
    y_pred: np.ndarray,
    test_split_df: pd.DataFrame | None = None,
) -> dict:
    eval_df = test_meta_frame.copy()
    eval_df["y_true"] = eval_df["label"].astype(int).to_numpy()
    event_metrics = week1.evaluate_event_level(
        eval_df.drop(columns=["y_true"], errors="ignore"),
        eval_df["y_true"].to_numpy(),
        y_pred,
    )
    if test_split_df is None:
        test_split_df = base.split_modules_stratified()["test"]
    event_metrics.update(
        week1.evaluate_event_level_all_modules(
            eval_df.drop(columns=["y_true"], errors="ignore"),
            eval_df["y_true"].to_numpy(),
            y_pred,
            test_split_df,
        )
    )
    return event_metrics


def choose_threshold(y_true: np.ndarray, scores: np.ndarray):
    return ml_baselines.choose_threshold(y_true, scores)


def _ofp_thresholds(train_cfg: TrainCfg) -> np.ndarray:
    if train_cfg.ofp_threshold_grid == "coarse":
        return np.linspace(0.05, 0.95, 19)
    if train_cfg.ofp_threshold_grid == "fine":
        return np.linspace(0.01, 0.99, 99)
    raise ValueError(f"Unknown ofp_threshold_grid: {train_cfg.ofp_threshold_grid}")


DEFAULT_HORIZON_LOSS_WEIGHTS = {
    12: 0.2,
    16: 0.2,
    24: 0.3,
    72: 0.5,
    120: 1.0,
}


def _horizon_from_label_column(column: str) -> int | None:
    prefix = "label_ahead_"
    suffix = "h"
    if not (column.startswith(prefix) and column.endswith(suffix)):
        return None
    try:
        return int(column[len(prefix):-len(suffix)])
    except ValueError:
        return None


def _resolve_target_columns(frame: pd.DataFrame, train_cfg: TrainCfg) -> list[str]:
    if train_cfg.use_multi_horizon_aux:
        horizon_cols = [
            col for col in frame.columns
            if _horizon_from_label_column(str(col)) is not None
        ]
        if horizon_cols:
            return sorted(horizon_cols, key=lambda col: _horizon_from_label_column(str(col)) or 0)
    return ["label"]


def _primary_index(target_columns: list[str], primary_horizon_hours: int) -> int:
    primary_col = f"label_ahead_{int(primary_horizon_hours)}h"
    if primary_col in target_columns:
        return target_columns.index(primary_col)
    if "label" in target_columns:
        return target_columns.index("label")
    return len(target_columns) - 1


def _target_loss_weights(target_columns: list[str], train_cfg: TrainCfg) -> np.ndarray:
    weights = []
    for col in target_columns:
        horizon = _horizon_from_label_column(str(col))
        base_weight = DEFAULT_HORIZON_LOSS_WEIGHTS.get(horizon, 1.0)
        if horizon is not None and horizon != int(train_cfg.primary_horizon_hours):
            base_weight *= float(train_cfg.aux_loss_scale)
        weights.append(float(base_weight))
    return np.asarray(weights, dtype=np.float32)


class MultiHorizonAheadLoss(nn.Module):
    """BCE over ahead horizons plus an optional monotonic risk constraint."""

    def __init__(
        self,
        pos_weight: torch.Tensor,
        horizon_weight: torch.Tensor,
        monotonic_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.float())
        self.register_buffer("horizon_weight", horizon_weight.float())
        self.monotonic_weight = float(monotonic_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )
        loss = (bce * self.horizon_weight).mean()
        if self.monotonic_weight > 0 and logits.shape[1] > 1:
            probs = torch.sigmoid(logits)
            monotonic_violation = F.relu(probs[:, :-1] - probs[:, 1:]).mean()
            loss = loss + self.monotonic_weight * monotonic_violation
        return loss


def _primary_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: str,
    grad_clip: float = 1.0,
    primary_index: int = 0,
    scaler: object | None = None,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_n = 0
    scores_all = []
    y_all = []
    for xs, ms, ys, _meta in loader:
        xs = xs.to(device, non_blocking=True)
        ms = ms.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True).float()
        if is_train:
            optimizer.zero_grad()
        amp_enabled = bool(use_amp and str(device).startswith("cuda"))
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = _primary_logits(model(xs, ms))
                loss = loss_fn(logits, ys)
        if is_train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += float(loss.item()) * xs.size(0)
        total_n += xs.size(0)
        if logits.dim() == 1:
            primary_logits = logits
        else:
            primary_logits = logits[:, primary_index]
        if ys.dim() == 1:
            primary_targets = ys
        else:
            primary_targets = ys[:, primary_index]
        scores_all.append(torch.sigmoid(primary_logits).detach().cpu().numpy())
        y_all.append(primary_targets.detach().cpu().numpy().astype(int))
    return total_loss / max(total_n, 1), np.concatenate(scores_all), np.concatenate(y_all)


def fit_classifier(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    model_factory,  # callable: (n_sensors, obs_steps, output_dim) -> (nn.Module, cfg_dict, label)
    train_cfg: TrainCfg | None = None,
    save_filename: str = "model.pt",
    split_map: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Generic train/eval loop. `model_factory` returns (module, cfg_dict, label)."""
    train_cfg = train_cfg or TrainCfg()
    set_seed(train_cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer obs_steps from the actual feature frame (so this works for any
    # Week1TaskConfig, including R1/R2/R3/R5 with different obs windows).
    _row = frames["train"].iloc[0]
    obs_steps = int(_row["obs_end_idx"] - _row["obs_start_idx"] + 1)
    print(f"[task] inferred obs_steps={obs_steps} (~{obs_steps * 5} min at 5-min cadence)")
    slice_cache = WindowSliceCache()

    target_columns = _resolve_target_columns(frames["train"], train_cfg)
    primary_index = _primary_index(target_columns, train_cfg.primary_horizon_hours)
    primary_target = target_columns[primary_index]
    output_dim = len(target_columns)
    print(
        f"[task] targets={target_columns} primary={primary_target} "
        f"(index={primary_index}, output_dim={output_dim})"
    )

    print("[stats] computing per-sensor mean/std on train modules ...")
    norm = compute_norm_stats(frames["train"], slice_cache, obs_steps, max_modules=400)
    n_sensors = len(norm.sensors)
    print(f"        sensors={n_sensors} ({norm.sensors})")

    try:
        model, model_cfg_dict, model_label = model_factory(n_sensors, obs_steps, output_dim)
    except TypeError as exc:
        if output_dim != 1:
            raise TypeError(
                "model_factory must accept output_dim when multi-horizon targets are enabled"
            ) from exc
        model, model_cfg_dict, model_label = model_factory(n_sensors, obs_steps)

    ds_train = FailureWindowDataset(frames["train"], obs_steps, slice_cache, norm, target_columns=target_columns)
    ds_val = FailureWindowDataset(frames["val"], obs_steps, slice_cache, norm, target_columns=target_columns)
    ds_test = FailureWindowDataset(frames["test"], obs_steps, slice_cache, norm, target_columns=target_columns)

    print(f"[ds] train={len(ds_train)} val={len(ds_val)} test={len(ds_test)} obs_steps={obs_steps}")

    dl_kw = dict(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
                 collate_fn=collate, pin_memory=(train_cfg.device == "cuda"))
    train_loader = DataLoader(ds_train, shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(ds_val, shuffle=False, drop_last=False, **dl_kw)
    test_loader = DataLoader(ds_test, shuffle=False, drop_last=False, **dl_kw)

    device = train_cfg.device
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model_label} params={n_params/1e6:.2f}M, device={device}")

    # Class imbalance is estimated per horizon; scores/thresholds still use
    # the primary 120-hour OFP label by default.
    y_train_all = frames["train"][target_columns].to_numpy(dtype=np.float32)
    n_pos = y_train_all.sum(axis=0)
    n_neg = y_train_all.shape[0] - n_pos
    pos_weight = np.maximum(1.0, n_neg / np.maximum(n_pos, 1.0)).astype(np.float32)
    horizon_weight = _target_loss_weights(target_columns, train_cfg)
    loss_fn = MultiHorizonAheadLoss(
        pos_weight=torch.tensor(pos_weight, device=device),
        horizon_weight=torch.tensor(horizon_weight, device=device),
        monotonic_weight=train_cfg.monotonic_loss_weight if output_dim > 1 else 0.0,
    )
    loss_parts = [
        f"{col}:pos={int(n_pos[i])},neg={int(n_neg[i])},pos_w={pos_weight[i]:.2f},loss_w={horizon_weight[i]:.2f}"
        for i, col in enumerate(target_columns)
    ]
    print(f"[loss] multi-horizon BCE ({'; '.join(loss_parts)})")
    if output_dim > 1:
        print(f"[loss] monotonic_weight={train_cfg.monotonic_loss_weight:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(train_cfg.use_amp and str(device).startswith("cuda"))
    )
    if scaler.is_enabled():
        print("[train] AMP mixed precision enabled")
    warmup_epochs = train_cfg.warmup_epochs
    total_epochs = train_cfg.epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_state = None
    best_val_f1 = -1.0
    best_selection_score = -1.0
    best_val_auc = -1.0
    history = []
    epochs_since_improve = 0

    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.time()
        tr_loss, _tr_scores, _tr_y = run_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            grad_clip=train_cfg.grad_clip,
            primary_index=primary_index,
            scaler=scaler,
            use_amp=train_cfg.use_amp,
        )
        scheduler.step()
        va_loss, va_scores, va_y = run_one_epoch(
            model,
            val_loader,
            None,
            loss_fn,
            device,
            primary_index=primary_index,
            use_amp=train_cfg.use_amp,
        )
        # Window-level metrics on val with best threshold for early stopping.
        _, thr_meta = choose_threshold(va_y, va_scores)
        try:
            from sklearn.metrics import roc_auc_score
            va_auc = float(roc_auc_score(va_y, va_scores)) if len(set(va_y)) > 1 else float("nan")
        except Exception:
            va_auc = float("nan")
        selection_score = float(thr_meta["f1"])
        selection_metric = "window_f1"
        ofp_epoch_meta = None
        use_threshold_metric_for_early_stop = (
            train_cfg.early_stop_metric == "threshold_metric"
            and train_cfg.threshold_metric in {"ofp_final_score", "ofp_f1_score"}
        )
        if use_threshold_metric_for_early_stop:
            optimize_metric = "final_score" if train_cfg.threshold_metric == "ofp_final_score" else "f1_score"
            _, ofp_epoch_meta = ofp_task.choose_threshold_by_ofp_score(
                frames["val"],
                va_scores,
                split_df=split_map.get("val") if split_map else None,
                thresholds=_ofp_thresholds(train_cfg),
                optimize_metric=optimize_metric,
                score_postprocess=train_cfg.ofp_score_postprocess,
                postprocess_window=train_cfg.ofp_postprocess_window,
            )
            selection_metric = train_cfg.threshold_metric
            selection_score = float(ofp_epoch_meta[optimize_metric])

        epoch_record = {
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "val_f1": thr_meta["f1"], "val_precision": thr_meta["precision"],
            "val_recall": thr_meta["recall"], "val_threshold": thr_meta["threshold"],
            "val_auc": va_auc, "lr": optimizer.param_groups[0]["lr"],
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            "epoch_seconds": time.time() - t0,
        }
        if ofp_epoch_meta is not None:
            epoch_record.update({
                "val_ofp_threshold": ofp_epoch_meta["threshold"],
                "val_ofp_final_score": ofp_epoch_meta["final_score"],
                "val_ofp_f1_score": ofp_epoch_meta["f1_score"],
                "val_ofp_precision": ofp_epoch_meta["precision"],
                "val_ofp_recall": ofp_epoch_meta["recall"],
                "val_ofp_accuracy": ofp_epoch_meta["accuracy"],
            })
        history.append(epoch_record)
        improved = selection_score > best_selection_score + 1e-4
        if improved:
            best_val_f1 = thr_meta["f1"]
            best_selection_score = selection_score
            best_val_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        print(f"  ep{epoch:02d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_F1={thr_meta['f1']:.3f} (thr={thr_meta['threshold']:.2f}) "
              f"select={selection_metric}:{selection_score:.3f} "
              f"AUC={va_auc:.3f} | {time.time()-t0:.1f}s"
              f"{'  *' if improved else ''}")
        if epochs_since_improve >= train_cfg.patience:
            print(f"  early-stop at ep{epoch} (no improvement for {train_cfg.patience} epochs)")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    # Re-derive threshold on val using best model.
    _, va_scores, va_y = run_one_epoch(
        model,
        val_loader,
        None,
        loss_fn,
        device,
        primary_index=primary_index,
        use_amp=train_cfg.use_amp,
    )
    if train_cfg.threshold_metric == "ofp_final_score":
        threshold, threshold_meta = ofp_task.choose_threshold_by_ofp_score(
            frames["val"],
            va_scores,
            split_df=split_map.get("val") if split_map else None,
            thresholds=_ofp_thresholds(train_cfg),
            optimize_metric="final_score",
            score_postprocess=train_cfg.ofp_score_postprocess,
            postprocess_window=train_cfg.ofp_postprocess_window,
        )
    elif train_cfg.threshold_metric == "ofp_f1_score":
        threshold, threshold_meta = ofp_task.choose_threshold_by_ofp_score(
            frames["val"],
            va_scores,
            split_df=split_map.get("val") if split_map else None,
            thresholds=_ofp_thresholds(train_cfg),
            optimize_metric="f1_score",
            score_postprocess=train_cfg.ofp_score_postprocess,
            postprocess_window=train_cfg.ofp_postprocess_window,
        )
    elif train_cfg.threshold_metric == "window_f1":
        threshold, threshold_meta = choose_threshold(va_y, va_scores)
    else:
        raise ValueError(f"Unknown threshold_metric: {train_cfg.threshold_metric}")
    _, te_scores, te_y = run_one_epoch(
        model,
        test_loader,
        None,
        loss_fn,
        device,
        primary_index=primary_index,
        use_amp=train_cfg.use_amp,
    )
    te_eval_scores = ofp_task.postprocess_scores(
        frames["test"],
        te_scores,
        mode=train_cfg.ofp_score_postprocess,
        window=train_cfg.ofp_postprocess_window,
    )
    test_pred = (te_eval_scores >= threshold).astype(int)

    window_metrics = base.evaluate_binary_scores(te_y, te_eval_scores, threshold)
    event_metrics = event_metrics_from_preds(
        frames["test"],
        test_pred,
        test_split_df=split_map.get("test") if split_map else None,
    )
    ofp_metrics = ofp_task.evaluate_ofp_scores(
        frames["test"],
        te_scores,
        threshold,
        split_df=split_map.get("test") if split_map else None,
        score_postprocess=train_cfg.ofp_score_postprocess,
        postprocess_window=train_cfg.ofp_postprocess_window,
    )

    if train_cfg.export_ofp_predictions:
        ofp_task.export_ofp_prediction_files(
            frames["test"],
            te_eval_scores,
            threshold,
            output_dir / "results" / "ofp_predict_holdout",
            source_dir=base.TRAINING_DIR,
        )

    summary = {
        "model": model_label,
        "model_cfg": model_cfg_dict,
        "train_cfg": asdict(train_cfg),
        "target_columns": target_columns,
        "primary_target": primary_target,
        "primary_index": primary_index,
        "pos_weight": pos_weight.tolist(),
        "horizon_loss_weight": horizon_weight.tolist(),
        "n_params": n_params,
        "feature_count": n_sensors,
        "threshold_selection": threshold_meta,
        "threshold_metric": train_cfg.threshold_metric,
        "threshold": threshold,
        "best_val_f1": best_val_f1,
        "best_selection_score": best_selection_score,
        "best_val_auc": best_val_auc,
        "window_metrics": {k: float(v) if isinstance(v, (int, float, np.floating))
                            else v for k, v in window_metrics.items()
                            if k not in {"y_pred", "y_score"}},
        "event_metrics": event_metrics,
        "ofp_metrics": ofp_metrics,
        "history": history,
    }

    (output_dir / "results").mkdir(exist_ok=True, parents=True)
    with open(output_dir / "results" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    (output_dir / "models").mkdir(exist_ok=True, parents=True)
    torch.save({"state_dict": best_state, "model_cfg": model_cfg_dict},
               output_dir / "models" / save_filename)
    return summary


def fit_itransformer(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    model_cfg: ITransformerCfg | None = None,
    train_cfg: TrainCfg | None = None,
    split_map: dict[str, pd.DataFrame] | None = None,
) -> dict:
    def factory(n_sensors: int, obs_steps: int, output_dim: int = 1):
        cfg = model_cfg or ITransformerCfg(seq_len=obs_steps, n_sensors=n_sensors)
        cfg.n_sensors = n_sensors
        cfg.seq_len = obs_steps
        cfg.output_dim = output_dim
        return ITransformerClassifier(cfg), asdict(cfg), "iTransformerClassifier"
    return fit_classifier(frames, output_dir, factory, train_cfg=train_cfg,
                          save_filename="itransformer.pt", split_map=split_map)


def fit_patchtst(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    model_cfg=None,
    train_cfg: TrainCfg | None = None,
    split_map: dict[str, pd.DataFrame] | None = None,
) -> dict:
    from model.Optical_prediction_model.deep_learning.patchtst_wrapper import (
        PatchTSTCfg, PatchTSTClassifier,
    )
    def factory(n_sensors: int, obs_steps: int, output_dim: int = 1):
        cfg = model_cfg or PatchTSTCfg(seq_len=obs_steps, n_sensors=n_sensors)
        cfg.n_sensors = n_sensors
        cfg.seq_len = obs_steps
        cfg.output_dim = output_dim
        return PatchTSTClassifier(cfg), asdict(cfg), "PatchTSTClassifier"
    return fit_classifier(frames, output_dir, factory, train_cfg=train_cfg,
                          save_filename="patchtst.pt", split_map=split_map)


def fit_moderntcn(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    model_cfg=None,
    train_cfg: TrainCfg | None = None,
    split_map: dict[str, pd.DataFrame] | None = None,
) -> dict:
    from model.Optical_prediction_model.deep_learning.moderntcn_wrapper import (
        ModernTCNCfg,
        ModernTCNClassifier,
    )

    def factory(n_sensors: int, obs_steps: int, output_dim: int = 1):
        cfg = model_cfg or ModernTCNCfg(seq_len=obs_steps, n_sensors=n_sensors)
        cfg.n_sensors = n_sensors
        cfg.seq_len = obs_steps
        cfg.output_dim = output_dim
        return ModernTCNClassifier(cfg), asdict(cfg), "ModernTCNClassifier"

    return fit_classifier(
        frames,
        output_dir,
        factory,
        train_cfg=train_cfg,
        save_filename="moderntcn.pt",
        split_map=split_map,
    )


def fit_fits(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    model_cfg=None,
    train_cfg: TrainCfg | None = None,
    split_map: dict[str, pd.DataFrame] | None = None,
) -> dict:
    from model.Optical_prediction_model.deep_learning.interpretable_fits import (
        InterpFITSCfg,
        InterpFITSClassifier,
    )

    def factory(n_sensors: int, obs_steps: int, output_dim: int = 1):
        cfg = model_cfg or InterpFITSCfg(seq_len=obs_steps, n_sensors=n_sensors)
        cfg.n_sensors = n_sensors
        cfg.seq_len = obs_steps
        cfg.output_dim = output_dim
        cfg.cut_freq = min(int(cfg.cut_freq), max(1, obs_steps // 2 + 1))
        return InterpFITSClassifier(cfg), asdict(cfg), "FITSClassifier"

    return fit_classifier(
        frames,
        output_dir,
        factory,
        train_cfg=train_cfg,
        save_filename="fits.pt",
        split_map=split_map,
    )
