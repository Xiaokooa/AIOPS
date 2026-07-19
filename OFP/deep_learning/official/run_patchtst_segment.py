from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.official.ofp_formal_protocol.protocol import SENSORS
from OFP.deep_learning.official.ofp_protocol.evaluator import evaluate_prediction_folder

from OFP.deep_learning.official.PatchTST.segment_model import build_segment_model
from OFP.deep_learning.official.common.formal_data import (
    FormalModuleCache,
    HORIZON_HOURS,
    PRIMARY_HORIZON_INDEX,
    assert_prediction_coverage,
    compute_norm_stats_all_rows,
    set_seed,
    write_predictions,
)
from OFP.deep_learning.official.common.index_split import files_for_index_fold, read_index
from OFP.deep_learning.official.common.segment_data import (
    SegmentForecastDataset,
    collect_segment_module_scores,
    make_segment_loader,
)
from OFP.deep_learning.official.common.trainer import (
    cuda_autocast,
    cuda_grad_scaler,
    format_metric_summary,
    metrics_to_dict,
    resolve_runtime_device,
    should_use_tqdm,
)


def runtime_memory_summary(device: str) -> str:
    pieces: list[str] = []
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        pieces.append(f"rss_gb={mem.rss / (1024 ** 3):.2f}")
        pieces.append(f"vms_gb={mem.vms / (1024 ** 3):.2f}")
    except Exception:
        pass
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            pieces.append(f"cuda_alloc_gb={allocated:.2f}")
            pieces.append(f"cuda_reserved_gb={reserved:.2f}")
        except Exception:
            pass
    return " ".join(pieces)


@dataclass
class SegmentCfg:
    input_len: int = 288
    pred_len: int = 12
    train_stride: int = 12
    test_stride: int = 12
    epochs: int = 3
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    negative_ratio: float = 10.0
    pos_weight_cap: float = 5.0
    fixed_threshold: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_cached_files: int = 512
    num_workers: int = 0
    timestamp_mode: str = "legacy_float32"
    module_cache_dir: str = ""
    amp: bool = False
    allow_tf32: bool = True
    log_batches: int = 50
    use_tqdm: bool = True
    force_tqdm: bool = False
    aggregation: str = "max"
    save_checkpoint: bool = True
    shuffle_segments: bool = True
    train_score_diag_batches: int = 20
    alarm_smoothing: str = "ema"
    alarm_smooth_window: int = 1
    alarm_consecutive_k: int = 1
    first_alarm_only: bool = True


def configure_torch(cfg: SegmentCfg) -> None:
    cfg.device = resolve_runtime_device(cfg.device)
    if str(cfg.device).startswith("cuda") and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def effective_pos_weight(pos_rows: int | np.ndarray, neg_rows: int | np.ndarray, cfg: SegmentCfg) -> np.ndarray:
    pos = np.asarray(pos_rows, dtype=np.float64)
    neg = np.asarray(neg_rows, dtype=np.float64)
    value = np.maximum(1.0, neg / np.maximum(pos, 1.0))
    if float(cfg.pos_weight_cap) > 0:
        value = np.minimum(value, float(cfg.pos_weight_cap))
    return value.astype(np.float32)


def train_segment_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    cfg: SegmentCfg,
    epoch: int,
    max_batches: int = 0,
) -> dict[str, float]:
    model.train()
    started = time.time()
    total_loss = 0.0
    target_rows = 0
    batch_idx = 0
    total_batches = len(loader)
    display_batches = min(total_batches, int(max_batches)) if int(max_batches) > 0 else total_batches
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
    scaler = cuda_grad_scaler(amp_enabled)
    iterator = enumerate(loader, 1)
    progress_bar = None
    desc = f"[segment patchtst ep={epoch:02d}]"
    if should_use_tqdm(cfg.use_tqdm, cfg.force_tqdm):
        from tqdm.auto import tqdm

        progress_bar = tqdm(
            iterator,
            total=display_batches,
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
            file=sys.stdout,
            ascii=True,
        )
        iterator = progress_bar

    for batch_idx, (xs, masks, ys, ymask) in iterator:
        xs = xs.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        ys = ys.to(cfg.device, non_blocking=True).float()
        ymask = ymask.to(cfg.device, non_blocking=True).float()
        if not torch.isfinite(xs).all() or not torch.isfinite(masks).all():
            raise FloatingPointError(f"non-finite segment input at epoch={epoch} batch={batch_idx}")
        if not torch.isfinite(ys).all() or not torch.isfinite(ymask).all():
            raise FloatingPointError(f"non-finite segment labels at epoch={epoch} batch={batch_idx}")

        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(amp_enabled):
            logits = model(xs, masks)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            if logits.shape != ys.shape:
                raise ValueError(
                    f"segment logits shape {tuple(logits.shape)} does not match labels {tuple(ys.shape)}"
                )
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite segment logits at epoch={epoch} batch={batch_idx}")
            raw_loss = loss_fn(logits, ys)
            denom = ymask.sum().clamp(min=1.0)
            loss = (raw_loss * ymask).sum() / denom
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite segment loss at epoch={epoch} batch={batch_idx}")

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {name} at epoch={epoch} batch={batch_idx}")
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {name} at epoch={epoch} batch={batch_idx}")
            optimizer.step()

        valid_rows = int(ymask.sum().detach().cpu())
        target_rows += valid_rows
        total_loss += float(loss.detach().cpu()) * valid_rows
        if int(cfg.log_batches) > 0 and (batch_idx % int(cfg.log_batches) == 0 or batch_idx == display_batches):
            avg_loss = total_loss / max(target_rows, 1)
            elapsed_min = (time.time() - started) / 60.0
            if progress_bar is not None:
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}", rows=target_rows, min=f"{elapsed_min:.1f}")
            else:
                print(
                    f"{desc} batch={batch_idx}/{display_batches} target_rows={target_rows} "
                    f"loss={avg_loss:.4f} elapsed_min={elapsed_min:.1f} "
                    f"{runtime_memory_summary(cfg.device)}",
                    flush=True,
                )
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
    if progress_bar is not None:
        progress_bar.close()

    elapsed = time.time() - started
    avg_loss = total_loss / max(target_rows, 1)
    rows_per_second = float(target_rows) / elapsed if elapsed > 0 else 0.0
    batches_per_second = float(batch_idx) / elapsed if elapsed > 0 else 0.0
    print(
        f"{desc} done loss={avg_loss:.6f} target_rows={target_rows} "
        f"batches={batch_idx}/{display_batches} elapsed_min={elapsed/60.0:.2f} "
        f"rows_per_sec={rows_per_second:.2f} batches_per_sec={batches_per_second:.3f} "
        f"{runtime_memory_summary(cfg.device)}",
        flush=True,
    )
    return {
        "loss": float(avg_loss),
        "target_rows_seen": float(target_rows),
        "batches_seen": float(batch_idx),
        "elapsed_seconds": float(elapsed),
        "rows_per_second": rows_per_second,
        "batches_per_second": batches_per_second,
    }


@torch.no_grad()
def diagnostic_train_scores(
    model: nn.Module,
    loader,
    cfg: SegmentCfg,
    max_batches: int = 20,
) -> dict[str, object]:
    model.eval()
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    amp_enabled = bool(cfg.amp) and str(cfg.device).startswith("cuda")
    for batch_idx, (xs, masks, ys, ymask) in enumerate(loader, 1):
        xs = xs.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        ys = ys.to(cfg.device, non_blocking=True).float()
        ymask = ymask.to(cfg.device, non_blocking=True).float()
        with cuda_autocast(amp_enabled):
            logits = model(xs, masks)
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        scores = torch.sigmoid(logits)
        if scores.ndim == 3:
            scores = scores[:, :, PRIMARY_HORIZON_INDEX]
            ys = ys[:, :, PRIMARY_HORIZON_INDEX]
            ymask = ymask[:, :, PRIMARY_HORIZON_INDEX]
        valid = ymask > 0
        pos = (ys > 0) & valid
        neg = (ys <= 0) & valid
        if pos.any():
            pos_scores.extend(scores[pos].detach().cpu().numpy().astype(float).tolist())
        if neg.any():
            neg_scores.extend(scores[neg].detach().cpu().numpy().astype(float).tolist())
        if batch_idx >= int(max_batches):
            break

    def summarize(values: list[float]) -> dict[str, object]:
        if not values:
            return {"count": 0}
        arr = torch.tensor(values, dtype=torch.float64).numpy()
        qs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        return {
            "count": int(len(values)),
            "mean": float(arr.mean()),
            "quantiles": {str(q): float(v) for q, v in zip(qs, np.quantile(arr, qs))},
            "ge_0p5": int((arr >= 0.5).sum()),
        }

    return {"pos": summarize(pos_scores), "neg": summarize(neg_scores), "max_batches": int(max_batches)}


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    cfg: SegmentCfg,
    smoke_max_batches: int = 0,
    max_train_files: int = 0,
    max_test_files: int = 0,
    skip_eval: bool = False,
) -> dict:
    if cfg.train_stride > cfg.pred_len:
        raise ValueError("train_stride must be <= pred_len for timestamp coverage")
    configure_torch(cfg)
    set_seed(cfg.seed + int(fold))
    t0 = time.time()
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    if int(max_train_files) > 0:
        train_files = train_files[: int(max_train_files)]
    if int(max_test_files) > 0:
        test_files = test_files[: int(max_test_files)]

    run_dir = Path(out_root) / "patchtst_segment" / f"fold_{int(fold)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = FormalModuleCache(
        data_dir,
        max_cached_files=cfg.max_cached_files,
        timestamp_mode=cfg.timestamp_mode,
        module_cache_dir=Path(cfg.module_cache_dir) if cfg.module_cache_dir else None,
    )

    print(f"[segment fold={fold}] computing train-only normalization files={len(train_files)}", flush=True)
    norm = compute_norm_stats_all_rows(data_dir, train_files, timestamp_mode=cfg.timestamp_mode)
    mean, std = norm.arrays()
    (run_dir / "norm_stats.json").write_text(json.dumps(asdict(norm), indent=2), encoding="utf-8")

    dataset = SegmentForecastDataset(
        train_files,
        cache=cache,
        input_len=cfg.input_len,
        pred_len=cfg.pred_len,
        train_stride=cfg.train_stride,
        mean=mean,
        std=std,
        batch_segments=cfg.batch_size,
        negative_ratio=cfg.negative_ratio,
        seed=cfg.seed + int(fold),
        shuffle_segments=cfg.shuffle_segments,
    )
    loader = make_segment_loader(dataset, num_workers=cfg.num_workers)
    model, model_cfg = build_segment_model(cfg.input_len, cfg.pred_len, len(SENSORS))
    model = model.to(cfg.device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    pos_weight = effective_pos_weight(dataset.pos_rows_by_horizon, dataset.neg_rows_by_horizon, cfg)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight, device=cfg.device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(
        f"[segment fold={fold}] train_segments={dataset.total_segments} batches={len(loader)} "
        f"target_rows={dataset.total_target_rows} pos_rows={dataset.pos_rows} neg_rows={dataset.neg_rows} "
        f"pos_weight={pos_weight.tolist()} horizons={list(HORIZON_HOURS)} "
        f"primary_horizon={HORIZON_HOURS[PRIMARY_HORIZON_INDEX]} amp={cfg.amp}",
        flush=True,
    )

    history: list[dict] = []
    for epoch in range(1, int(cfg.epochs) + 1):
        parts = train_segment_one_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            cfg,
            epoch,
            max_batches=smoke_max_batches,
        )
        history.append({"epoch": epoch, **parts})
        if int(smoke_max_batches) > 0:
            break

    train_score_diag = diagnostic_train_scores(
        model,
        loader,
        cfg,
        max_batches=max(1, int(cfg.train_score_diag_batches)),
    )
    print(
        f"[segment fold={fold}] train score diag "
        f"pos_mean={train_score_diag.get('pos', {}).get('mean', 'NA')} "
        f"neg_mean={train_score_diag.get('neg', {}).get('mean', 'NA')}",
        flush=True,
    )

    if int(smoke_max_batches) > 0 or bool(skip_eval):
        result = {
            "model": "patchtst_segment",
            "fold": int(fold),
            "formal_result": False,
            "purpose": "smoke_train_only" if int(smoke_max_batches) > 0 else "train_only_skip_eval",
            "protocol": "ofp_index_segment_forecasting_24h_to_1h",
            "input_len": cfg.input_len,
            "pred_len": cfg.pred_len,
            "train_stride": cfg.train_stride,
            "test_stride": cfg.test_stride,
            "train_modules": len(train_files),
            "test_modules": len(test_files),
            "train_segments": dataset.total_segments,
            "train_pos_segments": dataset.pos_segments,
            "train_neg_segments": dataset.neg_segments,
            "train_target_rows": dataset.total_target_rows,
            "train_pos_rows": dataset.pos_rows,
            "train_neg_rows": dataset.neg_rows,
            "train_pos_rows_by_horizon": dataset.pos_rows_by_horizon.astype(int).tolist(),
            "train_neg_rows_by_horizon": dataset.neg_rows_by_horizon.astype(int).tolist(),
            "pos_weight": pos_weight.astype(float).tolist(),
            "horizon_hours": list(HORIZON_HOURS),
            "primary_horizon_index": int(PRIMARY_HORIZON_INDEX),
            "n_params": n_params,
            "shuffle_segments": cfg.shuffle_segments,
            "seconds": time.time() - t0,
            "run_cfg": asdict(cfg),
            "model_cfg": model_cfg,
            "train_score_diag": train_score_diag,
            "history": history,
        }
        (run_dir / "segment_smoke_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    print(f"[segment fold={fold}] scoring full test modules={len(test_files)}", flush=True)
    test_rows = collect_segment_module_scores(
        model,
        cache,
        test_files,
        cfg.input_len,
        cfg.pred_len,
        cfg.test_stride,
        mean,
        std,
        cfg.batch_size,
        cfg.device,
        use_amp=cfg.amp,
        aggregation=cfg.aggregation,
        use_tqdm=cfg.use_tqdm,
        force_tqdm=cfg.force_tqdm,
        progress_desc=f"[segment fold={fold} test]",
    )
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(
        test_rows,
        pred_dir,
        cfg.fixed_threshold,
        alarm_smoothing=cfg.alarm_smoothing,
        alarm_smooth_window=cfg.alarm_smooth_window,
        alarm_consecutive_k=cfg.alarm_consecutive_k,
        first_alarm_only=cfg.first_alarm_only,
    )
    assert_prediction_coverage(pred_dir, test_files)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    metrics = metrics_to_dict(summary_df)
    print(
        f"[segment-final] fold={fold} threshold={cfg.fixed_threshold:.6g} "
        f"epochs={cfg.epochs} elapsed_min={(time.time() - t0) / 60.0:.2f} "
        f"{format_metric_summary(metrics)}",
        flush=True,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)

    result = {
        "model": "patchtst_segment",
        "fold": int(fold),
        "paper_ready": True,
        "protocol": "ofp_index_segment_forecasting_24h_to_1h",
        "threshold": float(cfg.fixed_threshold),
        "alarm_strategy": {
            "alarm_smoothing": cfg.alarm_smoothing,
            "alarm_smooth_window": int(cfg.alarm_smooth_window),
            "alarm_consecutive_k": int(cfg.alarm_consecutive_k),
        },
        "first_alarm_only": bool(cfg.first_alarm_only),
        "input_len": cfg.input_len,
        "pred_len": cfg.pred_len,
        "train_stride": cfg.train_stride,
        "test_stride": cfg.test_stride,
        "aggregation": cfg.aggregation,
        "train_modules": len(train_files),
        "test_modules": len(test_files),
        "train_segments": dataset.total_segments,
        "train_pos_segments": dataset.pos_segments,
        "train_neg_segments": dataset.neg_segments,
        "train_target_rows": dataset.total_target_rows,
        "train_pos_rows": dataset.pos_rows,
        "train_neg_rows": dataset.neg_rows,
        "train_pos_rows_by_horizon": dataset.pos_rows_by_horizon.astype(int).tolist(),
        "train_neg_rows_by_horizon": dataset.neg_rows_by_horizon.astype(int).tolist(),
        "pos_weight": pos_weight.astype(float).tolist(),
        "horizon_hours": list(HORIZON_HOURS),
        "primary_horizon_index": int(PRIMARY_HORIZON_INDEX),
        "pos_weight_cap": float(cfg.pos_weight_cap),
        "n_params": n_params,
        "shuffle_segments": cfg.shuffle_segments,
        "seconds": time.time() - t0,
        "run_cfg": asdict(cfg),
        "model_cfg": model_cfg,
        "train_score_diag": train_score_diag,
        "metrics": metrics,
        "history": history,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    if cfg.save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "run_cfg": asdict(cfg),
                "threshold": cfg.fixed_threshold,
                "alarm_strategy": {
                    "alarm_smoothing": cfg.alarm_smoothing,
                    "alarm_smooth_window": int(cfg.alarm_smooth_window),
                    "alarm_consecutive_k": int(cfg.alarm_consecutive_k),
                },
                "first_alarm_only": bool(cfg.first_alarm_only),
            },
            run_dir / "model.pt",
        )
    return result


def aggregate(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        if "metrics" not in item:
            continue
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "paper_ready": item.get("paper_ready", False),
            "protocol": item["protocol"],
            "train_segments": item["train_segments"],
            "train_target_rows": item["train_target_rows"],
            "seconds": item["seconds"],
        }
        row.update(item["metrics"])
        rows.append(row)
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    metrics_path = out_root / "patchtst_segment_fold_metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        df_new = pd.concat([old, df_new], ignore_index=True)
        df_new.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df_new.sort_values(["model", "fold"], inplace=True)
    df_new.to_csv(metrics_path, index=False)
    numeric = [
        col
        for col in df_new.columns
        if col not in {"model", "paper_ready", "protocol"} and pd.api.types.is_numeric_dtype(df_new[col])
    ]
    summary = df_new.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    summary.reset_index().to_csv(out_root / "patchtst_segment_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PatchTST segment forecasting under OFP index protocol.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_official_results/ofp_index_segment_patchtst"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--input_len", type=int, default=288)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--train_stride", type=int, default=12)
    parser.add_argument("--test_stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=5.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--timestamp_mode", choices=["legacy_float32", "strict_int64"], default="legacy_float32")
    parser.add_argument("--module_cache_dir", default="")
    parser.add_argument("--amp", action="store_true", help="enable CUDA autocast mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="disable CUDA autocast mixed precision")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--log_batches", type=int, default=50)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--force_tqdm", action="store_true")
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--no_save_checkpoint", action="store_true")
    parser.add_argument("--no_shuffle_segments", action="store_true")
    parser.add_argument("--train_score_diag_batches", type=int, default=20)
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--smoke_max_batches", type=int, default=0)
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_test_files", type=int, default=0)
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> SegmentCfg:
    return SegmentCfg(
        input_len=args.input_len,
        pred_len=args.pred_len,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        fixed_threshold=args.fixed_threshold,
        seed=args.seed,
        device=args.device,
        max_cached_files=args.max_cached_files,
        num_workers=args.num_workers,
        timestamp_mode=args.timestamp_mode,
        module_cache_dir=args.module_cache_dir,
        amp=bool(args.amp) and not bool(args.no_amp),
        allow_tf32=not args.no_tf32,
        log_batches=args.log_batches,
        use_tqdm=not args.no_tqdm,
        force_tqdm=args.force_tqdm,
        aggregation=args.aggregation,
        save_checkpoint=not args.no_save_checkpoint,
        shuffle_segments=not args.no_shuffle_segments,
        train_score_diag_batches=args.train_score_diag_batches,
        alarm_smoothing=args.alarm_smoothing,
        alarm_smooth_window=args.alarm_smooth_window,
        alarm_consecutive_k=args.alarm_consecutive_k,
        first_alarm_only=not args.no_first_alarm_only,
    )


def main() -> None:
    args = parse_args()
    cfg = cfg_from_args(args)
    index_df = read_index(args.index_path)
    results = []
    for fold in args.folds:
        result = run_fold(
            int(fold),
            args.data_dir,
            index_df,
            args.out_root,
            cfg,
            smoke_max_batches=args.smoke_max_batches,
            max_train_files=args.max_train_files,
            max_test_files=args.max_test_files,
            skip_eval=args.skip_eval,
        )
        results.append(result)
        aggregate([result], args.out_root)
        if "metrics" in result:
            print(
                f"[segment done] fold={fold} final={result['metrics'].get('final_score', 0):.4f} "
                f"f1={result['metrics'].get('f1_score', 0):.4f}",
                flush=True,
            )
        else:
            print(f"[segment done] fold={fold} smoke/train-only complete", flush=True)


if __name__ == "__main__":
    main()
