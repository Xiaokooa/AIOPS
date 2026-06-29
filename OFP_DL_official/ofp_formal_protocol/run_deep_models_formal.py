from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_formal_protocol.protocol import (
    DEFAULT_HORIZON_HOURS,
    SENSORS,
    ahead_labels_and_valid_mask_multi_first_event,
    choose_threshold_by_final_score,
    files_for_fold,
    primary_horizon_index,
    read_manifest,
)
from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_deep_models import (
    DeepRunCfg,
    OFPWindowDataset,
    collate,
    collect_module_scores,
    compute_norm_stats,
    make_model,
    point_metrics,
    set_seed,
    train_epoch,
    write_predictions,
)
from OFP_DL_official.common.formal_data import choose_alarm_strategy_by_final_score


HORIZON_HOURS = DEFAULT_HORIZON_HOURS
PRIMARY_HORIZON_INDEX = primary_horizon_index(HORIZON_HOURS)


def primary_labels(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 1:
        return arr.astype(np.int8, copy=False)
    return arr[:, PRIMARY_HORIZON_INDEX].astype(np.int8, copy=False)


class FormalModuleArrayCache:
    """Module cache using first-event 1-hour labels."""

    def __init__(self, data_dir: Path, max_cached_files: int = 2048) -> None:
        self.data_dir = Path(data_dir)
        self.max_cached_files = int(max_cached_files)
        self.cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if file_name in self.cache:
            item = self.cache.pop(file_name)
            self.cache[file_name] = item
            return item

        df = pd.read_csv(self.data_dir / file_name, usecols=lambda c: c in {"timestamp", "anomaly", *SENSORS})
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.int64)
        values = df[SENSORS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        anomaly = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        labels, valid_mask = ahead_labels_and_valid_mask_multi_first_event(
            timestamps.astype(float),
            anomaly,
            HORIZON_HOURS,
        )
        item = (timestamps, values, anomaly, labels, valid_mask)

        self.cache[file_name] = item
        if len(self.cache) > self.max_cached_files:
            self.cache.popitem(last=False)
        return item


def sample_training_windows(
    cache: FormalModuleArrayCache,
    file_names: list[str],
    pos_per_file: int,
    neg_per_file: int,
    seed: int,
    max_files: int | None = None,
) -> list[tuple[str, int, int]]:
    rng = np.random.default_rng(seed)
    files = file_names[: max_files if max_files is not None else None]
    samples: list[tuple[str, int, int]] = []
    for name in files:
        _timestamps, _values, _anomaly, labels, valid_mask = cache.get(name)
        primary = primary_labels(labels)
        pos_idx = np.where((primary > 0) & valid_mask)[0]
        neg_idx = np.where((primary == 0) & valid_mask)[0]
        if len(pos_idx):
            take = min(int(pos_per_file), len(pos_idx))
            chosen = rng.choice(pos_idx, size=take, replace=False)
            samples.extend((name, int(i), tuple(labels[int(i)].astype(float).tolist())) for i in chosen)
        if len(neg_idx):
            take = min(int(neg_per_file), len(neg_idx))
            chosen = rng.choice(neg_idx, size=take, replace=False)
            samples.extend((name, int(i), tuple(labels[int(i)].astype(float).tolist())) for i in chosen)
    rng.shuffle(samples)
    return samples


def run_model_fold(
    model_name: str,
    fold: int,
    data_dir: Path,
    manifest_df: pd.DataFrame,
    out_root: Path,
    cfg: DeepRunCfg,
    max_train_files: int | None,
    max_val_files: int | None,
    max_test_files: int | None,
) -> dict:
    t0 = time.time()
    set_seed(cfg.seed + int(fold))
    train_files, val_files, test_files = files_for_fold(manifest_df, fold)
    if max_val_files is not None:
        val_files = val_files[: int(max_val_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]

    run_dir = out_root / model_name / f"fold_{fold}"
    cache = FormalModuleArrayCache(data_dir, max_cached_files=cfg.max_cached_files)
    mean, std = compute_norm_stats(data_dir, train_files)
    samples = sample_training_windows(
        cache,
        train_files,
        cfg.train_pos_per_file,
        cfg.train_neg_per_file,
        seed=cfg.seed + int(fold),
        max_files=max_train_files,
    )
    sample_labels = np.asarray([label for *_rest, label in samples], dtype=np.float32) if samples else np.empty((0, len(HORIZON_HOURS)), dtype=np.float32)
    primary_sample_labels = sample_labels[:, PRIMARY_HORIZON_INDEX] if sample_labels.ndim == 2 and len(sample_labels) else np.empty((0,), dtype=np.float32)
    pos = int(primary_sample_labels.sum())
    neg = int(len(samples) - pos)
    if not samples:
        raise ValueError("No training windows sampled; check manifest and data_dir")

    dataset = OFPWindowDataset(samples, cache, cfg.seq_len, mean, std)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    model, model_cfg = make_model(model_name, cfg.seq_len, len(SENSORS))
    model = model.to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    pos_by_horizon = sample_labels.sum(axis=0) if len(sample_labels) else np.ones(len(HORIZON_HOURS), dtype=np.float32)
    neg_by_horizon = len(sample_labels) - pos_by_horizon
    pos_weight = np.maximum(1.0, neg_by_horizon / np.maximum(pos_by_horizon, 1.0)).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state, best_train_f1, best_epoch = None, -1.0, 0
    stale = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        ep0 = time.time()
        train_loss, train_scores, train_y = train_epoch(model, loader, optimizer, loss_fn, cfg.device, cfg.grad_clip)
        train_metrics = point_metrics(train_y, train_scores, threshold=0.5)
        improved = train_metrics["f1"] > best_train_f1 + 1e-5
        if improved:
            best_train_f1 = train_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "seconds": time.time() - ep0,
            }
        )
        print(
            f"[formal {model_name} fold={fold}] ep{epoch:02d} "
            f"loss={train_loss:.4f} train_F1={train_metrics['f1']:.4f}"
        )
        if stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[formal {model_name} fold={fold}] scoring val modules={len(val_files)}")
    val_rows = collect_module_scores(model, data_dir, val_files, cache, cfg, mean, std)
    threshold, alarm_strategy, val_module_metrics = choose_alarm_strategy_by_final_score(
        val_rows,
        grid_size=cfg.threshold_grid_size,
        smoothing=cfg.alarm_smoothing,
        smooth_windows=cfg.alarm_search_smooth_windows,
        consecutive_ks=cfg.alarm_search_consecutive_ks,
    )
    print(
        f"[formal {model_name} fold={fold}] val final={val_module_metrics['final_score']:.4f} "
        f"F1={val_module_metrics['f1_score']:.4f} thr={threshold:.3f} alarm={alarm_strategy}"
    )

    print(f"[formal {model_name} fold={fold}] scoring test modules={len(test_files)}")
    test_rows = collect_module_scores(model, data_dir, test_files, cache, cfg, mean, std)
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    write_predictions(test_rows, pred_dir, threshold, cfg=cfg, alarm_strategy=alarm_strategy)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    test_metrics = {str(r["Item"]): float(r["Value"]) for _, r in summary_df.iterrows()}

    result = {
        "model": model_name,
        "fold": int(fold),
        "threshold": threshold,
        "alarm_strategy": alarm_strategy,
        "first_alarm_only": bool(cfg.first_alarm_only),
        "best_train_f1": best_train_f1,
        "best_epoch": best_epoch,
        "train_modules": len(train_files),
        "val_modules": len(val_files),
        "test_modules": len(test_files),
        "train_samples": len(samples),
        "train_pos_samples": int(pos),
        "train_neg_samples": int(neg),
        "train_pos_samples_by_horizon": pos_by_horizon.astype(float).tolist(),
        "train_neg_samples_by_horizon": neg_by_horizon.astype(float).tolist(),
        "horizon_hours": list(HORIZON_HOURS),
        "primary_horizon_index": int(PRIMARY_HORIZON_INDEX),
        "pos_weight": pos_weight.astype(float).tolist(),
        "n_params": int(n_params),
        "seconds": time.time() - t0,
        "model_cfg": model_cfg,
        "run_cfg": asdict(cfg),
        "val_module_metrics": val_module_metrics,
        "metrics": test_metrics,
        "history": history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "run_cfg": asdict(cfg)}, run_dir / "model.pt")
    return result


def aggregate(results: list[dict], out_root: Path) -> None:
    rows = []
    for item in results:
        row = {
            "model": item["model"],
            "fold": item["fold"],
            "threshold": item["threshold"],
            "alarm_smoothing": item.get("alarm_strategy", {}).get("alarm_smoothing", ""),
            "alarm_smooth_window": item.get("alarm_strategy", {}).get("alarm_smooth_window", 1),
            "alarm_consecutive_k": item.get("alarm_strategy", {}).get("alarm_consecutive_k", 1),
            "first_alarm_only": item.get("first_alarm_only", False),
            "train_modules": item["train_modules"],
            "val_modules": item["val_modules"],
            "test_modules": item["test_modules"],
            "train_samples": item["train_samples"],
            "train_pos_samples": item["train_pos_samples"],
            "train_neg_samples": item["train_neg_samples"],
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
    numeric = [c for c in df.columns if c not in {"model", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deep models under the confirmed formal OFP protocol.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--manifest_path", type=Path, default=Path("output/ofp_formal_protocol/splits/formal_manifest.csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_formal_protocol/deep_models"))
    parser.add_argument("--models", nargs="+", default=["itransformer", "patchtst", "moderntcn", "fits", "fteformer"],
                        choices=["itransformer", "patchtst", "moderntcn", "fits", "fteformer"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train_pos_per_file", type=int, default=96)
    parser.add_argument("--train_neg_per_file", type=int, default=24)
    parser.add_argument("--threshold_grid_size", type=int, default=99)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--alarm_search_smooth_windows", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--alarm_search_consecutive_ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no_first_alarm_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = read_manifest(args.manifest_path)
    cfg = DeepRunCfg(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        train_pos_per_file=args.train_pos_per_file,
        train_neg_per_file=args.train_neg_per_file,
        threshold_grid_size=args.threshold_grid_size,
        seed=args.seed,
        device=args.device,
        max_cached_files=args.max_cached_files,
        alarm_smoothing=args.alarm_smoothing,
        alarm_smooth_window=args.alarm_smooth_window,
        alarm_consecutive_k=args.alarm_consecutive_k,
        alarm_search_smooth_windows=tuple(int(x) for x in args.alarm_search_smooth_windows),
        alarm_search_consecutive_ks=tuple(int(x) for x in args.alarm_search_consecutive_ks),
        first_alarm_only=not args.no_first_alarm_only,
    )
    all_results: list[dict] = []
    for model_name in args.models:
        for fold in args.folds:
            print(f"[run formal] model={model_name} fold={fold} device={cfg.device}")
            result = run_model_fold(
                model_name,
                fold,
                args.data_dir,
                manifest_df,
                args.out_root,
                cfg,
                args.max_train_files,
                args.max_val_files,
                args.max_test_files,
            )
            all_results.append(result)
            aggregate([result], args.out_root)
            print(
                f"[done formal] {model_name} fold={fold} "
                f"final={result['metrics'].get('final_score', 0):.4f} "
                f"F1={result['metrics'].get('f1_score', 0):.4f} "
                f"P={result['metrics'].get('precision', 0):.4f} "
                f"R={result['metrics'].get('recall', 0):.4f}"
            )
    aggregate(all_results, args.out_root)


if __name__ == "__main__":
    main()


