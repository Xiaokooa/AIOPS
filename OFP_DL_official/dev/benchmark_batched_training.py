from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_formal_protocol.protocol import SENSORS

from OFP_DL_official.FTEformer.model import build_model as build_fteformer
from OFP_DL_official.PatchTST.model import build_model as build_patchtst
from OFP_DL_official.common.formal_data import FormalModuleCache, NormStats, TYPE_NAMES
from OFP_DL_official.common.index_split import files_for_index_fold, read_index
from OFP_DL_official.common.losses import FTEformerLossCfg, make_sensor_prior
from OFP_DL_official.common.trainer import (
    OfficialDeepCfg,
    build_train_loader,
    compute_norm_stats_all_rows,
    compute_type_reference_all_train_negatives,
    dataset_row_count,
    effective_pos_weight,
    train_one_epoch,
)
from OFP_DL_official.iTransformer.model import build_model as build_itransformer


BUILDERS = {
    "patchtst": build_patchtst,
    "itransformer": build_itransformer,
    "fteformer": build_fteformer,
}


def load_norm(path: Path) -> NormStats | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NormStats(
        mean=payload["mean"],
        std=payload["std"],
        rows_seen=int(payload["rows_seen"]),
        files_seen=int(payload["files_seen"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure OFP DL batched-window training throughput.")
    parser.add_argument("--model", choices=sorted(BUILDERS), default="patchtst")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_official_results/ofp_index_legacy_ts_benchmark"))
    parser.add_argument("--result_dir", type=Path, default=Path("OFP_DL_official_dev_results/throughput"))
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timestamp_mode", choices=["legacy_float32", "strict_int64"], default="legacy_float32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_cached_files", type=int, default=512)
    parser.add_argument("--log_batches", type=int, default=50)
    parser.add_argument("--reuse_norm_stats", action="store_true")
    parser.add_argument("--module_cache_dir", default="")
    parser.add_argument("--train_sampling", choices=["full", "pos_all_neg_ratio"], default="full")
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--amp", action="store_true", help="enable CUDA autocast mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="disable CUDA autocast mixed precision")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--force_tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OfficialDeepCfg(
        seq_len=args.seq_len,
        epochs=1,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_cached_files=args.max_cached_files,
        device=args.device,
        num_workers=args.num_workers,
        timestamp_mode=args.timestamp_mode,
        batched_windows=True,
        amp=bool(args.amp) and not bool(args.no_amp),
        allow_tf32=not args.no_tf32,
        log_batches=args.log_batches,
        save_checkpoint=False,
        module_cache_dir=args.module_cache_dir,
        train_sampling=args.train_sampling,
        negative_ratio=args.negative_ratio,
        pos_weight_cap=args.pos_weight_cap,
        use_tqdm=not args.no_tqdm,
        force_tqdm=args.force_tqdm,
    )
    if str(cfg.device).startswith("cuda") and cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    index_df = read_index(args.index_path)
    train_files, _test_files = files_for_index_fold(index_df, int(args.fold))
    run_dir = args.out_root / args.model / f"fold_{int(args.fold)}"
    norm_path = run_dir / "norm_stats.json"
    norm = load_norm(norm_path) if args.reuse_norm_stats else None
    if norm is None:
        print(f"[throughput] computing norm stats for model={args.model} fold={args.fold}")
        norm = compute_norm_stats_all_rows(args.data_dir, train_files, timestamp_mode=args.timestamp_mode)
    else:
        print(f"[throughput] reused norm stats: {norm_path}")
    mean, std = norm.arrays()

    model, model_cfg, use_fteformer_losses = BUILDERS[args.model](cfg.seq_len, len(SENSORS))
    if use_fteformer_losses:
        print("[throughput] computing FTEformer type reference")
        type_ref = compute_type_reference_all_train_negatives(
            args.data_dir,
            train_files,
            timestamp_mode=cfg.timestamp_mode,
        )
    else:
        type_ref = None

    cache = FormalModuleCache(
        args.data_dir,
        max_cached_files=cfg.max_cached_files,
        timestamp_mode=cfg.timestamp_mode,
        module_cache_dir=Path(cfg.module_cache_dir) if cfg.module_cache_dir else None,
    )
    dataset, loader = build_train_loader(
        train_files,
        cache=cache,
        seq_len=cfg.seq_len,
        mean=mean,
        std=std,
        type_reference=type_ref,
        cfg=cfg,
    )
    model = model.to(cfg.device)
    pos_weight = effective_pos_weight(dataset, cfg)
    fault_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sensor_prior = make_sensor_prior(TYPE_NAMES, SENSORS).to(cfg.device)
    fte_loss_cfg = FTEformerLossCfg(sensor_contrastive_weight=float(model_cfg.get("sensor_contrastive_weight", 0.03)))

    print(
        f"[throughput] model={args.model} fold={args.fold} rows={dataset_row_count(dataset)} "
        f"batches={len(loader)} batch_size={cfg.batch_size} max_batches={args.max_batches} "
        f"device={cfg.device} amp={cfg.amp} pos_weight={pos_weight.tolist()}"
    )
    started = time.time()
    history = []
    rows_seen = 0
    batches_seen = 0
    parts = {}
    for epoch in range(1, int(args.epochs) + 1):
        parts = train_one_epoch(
            model,
            loader,
            optimizer,
            fault_loss_fn,
            cfg.device,
            cfg.grad_clip,
            use_fteformer_losses,
            sensor_prior,
            fte_loss_cfg,
            log_prefix=f"[throughput {args.model} fold={args.fold} ep={epoch:02d}]",
            log_batches=args.log_batches,
            use_amp=cfg.amp,
            max_batches=args.max_batches,
            use_tqdm=cfg.use_tqdm,
            force_tqdm=cfg.force_tqdm,
        )
        history.append({"epoch": epoch, **parts})
        rows_seen += int(parts.get("rows_seen", 0))
        batches_seen += int(parts.get("batches_seen", 0))
    elapsed = time.time() - started
    rows_per_second = rows_seen / elapsed if elapsed > 0 else 0.0
    total_rows = dataset_row_count(dataset)
    estimated_epoch_seconds = total_rows / rows_per_second if rows_per_second > 0 else None
    payload = {
        "model": args.model,
        "fold": int(args.fold),
        "formal_result": False,
        "purpose": "throughput_probe_only",
        "data_dir": str(args.data_dir),
        "index_path": str(args.index_path),
        "timestamp_mode": args.timestamp_mode,
        "train_sampling": cfg.train_sampling,
        "negative_ratio": cfg.negative_ratio,
        "pos_weight_cap": cfg.pos_weight_cap,
        "pos_weight": pos_weight,
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "device": cfg.device,
        "amp": cfg.amp,
        "allow_tf32": cfg.allow_tf32,
        "total_rows": total_rows,
        "total_batches": len(loader),
        "epochs": int(args.epochs),
        "rows_seen": rows_seen,
        "batches_seen": batches_seen,
        "elapsed_seconds": elapsed,
        "rows_per_second": rows_per_second,
        "estimated_epoch_seconds": estimated_epoch_seconds,
        "estimated_epoch_hours": estimated_epoch_seconds / 3600.0 if estimated_epoch_seconds else None,
        "train_parts": parts,
        "history": history,
        "run_cfg": asdict(cfg),
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.result_dir / f"{args.model}_fold{args.fold}_bs{cfg.batch_size}_b{args.max_batches}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print(f"[throughput] wrote {out_path}")


if __name__ == "__main__":
    main()

