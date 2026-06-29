from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_MODELS = ["rf", "xgboost", "patchtst", "itransformer", "fteformer"]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tail_text(path: Path, n_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n_lines:]
    except OSError:
        return []


def run_one(args: argparse.Namespace, model: str, fold: int, log_dir: Path, progress_path: Path) -> int:
    log_path = log_dir / f"{model}_fold{fold}.log"
    command = [
        sys.executable,
        "-u",
        "-B",
        "OFP_DL_official/run_official_benchmark.py",
        "--models",
        model,
        "--folds",
        str(fold),
        "--data_dir",
        str(args.data_dir),
        "--index_path",
        str(args.index_path),
        "--out_root",
        str(args.out_root),
        "--n_jobs",
        str(args.n_jobs),
        "--rf_estimators",
        str(args.rf_estimators),
        "--xgb_estimators",
        str(args.xgb_estimators),
        "--rf_threshold",
        str(args.rf_threshold),
        "--xgb_threshold",
        str(args.xgb_threshold),
        "--fixed_threshold",
        str(args.fixed_threshold),
        "--timestamp_mode",
        str(args.timestamp_mode),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--seq_len",
        str(args.seq_len),
        "--device",
        str(args.device),
        "--num_workers",
        str(args.num_workers),
        "--log_batches",
        str(args.log_batches),
        "--train_sampling",
        str(args.train_sampling),
        "--negative_ratio",
        str(args.negative_ratio),
        "--pos_weight_cap",
        str(args.pos_weight_cap),
        "--aux_bce_weight",
        str(args.aux_bce_weight),
        "--event_loss_weight",
        str(args.event_loss_weight),
        "--event_modules_per_epoch",
        str(args.event_modules_per_epoch),
        "--event_max_windows_per_module",
        str(args.event_max_windows_per_module),
        "--event_module_batch_size",
        str(args.event_module_batch_size),
        "--event_positive_fraction",
        str(args.event_positive_fraction),
        "--alarm_smoothing",
        str(args.alarm_smoothing),
        "--alarm_smooth_window",
        str(args.alarm_smooth_window),
        "--alarm_consecutive_k",
        str(args.alarm_consecutive_k),
    ]
    command.extend(["--alarm_search_smooth_windows", *[str(x) for x in args.alarm_search_smooth_windows]])
    command.extend(["--alarm_search_consecutive_ks", *[str(x) for x in args.alarm_search_consecutive_ks]])
    if args.module_cache_dir:
        command.extend(["--module_cache_dir", str(args.module_cache_dir)])
    if args.no_save_checkpoint:
        command.append("--no_save_checkpoint")
    if args.shuffle_train_rows:
        command.append("--shuffle_train_rows")
    if args.no_batched_windows:
        command.append("--no_batched_windows")
    if args.amp:
        command.append("--amp")
    if args.no_amp:
        command.append("--no_amp")
    if args.no_tf32:
        command.append("--no_tf32")
    if args.no_first_alarm_only:
        command.append("--no_first_alarm_only")
    if args.no_tqdm:
        command.append("--no_tqdm")
    if args.force_tqdm:
        command.append("--force_tqdm")

    started = time.time()
    write_progress(
        progress_path,
        {
            "status": "running",
            "model": model,
            "fold": fold,
            "started_at": now(),
            "elapsed_seconds": 0,
            "command": command,
            "log_path": str(log_path),
        },
    )
    print(f"[sequence] {now()} start model={model} fold={fold}", flush=True)
    print(f"[sequence] log={log_path}", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_f:
        log_f.write(f"\n===== {now()} START model={model} fold={fold} =====\n")
        log_f.write("COMMAND: " + " ".join(command) + "\n")
        log_f.flush()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        last_progress = time.time()
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
            if time.time() - last_progress >= float(args.progress_update_seconds):
                write_progress(
                    progress_path,
                    {
                        "status": "running",
                        "model": model,
                        "fold": fold,
                        "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
                        "last_update_at": now(),
                        "elapsed_seconds": time.time() - started,
                        "command": command,
                        "log_path": str(log_path),
                        "recent_log_tail": tail_text(log_path, 20),
                    },
                )
                last_progress = time.time()
        return_code = proc.wait()
        log_f.write(f"===== {now()} END model={model} fold={fold} return_code={return_code} =====\n")
        log_f.flush()

    elapsed = time.time() - started
    write_progress(
        progress_path,
        {
            "status": "completed" if return_code == 0 else "failed",
            "model": model,
            "fold": fold,
            "finished_at": now(),
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "log_path": str(log_path),
            "recent_log_tail": tail_text(log_path, 40),
        },
    )
    print(
        f"[sequence] {now()} end model={model} fold={fold} "
        f"return_code={return_code} elapsed_min={elapsed/60:.1f}",
        flush=True,
    )
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OFP benchmark sequentially with progress logs.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_DL_official_results/ofp_index_legacy_ts_benchmark"))
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--xgb_estimators", type=int, default=10)
    parser.add_argument("--rf_threshold", type=float, default=0.5)
    parser.add_argument("--xgb_threshold", type=float, default=0.5)
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument(
        "--timestamp_mode",
        choices=["legacy_float32", "strict_int64"],
        default="legacy_float32",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_batches", type=int, default=1000)
    parser.add_argument("--shuffle_train_rows", action="store_true")
    parser.add_argument("--module_cache_dir", default="")
    parser.add_argument("--train_sampling", choices=["full", "pos_all_neg_ratio"], default="pos_all_neg_ratio")
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--aux_bce_weight", type=float, default=0.3)
    parser.add_argument("--event_loss_weight", type=float, default=1.0)
    parser.add_argument("--event_modules_per_epoch", type=int, default=256)
    parser.add_argument("--event_max_windows_per_module", type=int, default=256)
    parser.add_argument("--event_module_batch_size", type=int, default=4)
    parser.add_argument("--event_positive_fraction", type=float, default=0.5)
    parser.add_argument("--alarm_smoothing", choices=["none", "ema"], default="ema")
    parser.add_argument("--alarm_smooth_window", type=int, default=1)
    parser.add_argument("--alarm_consecutive_k", type=int, default=1)
    parser.add_argument("--alarm_search_smooth_windows", nargs="+", type=int, default=[1, 3, 6, 12])
    parser.add_argument("--alarm_search_consecutive_ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--no_save_checkpoint", action="store_true")
    parser.add_argument("--no_batched_windows", action="store_true")
    parser.add_argument("--amp", action="store_true", help="enable CUDA autocast mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="disable CUDA autocast mixed precision")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--force_tqdm", action="store_true")
    parser.add_argument("--progress_update_seconds", type=float, default=300.0)
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.out_root) / "logs"
    progress_path = Path(args.out_root) / "progress.json"
    summary_path = Path(args.out_root) / "sequence_summary.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.time()
    write_progress(
        progress_path,
        {
            "status": "starting",
            "models": args.models,
            "folds": args.folds,
            "started_at": now(),
            "out_root": str(args.out_root),
        },
    )
    for model in args.models:
        for fold in args.folds:
            fold_summary_path = Path(args.out_root) / model / f"fold_{int(fold)}" / "fold_summary.json"
            if args.skip_existing and fold_summary_path.exists():
                record = {
                    "model": model,
                    "fold": int(fold),
                    "return_code": 0,
                    "skipped_existing": True,
                    "summary_path": str(fold_summary_path),
                    "timestamp": now(),
                }
                with summary_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                write_progress(
                    progress_path,
                    {
                        "status": "skipped_existing",
                        "model": model,
                        "fold": int(fold),
                        "summary_path": str(fold_summary_path),
                        "timestamp": now(),
                    },
                )
                print(
                    f"[sequence] {now()} skip existing model={model} fold={int(fold)} "
                    f"summary={fold_summary_path}",
                    flush=True,
                )
                continue
            rc = run_one(args, model, int(fold), log_dir, progress_path)
            record = {
                "model": model,
                "fold": int(fold),
                "return_code": rc,
                "timestamp": now(),
            }
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if rc != 0 and args.stop_on_failure:
                write_progress(
                    progress_path,
                    {
                        "status": "stopped_on_failure",
                        "model": model,
                        "fold": int(fold),
                        "return_code": rc,
                        "elapsed_seconds": time.time() - started_all,
                        "finished_at": now(),
                    },
                )
                raise SystemExit(rc)
    write_progress(
        progress_path,
        {
            "status": "all_completed",
            "models": args.models,
            "folds": args.folds,
            "elapsed_seconds": time.time() - started_all,
            "finished_at": now(),
            "out_root": str(args.out_root),
        },
    )


if __name__ == "__main__":
    main()
