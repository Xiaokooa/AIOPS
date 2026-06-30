from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_model2_compat.run_model2_compat_deep import format_duration, progress_bar


METHODS = ("pure_deep", "ofp_compat_deep", "hybrid_fusion")
DEFAULT_MODELS = ["fits", "itransformer", "moderntcn", "patchtst"]
DEFAULT_THRESHOLD_GRID = "0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50"

PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    "smoke": {
        "epochs": 1,
        "batch_size": 64,
        "seq_len": 16,
        "positive_windows_per_module": 8,
        "negative_windows_per_faulty_module": 4,
        "normal_windows_per_module": 4,
        "max_cached_files": 64,
        "max_train_files": 80,
        "max_test_files": 40,
        "ml_n_estimators": 20,
        "select_k": 32,
        "log_batches": 20,
    },
    "quick": {
        "epochs": 3,
        "batch_size": 192,
        "seq_len": 32,
        "positive_windows_per_module": 32,
        "negative_windows_per_faulty_module": 8,
        "normal_windows_per_module": 8,
        "max_cached_files": 128,
        "max_train_files": 2500,
        "max_test_files": 1200,
        "ml_n_estimators": 300,
        "select_k": 96,
        "log_batches": 100,
    },
    "formal": {
        "epochs": 8,
        "batch_size": 256,
        "seq_len": 64,
        "positive_windows_per_module": 96,
        "negative_windows_per_faulty_module": 24,
        "normal_windows_per_module": 24,
        "max_cached_files": 512,
        "max_train_files": 0,
        "max_test_files": 0,
        "ml_n_estimators": 500,
        "select_k": 128,
        "log_batches": 200,
    },
}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def profile_int(args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    return int(PROFILE_DEFAULTS[str(args.profile)][name])


def selected_methods(raw: list[str]) -> list[str]:
    lowered = [str(item).lower() for item in raw]
    if "all" in lowered:
        return list(METHODS)
    out: list[str] = []
    for item in lowered:
        if item not in METHODS:
            raise ValueError(f"Unknown method={item!r}; choose from all, {', '.join(METHODS)}")
        if item not in out:
            out.append(item)
    return out


def csv_words(text: str) -> list[str]:
    words: list[str] = []
    for part in str(text).replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if part:
            words.append(part)
    return words


def common_args(args: argparse.Namespace, out_root: Path, target_mode: str, feature_mode: str, rule_mode: str) -> list[str]:
    cmd: list[str] = [
        "--models",
        *args.models,
        "--folds",
        *[str(fold) for fold in args.folds],
        "--data_dir",
        str(args.data_dir),
        "--index_path",
        str(args.index_path),
        "--out_root",
        str(out_root),
        "--epochs",
        str(profile_int(args, "epochs")),
        "--batch_size",
        str(profile_int(args, "batch_size")),
        "--seq_len",
        str(profile_int(args, "seq_len")),
        "--device",
        str(args.device),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--grad_clip",
        str(args.grad_clip),
        "--negative_ratio",
        str(args.negative_ratio),
        "--pos_weight_cap",
        str(args.pos_weight_cap),
        "--fixed_threshold",
        str(args.fixed_threshold),
        "--target_mode",
        target_mode,
        "--feature_mode",
        feature_mode,
        "--sampling_mode",
        str(args.sampling_mode),
        "--positive_windows_per_module",
        str(profile_int(args, "positive_windows_per_module")),
        "--negative_windows_per_faulty_module",
        str(profile_int(args, "negative_windows_per_faulty_module")),
        "--normal_windows_per_module",
        str(profile_int(args, "normal_windows_per_module")),
        "--rule_mode",
        rule_mode,
        "--sample_selection",
        str(args.sample_selection),
        "--sample_topk_fraction",
        str(args.sample_topk_fraction),
        "--temporal_positive_weight",
        str(args.temporal_positive_weight),
        "--temporal_weight_horizon_hours",
        str(args.temporal_weight_horizon_hours),
        "--adaptive_negative_weight",
        str(args.adaptive_negative_weight),
        "--adaptive_warmup_epochs",
        str(args.adaptive_warmup_epochs),
        "--min_hit_lead_hours",
        str(args.min_hit_lead_hours),
        "--val_fraction",
        str(args.val_fraction),
        "--threshold_grid",
        str(args.threshold_grid),
        "--threshold_metric",
        str(args.threshold_metric),
        "--max_cached_files",
        str(profile_int(args, "max_cached_files")),
        "--max_train_files",
        str(profile_int(args, "max_train_files")),
        "--max_test_files",
        str(profile_int(args, "max_test_files")),
        "--num_workers",
        str(args.num_workers),
        "--log_batches",
        str(profile_int(args, "log_batches")),
    ]
    if not bool(args.threshold_search):
        cmd.append("--no_threshold_search")
    if bool(args.amp):
        cmd.append("--amp")
    if bool(args.no_tf32):
        cmd.append("--no_tf32")
    return cmd


def build_method_command(args: argparse.Namespace, method: str, method_root: Path) -> list[str]:
    py = str(args.python_executable)
    if method == "pure_deep":
        return [
            py,
            "-u",
            "-B",
            "OFP_DL_model2_compat/run_model2_compat_deep.py",
            *common_args(args, method_root, args.pure_target_mode, args.pure_feature_mode, args.pure_rule_mode),
        ]
    if method == "ofp_compat_deep":
        return [
            py,
            "-u",
            "-B",
            "OFP_DL_model2_compat/run_model2_compat_deep.py",
            *common_args(args, method_root, args.compat_target_mode, args.compat_feature_mode, args.compat_rule_mode),
        ]
    if method == "hybrid_fusion":
        ml_models = csv_words(args.ml_models)
        cmd = [
            py,
            "-u",
            "-B",
            "OFP_DL_model2_compat/run_model2_compat_tabular_fusion.py",
            *common_args(args, method_root, args.hybrid_target_mode, args.hybrid_feature_mode, args.hybrid_rule_mode),
            "--ml_models",
            *ml_models,
            "--ml_feature_set",
            str(args.ml_feature_set),
            "--deep_feature_parts",
            str(args.deep_feature_parts),
            "--selector",
            str(args.selector),
            "--select_k",
            str(profile_int(args, "select_k")),
            "--ml_n_estimators",
            str(profile_int(args, "ml_n_estimators")),
            "--rf_max_depth",
            str(args.rf_max_depth),
            "--rf_min_samples_leaf",
            str(args.rf_min_samples_leaf),
            "--xgb_max_depth",
            str(args.xgb_max_depth),
            "--xgb_lr",
            str(args.xgb_lr),
            "--xgb_subsample",
            str(args.xgb_subsample),
            "--xgb_colsample_bytree",
            str(args.xgb_colsample_bytree),
            "--xgb_tree_method",
            str(args.xgb_tree_method),
            "--n_jobs",
            str(args.n_jobs),
        ]
        if bool(args.require_all_ml):
            cmd.append("--require_all_ml")
        return cmd
    raise ValueError(method)


def display_command(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def run_logged(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    rendered = display_command(cmd)
    print(f"[suite-cmd] {rendered}")
    if dry_run:
        return {"status": "dry_run", "command": rendered, "seconds": 0.0, "returncode": None}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(rendered + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = proc.wait()
    seconds = time.time() - started
    return {
        "status": "ok" if returncode == 0 else "failed",
        "command": rendered,
        "seconds": seconds,
        "returncode": int(returncode),
        "log_path": str(log_path),
    }


def method_description(args: argparse.Namespace, method: str) -> dict[str, str]:
    if method == "pure_deep":
        return {
            "driver": "deep",
            "target_mode": args.pure_target_mode,
            "feature_mode": args.pure_feature_mode,
            "rule_mode": args.pure_rule_mode,
            "meaning": "deep backbone only; no rule OR and no tabular ML",
        }
    if method == "ofp_compat_deep":
        return {
            "driver": "deep",
            "target_mode": args.compat_target_mode,
            "feature_mode": args.compat_feature_mode,
            "rule_mode": args.compat_rule_mode,
            "meaning": "deep row classifier with model2-compatible objective and optional rule OR",
        }
    return {
        "driver": "tabular_fusion",
        "target_mode": args.hybrid_target_mode,
        "feature_mode": args.hybrid_feature_mode,
        "rule_mode": args.hybrid_rule_mode,
        "meaning": "deep embedding/deep score plus model2 features, followed by RF/XGB-style tabular models",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-oriented OFP model2-compatible experiment suite.")
    parser.add_argument("--methods", nargs="+", default=["all"], help="all, pure_deep, ofp_compat_deep, hybrid_fusion")
    parser.add_argument("--profile", choices=["smoke", "quick", "formal"], default="formal")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu_id", default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_going", action="store_true")

    parser.add_argument("--pure_target_mode", choices=["ahead120", "anomaly", "module_fault"], default="ahead120")
    parser.add_argument("--pure_feature_mode", choices=["model2", "model2_plus"], default="model2_plus")
    parser.add_argument("--pure_rule_mode", choices=["none", "temp", "model2_simple"], default="none")
    parser.add_argument("--compat_target_mode", choices=["ahead120", "anomaly", "module_fault"], default="module_fault")
    parser.add_argument("--compat_feature_mode", choices=["model2", "model2_plus"], default="model2_plus")
    parser.add_argument("--compat_rule_mode", choices=["none", "temp", "model2_simple"], default="model2_simple")
    parser.add_argument("--hybrid_target_mode", choices=["ahead120", "anomaly", "module_fault"], default="module_fault")
    parser.add_argument("--hybrid_feature_mode", choices=["model2", "model2_plus"], default="model2_plus")
    parser.add_argument("--hybrid_rule_mode", choices=["none", "temp", "model2_simple"], default="model2_simple")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--positive_windows_per_module", type=int, default=None)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=None)
    parser.add_argument("--normal_windows_per_module", type=int, default=None)
    parser.add_argument("--max_cached_files", type=int, default=None)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--ml_n_estimators", type=int, default=None)
    parser.add_argument("--select_k", type=int, default=None)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--negative_ratio", type=float, default=10.0)
    parser.add_argument("--pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--fixed_threshold", type=float, default=0.3)
    parser.add_argument("--sampling_mode", choices=["row_ratio", "module_balanced"], default="module_balanced")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_grid", default=DEFAULT_THRESHOLD_GRID)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--no_threshold_search", dest="threshold_search", action="store_false")
    parser.set_defaults(threshold_search=True)
    parser.add_argument("--min_hit_lead_hours", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_batches", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")

    parser.add_argument("--sample_selection", choices=["random", "signal_topk", "hybrid"], default="hybrid")
    parser.add_argument("--sample_topk_fraction", type=float, default=0.5)
    parser.add_argument("--temporal_positive_weight", type=float, default=2.0)
    parser.add_argument("--temporal_weight_horizon_hours", type=float, default=120.0)
    parser.add_argument("--adaptive_negative_weight", type=float, default=1.0)
    parser.add_argument("--adaptive_warmup_epochs", type=int, default=1)
    parser.add_argument("--ml_models", default="rf,xgb,lgbm,catboost")
    parser.add_argument("--ml_feature_set", choices=["model2", "embedding", "fusion"], default="fusion")
    parser.add_argument("--deep_feature_parts", default="embedding,score")
    parser.add_argument("--selector", choices=["none", "variance", "f_classif", "mutual_info", "extra_trees", "model_importance"], default="extra_trees")
    parser.add_argument("--rf_max_depth", type=int, default=0)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--require_all_ml", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suite_root = Path(args.out_root) if args.out_root is not None else Path("OFP_DL_model2_compat_results") / f"paper_suite_{args.profile}_{now_tag()}"
    suite_root.mkdir(parents=True, exist_ok=True)
    methods = selected_methods(args.methods)
    env = os.environ.copy()
    if str(args.gpu_id).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip()

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "models": args.models,
        "folds": args.folds,
        "methods": methods,
        "suite_root": str(suite_root),
        "dry_run": bool(args.dry_run),
        "profile_defaults": PROFILE_DEFAULTS[args.profile],
        "runs": [],
    }
    print("=" * 72)
    print("OFP model2-compatible paper suite")
    print("=" * 72)
    print(f"profile: {args.profile}")
    print(f"methods: {' '.join(methods)}")
    print(f"models: {' '.join(args.models)}")
    print(f"folds: {' '.join(str(x) for x in args.folds)}")
    print(f"out_root: {suite_root}")
    print(f"dry_run: {args.dry_run}")
    print("=" * 72)

    status = 0
    suite_started = time.time()
    for method_idx, method in enumerate(methods, 1):
        method_root = suite_root / method
        method_root.mkdir(parents=True, exist_ok=True)
        cmd = build_method_command(args, method, method_root)
        log_path = suite_root / "logs" / f"{method}.log"
        print(
            f"[suite-start] method={method_idx}/{len(methods)} "
            f"{progress_bar(method_idx - 1, len(methods), width=18)} name={method} out={method_root}"
        )
        run_info = {
            "method": method,
            "out_root": str(method_root),
            "description": method_description(args, method),
        }
        run_info.update(run_logged(cmd, repo_root, log_path, env, bool(args.dry_run)))
        print(
            f"[suite-finish] method={method_idx}/{len(methods)} name={method} "
            f"status={run_info.get('status')} elapsed={format_duration(float(run_info.get('seconds', 0.0)))}"
        )
        manifest["runs"].append(run_info)
        (suite_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if run_info.get("returncode") not in (None, 0):
            status = int(run_info["returncode"])
            if not bool(args.keep_going):
                break

    (suite_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if status != 0:
        raise SystemExit(status)
    print(
        f"[suite-done] {progress_bar(len(methods), len(methods), width=18)} "
        f"elapsed={format_duration(time.time() - suite_started)} manifest={suite_root / 'suite_manifest.json'}"
    )


if __name__ == "__main__":
    main()
