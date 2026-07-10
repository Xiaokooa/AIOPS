from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


RUNNER = Path("OFP_DL_model2_compat/run_patchtst_stat_aligned_xgb.py")
SUMMARIZER = Path("OFP_DL_model2_compat/summarize_htsf_experiments.py")
DEFAULT_LEAD_GRID = "1m,5m,15m,30m,1h,2h,5h,12h,24h"
CANONICAL_CONTROL_SUITES = {
    "fusion",
    "sampling",
    "feature_groups",
    "feature_selection",
    "replacement",
    "lead_time",
}
DEPRECATED_DUPLICATE_IDS = {
    "fusion_htsf",
    "sampling_module_hybrid",
    "feature_full",
    "selection_all",
    "backbone_patchtst",
    "lead_time_htsf",
}

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "folds": [1],
        "seq_len": 16,
        "epochs": 1,
        "batch_size": 32,
        "max_train_files": 120,
        "max_test_files": 40,
        "ml_n_estimators": 20,
        "stat_selector_estimators": 20,
        "stat_selector_max_rows": 2000,
        "latent_probe_estimators": 20,
        "latent_probe_max_rows": 2000,
        "explain_max_rows": 64,
    },
    "quick": {
        "folds": [1],
        "seq_len": 32,
        "epochs": 3,
        "batch_size": 192,
        "max_train_files": 2500,
        "max_test_files": 1200,
        "ml_n_estimators": 300,
        "stat_selector_estimators": 200,
        "stat_selector_max_rows": 200000,
        "latent_probe_estimators": 200,
        "latent_probe_max_rows": 200000,
        "explain_max_rows": 2048,
    },
    "formal": {
        "folds": [1, 2, 3],
        "seq_len": 32,
        "epochs": 8,
        "batch_size": 192,
        "max_train_files": 0,
        "max_test_files": 0,
        "ml_n_estimators": 500,
        "stat_selector_estimators": 300,
        "stat_selector_max_rows": 300000,
        "latent_probe_estimators": 300,
        "latent_probe_max_rows": 300000,
        "explain_max_rows": 4096,
    },
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    suite: str
    method_label: str
    claim: str
    overrides: dict[str, Any] = field(default_factory=dict)


def spec(
    experiment_id: str,
    suite: str,
    method_label: str,
    claim: str,
    **overrides: Any,
) -> ExperimentSpec:
    return ExperimentSpec(experiment_id, suite, method_label, claim, overrides)


def experiment_catalog() -> dict[str, list[ExperimentSpec]]:
    full = {
        "fusion_mode": "gated_attn",
        "stat_feature_groups": "statistical,expert",
        "stat_selector": "none",
        "stat_select_k": 0,
        "sampling_mode": "module_balanced",
        "sample_selection": "hybrid",
        "sample_topk_fraction": 0.5,
        "sample_signal_mode": "expert",
        "sample_signal_temporal_fraction": 0.5,
        "branch_warmup_epochs": 0,
        "temporal_aux_weight": 0.0,
        "stat_aux_weight": 0.0,
        "temporal_modality_dropout": 0.0,
        "stat_modality_dropout": 0.0,
        "adaptive_negative_weight": 0.0,
    }
    branch_warmup = {
        **full,
        "branch_warmup_epochs": 2,
        "temporal_aux_weight": 0.5,
        "stat_aux_weight": 0.25,
    }
    modality_dropout = {
        **branch_warmup,
        "stat_modality_dropout": 0.25,
    }
    balanced_signal = {
        **modality_dropout,
        "sample_signal_mode": "mixed",
        "sample_signal_temporal_fraction": 0.5,
    }
    hard_negative = {
        **balanced_signal,
        "adaptive_negative_weight": 1.0,
        "adaptive_warmup_epochs": 3,
        "no_xgb_sample_weight": False,
        "xgb_use_sample_weight": True,
    }
    return {
        "main": [
            spec(
                "main_htsf",
                "main",
                "HTSF",
                "Main proposed framework under the OFP first-warning protocol.",
                lead_time_grid=DEFAULT_LEAD_GRID,
                **full,
            ),
        ],
        "fusion": [
            spec("fusion_temporal_only", "fusion", "Temporal view only", "Tests the raw temporal representation alone.", fusion_mode="temporal_only", **{k: v for k, v in full.items() if k != "fusion_mode"}),
            spec("fusion_expert_stat_only", "fusion", "Expert-statistical view only", "Tests engineered OFP evidence alone.", fusion_mode="stat_only", **{k: v for k, v in full.items() if k != "fusion_mode"}),
            spec("fusion_latent_concat", "fusion", "Latent concatenation", "Controls for gains from simple two-view concatenation.", fusion_mode="latent_concat", **{k: v for k, v in full.items() if k != "fusion_mode"}),
            spec("fusion_cross_attention", "fusion", "Cross attention", "Tests cross attention without learned modality gating.", fusion_mode="attn_mean", **{k: v for k, v in full.items() if k != "fusion_mode"}),
        ],
        "training": [
            spec("training_baseline", "training", "HTSF training baseline", "Current end-to-end HTSF training control.", **full),
            spec("training_branch_warmup", "training", "Branch warm-up", "Adds branch-specific auxiliary supervision before fused training.", **branch_warmup),
            spec("training_modality_dropout", "training", "Warm-up + modality dropout", "Prevents immediate reliance on the expert-statistical branch.", **modality_dropout),
            spec("training_balanced_signal", "training", "Balanced signal sampling", "Adds raw temporal-change evidence to hybrid sample selection.", **balanced_signal),
            spec("training_hard_negative", "training", "Two-stage HTSF", "Adds adaptive hard-negative weighting to the complete training strategy.", **hard_negative),
        ],
        "sampling": [
            spec("sampling_row_random", "sampling", "Row-ratio random", "Controls for row-level sampling without trace balancing.", sampling_mode="row_ratio", sample_selection="random", fusion_mode="gated_attn", stat_feature_groups="statistical,expert"),
            spec("sampling_module_random", "sampling", "Trace-balanced random", "Isolates trace-balanced selection.", sampling_mode="module_balanced", sample_selection="random", fusion_mode="gated_attn", stat_feature_groups="statistical,expert"),
            spec("sampling_module_topk", "sampling", "Signal Top-K", "Tests signal-only sample selection.", sampling_mode="module_balanced", sample_selection="signal_topk", fusion_mode="gated_attn", stat_feature_groups="statistical,expert"),
            spec("sampling_hybrid_25", "sampling", "Hybrid (25% signal)", "Sensitivity to the signal-selected fraction.", sampling_mode="module_balanced", sample_selection="hybrid", sample_topk_fraction=0.25, fusion_mode="gated_attn", stat_feature_groups="statistical,expert"),
            spec("sampling_hybrid_75", "sampling", "Hybrid (75% signal)", "Sensitivity to the signal-selected fraction.", sampling_mode="module_balanced", sample_selection="hybrid", sample_topk_fraction=0.75, fusion_mode="gated_attn", stat_feature_groups="statistical,expert"),
        ],
        "feature_groups": [
            spec("feature_statistics_only", "feature_groups", "Statistical only", "Separates statistical summaries from expert evidence.", stat_feature_groups="statistical", fusion_mode="gated_attn"),
            spec("feature_expert_only", "feature_groups", "Expert only", "Separates rule and lane evidence from statistical summaries.", stat_feature_groups="expert", fusion_mode="gated_attn"),
            spec("feature_without_prefix", "feature_groups", "w/o prefix statistics", "Tests cumulative min/max/range/std/skew/kurt features.", stat_feature_groups="statistical,expert", exclude_stat_feature_groups="stat_prefix", fusion_mode="gated_attn"),
            spec("feature_without_correlation", "feature_groups", "w/o correlations", "Tests rolling Pearson correlations.", stat_feature_groups="statistical,expert", exclude_stat_feature_groups="stat_correlation", fusion_mode="gated_attn"),
            spec("feature_without_lane", "feature_groups", "w/o lane consistency", "Tests P0-lane consistency evidence.", stat_feature_groups="statistical,expert", exclude_stat_feature_groups="expert_lane,expert_lane_extended", fusion_mode="gated_attn"),
            spec("feature_without_rules", "feature_groups", "w/o rule triggers", "Tests deterministic OFP rule-trigger features.", stat_feature_groups="statistical,expert", exclude_stat_feature_groups="expert_rule,expert_pattern", fusion_mode="gated_attn"),
        ],
        "feature_selection": [
            spec("selection_top32", "feature_selection", "Top-32", "ExtraTrees selection on training traces only.", stat_feature_groups="statistical,expert", stat_selector="extra_trees", stat_select_k=32, fusion_mode="gated_attn"),
            spec("selection_top64", "feature_selection", "Top-64", "ExtraTrees selection on training traces only.", stat_feature_groups="statistical,expert", stat_selector="extra_trees", stat_select_k=64, fusion_mode="gated_attn"),
            spec("selection_top128", "feature_selection", "Top-128", "ExtraTrees selection on training traces only.", stat_feature_groups="statistical,expert", stat_selector="extra_trees", stat_select_k=128, fusion_mode="gated_attn"),
        ],
        "latent_probe": [
            spec("latent_probe_top32", "latent_probe", "Pre-fusion latent Top-32", "Ranks temporal and expert-statistical latent dimensions with one training-only selector.", latent_probe_topk=32, **full),
            spec("latent_probe_top64", "latent_probe", "Pre-fusion latent Top-64", "Ranks temporal and expert-statistical latent dimensions with one training-only selector.", latent_probe_topk=64, **full),
            spec("latent_probe_top128", "latent_probe", "Pre-fusion latent Top-128", "Ranks temporal and expert-statistical latent dimensions with one training-only selector.", latent_probe_topk=128, **full),
        ],
        "replacement": [
            spec("backbone_itransformer", "replacement", "HTSF", "Temporal encoder replacement.", temporal_encoder="itransformer", decision_layer="xgb", **full),
            spec("backbone_moderntcn", "replacement", "HTSF", "Temporal encoder replacement.", temporal_encoder="moderntcn", decision_layer="xgb", **full),
            spec("backbone_fits", "replacement", "HTSF", "Temporal encoder replacement.", temporal_encoder="fits", decision_layer="xgb", **full),
            spec("classifier_rf", "replacement", "HTSF", "Decision-layer replacement.", temporal_encoder="patchtst", decision_layer="rf", **full),
            spec("classifier_lgbm", "replacement", "HTSF", "Decision-layer replacement.", temporal_encoder="patchtst", decision_layer="lgbm", **full),
            spec("classifier_catboost", "replacement", "HTSF", "Decision-layer replacement.", temporal_encoder="patchtst", decision_layer="catboost", **full),
        ],
        "lead_time": [],
    }


def selected_specs(suites: list[str], only: str) -> list[ExperimentSpec]:
    catalog = experiment_catalog()
    requested = list(catalog) if "all" in suites else suites
    unknown = sorted(set(requested) - set(catalog))
    if unknown:
        raise ValueError(f"Unknown suites: {unknown}; choose from {sorted(catalog)} or all")
    keep = {value.strip() for value in only.replace(";", ",").split(",") if value.strip()}
    available = {item.experiment_id for items in catalog.values() for item in items}
    unknown_keep = sorted(keep - available)
    if unknown_keep:
        raise ValueError(f"Unknown experiment IDs: {unknown_keep}")
    out: list[ExperimentSpec] = []
    seen: set[str] = set()
    if set(requested) & ({"main"} | CANONICAL_CONTROL_SUITES):
        main_spec = catalog["main"][0]
        out.append(main_spec)
        seen.add(main_spec.experiment_id)
    for suite in requested:
        for item in catalog[suite]:
            if keep and item.experiment_id not in keep:
                continue
            if item.experiment_id not in seen:
                out.append(item)
                seen.add(item.experiment_id)
    return out


def add_option(command: list[str], name: str, value: Any) -> None:
    option = f"--{name}"
    if isinstance(value, bool):
        if value:
            command.append(option)
        return
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        command.append(option)
        command.extend(str(item) for item in value)
        return
    command.extend([option, str(value)])


def build_command(
    args: argparse.Namespace,
    item: ExperimentSpec,
    profile: dict[str, Any],
    experiment_root: Path,
) -> list[str]:
    feature_cache_dir = args.feature_cache_dir or (experiment_root.parent.parent / "feature_cache")
    values: dict[str, Any] = {
        "experiment_id": item.experiment_id,
        "method_label": item.method_label,
        "folds": args.folds or profile["folds"],
        "data_dir": args.data_dir,
        "index_path": args.index_path,
        "out_root": experiment_root,
        "module_cache_dir": feature_cache_dir,
        "seq_len": args.seq_len or profile["seq_len"],
        "epochs": args.epochs or profile["epochs"],
        "batch_size": args.batch_size or profile["batch_size"],
        "max_train_files": args.max_train_files if args.max_train_files is not None else profile["max_train_files"],
        "max_test_files": args.max_test_files if args.max_test_files is not None else profile["max_test_files"],
        "ml_n_estimators": args.ml_n_estimators or profile["ml_n_estimators"],
        "stat_selector_estimators": profile["stat_selector_estimators"],
        "stat_selector_max_rows": profile["stat_selector_max_rows"],
        "latent_probe_estimators": profile["latent_probe_estimators"],
        "latent_probe_max_rows": profile["latent_probe_max_rows"],
        "explain_max_rows": profile["explain_max_rows"],
        "target_mode": "ahead120",
        "feature_mode": "ofp",
        "stat_feature_mode": "ofp_expert_stat",
        "stat_feature_groups": "statistical,expert",
        "temporal_encoder": "patchtst",
        "decision_layer": "xgb",
        "fusion_mode": "gated_attn",
        "tsf_ablation": "full",
        "sampling_mode": "module_balanced",
        "sample_selection": "hybrid",
        "sample_topk_fraction": 0.5,
        "sample_signal_mode": "expert",
        "sample_signal_temporal_fraction": 0.5,
        "branch_warmup_epochs": 0,
        "temporal_aux_weight": 0.0,
        "stat_aux_weight": 0.0,
        "temporal_modality_dropout": 0.0,
        "stat_modality_dropout": 0.0,
        "temporal_positive_weight": 0.0,
        "adaptive_negative_weight": 0.0,
        "adaptive_warmup_epochs": 1,
        "rule_mode": "none",
        "positive_windows_per_module": args.positive_windows_per_module,
        "negative_windows_per_faulty_module": args.negative_windows_per_faulty_module,
        "normal_windows_per_module": args.normal_windows_per_module,
        "threshold_grid": args.threshold_grid,
        "threshold_metric": "f1_score",
        "xgb_balance_mode": args.xgb_balance_mode,
        "xgb_tree_method": args.xgb_tree_method,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "device": args.device,
        "amp": args.amp,
        "no_xgb_sample_weight": True,
    }
    values.update(item.overrides)
    command = [args.python_executable, "-u", "-B", str(RUNNER)]
    for name, value in values.items():
        add_option(command, name, value)
    return command


def render_shell_command(command: list[str], gpu: str) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} " if gpu else ""
    return prefix + shlex.join(command)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def command_fingerprint(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def merge_manifest_entries(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entry in [*previous, *current]:
        experiment_id = str(entry.get("experiment_id", ""))
        if not experiment_id or experiment_id in DEPRECATED_DUPLICATE_IDS:
            continue
        merged[experiment_id] = entry
    return list(merged.values())


def run_context(args: argparse.Namespace, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": args.profile,
        "folds": [int(value) for value in (args.folds or profile["folds"])],
        "data_dir": str(args.data_dir),
        "index_path": str(args.index_path),
        "seq_len": int(args.seq_len or profile["seq_len"]),
        "epochs": int(args.epochs or profile["epochs"]),
        "batch_size": int(args.batch_size or profile["batch_size"]),
        "max_train_files": int(args.max_train_files if args.max_train_files is not None else profile["max_train_files"]),
        "max_test_files": int(args.max_test_files if args.max_test_files is not None else profile["max_test_files"]),
        "ml_n_estimators": int(args.ml_n_estimators or profile["ml_n_estimators"]),
        "positive_windows_per_module": int(args.positive_windows_per_module),
        "negative_windows_per_faulty_module": int(args.negative_windows_per_faulty_module),
        "normal_windows_per_module": int(args.normal_windows_per_module),
        "threshold_grid": str(args.threshold_grid),
        "xgb_balance_mode": str(args.xgb_balance_mode),
        "xgb_tree_method": str(args.xgb_tree_method),
        "seed": int(args.seed),
        "device": str(args.device),
        "amp": bool(args.amp),
    }


def experiment_is_complete(
    result_path: Path,
    expected_folds: list[int],
    previous_entry: dict[str, Any] | None,
    expected_fingerprint: str,
) -> bool:
    if not result_path.exists():
        return False
    if not previous_entry or previous_entry.get("command_fingerprint") != expected_fingerprint:
        return False
    try:
        import pandas as pd

        frame = pd.read_csv(result_path, usecols=["fold"])
    except Exception:
        return False
    completed = {int(value) for value in frame["fold"].dropna().tolist()}
    return set(int(value) for value in expected_folds) <= completed


def run_one(
    item: ExperimentSpec,
    command: list[str],
    gpu: str,
    log_path: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(render_shell_command(command, gpu) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "experiment_id": item.experiment_id,
        "returncode": int(completed.returncode),
        "status": "completed" if completed.returncode == 0 else "failed",
        "elapsed_seconds": float(time.time() - started),
        "log": str(log_path),
        "gpu": gpu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the claim-aligned HTSF experiment and ablation matrix.")
    parser.add_argument("--suites", nargs="+", default=["main", "fusion", "sampling", "feature_groups", "feature_selection"])
    parser.add_argument("--only", default="", help="Optional comma-separated experiment IDs.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="formal")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="0", help="Comma-separated CUDA device IDs used round-robin.")
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--no_summarize", action="store_true")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--ml_n_estimators", type=int, default=None)
    parser.add_argument("--positive_windows_per_module", type=int, default=32)
    parser.add_argument("--negative_windows_per_faulty_module", type=int, default=8)
    parser.add_argument("--normal_windows_per_module", type=int, default=8)
    parser.add_argument(
        "--threshold_grid",
        default="0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99",
    )
    parser.add_argument("--xgb_balance_mode", choices=["auto", "sqrt", "none"], default="none")
    parser.add_argument("--xgb_tree_method", default="hist")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or Path(f"OFP_DL_model2_compat_results/htsf_evidence_{args.profile}_{tag}")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "suite_manifest.json"
    previous_manifest = read_manifest(manifest_path)
    current_context = run_context(args, profile)
    previous_context = previous_manifest.get("run_context")
    if previous_context and previous_context != current_context:
        raise ValueError(
            "The output root already contains a different experiment context. "
            "Use the original arguments or choose a new --out_root."
        )
    previous_entries = {
        str(entry.get("experiment_id", "")): entry
        for entry in previous_manifest.get("experiments", [])
        if str(entry.get("experiment_id", "")) not in DEPRECATED_DUPLICATE_IDS
    }
    specs = selected_specs(args.suites, args.only)
    if not specs:
        raise ValueError("No experiments selected")
    gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
    if str(args.device).lower() == "cpu":
        gpus = [""]
    if not gpus:
        gpus = [""]

    entries: list[dict[str, Any]] = []
    jobs: list[tuple[ExperimentSpec, list[str], str, Path]] = []
    expected_folds = [int(value) for value in (args.folds or profile["folds"])]
    for index, item in enumerate(specs):
        experiment_root = out_root / "experiments" / item.experiment_id
        command = build_command(args, item, profile, experiment_root)
        fingerprint = command_fingerprint(command)
        gpu = gpus[index % len(gpus)]
        result_path = experiment_root / "fold_metrics.csv"
        status = (
            "skipped_existing"
            if args.resume
            and experiment_is_complete(
                result_path,
                expected_folds,
                previous_entries.get(item.experiment_id),
                fingerprint,
            )
            else "planned"
        )
        entry = {
            **asdict(item),
            "experiment_root": str(experiment_root),
            "command": command,
            "shell_command": render_shell_command(command, gpu),
            "gpu": gpu,
            "status": status,
            "command_fingerprint": fingerprint,
        }
        entries.append(entry)
        if status == "planned":
            jobs.append((item, command, gpu, out_root / "logs" / f"{item.experiment_id}.log"))

    requested_suites = list(experiment_catalog()) if "all" in args.suites else list(args.suites)
    merged_suites = list(dict.fromkeys([*previous_manifest.get("suites", []), *requested_suites]))
    manifest: dict[str, Any] = {
        "created_at": previous_manifest.get("created_at", datetime.now().isoformat(timespec="seconds")),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "run_context": current_context,
        "protocol": {
            "target_mode": "ahead120",
            "feature_mode": "ofp",
            "evaluator": "trace-level OFP first warning",
            "threshold_selection": "validation F1",
            "rule_or_for_learned_models": False,
        },
        "suites": merged_suites,
        "experiments": merge_manifest_entries(list(previous_entries.values()), entries),
    }
    write_manifest(manifest_path, manifest)
    commands_path = out_root / "commands.sh"
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(entry["shell_command"] for entry in entries) + "\n", encoding="utf-8")

    print(f"[suite] profile={args.profile} experiments={len(specs)} out={out_root}", flush=True)
    for entry in entries:
        print(f"[{entry['status']}] {entry['experiment_id']}: {entry['shell_command']}", flush=True)
    if args.dry_run:
        return

    results: list[dict[str, Any]] = []
    max_parallel = max(1, int(args.max_parallel))
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_map = {
            executor.submit(run_one, item, command, gpu, log_path): item
            for item, command, gpu, log_path in jobs
        }
        for future in as_completed(future_map):
            item = future_map[future]
            result = future.result()
            results.append(result)
            print(
                f"[{result['status']}] {item.experiment_id} rc={result['returncode']} "
                f"elapsed={result['elapsed_seconds']:.1f}s log={result['log']}",
                flush=True,
            )
            if result["returncode"] != 0 and not args.keep_going:
                for pending in future_map:
                    pending.cancel()
                break

    by_id = {result["experiment_id"]: result for result in results}
    for entry in entries:
        if entry["experiment_id"] in by_id:
            entry.update(by_id[entry["experiment_id"]])
    manifest["experiments"] = merge_manifest_entries(manifest["experiments"], entries)
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_manifest(manifest_path, manifest)

    failed = [entry for entry in entries if entry.get("status") == "failed"]
    if not args.no_summarize:
        summary_cmd = [args.python_executable, "-u", "-B", str(SUMMARIZER), "--root", str(out_root)]
        subprocess.run(summary_cmd, cwd=Path.cwd(), check=False)
    if failed:
        raise SystemExit(f"{len(failed)} HTSF experiments failed; inspect {out_root / 'logs'}")


if __name__ == "__main__":
    main()
