"""Run the compact, paper-facing FORT module ablation in one command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "artifacts" / "FORT_module_ablation"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fgofp.ablation import LEARNED_SPECS, ModuleAblationSpec, write_module_ablation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compact FORT module ablations and pool protocol-aligned "
            "three-fold OFP decisions."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=BASE_DIR / "configs" / "default.json"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE_ROOT / "dataset" / "training"
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=WORKSPACE_ROOT / "dataset" / "train_test_set_index(in).csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument(
        "--skip-static-xgb",
        action="store_true",
        help="skip training S0; an existing static_xgb/comparison.csv is still summarized",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse sub-runs that already contain comparison.csv",
    )
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    return args


def build_static_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(BASE_DIR / "run_static_xgb_threefold.py"),
        "--device",
        str(args.device),
        "--data-dir",
        str(args.data_dir),
        "--index-path",
        str(args.index_path),
        "--output-dir",
        str(args.output_dir / "static_xgb"),
        "--experiment-name",
        "S0_static_xgb",
    ]
    if args.smoke:
        command.append("--smoke")
    if args.overwrite:
        command.append("--overwrite")
    return command


def build_learned_command(
    args: argparse.Namespace, spec: ModuleAblationSpec
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(BASE_DIR / "run_threefold.py"),
        "--config",
        str(args.config),
        "--device",
        str(args.device),
        "--data-dir",
        str(args.data_dir),
        "--index-path",
        str(args.index_path),
        "--output-dir",
        str(args.output_dir / spec.experiment_id),
        "--experiment-name",
        spec.experiment_id,
        "--architecture",
        spec.architecture,
        "--positive-weight-mode",
        spec.positive_weight_mode,
        "--disable-safe-rule-fallback",
    ]
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    if args.inference_batch_size is not None:
        command.extend(("--inference-batch-size", str(args.inference_batch_size)))
    if args.no_mixed_precision:
        command.append("--no-mixed-precision")
    if args.smoke:
        command.append("--smoke")
    if args.overwrite:
        command.append("--overwrite")
    return command


def _run_or_resume(
    command: list[str], output_dir: Path, *, resume: bool, label: str
) -> None:
    if resume and (output_dir / "comparison.csv").is_file():
        print(f"[ablation] resume: reuse {label}", flush=True)
        return
    if resume and output_dir.exists() and any(output_dir.iterdir()):
        # The inner three-fold runners do not checkpoint aggregation state.
        # Reuse only a fully audited comparison; restart an interrupted
        # sub-run explicitly so --resume never fails on its partial files.
        print(f"[ablation] resume: restart incomplete {label}", flush=True)
        if "--overwrite" not in command:
            command = [*command, "--overwrite"]
    print(f"[ablation] running {label}", flush=True)
    subprocess.run(command, cwd=str(WORKSPACE_ROOT), check=True)


def main() -> None:
    args = parse_args()
    # Child processes run from WORKSPACE_ROOT so imports are identical on the
    # server.  Freeze every user path first; otherwise a relative output path
    # would be interpreted once by this process and again from a different cwd.
    args.config = args.config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.index_path = args.index_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static_script = BASE_DIR / "run_static_xgb_threefold.py"
    if not args.skip_static_xgb:
        if not static_script.is_file():
            raise FileNotFoundError(
                f"static XGB runner is missing: {static_script}; "
                "use --skip-static-xgb only if the baseline is intentionally omitted"
            )
        _run_or_resume(
            build_static_command(args),
            args.output_dir / "static_xgb",
            resume=args.resume,
            label="S0_static_xgb",
        )

    for spec in LEARNED_SPECS:
        _run_or_resume(
            build_learned_command(args, spec),
            args.output_dir / spec.experiment_id,
            resume=args.resume,
            label=spec.experiment_id,
        )

    summary = write_module_ablation_outputs(
        args.output_dir,
        args.index_path,
        formal=not args.smoke,
    )
    columns = [
        "experiment_id",
        "final_score",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
        "avg_lead_hour",
        "early_hit_rate_at_24h",
    ]
    print(summary.loc[:, columns].to_string(index=False), flush=True)
    print(
        f"[ablation] summary={args.output_dir / 'module_ablation.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
