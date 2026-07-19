from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP.deep_learning.compat.run_model2_compat_deep import (
    CompatCfg,
    _module_feature_cache_path,
    _read_feature_arrays,
    file_label_map,
)
from OFP.deep_learning.official.common.index_split import read_index


def prepare_one(
    file_name: str,
    data_dir: Path,
    cfg: CompatCfg,
    labels: dict[str, int],
) -> dict[str, Any]:
    cache_path = _module_feature_cache_path(cfg, file_name)
    existed = bool(cache_path and cache_path.exists())
    started = time.time()
    arrays = _read_feature_arrays(data_dir, file_name, cfg, labels)
    return {
        "file_name": file_name,
        "status": "cached" if existed else "created",
        "rows": int(len(arrays[0])),
        "features": int(arrays[1].shape[1]) if arrays[1].ndim == 2 else 0,
        "elapsed_seconds": float(time.time() - started),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute shared OFP expert-statistical feature arrays.")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--cache_dir", type=Path, default=Path("OFP_DL_model2_compat_results/htsf_feature_cache"))
    parser.add_argument("--feature_mode", choices=["ofp", "ofp_plus"], default="ofp")
    parser.add_argument("--target_mode", choices=["ahead120", "anomaly", "module_fault"], default="ahead120")
    parser.add_argument("--rule_mode", choices=["none", "ofp_rules", "model2_simple"], default="none")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--keep_going", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    labels = file_label_map(index_df)
    files = sorted(labels)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    cfg = CompatCfg(
        target_mode=args.target_mode,
        feature_mode=args.feature_mode,
        rule_mode=args.rule_mode,
        module_cache_dir=str(args.cache_dir),
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    created = 0
    cached = 0
    failed: list[dict[str, str]] = []
    processed_rows = 0
    feature_count = 0
    print(
        f"[cache] files={len(files)} workers={args.workers} mode={args.feature_mode} "
        f"target={args.target_mode} out={args.cache_dir}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(prepare_one, file_name, args.data_dir, cfg, labels): file_name
            for file_name in files
        }
        for index, future in enumerate(as_completed(futures), 1):
            file_name = futures[future]
            try:
                result = future.result()
                created += int(result["status"] == "created")
                cached += int(result["status"] == "cached")
                processed_rows += int(result["rows"])
                feature_count = max(feature_count, int(result["features"]))
            except Exception as exc:
                failed.append({"file_name": file_name, "error": repr(exc)})
                if not args.keep_going:
                    for pending in futures:
                        pending.cancel()
                    break
            if index % max(1, int(args.progress_every)) == 0 or index == len(files):
                elapsed = time.time() - started
                print(
                    f"[cache-progress] {index}/{len(files)} created={created} reused={cached} "
                    f"failed={len(failed)} rows={processed_rows} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "index_path": str(args.index_path),
        "cache_dir": str(args.cache_dir),
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
        "rule_mode": args.rule_mode,
        "requested_files": len(files),
        "created_files": created,
        "reused_files": cached,
        "failed_files": len(failed),
        "processed_rows": processed_rows,
        "feature_count": feature_count,
        "elapsed_seconds": float(time.time() - started),
        "failures": failed,
    }
    (args.cache_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[cache-done] created={created} reused={cached} failed={len(failed)} "
        f"rows={processed_rows} elapsed={manifest['elapsed_seconds']:.1f}s",
        flush=True,
    )
    if failed:
        raise SystemExit(f"Feature cache preparation failed for {len(failed)} files")


if __name__ == "__main__":
    main()
