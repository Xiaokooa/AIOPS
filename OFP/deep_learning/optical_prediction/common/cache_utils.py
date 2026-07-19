from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from OFP.deep_learning.optical_prediction.common import base_utils as base


SHARED_CACHE_ROOT = base.PROJECT_ROOT / "output" / "Optical_prediction_model" / "_shared_cache"
RESAMPLED_CACHE_DIR = SHARED_CACHE_ROOT / "resampled_modules"
FEATURE_FRAME_CACHE_DIR = SHARED_CACHE_ROOT / "feature_frames"
WINDOW_INDEX_CACHE_DIR = SHARED_CACHE_ROOT / "window_index"
LONG_TABLE_PATH = SHARED_CACHE_ROOT / "long_table.parquet"
LONG_TABLE_META_PATH = SHARED_CACHE_ROOT / "long_table_manifest.json"


def split_fingerprint(split_df: pd.DataFrame) -> str:
    h = hashlib.sha1()
    cols = [c for c in ["file_name", "Label"] if c in split_df.columns]
    for item in split_df[cols].astype(str).agg("|".join, axis=1).tolist():
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def window_index_cache_path(split_name: str, split_df: pd.DataFrame, cache_tag: str) -> Path:
    split_sig = split_fingerprint(split_df)
    return WINDOW_INDEX_CACHE_DIR / f"{split_name}_{split_sig}_{cache_tag}_windows.parquet"


def feature_frame_cache_path(split_name: str, split_df: pd.DataFrame, cache_tag: str) -> Path:
    split_sig = split_fingerprint(split_df)
    return FEATURE_FRAME_CACHE_DIR / f"{split_name}_{split_sig}_{cache_tag}_features.pkl"


def _module_cache_path(file_name: str, cache_dir: Path = RESAMPLED_CACHE_DIR) -> Path:
    return cache_dir / f"{Path(file_name).stem}.pkl"


def load_resampled_module(
    file_name: str,
    force_rebuild: bool = False,
    cache_dir: Path = RESAMPLED_CACHE_DIR,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Load a module once as a 5-minute aligned frame and cache it.

    Window experiments can change observation, lead, and prediction lengths many
    times. The expensive, reusable part is CSV IO plus cleaning/resampling, so we
    cache that layer independently from any window configuration.
    """
    raw_path = base.TRAINING_DIR / file_name
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    stat = raw_path.stat()
    cache_path = _module_cache_path(file_name, cache_dir)
    if cache_path.exists() and not force_rebuild:
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if (
                cached.get("source_mtime_ns") == stat.st_mtime_ns
                and cached.get("source_size") == stat.st_size
                and cached.get("resample_version") == base.RESAMPLE_VERSION
                and "resampled" in cached
            ):
                return cached["resampled"], cached["sensors"], cached["meta"]
        except Exception:
            pass

    raw = pd.read_csv(
        raw_path,
        usecols=lambda c: c == "timestamp" or c == "anomaly" or c in base.SENSORS,
        low_memory=False,
    )
    resampled, sensors, meta = base.resample_module(raw)
    payload = {
        "file_name": file_name,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "resample_version": base.RESAMPLE_VERSION,
        "resampled": resampled,
        "sensors": sensors,
        "meta": meta,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return resampled, sensors, meta


def prepare_resampled_cache(
    index_df: pd.DataFrame,
    force_rebuild: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    built = 0
    skipped_missing = 0
    failed = 0
    rows = index_df if limit is None else index_df.head(limit)
    for row in rows.itertuples(index=False):
        file_name = getattr(row, "file_name")
        try:
            load_resampled_module(file_name, force_rebuild=force_rebuild)
            built += 1
        except FileNotFoundError:
            skipped_missing += 1
        except Exception:
            failed += 1
    return {
        "requested": int(len(rows)),
        "built_or_valid": built,
        "skipped_missing": skipped_missing,
        "failed": failed,
    }


def build_long_table_cache(
    index_df: pd.DataFrame,
    force_rebuild: bool = False,
    limit: int | None = None,
    batch_modules: int = 256,
) -> dict[str, object]:
    """Build one module-aware long-table cache.

    The output is a single Parquet file, but every row keeps `file_name` and
    `module_row_idx`, so downstream windows never cross module boundaries.
    """
    if LONG_TABLE_PATH.exists() and LONG_TABLE_META_PATH.exists() and not force_rebuild:
        with open(LONG_TABLE_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    import pyarrow as pa
    import pyarrow.parquet as pq

    SHARED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = LONG_TABLE_PATH.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except PermissionError:
            pass
    if LONG_TABLE_PATH.exists():
        LONG_TABLE_PATH.unlink()

    rows = index_df if limit is None else index_df.head(limit)
    table_cols = [
        "file_name",
        "module_label",
        "module_row_idx",
        "timestamp",
        "observed",
        "anomaly",
        "first_failure_ts",
        "event_label",
        *base.SENSORS,
    ]
    writer = None
    buffers: list[pd.DataFrame] = []
    modules_written = 0
    rows_written = 0
    skipped_missing = 0
    failed = 0

    def flush() -> None:
        nonlocal writer, rows_written
        if not buffers:
            return
        batch = pd.concat(buffers, ignore_index=True)
        arrow_table = pa.Table.from_pandas(batch[table_cols], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(LONG_TABLE_PATH, arrow_table.schema, compression="snappy")
        writer.write_table(arrow_table)
        rows_written += int(len(batch))
        buffers.clear()

    try:
        for module_id, row in enumerate(rows.itertuples(index=False)):
            file_name = getattr(row, "file_name")
            module_label = int(getattr(row, "Label"))
            try:
                resampled, _, _ = load_resampled_module(file_name)
            except FileNotFoundError:
                skipped_missing += 1
                continue
            except Exception:
                failed += 1
                continue

            frame = resampled.copy()
            for sensor in base.SENSORS:
                if sensor not in frame.columns:
                    frame[sensor] = pd.NA
                frame[sensor] = pd.to_numeric(frame[sensor], errors="coerce").astype("float32")
            first_ts = base.first_failure_timestamp(frame)
            frame["file_name"] = str(file_name)
            frame["module_label"] = module_label
            frame["module_row_idx"] = range(len(frame))
            frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").astype("int64")
            frame["observed"] = pd.to_numeric(frame["observed"], errors="coerce").fillna(0).astype("int8")
            frame["anomaly"] = pd.to_numeric(frame["anomaly"], errors="coerce").fillna(0).astype("int8")
            frame["first_failure_ts"] = int(first_ts) if first_ts is not None else -1
            frame["event_label"] = 1 if first_ts is not None else 0
            buffers.append(frame[table_cols])
            modules_written += 1
            if len(buffers) >= batch_modules:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    meta = {
        "path": str(LONG_TABLE_PATH),
        "requested_modules": int(len(rows)),
        "modules_written": int(modules_written),
        "rows_written": int(rows_written),
        "skipped_missing": int(skipped_missing),
        "failed": int(failed),
        "columns": table_cols,
        "format": "parquet",
    }
    with open(LONG_TABLE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def load_long_table(columns: list[str] | None = None) -> pd.DataFrame:
    if not LONG_TABLE_PATH.exists():
        raise FileNotFoundError(LONG_TABLE_PATH)
    return pd.read_parquet(LONG_TABLE_PATH, columns=columns)
