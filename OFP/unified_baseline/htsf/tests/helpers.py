from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HTSF_DIR = Path(__file__).resolve().parents[1]
UNIFIED_DIR = HTSF_DIR.parent


def tiny_payload() -> dict:
    payload = json.loads((HTSF_DIR / "configs" / "protocol.json").read_text(encoding="utf-8"))
    payload["folds"] = [1]
    payload["window"].update(
        {"sequence_length": 6, "patch_length": 3, "patch_stride": 1}
    )
    payload["sampling"].update(
        {
            "mode": "hss",
            "windows_per_module": 4,
            "positive_windows_per_faulty_module": 4,
            "negative_windows_per_faulty_module": 2,
            "normal_windows_per_module": 2,
        }
    )
    payload["representation"].update(
        {
            "d_model": 8,
            "latent_dim": 8,
            "transformer_layers": 1,
            "attention_heads": 2,
            "engineered_hidden": 16,
            "dropout": 0.0,
        }
    )
    payload["training"].update(
        {"epochs": 1, "batch_size": 8, "device": "cpu"}
    )
    payload["decision"]["xgboost"].update(
        {"device": "cpu", "num_boost_round": 2, "nthread": 1, "max_depth": 2}
    )
    return payload


def write_module(path: Path, *, length: int = 12, fault_row: int | None = None, offset: float = 0.0) -> None:
    from ofp_unified.features import RAW_FEATURES

    row = np.arange(length, dtype=np.float32)
    data = {"timestamp": 1_700_000_000 + np.arange(length, dtype=np.int64) * 3600}
    for index, name in enumerate(RAW_FEATURES):
        data[name] = offset + (index + 1) * 10.0 + row * (0.2 + index * 0.01)
    anomaly = np.zeros(length, dtype=np.int8)
    if fault_row is not None:
        anomaly[int(fault_row)] = 1
    data["anomaly"] = anomaly
    pd.DataFrame(data).to_csv(path, index=False)
