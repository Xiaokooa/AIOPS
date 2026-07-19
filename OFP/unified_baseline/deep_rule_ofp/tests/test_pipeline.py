from __future__ import annotations

from pathlib import Path

import pandas as pd

from drfp.config import DRFPConfig
from drfp.pipeline import (
    _fingerprint_records,
    _write_primary_predictions,
    build_model,
)


def test_split_fingerprint_includes_membership() -> None:
    base = pd.DataFrame(
        {
            "file_name": ["a.csv", "b.csv"],
            "folder_index": [1, 2],
            "Label": [0, 1],
            "split": ["train", "validation"],
        }
    )
    swapped = base.copy()
    swapped["split"] = ["validation", "train"]
    assert _fingerprint_records(base) != _fingerprint_records(swapped)


def test_prediction_writer_rebuilds_exact_file_set(tmp_path: Path) -> None:
    output = tmp_path / "predictions"
    output.mkdir()
    (output / "stale.csv").write_text("timestamp,predict\n0,1\n", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "module_id": [0, 0, 1],
            "timestamp": [2, 1, 1],
            "fusion_score": [0.8, 0.2, 0.5],
        }
    )
    _write_primary_predictions(
        frame,
        ["a.csv", "b.csv"],
        output,
        score_col="fusion_score",
        threshold=0.5,
        overwrite=True,
    )
    assert {path.name for path in output.iterdir()} == {"a.csv", "b.csv"}
    assert pd.read_csv(output / "a.csv")["timestamp"].tolist() == [1, 2]


def test_model_factory_matches_frozen_dimensions() -> None:
    config = DRFPConfig()
    model = build_model(config)
    assert model.sequence_length == 168
    assert model.stats_dim == 76
    assert model.rule_dim == 156
