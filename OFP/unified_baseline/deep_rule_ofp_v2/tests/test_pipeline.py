from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fgofp.config import FGOFPConfig
from fgofp.data import ModuleRecord, NativeSequenceDataset, Standardizer
from fgofp.features import RAW_FEATURES
from fgofp.pipeline import infer_student, run_experiment
from fgofp.model import build_teacher_student
from fgofp.training import seed_everything


class FixedStudent(torch.nn.Module):
    def forward(self, raw, raw_mask, delta_steps, padding_mask):
        score_logit = raw[..., 0]
        return torch.stack((-score_logit, score_logit), dim=-1) * padding_mask.unsqueeze(-1)


def test_seeded_student_initialization_is_suite_reproducible() -> None:
    config = FGOFPConfig()
    seed_everything(config.seed)
    first, _ = build_teacher_student(config.model)
    seed_everything(config.seed)
    second, _ = build_teacher_student(config.model)
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), second.state_dict().values())
    )


def _record(name: str, rows: int) -> ModuleRecord:
    raw = np.zeros((rows, len(RAW_FEATURES)), dtype=np.float32)
    raw[:, 0] = np.linspace(-1.0, 1.0, rows)
    return ModuleRecord(
        file_name=name,
        timestamps=np.arange(rows, dtype=np.int64) * 300 + 1_700_000_000,
        raw=raw,
        raw_mask=np.ones_like(raw, dtype=bool),
        anomaly=np.zeros(rows, dtype=np.float32),
        first_fault_index=None,
    )


def test_student_only_inference_returns_every_native_int64_row() -> None:
    records = [_record("a.csv", 3), _record("b.csv", 5)]
    dataset = NativeSequenceDataset(
        records,
        standardizer=Standardizer.fit(records),
        student_horizon_hours=120,
        teacher_horizon_hours=114,
        offset_hours=6,
    )
    config = FGOFPConfig()
    config.training.device = "cpu"
    config.training.inference_batch_size = 2
    config.training.num_workers = 0
    frame = infer_student(FixedStudent(), dataset, config, progress_label="unit")
    assert len(frame) == 8
    assert frame["timestamp"].dtype == np.int64
    assert frame.groupby("file_name", observed=True).size().to_dict() == {
        "a.csv": 3,
        "b.csv": 5,
    }


def _write_module(path: Path, faulty: bool, first_row_fault: bool = False) -> None:
    rows = 4
    timestamps = np.arange(rows, dtype=np.int64) * 21_600 + 1_700_000_000
    anomaly = np.zeros(rows, dtype=np.int8)
    if faulty:
        anomaly[0 if first_row_fault else -1] = 1
    payload = {
        "timestamp": timestamps,
        **{
            name: np.linspace(index + 1.0, index + 1.3, rows)
            for index, name in enumerate(RAW_FEATURES)
        },
        "anomaly": anomaly,
    }
    pd.DataFrame(payload).to_csv(path, index=False)


def test_tiny_end_to_end_writes_student_only_deployment_checkpoint(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    rows: list[dict[str, int | str]] = []
    module_index = 0
    for fold in (1, 2, 3):
        for faulty in (False, True, False, True):
            name = f"m{module_index:03d}.csv"
            _write_module(
                data_dir / name,
                faulty=faulty,
                first_row_fault=faulty and module_index % 4 == 1,
            )
            rows.append(
                {"file_name": name, "folder_index": fold, "Label": int(faulty)}
            )
            module_index += 1
    index_path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)

    config = FGOFPConfig()
    config.experiment_name = "pipeline_unit"
    config.split.validation_fraction = 0.25
    config.model.hidden_channels = 8
    config.model.dropout = 0.0
    config.training.device = "cpu"
    config.training.mixed_precision = False
    config.training.fgl_alpha = 1.0
    config.training.positive_weight_mode = "none"
    config.training.teacher_epochs = 1
    config.training.student_epochs = 1
    config.training.batch_size = 4
    config.training.inference_batch_size = 4
    config.training.early_stopping_patience = 0
    config.decision.threshold_grid_size = 3
    config.decision.threshold_quantile_count = 3

    output = tmp_path / "artifacts"
    stale_checkpoints = output / "checkpoints"
    stale_checkpoints.mkdir(parents=True)
    (stale_checkpoints / "teacher_train_only.pt").write_bytes(b"stale")
    stale_original_predictions = output / "ofp_predictions_original_fixed_0.5"
    stale_original_predictions.mkdir(parents=True)
    (stale_original_predictions / "stale.csv").write_text(
        "timestamp,score,predict\n0,1,1\n", encoding="utf-8"
    )
    manifest = run_experiment(
        config,
        data_dir,
        index_path,
        output,
        overwrite=True,
        progress=False,
    )
    checkpoint = torch.load(
        output / "checkpoints" / "student_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["teacher_used_at_inference"] is False
    assert checkpoint["input_config"]["expected_cadence_seconds"] == 300
    assert checkpoint["input_config"]["delta_clip_steps"] == 288.0
    assert checkpoint["task_config"]["student_horizon_hours"] == 120.0
    assert len(checkpoint["config_fingerprint"]) == 64
    assert not any(name.startswith("teacher") for name in checkpoint["state_dict"])
    assert not (output / "checkpoints" / "teacher_train_only.pt").exists()
    assert manifest["data_access_order"].index("validation_threshold_frozen") < manifest[
        "data_access_order"
    ].index("test_csv")
    assert manifest["deployment"]["teacher_used_at_inference"] is False
    assert pd.read_csv(output / "result.csv").loc[0, "protocol"] == "legacy_inclusive_v1"
    original_result = pd.read_csv(output / "result_ofp_original.csv").iloc[0]
    assert original_result["protocol"] == "ofp_original_strict_v1"
    assert original_result["decision_threshold"] == 0.5
    assert (output / "test_module_decisions_ofp_original.csv").is_file()
    expected_test_files = {
        row["file_name"] for row in rows if row["folder_index"] == 3
    }
    assert {
        path.name for path in (output / "ofp_predictions").glob("*.csv")
    } == expected_test_files
    original_prediction_dir = output / "ofp_predictions_original_fixed_0.5"
    assert {
        path.name for path in original_prediction_dir.glob("*.csv")
    } == expected_test_files
    assert not (original_prediction_dir / "stale.csv").exists()
    for path in original_prediction_dir.glob("*.csv"):
        prediction = pd.read_csv(path)
        expected_predict = (prediction["score"] >= 0.5).astype(np.int8)
        assert prediction["predict"].astype(np.int8).equals(expected_predict)
    persisted = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert persisted["task"]["student_horizon_hours"] == 120.0
    assert persisted["ofp_original_compatibility"]["lead_denominator"] == "all_faulty_modules"
