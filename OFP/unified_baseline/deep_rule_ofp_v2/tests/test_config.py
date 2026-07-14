from __future__ import annotations

import json
from pathlib import Path

import pytest

from fgofp.config import config_from_dict, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_freezes_native_legacy_fgl_contract() -> None:
    config = load_config(ROOT / "configs" / "default.json")
    assert config.protocol.name == "legacy_inclusive_v1"
    assert config.inputs.cadence_mode == "native_observed_rows"
    assert config.task.teacher_horizon_hours + config.task.future_offset_hours == 120.0
    assert config.model.architecture == "rule_guided_residual_tcn"
    assert config.decision.fallback_policy == "validation_safe_rule"
    assert config.receptive_field_steps >= config.student_horizon_steps
    assert len(config.fingerprint()) == 64


def test_misaligned_fgl_horizons_are_rejected() -> None:
    payload = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    payload["task"]["future_offset_hours"] = 24.0
    with pytest.raises(ValueError, match="teacher_horizon_hours"):
        config_from_dict(payload)


def test_student_window_is_frozen_at_120_hours() -> None:
    payload = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    payload["task"].update(
        {
            "student_horizon_hours": 72.0,
            "teacher_horizon_hours": 66.0,
            "future_offset_hours": 6.0,
        }
    )
    with pytest.raises(ValueError, match="120 hours"):
        config_from_dict(payload)


def test_only_legacy_inclusive_and_native_rows_are_supported() -> None:
    payload = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    payload["protocol"]["name"] = "strict"
    with pytest.raises(ValueError, match="only legacy_inclusive"):
        config_from_dict(payload)

    payload = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    payload["inputs"]["cadence_mode"] = "hourly"
    with pytest.raises(ValueError, match="only native_observed_rows"):
        config_from_dict(payload)


@pytest.mark.parametrize("test_fold", (1, 2, 3))
def test_each_outer_test_fold_is_supported(test_fold: int) -> None:
    payload = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    payload["split"]["test_fold"] = test_fold
    assert config_from_dict(payload).split.test_fold == test_fold


@pytest.mark.parametrize("test_fold", (0, 4))
def test_unknown_outer_test_fold_is_rejected(test_fold: int) -> None:
    payload = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    payload["split"]["test_fold"] = test_fold
    with pytest.raises(ValueError, match="one of 1, 2, or 3"):
        config_from_dict(payload)
