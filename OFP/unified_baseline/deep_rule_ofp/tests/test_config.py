from __future__ import annotations

import pytest

from drfp.config import DRFPConfig


def test_default_protocol_is_the_frozen_single_split() -> None:
    config = DRFPConfig()
    config.validate()
    assert config.split.test_fold == 3
    assert config.task.horizons_hours == (16.0, 24.0, 72.0, 120.0)
    assert config.task.primary_horizon_hours == 120.0
    assert config.sequence_length == 168


def test_horizons_must_be_increasing_and_include_primary() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        DRFPConfig.from_dict({"task": {"horizons_hours": [24, 16, 120]}})
    with pytest.raises(ValueError, match="primary_horizon"):
        DRFPConfig.from_dict(
            {"task": {"horizons_hours": [16, 24, 72], "primary_horizon_hours": 120}}
        )


def test_non_comparable_test_fold_is_rejected() -> None:
    with pytest.raises(ValueError, match="test_fold=3"):
        DRFPConfig.from_dict({"split": {"test_fold": 2}})
