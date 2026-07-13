from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from fgofp.config import FGOFPConfig
from fgofp.training import (
    module_normalized_positive_weight,
    seed_everything,
    train_models,
    training_coverage,
)


def _sample(
    name: str,
    student_targets: list[int],
    *,
    teacher_targets: list[int] | None = None,
    fgl_sources: dict[int, int] | None = None,
    first_fault_index: int = -1,
) -> dict[str, object]:
    length = len(student_targets)
    generator = torch.Generator().manual_seed(sum(ord(char) for char in name))
    fgl_sources = fgl_sources or {}
    teacher_index = torch.full((length,), -1, dtype=torch.long)
    fgl_mask = torch.zeros(length, dtype=torch.bool)
    for source, target in fgl_sources.items():
        teacher_index[source] = target
        fgl_mask[source] = True
    anomaly = torch.zeros(length)
    if first_fault_index >= 0:
        anomaly[first_fault_index] = 1.0
    return {
        "raw": torch.randn(length, 12, generator=generator),
        "raw_mask": torch.ones(length, 12, dtype=torch.bool),
        "delta_steps": torch.cat((torch.zeros(1), torch.ones(length - 1))).unsqueeze(-1),
        "targets_teacher": torch.tensor(
            teacher_targets if teacher_targets is not None else student_targets,
            dtype=torch.long,
        ),
        "targets_student": torch.tensor(student_targets, dtype=torch.long),
        "loss_mask": torch.ones(length, dtype=torch.bool),
        "teacher_index": teacher_index,
        "fgl_mask": fgl_mask,
        "timestamps": torch.arange(length, dtype=torch.long) * 300,
        "anomaly": anomaly,
        "length": length,
        "file_name": name,
        "first_fault_index": first_fault_index,
    }


class SyntheticSequenceDataset(Dataset):
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self.samples = samples
        self.lengths = np.asarray([int(sample["length"]) for sample in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


class CountingRowModel(nn.Module):
    """Small model with the production forward signature and call audit."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(25, 2)
        self.forward_training_modes: list[bool] = []

    def forward(
        self,
        raw: torch.Tensor,
        raw_mask: torch.Tensor,
        delta_steps: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_training_modes.append(bool(self.training))
        inputs = torch.cat(
            (
                raw,
                raw_mask.to(dtype=raw.dtype),
                delta_steps,
            ),
            dim=-1,
        )
        return self.projection(inputs) * padding_mask.unsqueeze(-1)


class BombTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(()))

    def forward(self, *args, **kwargs):  # pragma: no cover - must never execute
        raise AssertionError("CE-only training called the teacher")


def _datasets() -> tuple[SyntheticSequenceDataset, SyntheticSequenceDataset]:
    train = SyntheticSequenceDataset(
        [
            _sample(
                "faulty.csv",
                [0, 1, 1, 1],
                teacher_targets=[0, 0, 1, 1],
                fgl_sources={1: 2},
                first_fault_index=3,
            ),
            _sample(
                "normal.csv",
                [0, 0, 0, 0, 0],
                fgl_sources={0: 2},
            ),
            _sample("first_row_fault.csv", [1, 0, 0], first_fault_index=0),
        ]
    )
    validation = SyntheticSequenceDataset(
        [
            _sample(
                "validation.csv",
                [0, 1, 1, 1],
                teacher_targets=[0, 0, 1, 1],
                fgl_sources={1: 2},
                first_fault_index=3,
            )
        ]
    )
    return train, validation


def _config(alpha: float) -> FGOFPConfig:
    config = FGOFPConfig()
    config.seed = 19
    config.training.device = "cpu"
    config.training.mixed_precision = False
    config.training.teacher_epochs = 1
    config.training.student_epochs = 1
    config.training.batch_size = 16
    config.training.num_workers = 0
    config.training.early_stopping_patience = 0
    config.training.teacher_learning_rate = 1e-2
    config.training.student_learning_rate = 1e-2
    config.training.weight_decay = 0.0
    config.training.positive_weight_mode = "none"
    config.training.fgl_alpha = alpha
    return config


def test_module_normalized_posweight_and_coverage_diagnostics():
    dataset = SyntheticSequenceDataset(
        [
            _sample("half-positive.csv", [0, 0, 1, 1], fgl_sources={2: 3}),
            _sample("normal.csv", [0, 0, 0, 0]),
        ]
    )
    # Equal-module mass is positive=.5+0 and negative=.5+1, hence 3.
    assert module_normalized_positive_weight(
        dataset, target="student", maximum=10.0
    ) == pytest.approx(3.0)
    assert module_normalized_positive_weight(
        dataset, target="student", mode="none", maximum=10.0
    ) == 1.0
    assert module_normalized_positive_weight(
        dataset, target="student", maximum=2.0
    ) == 2.0
    coverage = training_coverage(dataset)
    assert coverage == {
        "modules": 2,
        "ce_modules": 2,
        "student_ce_rows": 8,
        "student_ce_positive_rows": 2,
        "teacher_ce_rows": 8,
        "teacher_ce_positive_rows": 2,
        "fgl_modules": 1,
        "fgl_pairs": 1,
        "positive_fgl_modules": 1,
        "positive_fgl_pairs": 1,
        "first_row_faults": 0,
    }


def test_ce_only_never_calls_or_returns_teacher():
    train, validation = _datasets()
    seed_everything(3)
    result = train_models(
        CountingRowModel(), BombTeacher(), train, validation, _config(alpha=1.0)
    )
    assert result.teacher_train_only is None
    assert set(result.history["stage"]) == {"student"}
    assert result.best_epochs == {"teacher": None, "student": 1}
    assert result.best_validation_losses["teacher"] is None
    assert result.coverage["positive_fgl_pairs"] == 1


def test_teacher_is_frozen_and_student_validation_never_calls_it():
    train, validation = _datasets()
    seed_everything(4)
    teacher = CountingRowModel()
    result = train_models(
        CountingRowModel(), teacher, train, validation, _config(alpha=0.7)
    )
    assert result.teacher_train_only is teacher
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    # Exactly one teacher-train batch, one teacher-validation batch, and one
    # future-teacher batch during student training. Student validation is absent.
    assert teacher.forward_training_modes == [True, False, False]
    assert list(result.history["stage"]) == ["teacher", "student"]
    assert result.history.iloc[-1]["validation_ce_loss"] > 0
    assert result.history.iloc[-1]["train_fgl_pairs"] == 2
    assert result.coverage["first_row_faults"] == 1


def test_best_student_checkpoint_can_be_reloaded_exactly():
    train, validation = _datasets()
    seed_everything(5)
    result = train_models(
        CountingRowModel(), BombTeacher(), train, validation, _config(alpha=1.0)
    )
    checkpoint = copy.deepcopy(result.student.state_dict())
    reloaded = CountingRowModel()
    reloaded.load_state_dict(checkpoint)
    reloaded.eval()
    sample = validation[0]
    raw = sample["raw"].unsqueeze(0)
    raw_mask = sample["raw_mask"].unsqueeze(0)
    delta = sample["delta_steps"].unsqueeze(0)
    padding = torch.ones(1, int(sample["length"]), dtype=torch.bool)
    with torch.no_grad():
        expected = result.student(raw, raw_mask, delta, padding)
        actual = reloaded(raw, raw_mask, delta, padding)
    assert torch.equal(expected, actual)
    assert isinstance(result.history, pd.DataFrame)


def test_fgl_fails_early_without_positive_aligned_pairs():
    train = SyntheticSequenceDataset([_sample("normal.csv", [0, 0, 0])])
    validation = SyntheticSequenceDataset([_sample("validation.csv", [0, 0])])
    with pytest.raises(ValueError, match="zero positive FGL pairs"):
        train_models(
            CountingRowModel(), CountingRowModel(), train, validation, _config(0.7)
        )
