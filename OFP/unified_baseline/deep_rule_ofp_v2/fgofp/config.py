"""Validated configuration for native sequence-to-sequence FGL-OFP."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar


@dataclass
class ProtocolConfig:
    name: str = "legacy_inclusive_v1"
    hit_condition: str = "first_alarm_timestamp <= first_failure_timestamp"
    timestamp_dtype: str = "int64_seconds"


@dataclass
class TaskConfig:
    name: str = "optical_module_failure_prediction"
    student_horizon_hours: float = 120.0
    teacher_horizon_hours: float = 114.0
    future_offset_hours: float = 6.0
    include_first_failure_row: bool = True


@dataclass
class SplitConfig:
    test_fold: int = 3
    validation_fraction: float = 0.1
    seed: int = 42


@dataclass
class InputConfig:
    cadence_mode: str = "native_observed_rows"
    expected_cadence_seconds: int = 300
    delta_clip_steps: float = 288.0
    alignment_tolerance_seconds: int = 300


@dataclass
class ModelConfig:
    architecture: str = "causal_depthwise_tcn"
    hidden_channels: int = 48
    dilation_blocks: int = 10
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    teacher_epochs: int = 5
    student_epochs: int = 8
    batch_size: int = 8
    inference_batch_size: int = 8
    teacher_learning_rate: float = 3e-4
    student_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    fgl_alpha: float = 0.7
    temperature: float = 4.0
    positive_weight_mode: str = "module_normalized_auto"
    max_positive_weight: float = 5.0
    mixed_precision: bool = True
    early_stopping_patience: int = 2
    num_workers: int = 0
    device: str = "auto"


@dataclass
class DecisionConfig:
    fixed_threshold: float = 0.5
    tune_on_validation: bool = True
    threshold_grid_size: int = 201
    threshold_quantile_count: int = 201
    selection_metric: str = "final_score"


@dataclass
class FGOFPConfig:
    schema_version: str = "fgofp-native-seq-v1"
    experiment_name: str = "FGLSeq_LegacyInclusive"
    seed: int = 42
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def __post_init__(self) -> None:
        if self.schema_version != "fgofp-native-seq-v1":
            raise ValueError("schema_version must be 'fgofp-native-seq-v1'")
        if self.protocol.name != "legacy_inclusive_v1":
            raise ValueError("v2 supports only legacy_inclusive_v1")
        if self.protocol.timestamp_dtype != "int64_seconds":
            raise ValueError("timestamps must use int64 seconds")
        if not self.task.include_first_failure_row:
            raise ValueError("Legacy-Inclusive training must include the first failure row")
        horizons = (
            float(self.task.teacher_horizon_hours),
            float(self.task.future_offset_hours),
            float(self.task.student_horizon_hours),
        )
        if min(horizons) <= 0:
            raise ValueError("teacher horizon, future offset, and student horizon must be positive")
        if abs(horizons[0] + horizons[1] - horizons[2]) > 1e-9:
            raise ValueError(
                "FGL requires teacher_horizon_hours + future_offset_hours "
                "== student_horizon_hours"
            )
        if abs(horizons[2] - 120.0) > 1e-9:
            raise ValueError("the frozen v2 student future window is 120 hours")
        if self.split.test_fold != 3:
            raise ValueError("the comparable OFP protocol fixes test_fold=3")
        if not 0.0 < self.split.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.inputs.cadence_mode != "native_observed_rows":
            raise ValueError("v2 supports only native_observed_rows")
        if self.inputs.expected_cadence_seconds <= 0:
            raise ValueError("expected_cadence_seconds must be positive")
        if self.inputs.delta_clip_steps <= 0:
            raise ValueError("delta_clip_steps must be positive")
        if self.inputs.alignment_tolerance_seconds < 0:
            raise ValueError("alignment_tolerance_seconds cannot be negative")
        if self.inputs.alignment_tolerance_seconds > self.inputs.expected_cadence_seconds:
            raise ValueError(
                "alignment_tolerance_seconds cannot exceed one expected cadence step"
            )
        if self.model.architecture != "causal_depthwise_tcn":
            raise ValueError("v2 supports only causal_depthwise_tcn")
        if self.model.hidden_channels < 8:
            raise ValueError("hidden_channels must be at least 8")
        if self.model.dilation_blocks < 1:
            raise ValueError("dilation_blocks must be positive")
        if self.model.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.receptive_field_steps < self.student_horizon_steps:
            raise ValueError(
                "model receptive field is shorter than the 120-hour native-cadence horizon"
            )
        for name in ("teacher_epochs", "student_epochs", "batch_size", "inference_batch_size"):
            if int(getattr(self.training, name)) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if not 0.0 < self.training.fgl_alpha <= 1.0:
            raise ValueError("fgl_alpha must be in (0, 1]")
        if self.training.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.training.positive_weight_mode not in {"none", "module_normalized_auto"}:
            raise ValueError(
                "positive_weight_mode must be none or module_normalized_auto"
            )
        if self.training.max_positive_weight < 1.0:
            raise ValueError("max_positive_weight must be at least 1")
        if self.training.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training.device must be auto, cpu, or cuda")
        if self.training.teacher_learning_rate <= 0 or self.training.student_learning_rate <= 0:
            raise ValueError("training learning rates must be positive")
        if self.training.weight_decay < 0 or self.training.grad_clip < 0:
            raise ValueError("weight_decay and grad_clip cannot be negative")
        if self.training.early_stopping_patience < 0 or self.training.num_workers < 0:
            raise ValueError("patience and num_workers cannot be negative")
        if not 0.0 <= self.decision.fixed_threshold <= 1.0:
            raise ValueError("fixed_threshold must be in [0, 1]")
        if self.decision.selection_metric != "final_score":
            raise ValueError("Legacy-Inclusive threshold selection uses final_score")
        if (
            self.decision.threshold_grid_size < 2
            or self.decision.threshold_quantile_count < 2
        ):
            raise ValueError("threshold candidate counts must be at least 2")

    @property
    def student_horizon_steps(self) -> int:
        seconds = self.task.student_horizon_hours * 3600.0
        return int(round(seconds / self.inputs.expected_cadence_seconds)) + 1

    @property
    def receptive_field_steps(self) -> int:
        dilation_sum = (2 ** int(self.model.dilation_blocks)) - 1
        return 1 + (int(self.model.kernel_size) - 1) * dilation_sum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


T = TypeVar("T")


def _construct(cls: Type[T], payload: Mapping[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**dict(payload))


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def config_from_dict(payload: Mapping[str, Any]) -> FGOFPConfig:
    data = dict(payload)
    nested = {
        "protocol": ProtocolConfig,
        "task": TaskConfig,
        "split": SplitConfig,
        "inputs": InputConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "decision": DecisionConfig,
    }
    for name, cls in nested.items():
        if name in data:
            value = data[name]
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
            data[name] = _construct(cls, value)
    return _construct(FGOFPConfig, data)


def load_config(path: Path | str, overrides: Mapping[str, Any] | None = None) -> FGOFPConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if overrides:
        payload = _deep_merge(payload, overrides)
    return config_from_dict(payload)


__all__ = [
    "DecisionConfig",
    "FGOFPConfig",
    "InputConfig",
    "ModelConfig",
    "ProtocolConfig",
    "SplitConfig",
    "TaskConfig",
    "TrainingConfig",
    "config_from_dict",
    "load_config",
]
