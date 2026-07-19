"""Typed, protocol-locked configuration for Deep-Rule Failure Prediction.

The configuration intentionally separates task, data, architecture, training,
calibration, and decision axes so that an ablation changes one declared axis.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA_VERSION = "drfp-ofp-v1"
MODEL_VARIANTS = (
    "temporal_only",
    "deep_raw_stat",
    "rule_only",
    "fixed_fusion",
    "gated_fusion",
)


@dataclass(frozen=True)
class TaskConfig:
    name: str = "optical_module_failure_prediction"
    horizons_hours: tuple[float, ...] = (16.0, 24.0, 72.0, 120.0)
    primary_horizon_hours: float = 120.0
    first_anomaly_is_failure_time: bool = True
    exclude_at_or_after_failure: bool = True


@dataclass(frozen=True)
class SplitConfig:
    test_fold: int = 3
    validation_fraction: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class InputConfig:
    history_hours: int = 168
    grid_hours: int = 1
    rule_approach_hours: float = 12.0
    rule_persistence_hours: float = 6.0


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "uniform_per_module"
    endpoints_per_module: int = 24


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "gated_fusion"
    d_model: int = 64
    latent_dim: int = 64
    transformer_layers: int = 2
    attention_heads: int = 4
    patch_length: int = 8
    patch_stride: int = 4
    statistical_hidden: int = 128
    rule_hidden: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 8
    batch_size: int = 256
    inference_batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    max_positive_weight: float = 20.0
    fused_loss_weight: float = 1.0
    deep_auxiliary_weight: float = 0.3
    rule_auxiliary_weight: float = 0.2
    early_stopping_patience: int = 2
    num_workers: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class CalibrationConfig:
    method: str = "temperature"
    minimum_temperature: float = 0.05
    maximum_temperature: float = 20.0
    grid_size: int = 161


@dataclass(frozen=True)
class DecisionConfig:
    fixed_threshold: float = 0.5
    tune_on_validation: bool = True
    threshold_candidates: int = 201
    selection_metric: str = "final_score"


@dataclass(frozen=True)
class DRFPConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    experiment_name: str = "drfp_full"
    seed: int = 42
    task: TaskConfig = field(default_factory=TaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DRFPConfig":
        data = copy.deepcopy(dict(payload))
        allowed = {
            "schema_version",
            "experiment_name",
            "seed",
            "task",
            "split",
            "inputs",
            "sampling",
            "model",
            "training",
            "calibration",
            "decision",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown DRFP config fields: {unknown}")
        task_data = dict(data.get("task", {}))
        if "horizons_hours" in task_data:
            task_data["horizons_hours"] = tuple(
                float(value) for value in task_data["horizons_hours"]
            )
        config = cls(
            schema_version=str(data.get("schema_version", CONFIG_SCHEMA_VERSION)),
            experiment_name=str(data.get("experiment_name", "drfp_full")),
            seed=int(data.get("seed", 42)),
            task=TaskConfig(**task_data),
            split=SplitConfig(**data.get("split", {})),
            inputs=InputConfig(**data.get("inputs", {})),
            sampling=SamplingConfig(**data.get("sampling", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            calibration=CalibrationConfig(**data.get("calibration", {})),
            decision=DecisionConfig(**data.get("decision", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"config schema mismatch: {self.schema_version!r} != {CONFIG_SCHEMA_VERSION!r}"
            )
        horizons = self.task.horizons_hours
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("task.horizons_hours must contain positive values")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("task.horizons_hours must be strictly increasing")
        if self.task.primary_horizon_hours not in horizons:
            raise ValueError("primary_horizon_hours must be one configured horizon")
        if not self.task.first_anomaly_is_failure_time:
            raise ValueError("v1 fixes the first anomaly as the failure timestamp")
        if not self.task.exclude_at_or_after_failure:
            raise ValueError("post-failure training endpoints are forbidden")
        if self.split.test_fold != 3:
            raise ValueError("the comparable single-split protocol fixes test_fold=3")
        if not 0.0 < self.split.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.inputs.history_hours <= 0 or self.inputs.grid_hours <= 0:
            raise ValueError("history_hours and grid_hours must be positive")
        if self.inputs.history_hours % self.inputs.grid_hours:
            raise ValueError("history_hours must be divisible by grid_hours")
        if self.inputs.rule_approach_hours <= 0 or self.inputs.rule_persistence_hours <= 0:
            raise ValueError("rule history lengths must be positive")
        if (
            self.inputs.rule_approach_hours != 12.0
            or self.inputs.rule_persistence_hours != 6.0
        ):
            raise ValueError("v1 locks rule approach=12h and persistence=6h")
        if self.sampling.mode != "uniform_per_module":
            raise ValueError("v1 fixes module-uniform sampling; HSS is a later ablation axis")
        if self.sampling.endpoints_per_module <= 0:
            raise ValueError("endpoints_per_module must be positive")
        if self.model.variant not in MODEL_VARIANTS:
            raise ValueError(f"unknown model.variant={self.model.variant!r}")
        if self.model.d_model % self.model.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.model.latent_dim <= 0 or self.model.d_model <= 0:
            raise ValueError("model dimensions must be positive")
        sequence_length = self.inputs.history_hours // self.inputs.grid_hours
        if not 1 <= self.model.patch_length <= sequence_length:
            raise ValueError("patch_length must fit the temporal history")
        if not 1 <= self.model.patch_stride <= self.model.patch_length:
            raise ValueError("patch_stride must be within [1, patch_length]")
        if (
            self.training.epochs <= 0
            or self.training.batch_size <= 0
            or self.training.inference_batch_size <= 0
        ):
            raise ValueError("epochs and train/inference batch sizes must be positive")
        if self.training.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training.device must be auto, cpu, or cuda")
        if self.training.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.training.fused_loss_weight <= 0:
            raise ValueError("fused_loss_weight must be positive")
        if self.training.deep_auxiliary_weight < 0 or self.training.rule_auxiliary_weight < 0:
            raise ValueError("auxiliary loss weights cannot be negative")
        if self.calibration.method not in {"none", "temperature"}:
            raise ValueError("calibration.method must be none or temperature")
        if not 0 < self.calibration.minimum_temperature <= self.calibration.maximum_temperature:
            raise ValueError("invalid calibration temperature range")
        if self.calibration.grid_size < 3:
            raise ValueError("calibration.grid_size must be at least 3")
        if not 0.0 <= self.decision.fixed_threshold <= 1.0:
            raise ValueError("fixed_threshold must be in [0, 1]")
        if self.decision.threshold_candidates < 3:
            raise ValueError("threshold_candidates must be at least 3")
        if self.decision.selection_metric != "final_score":
            raise ValueError("v1 tunes the validation threshold by original OFP final_score")

    @property
    def sequence_length(self) -> int:
        return self.inputs.history_hours // self.inputs.grid_hours

    @property
    def primary_horizon_index(self) -> int:
        return self.task.horizons_hours.index(self.task.primary_horizon_hours)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> DRFPConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        payload = deep_merge(payload, overrides)
    return DRFPConfig.from_dict(payload)
