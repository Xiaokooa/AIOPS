"""Typed configuration and comparison-axis fingerprints for clean HTSF."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ofp_unified.features import FEATURE_SCHEMA_VERSION


CONFIG_SCHEMA_VERSION = "htsf-clean-v1"


@dataclass(frozen=True)
class TaskConfig:
    target: str = "first_event"
    horizon_hours: float = 120.0
    exclude_at_or_after_first_fault: bool = True


@dataclass(frozen=True)
class WindowConfig:
    sequence_length: int = 32
    patch_length: int = 8
    patch_stride: int = 4
    left_padding: str = "zero_with_mask"
    statistical_history: str = "module_prefix"


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "uniform_per_module"
    windows_per_module: int = 32
    positive_windows_per_faulty_module: int = 32
    negative_windows_per_faulty_module: int = 8
    normal_windows_per_module: int = 8


@dataclass(frozen=True)
class WeightingConfig:
    mode: str = "none"
    temporal_positive_max_extra: float = 0.0
    adaptive_negative_max_extra: float = 0.0
    adaptive_warmup_epoch: int = 1


@dataclass(frozen=True)
class RepresentationConfig:
    temporal_encoder: str = "patchtst_style"
    d_model: int = 64
    latent_dim: int = 64
    transformer_layers: int = 2
    attention_heads: int = 4
    engineered_hidden: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    device: str = "auto"


@dataclass(frozen=True)
class DecisionConfig:
    model: str = "xgboost"
    threshold_policy: str = "fixed"
    threshold: float = 0.5
    use_sample_weight: bool = False
    xgboost: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "base_score": 0.5,
            "tree_method": "hist",
            "device": "auto",
            "max_depth": 6,
            "eta": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "max_bin": 256,
            "seed": 42,
            "nthread": 4,
            "num_boost_round": 100,
        }
    )


@dataclass(frozen=True)
class HTSFConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    experiment_name: str = "htsf_clean"
    seed: int = 42
    folds: tuple[int, ...] = (1, 2, 3)
    task: TaskConfig = field(default_factory=TaskConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HTSFConfig":
        data = copy.deepcopy(dict(payload))
        allowed = {
            "schema_version",
            "feature_schema_version",
            "experiment_name",
            "seed",
            "folds",
            "task",
            "window",
            "sampling",
            "weighting",
            "representation",
            "training",
            "decision",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown top-level HTSF config fields: {sorted(unknown)}")
        config = cls(
            schema_version=str(data.get("schema_version", CONFIG_SCHEMA_VERSION)),
            feature_schema_version=str(
                data.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
            ),
            experiment_name=str(data.get("experiment_name", "htsf_clean")),
            seed=int(data.get("seed", 42)),
            folds=tuple(int(value) for value in data.get("folds", (1, 2, 3))),
            task=TaskConfig(**data.get("task", {})),
            window=WindowConfig(**data.get("window", {})),
            sampling=SamplingConfig(**data.get("sampling", {})),
            weighting=WeightingConfig(**data.get("weighting", {})),
            representation=RepresentationConfig(**data.get("representation", {})),
            training=TrainingConfig(**data.get("training", {})),
            decision=DecisionConfig(**data.get("decision", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"config schema mismatch: {self.schema_version!r} != {CONFIG_SCHEMA_VERSION!r}"
            )
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "unified feature schema mismatch: "
                f"{self.feature_schema_version!r} != {FEATURE_SCHEMA_VERSION!r}"
            )
        if self.task.target != "first_event":
            raise ValueError("clean HTSF fixes task.target='first_event'")
        if self.task.horizon_hours <= 0:
            raise ValueError("task.horizon_hours must be positive")
        if not self.task.exclude_at_or_after_first_fault:
            raise ValueError("post-fault training endpoints are forbidden")
        if self.window.sequence_length <= 0:
            raise ValueError("window.sequence_length must be positive")
        if not 1 <= self.window.patch_length <= self.window.sequence_length:
            raise ValueError("patch_length must be within [1, sequence_length]")
        if self.window.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if self.window.patch_stride > self.window.patch_length:
            raise ValueError("patch_stride cannot exceed patch_length because that leaves temporal gaps")
        if self.window.left_padding != "zero_with_mask":
            raise ValueError("only causal zero_with_mask left padding is supported")
        if self.window.statistical_history != "module_prefix":
            raise ValueError("statistical features must use the causal module prefix")
        if self.sampling.mode not in {
            "uniform_per_module",
            "module_quota",
            "hss",
        }:
            raise ValueError(
                "sampling.mode must be uniform_per_module, module_quota, or hss; "
                "use the strict B2 runner for all-row training"
            )
        for name in (
            "windows_per_module",
            "positive_windows_per_faulty_module",
            "negative_windows_per_faulty_module",
            "normal_windows_per_module",
        ):
            if int(getattr(self.sampling, name)) <= 0:
                raise ValueError(f"sampling.{name} must be positive")
        if self.weighting.mode not in {"none", "tpw", "anw", "tpw_anw"}:
            raise ValueError("weighting.mode must be none, tpw, anw, or tpw_anw")
        if self.weighting.mode == "none" and (
            self.weighting.temporal_positive_max_extra != 0
            or self.weighting.adaptive_negative_max_extra != 0
        ):
            raise ValueError("inactive weighting gains must be zero in mode='none'")
        if self.weighting.mode == "tpw" and self.weighting.adaptive_negative_max_extra != 0:
            raise ValueError("adaptive negative gain is only valid for mode='tpw_anw'")
        if self.weighting.mode == "anw" and self.weighting.temporal_positive_max_extra != 0:
            raise ValueError("temporal positive gain must be zero for mode='anw'")
        if self.weighting.temporal_positive_max_extra < 0:
            raise ValueError("temporal_positive_max_extra cannot be negative")
        if self.weighting.adaptive_negative_max_extra < 0:
            raise ValueError("adaptive_negative_max_extra cannot be negative")
        if self.weighting.mode in {"anw", "tpw_anw"} and not (
            1 <= self.weighting.adaptive_warmup_epoch < self.training.epochs
        ):
            raise ValueError("ANW warmup must occur before the final representation epoch")
        if self.representation.temporal_encoder != "patchtst_style":
            raise ValueError("the canonical clean implementation fixes patchtst_style")
        if self.representation.latent_dim <= 0 or self.representation.d_model <= 0:
            raise ValueError("representation dimensions must be positive")
        if self.representation.d_model % self.representation.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.representation.latent_dim % self.representation.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        if self.training.epochs <= 0 or self.training.batch_size <= 0:
            raise ValueError("training epochs and batch size must be positive")
        if self.training.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training.device must be auto, cpu, or cuda")
        if self.decision.model != "xgboost":
            raise ValueError("the canonical decision layer is one XGBoost model")
        if self.decision.threshold_policy != "fixed":
            raise ValueError("architecture comparison requires a fixed threshold policy")
        if not 0.0 <= self.decision.threshold <= 1.0:
            raise ValueError("decision.threshold must be in [0, 1]")
        if not self.folds or len(self.folds) != len(set(self.folds)):
            raise ValueError("folds must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def axis_signature(self, changed_axis: str) -> str:
        """Hash every axis except the one intentionally changed by a suite."""

        data = self.to_dict()
        data.pop("experiment_name", None)
        ignored = {
            "architecture": (),
            "sampling": ("sampling",),
            "weighting": ("weighting",),
        }
        if changed_axis not in ignored:
            raise ValueError(f"unknown comparison axis={changed_axis!r}")
        for key in ignored[changed_axis]:
            data.pop(key, None)
        return fingerprint(data)


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> HTSFConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        payload = deep_merge(payload, overrides)
    return HTSFConfig.from_dict(payload)
