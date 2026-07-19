"""Validation-only temperature calibration for cumulative horizon risks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


BRANCHES = ("deep", "rule", "fused")


def cumulative_probabilities(hazard_logits: np.ndarray) -> np.ndarray:
    """Convert conditional interval hazard logits to monotone cumulative risk."""

    logits = np.asarray(hazard_logits, dtype=np.float64)
    hazards = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    return 1.0 - np.cumprod(1.0 - hazards, axis=-1)


def binary_cross_entropy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same shape")
    return float(
        -np.mean(targets * np.log(probabilities) + (1.0 - targets) * np.log1p(-probabilities))
    )


def fit_temperature(
    hazard_logits: np.ndarray,
    targets: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 20.0,
    grid_size: int = 161,
) -> tuple[float, float]:
    """Fit one positive temperature without changing sample ordering.

    A deterministic log-spaced search avoids an additional SciPy dependency.
    The loss is the unweighted validation BCE, because calibration should undo
    the distortion introduced by class-weighted training rather than repeat it.
    """

    logits = np.asarray(hazard_logits, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    if logits.ndim != 2 or logits.shape != labels.shape or not len(logits):
        raise ValueError("calibration expects non-empty aligned [row, horizon] arrays")
    if minimum <= 0 or maximum < minimum or grid_size < 3:
        raise ValueError("invalid temperature search range")
    # A full 161-point scan over millions of validation timestamps is wasteful.
    # Use a deterministic coarse log-grid followed by bounded golden-section
    # refinement in log-temperature space.  The configured grid_size remains
    # the search budget; the formal default needs at most 45 loss evaluations.
    coarse_size = min(int(grid_size), 21)
    candidates = np.geomspace(float(minimum), float(maximum), coarse_size)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return binary_cross_entropy(
            cumulative_probabilities(logits / temperature), labels
        )

    losses = np.asarray([objective(float(np.log(value))) for value in candidates])
    best = int(np.nanargmin(losses))
    best_temperature = float(candidates[best])
    best_loss = float(losses[best])
    if grid_size <= coarse_size:
        return best_temperature, best_loss

    left_index = max(0, best - 1)
    right_index = min(len(candidates) - 1, best + 1)
    left = float(np.log(candidates[left_index]))
    right = float(np.log(candidates[right_index]))
    if left == right:
        return best_temperature, best_loss
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    refinements = min(24, int(grid_size) - coarse_size)
    for _ in range(refinements):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    refined_log, refined_loss = (x1, f1) if f1 <= f2 else (x2, f2)
    if refined_loss < best_loss:
        return float(np.exp(refined_log)), float(refined_loss)
    return best_temperature, best_loss


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperatures: dict[str, float]
    validation_bce: dict[str, float]
    method: str = "temperature"

    def apply(self, branch: str, hazard_logits: np.ndarray) -> np.ndarray:
        if branch not in self.temperatures:
            raise KeyError(f"calibrator has no branch={branch!r}")
        return cumulative_probabilities(
            np.asarray(hazard_logits, dtype=np.float64) / float(self.temperatures[branch])
        ).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "temperatures": {key: float(value) for key, value in self.temperatures.items()},
            "validation_bce": {key: float(value) for key, value in self.validation_bce.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TemperatureCalibrator":
        return cls(
            temperatures={
                str(key): float(value)
                for key, value in dict(payload["temperatures"]).items()  # type: ignore[arg-type]
            },
            validation_bce={
                str(key): float(value)
                for key, value in dict(payload.get("validation_bce", {})).items()  # type: ignore[arg-type]
            },
            method=str(payload.get("method", "temperature")),
        )


def fit_calibrator(
    logits_by_branch: Mapping[str, np.ndarray],
    targets: np.ndarray,
    method: str = "temperature",
    minimum: float = 0.05,
    maximum: float = 20.0,
    grid_size: int = 161,
) -> TemperatureCalibrator:
    if method not in {"none", "temperature"}:
        raise ValueError(f"unknown calibration method={method!r}")
    temperatures: dict[str, float] = {}
    losses: dict[str, float] = {}
    for branch, logits in logits_by_branch.items():
        if method == "none":
            temperature = 1.0
            loss = binary_cross_entropy(cumulative_probabilities(logits), targets)
        else:
            temperature, loss = fit_temperature(
                logits,
                targets,
                minimum=minimum,
                maximum=maximum,
                grid_size=grid_size,
            )
        temperatures[str(branch)] = float(temperature)
        losses[str(branch)] = float(loss)
    return TemperatureCalibrator(
        temperatures=temperatures,
        validation_bce=losses,
        method=method,
    )
