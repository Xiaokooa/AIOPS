from __future__ import annotations

import numpy as np

from drfp.calibration import cumulative_probabilities, fit_calibrator, fit_temperature


def test_cumulative_probabilities_are_monotone() -> None:
    probabilities = cumulative_probabilities(np.asarray([[0.1, -2.0, 1.0, -0.5]]))
    assert probabilities.shape == (1, 4)
    assert np.all(np.diff(probabilities, axis=1) >= 0.0)


def test_temperature_fit_and_round_trip() -> None:
    logits = np.asarray([[4.0, -1.0], [-4.0, -1.0], [3.0, 0.0], [-3.0, 0.0]])
    targets = np.asarray([[1, 1], [0, 0], [1, 1], [0, 0]], dtype=float)
    temperature, loss = fit_temperature(logits, targets, grid_size=21)
    assert temperature > 0
    assert np.isfinite(loss)
    calibrator = fit_calibrator({"fused": logits}, targets, grid_size=21)
    restored = type(calibrator).from_dict(calibrator.to_dict())
    assert np.allclose(restored.apply("fused", logits), calibrator.apply("fused", logits))
