from __future__ import annotations

import numpy as np

from drfp.training import positive_weights_from_targets


def test_positive_weights_are_per_horizon_and_capped() -> None:
    targets = np.asarray(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
        dtype=np.float32,
    )
    weights = positive_weights_from_targets(targets, maximum=3.0)
    assert weights.shape == (4,)
    assert weights[0] == 1.0  # no positive label: safe neutral fallback
    assert np.all((weights >= 1.0) & (weights <= 3.0))
