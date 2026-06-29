from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg,
) -> tuple[RandomForestClassifier, float]:
    model = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_split=cfg.rf_min_samples_split,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - t0
