from __future__ import annotations

import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def train_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg,
) -> tuple[XGBClassifier, float]:
    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    model = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample,
        scale_pos_weight=neg / pos,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        eval_metric="logloss",
        verbosity=0,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - t0
