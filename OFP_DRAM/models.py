from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from OFP_DRAM.config import OFPDRAMTrainConfig


def _class_balance(y: np.ndarray) -> tuple[int, int, float]:
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return pos, neg, float(neg / pos)


def available_models() -> dict[str, bool]:
    out = {"random_forest": True}
    try:
        import xgboost  # noqa: F401

        out["xgboost"] = True
    except Exception:
        out["xgboost"] = False
    try:
        import lightgbm  # noqa: F401

        out["lightgbm"] = True
    except Exception:
        out["lightgbm"] = False
    try:
        import catboost  # noqa: F401

        out["catboost"] = True
    except Exception:
        out["catboost"] = False
    return out


def build_model(model_name: str, y_train: np.ndarray, cfg: OFPDRAMTrainConfig):
    _, _, scale_pos_weight = _class_balance(y_train)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("xgboost is not installed") from exc
        return XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample,
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            eval_metric="logloss",
            verbosity=0,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError("lightgbm is not installed") from exc
        return LGBMClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample,
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            verbosity=-1,
        )
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise RuntimeError(
                "catboost is not installed. Install it with `pip install catboost` "
                "or omit `catboost` from --models."
            ) from exc
        return CatBoostClassifier(
            iterations=cfg.n_estimators,
            depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            loss_function="Logloss",
            random_seed=cfg.random_state,
            allow_writing_files=False,
            verbose=False,
        )
    raise ValueError(f"Unknown model: {model_name}")


def positive_lead_weights(meta: pd.DataFrame, y: np.ndarray, cfg: OFPDRAMTrainConfig) -> np.ndarray:
    weights = np.ones(len(y), dtype=float)
    if "lead_hours" not in meta.columns or "ahead_hours" not in meta.columns:
        return weights
    lead_hours = pd.to_numeric(meta["lead_hours"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ahead_hours = pd.to_numeric(meta["ahead_hours"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    lead_norm = np.clip(lead_hours / np.maximum(ahead_hours, 1.0), 0.0, 1.0)
    weights[y == 1] = 1.0 + cfg.positive_lead_alpha * lead_norm[y == 1]
    return weights


def _fit(model, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None):
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return model


def predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def train_one_stage(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    cfg: OFPDRAMTrainConfig,
):
    model = build_model(model_name, y_train, cfg)
    weights = positive_lead_weights(meta_train, y_train, cfg)
    t0 = time.time()
    _fit(model, X_train, y_train, weights)
    return model, {"train_time_s": time.time() - t0, "stage": "one_stage"}


def train_two_stage(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    cfg: OFPDRAMTrainConfig,
):
    rng = np.random.default_rng(cfg.random_state)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    if len(pos_idx) == 0 or len(neg_idx) < 4:
        return train_one_stage(model_name, X_train, y_train, meta_train, cfg)

    rng.shuffle(neg_idx)
    split = len(neg_idx) // 2
    neg_a = neg_idx[:split]
    neg_b = neg_idx[split:]
    stage1_idx = np.concatenate([pos_idx, neg_a])
    stage2_idx = np.concatenate([pos_idx, neg_b])

    base_weights = positive_lead_weights(meta_train, y_train, cfg)
    t0 = time.time()
    init_model = build_model(model_name, y_train[stage1_idx], cfg)
    _fit(init_model, X_train.iloc[stage1_idx], y_train[stage1_idx], base_weights[stage1_idx])

    neg_b_scores = predict_scores(init_model, X_train.iloc[neg_b])
    if len(neg_b_scores) and float(neg_b_scores.max() - neg_b_scores.min()) > 1e-12:
        hard_norm = (neg_b_scores - neg_b_scores.min()) / (neg_b_scores.max() - neg_b_scores.min())
    else:
        hard_norm = np.zeros_like(neg_b_scores)

    final_weights = base_weights[stage2_idx].copy()
    final_weights[len(pos_idx) :] = 1.0 + cfg.hard_negative_alpha * hard_norm
    final_model = build_model(model_name, y_train[stage2_idx], cfg)
    _fit(final_model, X_train.iloc[stage2_idx], y_train[stage2_idx], final_weights)
    return final_model, {
        "train_time_s": time.time() - t0,
        "stage": "two_stage",
        "stage1_windows": int(len(stage1_idx)),
        "stage2_windows": int(len(stage2_idx)),
        "stage2_hard_negative_score_mean": float(np.mean(neg_b_scores)) if len(neg_b_scores) else 0.0,
        "train_config": asdict(cfg),
    }


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    cfg: OFPDRAMTrainConfig,
):
    if cfg.two_stage:
        return train_two_stage(model_name, X_train, y_train, meta_train, cfg)
    return train_one_stage(model_name, X_train, y_train, meta_train, cfg)

