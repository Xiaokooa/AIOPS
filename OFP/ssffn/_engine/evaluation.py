from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from OFP.ssffn._engine.data import apply_threshold, evaluate_prediction_dir_compat, evaluate_prediction_frames, format_duration, parse_threshold_grid
from OFP.ssffn._engine.runtime import format_metric_summary
from OFP.ssffn._engine.metrics import first_positive_timestamp

class LastLinearInputHook:

    def __init__(self, model: nn.Module, *, detach: bool=True) -> None:
        linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if not linears:
            raise ValueError('Cannot extract embeddings because the model has no nn.Linear layers')
        self.value: torch.Tensor | None = None
        self.detach = bool(detach)
        self.handle = linears[-1].register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        value = inputs[0] if inputs and torch.is_tensor(inputs[0]) else output
        if torch.is_tensor(value):
            self.value = value.detach() if self.detach else value

    def clear(self) -> None:
        self.value = None

    def close(self) -> None:
        self.handle.remove()

def log_stage_start(stage: str, deep_model_name: str, fold: int | None, **fields: object) -> float:
    parts = [f'model={deep_model_name}']
    if fold is not None:
        parts.append(f'fold={fold}')
    parts.append(f'stage={stage}')
    parts.extend((f'{key}={value}' for key, value in fields.items()))
    print('[stage-start] ' + ' '.join(parts), flush=True)
    return time.time()

def log_stage_done(stage: str, started: float, deep_model_name: str, fold: int | None, **fields: object) -> None:
    parts = [f'model={deep_model_name}']
    if fold is not None:
        parts.append(f'fold={fold}')
    parts.append(f'stage={stage}')
    parts.append(f'elapsed={format_duration(time.time() - started)}')
    parts.extend((f'{key}={value}' for key, value in fields.items()))
    print('[stage-done] ' + ' '.join(parts), flush=True)

def positive_scores(estimator: Any, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    if hasattr(estimator, 'predict_proba'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names.*', category=UserWarning)
            proba = estimator.predict_proba(x)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float32)
        if proba.ndim == 2 and proba.shape[1] == 1:
            classes = np.asarray(getattr(estimator, 'classes_', []))
            if len(classes) == 1 and int(classes[0]) == 1:
                return proba[:, 0].astype(np.float32)
            return np.zeros(proba.shape[0], dtype=np.float32)
        return proba.reshape(-1).astype(np.float32)
    if hasattr(estimator, 'decision_function'):
        raw = np.asarray(estimator.decision_function(x), dtype=float).reshape(-1)
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    return np.asarray(estimator.predict(x), dtype=np.float32).reshape(-1)

def select_threshold_for_scores(score_frames: dict[str, pd.DataFrame], label_dir: Path, run_dir: Path, tag: str, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    candidates = parse_threshold_grid(args.threshold_grid)
    if not True or not score_frames or (not candidates):
        return (0.3, {})
    search_rows: list[dict[str, float]] = []
    best_threshold = 0.3
    best_metrics: dict[str, float] = {}
    best_detail: pd.DataFrame | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None
    metric_key = 'f1_score'
    confirm_k = max(1, int(getattr(args, 'confirm_k', 1)))
    confirm_m = max(confirm_k, int(getattr(args, 'confirm_m', 1)))
    truth_by_name = {name: first_positive_timestamp(Path(label_dir) / name, 'anomaly') for name in score_frames if (Path(label_dir) / name).exists()}
    constraint = str(getattr(args, 'threshold_constraint', 'none')).lower()
    constraint_tolerance = max(0.0, float(getattr(args, 'threshold_constraint_tolerance', 0.0)))
    rule_metrics: dict[str, float] = {}
    rule_detail: pd.DataFrame | None = None
    for threshold in candidates:
        pred_frames = {name: apply_threshold(frame, float(threshold), confirm_k=confirm_k, confirm_m=confirm_m) for name, frame in score_frames.items()}
        summary_df, detail_df = evaluate_prediction_frames(pred_frames, label_dir, min_hit_lead_hours=0.0, truth_by_name=truth_by_name)
        metrics = {str(row['Item']): float(row['Value']) for _, row in summary_df.iterrows()}
        value = float(metrics.get(metric_key, metrics.get('f1_score', 0.0)))
        feasible = True
        row = {'threshold': float(threshold), 'confirm_k': float(confirm_k), 'confirm_m': float(confirm_m), 'constraint_feasible': float(feasible)}
        row.update(metrics)
        search_rows.append(row)
        key = (value, float(metrics.get('afws', 0.0)), float(metrics.get('precision', 0.0)), float(metrics.get('accuracy', 0.0)), float(metrics.get('recall', 0.0)), float(threshold))
        if feasible and (best_key is None or key > best_key):
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
            best_detail = detail_df
    val_dir = Path(run_dir) / 'validation' / tag
    val_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search_rows).to_csv(val_dir / 'threshold_search.csv', index=False)
    pd.DataFrame({'Item': list(best_metrics.keys()), 'Value': list(best_metrics.values())}).to_csv(val_dir / 'best_evaluate_result.csv', index=False)
    if best_detail is not None:
        best_detail.to_csv(val_dir / 'best_module_decisions.csv', index=False)
    print(f'[val-select] tag={tag} threshold={best_threshold:.4f} confirm={confirm_k}-of-{confirm_m} constraint={constraint} {metric_key}={best_metrics.get(metric_key, 0.0):.5f} {format_metric_summary(best_metrics)}')
    return (best_threshold, best_metrics)

def write_threshold_predictions(score_frames: dict[str, pd.DataFrame], out_dir: Path, threshold: float, confirm_k: int=1, confirm_m: int=1) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for name, frame in sorted(score_frames.items()):
        pred = apply_threshold(frame, float(threshold), confirm_k=int(confirm_k), confirm_m=int(confirm_m))
        pred.to_csv(out_dir / name, index=False)
        total_rows += int(len(pred))
    return total_rows

def evaluate_prediction_output(out_dir: Path, label_dir: Path, eval_dir: Path, min_hit_lead_hours: float) -> tuple[dict[str, float], pd.DataFrame]:
    summary_df, detail_df = evaluate_prediction_dir_compat(out_dir, label_dir, min_hit_lead_hours=float(min_hit_lead_hours))
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / 'evaluate_result.csv', index=False)
    detail_df.to_csv(eval_dir / 'module_decisions.csv', index=False)
    metrics = {str(row['Item']): float(row['Value']) for _, row in summary_df.iterrows()}
    return (metrics, detail_df)
