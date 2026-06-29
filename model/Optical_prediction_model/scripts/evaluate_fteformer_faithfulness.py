"""Faithfulness diagnostics for FTEformer sensor attribution.

The script evaluates whether attributed sensors are behaviorally important:
masking top-k attributed sensors should reduce the risk score more than masking
bottom-k or random sensors. It also reports per-sensor ablation effects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.common import base_utils as base
from model.Optical_prediction_model.deep_learning.data import WindowSliceCache, compute_norm_stats
from model.Optical_prediction_model.deep_learning.fault_type_labels import TYPE_NAMES, annotate_fault_type_frames
from model.Optical_prediction_model.deep_learning.mf_transformer_v3 import MFTransformerV3, MFTransformerV3Cfg
from model.Optical_prediction_model.deep_learning.train import load_task_frames
from model.Optical_prediction_model.scripts.run_dl_l4_typeaware_R4 import (
    TypeAwareFailureWindowDataset,
    collate_type,
    r4_task_cfg,
)


def short_sensor_name(sensor: str) -> str:
    mapping = {
        "temperature": "Temp",
        "current": "Bias",
        "currentTXPower": "TX",
        "currentRXPower": "RX",
    }
    if sensor in mapping:
        return mapping[sensor]
    return (
        sensor.replace("currentMulti", "")
        .replace("Power", "")
        .replace("TX", "TX")
        .replace("RX", "RX")
    )


def load_model_and_test(exp_dir: Path, batch_size: int):
    summary_path = exp_dir / "results" / "summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    cfg = MFTransformerV3Cfg(**summary["model_cfg"])
    model = MFTransformerV3(cfg)
    ckpt = torch.load(exp_dir / "models" / "model.pt", map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    task_cfg = r4_task_cfg()
    frames = load_task_frames(task_cfg=task_cfg, tps_lead_minutes=task_cfg.lead_minutes)
    type_loss_cfg = summary.get("type_loss_cfg", {})
    type_names = summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES
    frames, _ = annotate_fault_type_frames(
        frames,
        threshold_quantile=float(type_loss_cfg.get("threshold_quantile", 0.90)),
        fallback_quantile=float(type_loss_cfg.get("fallback_quantile", 0.75)),
        type_names=type_names,
    )
    obs_steps = int(frames["train"].iloc[0]["obs_end_idx"] - frames["train"].iloc[0]["obs_start_idx"] + 1)
    slice_cache = WindowSliceCache()
    norm = compute_norm_stats(frames["train"], slice_cache, obs_steps, max_modules=400)
    test_ds = TypeAwareFailureWindowDataset(frames["test"], obs_steps, slice_cache, norm, type_names)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_type)
    return summary, model, loader, device


def metric_bundle(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def mask_selected_sensors(x: torch.Tensor, mask: torch.Tensor, selected: np.ndarray):
    x_masked = x.clone()
    m_masked = mask.clone()
    for row, sensor_ids in enumerate(selected):
        ids = torch.as_tensor(sensor_ids, device=x.device, dtype=torch.long)
        x_masked[row, :, ids] = 0.0
        m_masked[row, :, ids] = 0.0
    return x_masked, m_masked


def normalize_attribution(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    denom = values.sum(axis=1, keepdims=True)
    uniform = np.full_like(values, 1.0 / max(values.shape[1], 1))
    return np.where(denom > eps, values / np.maximum(denom, eps), uniform)


def summarize_occlusion(y_true: np.ndarray, base_score: np.ndarray, scores: np.ndarray,
                        threshold: float, true_positive: np.ndarray,
                        positive: np.ndarray, baseline_f1: float) -> dict[str, float]:
    drops_tp = base_score[true_positive] - scores[true_positive]
    drops_pos = base_score[positive] - scores[positive]
    occluded_metrics = metric_bundle(y_true, scores, threshold)
    return {
        **occluded_metrics,
        "mean_risk_drop_true_positive": float(np.mean(drops_tp)) if drops_tp.size else float("nan"),
        "median_risk_drop_true_positive": float(np.median(drops_tp)) if drops_tp.size else float("nan"),
        "mean_risk_drop_positive": float(np.mean(drops_pos)) if drops_pos.size else float("nan"),
        "f1_drop": float(baseline_f1 - occluded_metrics["f1"]),
    }


def run_faithfulness(exp_dir: Path, output_dir: Path, batch_size: int,
                     k_values: list[int], seed: int, max_batches: int | None = None):
    summary, model, loader, device = load_model_and_test(exp_dir, batch_size)
    threshold = float(summary["threshold"])
    sensors = summary.get("sensors", base.SENSORS)
    rng = np.random.default_rng(seed)

    y_all: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    base_weights: list[np.ndarray] = []
    gradient_weights: list[np.ndarray] = []
    method_scores: dict[str, dict[str, dict[int, list[np.ndarray]]]] = {}
    method_names = ["attention", "gradient", "attention_x_gradient", "attention_x_occlusion"]
    for method in method_names:
        method_scores[method] = {
            "top": {k: [] for k in k_values},
            "bottom": {k: [] for k in k_values},
        }
    random_scores: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    single_sensor_scores: dict[int, list[np.ndarray]] = {i: [] for i in range(len(sensors))}

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch[0].to(device)
        obs_mask = batch[1].to(device)
        y = batch[2].cpu().numpy().astype(int)

        x_grad = x.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x_grad, obs_mask)
        risk_prob = torch.sigmoid(out[0])
        score = risk_prob.detach().cpu().numpy()
        weights = out[1].detach().cpu().numpy()

        risk_prob.sum().backward()
        grad = x_grad.grad.detach().abs() * obs_mask
        grad_denom = obs_mask.sum(dim=1).clamp(min=1.0)
        grad_attr = (grad.sum(dim=1) / grad_denom).cpu().numpy()
        grad_attr = normalize_attribution(grad_attr)

        y_all.append(y)
        base_scores.append(score)
        base_weights.append(weights)
        gradient_weights.append(grad_attr)

        batch_single_scores = []
        with torch.no_grad():
            for sensor_id in range(len(sensors)):
                selected = np.full((x.size(0), 1), sensor_id, dtype=int)
                x_masked, m_masked = mask_selected_sensors(x, obs_mask, selected)
                score_masked = torch.sigmoid(model(x_masked, m_masked)[0]).cpu().numpy()
                single_sensor_scores[sensor_id].append(score_masked)
                batch_single_scores.append(score_masked)
        batch_single_scores_arr = np.stack(batch_single_scores, axis=1)
        batch_risk_drop = np.maximum(score[:, None] - batch_single_scores_arr, 0.0)

        method_attr = {
            "attention": normalize_attribution(weights),
            "gradient": grad_attr,
            "attention_x_gradient": normalize_attribution(weights * grad_attr),
            "attention_x_occlusion": normalize_attribution(weights * batch_risk_drop),
        }

        with torch.no_grad():
            for k in k_values:
                random_selected = np.stack([
                    rng.choice(len(sensors), size=k, replace=False)
                    for _ in range(weights.shape[0])
                ], axis=0)
                x_masked, m_masked = mask_selected_sensors(x, obs_mask, random_selected)
                random_scores[k].append(torch.sigmoid(model(x_masked, m_masked)[0]).cpu().numpy())

                for method, attr in method_attr.items():
                    order_desc = np.argsort(-attr, axis=1)
                    order_asc = np.argsort(attr, axis=1)
                    for side, selected in {
                        "top": order_desc[:, :k],
                        "bottom": order_asc[:, :k],
                    }.items():
                        x_masked, m_masked = mask_selected_sensors(x, obs_mask, selected)
                        score_masked = torch.sigmoid(model(x_masked, m_masked)[0]).cpu().numpy()
                        method_scores[method][side][k].append(score_masked)

    y_true = np.concatenate(y_all)
    base_score = np.concatenate(base_scores)
    weights = np.concatenate(base_weights)
    gradient_attr_all = np.concatenate(gradient_weights)
    base_pred = (base_score >= threshold).astype(int)
    true_positive = (y_true == 1) & (base_pred == 1)
    positive = y_true == 1
    baseline_metrics = metric_bundle(y_true, base_score, threshold)

    metrics = {
        "experiment": str(exp_dir),
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "n_positive": int(positive.sum()),
        "n_true_positive": int(true_positive.sum()),
        "baseline": baseline_metrics,
        "occlusion": {},
        "method_occlusion": {},
        "random_occlusion": {},
        "single_sensor_ablation": [],
    }

    arrays: dict[str, np.ndarray] = {
        "y_true": y_true,
        "base_scores": base_score,
        "base_weights": weights,
        "gradient_weights": gradient_attr_all,
        "true_positive_mask": true_positive.astype(np.int8),
    }

    baseline_f1 = baseline_metrics["f1"]
    for k in k_values:
        metrics["occlusion"][str(k)] = {}
        random = np.concatenate(random_scores[k])
        metrics["random_occlusion"][str(k)] = summarize_occlusion(
            y_true, base_score, random, threshold, true_positive, positive, baseline_f1
        )
        arrays[f"random_{k}_scores"] = random

        # Preserve the old field name for backward-compatible plotting:
        # occlusion[k] is attention top/bottom plus random.
        for side in ["top", "bottom"]:
            scores = np.concatenate(method_scores["attention"][side][k])
            arrays[f"attention_{side}_{k}_scores"] = scores
            metrics["occlusion"][str(k)][side] = summarize_occlusion(
                y_true, base_score, scores, threshold, true_positive, positive, baseline_f1
            )
        metrics["occlusion"][str(k)]["random"] = metrics["random_occlusion"][str(k)]

    for method in method_names:
        metrics["method_occlusion"][method] = {}
        for k in k_values:
            metrics["method_occlusion"][method][str(k)] = {}
            for side in ["top", "bottom"]:
                scores = np.concatenate(method_scores[method][side][k])
                arrays[f"{method}_{side}_{k}_scores"] = scores
                metrics["method_occlusion"][method][str(k)][side] = summarize_occlusion(
                    y_true, base_score, scores, threshold, true_positive, positive, baseline_f1
                )

    for sensor_id, sensor in enumerate(sensors):
        scores = np.concatenate(single_sensor_scores[sensor_id])
        arrays[f"sensor_{sensor_id}_scores"] = scores
        drops_tp = base_score[true_positive] - scores[true_positive]
        sensor_metrics = metric_bundle(y_true, scores, threshold)
        metrics["single_sensor_ablation"].append({
            "sensor": sensor,
            "label": short_sensor_name(sensor),
            **sensor_metrics,
            "mean_risk_drop_true_positive": float(np.mean(drops_tp)) if drops_tp.size else float("nan"),
            "f1_drop": float(metrics["baseline"]["f1"] - sensor_metrics["f1"]),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "faithfulness_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(output_dir / "faithfulness_scores.npz", **arrays)
    print(json.dumps(metrics["baseline"], indent=2))
    for k in k_values:
        pieces = []
        for method in method_names:
            drop = metrics["method_occlusion"][method][str(k)]["top"]["mean_risk_drop_true_positive"]
            pieces.append(f"{method}={drop:.4f}")
        rnd = metrics["random_occlusion"][str(k)]["mean_risk_drop_true_positive"]
        print(f"[k={k}] top risk drop: " + ", ".join(pieces) + f", random={rnd:.4f}")
    print(f"[saved] {output_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.exp_dir / "analysis")
    run_faithfulness(
        exp_dir=args.exp_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        k_values=args.k_values,
        seed=args.seed,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
