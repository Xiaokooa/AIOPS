"""Generate per-module diagnostic outputs for a trained FTEformer run.

The clean FTEformer experiment already stores binary risk scores, fault-mode
expert probabilities, fault-mode-to-sensor attention, sensor-time attention,
and dynamic sensor masks.  This script turns those arrays into:

1. Window-level diagnosis table: risk + top fault-mode candidate + evidence.
2. Module-level diagnosis table: one operator-facing row per alerted module.
3. Presentation figures for diagnosis validity and case inspection.

The fault-mode label is an operational diagnostic candidate, not a confirmed
physical root cause.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OFP.deep_learning.optical_prediction.common import base_utils as base
from OFP.deep_learning.optical_prediction.common import cache_utils
from OFP.deep_learning.optical_prediction.deep_learning.fault_type_labels import (
    TYPE_DISPLAY_NAMES,
    TYPE_NAMES,
    annotate_fault_type_frames,
    type_columns,
)
from OFP.deep_learning.optical_prediction.deep_learning.train import load_task_frames
from OFP.deep_learning.optical_prediction.scripts.run_dl_l4_typeaware_R4 import r4_task_cfg


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "green": "#8BCF8B",
    "red": "#B64342",
    "red_light": "#E9A6A1",
    "violet": "#9A4D8E",
    "gold": "#D59F00",
    "neutral": "#CFCECE",
    "dark": "#263238",
}

MODE_COLORS = {
    "thermal_anomaly": "#B64342",
    "current_bias_anomaly": "#D59F00",
    "tx_power_anomaly": "#42949E",
    "rx_power_anomaly": "#0F4D92",
    "lane_imbalance": "#9A4D8E",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(f"[fig] {path}")
    plt.close(fig)


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


def display_mode(name: str, display: dict[str, str] | None = None) -> str:
    display = display or TYPE_DISPLAY_NAMES
    return display.get(name, name.replace("_", " "))


def load_summary(exp_dir: Path) -> dict:
    path = exp_dir / "results" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_frame(summary: dict) -> pd.DataFrame:
    task_cfg = r4_task_cfg()
    frames = load_task_frames(task_cfg=task_cfg, tps_lead_minutes=task_cfg.lead_minutes)
    type_cfg = summary.get("type_loss_cfg", {})
    type_names = summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES
    frames, _ = annotate_fault_type_frames(
        frames,
        threshold_quantile=float(type_cfg.get("threshold_quantile", 0.90)),
        fallback_quantile=float(type_cfg.get("fallback_quantile", 0.75)),
        type_names=type_names,
    )
    return frames["test"].reset_index(drop=True)


def top_sensor_string(weights: np.ndarray, sensors: list[str], k: int = 3) -> str:
    order = np.argsort(-np.asarray(weights, dtype=float))[:k]
    parts = [f"{short_sensor_name(sensors[i])}:{weights[i]:.3f}" for i in order]
    return "; ".join(parts)


def status_label(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP alert"
    if y_true == 0 and y_pred == 1:
        return "FP alert"
    if y_true == 1 and y_pred == 0:
        return "Missed positive"
    return "Negative"


def build_diagnosis_tables(exp_dir: Path, summary: dict, diagnosis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sensor_data = np.load(exp_dir / "results" / "sensor_weights.npz", allow_pickle=True)
    maps = np.load(exp_dir / "results" / "cross_attn_maps.npz", allow_pickle=True)
    test_frame = load_test_frame(summary)

    y_true = sensor_data["y_true"].astype(int)
    y_pred = sensor_data["y_pred"].astype(int)
    scores = sensor_data["scores"].astype(float)
    weights = sensor_data["weights"].astype(float)
    type_probs = sensor_data["type_probs"].astype(float)
    type_targets = sensor_data["type_targets"].astype(float)
    type_masks = sensor_data["type_masks"].astype(float)
    primary = sensor_data["weak_type_primary"].astype(str)
    c2s = maps["concept_to_sensor"].astype(float)
    sensor_masks = maps["dynamic_sensor_mask"].astype(float) if "dynamic_sensor_mask" in maps.files else None

    type_names = summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES
    display = summary.get("type_display_names", TYPE_DISPLAY_NAMES)
    sensors = summary.get("sensors", base.SENSORS)
    threshold = float(summary.get("threshold", 0.5))

    if len(test_frame) != len(scores):
        raise ValueError(f"Test frame length {len(test_frame)} != stored outputs {len(scores)}")

    rows: list[dict[str, object]] = []
    target_cols = type_columns(type_names)
    for idx in range(len(scores)):
        prob_vec = type_probs[idx]
        order = np.argsort(-prob_vec)
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else top1
        target_vec = type_targets[idx] > 0.5
        top2_match = bool(target_vec[order[:2]].any()) if type_masks[idx] > 0 else False
        mode_sensor_weights = c2s[idx, top1]
        global_sensor_weights = weights[idx]
        dyn_density = np.nan
        if sensor_masks is not None:
            m = sensor_masks[idx]
            offdiag = ~np.eye(m.shape[0], dtype=bool)
            dyn_density = float(m[offdiag].mean())

        row_meta = test_frame.iloc[idx]
        row: dict[str, object] = {
            "row_idx": idx,
            "file_name": str(row_meta["file_name"]),
            "y_true": int(y_true[idx]),
            "y_pred": int(y_pred[idx]),
            "status": status_label(int(y_true[idx]), int(y_pred[idx])),
            "score": float(scores[idx]),
            "threshold": threshold,
            "alert": int(scores[idx] >= threshold),
            "time_to_first_minutes": float(row_meta.get("time_to_first_minutes", np.nan)),
            "obs_start_ts": int(row_meta.get("obs_start_ts", -1)),
            "obs_end_ts": int(row_meta.get("obs_end_ts", -1)),
            "pred_end_ts": int(row_meta.get("pred_end_ts", -1)),
            "event_label": int(row_meta.get("event_label", 0)),
            "module_label": int(row_meta.get("module_label", 0)),
            "top1_mode": type_names[top1],
            "top1_mode_display": display_mode(type_names[top1], display),
            "top1_prob": float(prob_vec[top1]),
            "top2_mode": type_names[top2],
            "top2_mode_display": display_mode(type_names[top2], display),
            "top2_prob": float(prob_vec[top2]),
            "top1_margin": float(prob_vec[top1] - prob_vec[top2]),
            "weak_type_mask": int(type_masks[idx] > 0),
            "weak_type_primary": str(primary[idx]),
            "top1_matches_weak_any": bool(type_masks[idx] > 0 and target_vec[top1]),
            "top2_matches_weak_any": top2_match,
            "top1_matches_primary": bool(type_masks[idx] > 0 and type_names[top1] == str(primary[idx])),
            "top_mode_sensors": top_sensor_string(mode_sensor_weights, sensors, k=3),
            "global_top_sensors": top_sensor_string(global_sensor_weights, sensors, k=3),
            "dynamic_mask_density": dyn_density,
        }
        for mode_idx, name in enumerate(type_names):
            row[f"prob_{name}"] = float(prob_vec[mode_idx])
            row[f"target_{name}"] = int(target_vec[mode_idx])
            if target_cols[mode_idx] in row_meta.index:
                row[f"frame_target_{name}"] = int(float(row_meta[target_cols[mode_idx]]) > 0.5)
        rows.append(row)

    window_df = pd.DataFrame(rows)
    module_rows: list[dict[str, object]] = []
    for file_name, group in window_df.groupby("file_name", sort=False):
        alerts = group[group["alert"] == 1]
        chosen = alerts.sort_values(["score", "top1_prob"], ascending=False).head(1)
        if chosen.empty:
            chosen = group.sort_values("score", ascending=False).head(1)
        row = chosen.iloc[0].to_dict()
        row["module_window_count"] = int(len(group))
        row["module_alert_count"] = int(len(alerts))
        row["module_has_positive_window"] = int((group["y_true"] == 1).any())
        row["module_hit"] = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).any())
        row["module_any_alert"] = int((group["y_pred"] == 1).any())
        module_rows.append(row)
    module_df = pd.DataFrame(module_rows)

    summary_dict = summarize_diagnosis(window_df, type_names, display)
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(diagnosis_dir / "window_diagnosis.csv", index=False)
    module_df.to_csv(diagnosis_dir / "module_diagnosis.csv", index=False)
    (diagnosis_dir / "diagnosis_summary.json").write_text(
        json.dumps(summary_dict, indent=2), encoding="utf-8"
    )
    print(f"[diag] {diagnosis_dir / 'window_diagnosis.csv'}")
    print(f"[diag] {diagnosis_dir / 'module_diagnosis.csv'}")
    print(f"[diag] {diagnosis_dir / 'diagnosis_summary.json'}")
    return window_df, module_df, summary_dict


def summarize_diagnosis(window_df: pd.DataFrame, type_names: list[str], display: dict[str, str]) -> dict:
    weak = window_df["weak_type_mask"].astype(int) == 1
    weak_alert = weak & (window_df["alert"].astype(int) == 1)
    weak_tp = weak & (window_df["status"] == "TP alert")

    def agreement(mask: pd.Series) -> dict[str, object]:
        if int(mask.sum()) == 0:
            return {"n": 0}
        sub = window_df.loc[mask]
        return {
            "n": int(len(sub)),
            "top1_any": float(sub["top1_matches_weak_any"].mean()),
            "top2_any": float(sub["top2_matches_weak_any"].mean()),
            "top1_primary": float(sub["top1_matches_primary"].mean()),
        }

    distributions = {}
    for status in ["TP alert", "FP alert", "Missed positive", "Negative"]:
        sub = window_df[window_df["status"] == status]
        counts = Counter(sub["top1_mode"].tolist())
        distributions[status] = {
            display_mode(name, display): int(counts.get(name, 0)) for name in type_names
        }

    return {
        "n_windows": int(len(window_df)),
        "n_alerts": int(window_df["alert"].sum()),
        "n_positive_windows": int((window_df["y_true"] == 1).sum()),
        "status_counts": {k: int(v) for k, v in window_df["status"].value_counts().to_dict().items()},
        "top_mode_distributions": distributions,
        "weak_label_agreement": {
            "weak_all": agreement(weak),
            "weak_alert": agreement(weak_alert),
            "weak_true_positive": agreement(weak_tp),
        },
        "mean_top1_prob": float(window_df["top1_prob"].mean()),
        "mean_alert_top1_prob": float(window_df.loc[window_df["alert"] == 1, "top1_prob"].mean())
        if int(window_df["alert"].sum()) else float("nan"),
    }


def plot_diagnosis_overview(window_df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    type_names = summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES
    display = summary.get("type_display_names", TYPE_DISPLAY_NAMES)
    statuses = ["TP alert", "FP alert", "Missed positive"]
    x = np.arange(len(statuses))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1.1, 1.0]})

    bottom = np.zeros(len(statuses))
    for name in type_names:
        vals = [
            int(((window_df["status"] == status) & (window_df["top1_mode"] == name)).sum())
            for status in statuses
        ]
        axes[0].bar(
            x,
            vals,
            bottom=bottom,
            color=MODE_COLORS.get(name, PALETTE["neutral"]),
            edgecolor="white",
            linewidth=0.5,
            label=display_mode(name, display),
        )
        bottom += np.asarray(vals)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["TP", "FP", "FN"], rotation=0)
    axes[0].set_ylabel("Windows")
    axes[0].grid(axis="y", color="#E0E0E0", linewidth=0.6)

    groups = [
        ("TP", window_df["status"] == "TP alert"),
        ("FP", window_df["status"] == "FP alert"),
        ("FN", window_df["status"] == "Missed positive"),
    ]
    prob_matrix = []
    for _, mask in groups:
        if mask.sum() == 0:
            prob_matrix.append([np.nan] * len(type_names))
        else:
            prob_matrix.append([
                float(window_df.loc[mask, f"prob_{name}"].mean()) for name in type_names
            ])
    prob_matrix = np.asarray(prob_matrix).T
    im = axes[1].imshow(prob_matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(groups)))
    axes[1].set_xticklabels([g[0] for g in groups])
    axes[1].set_yticks(np.arange(len(type_names)))
    axes[1].set_yticklabels([display_mode(name, display) for name in type_names])
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.02)
    cbar.set_label("Mean probability")

    agree = summary.get("diagnosis_summary", {}).get("weak_label_agreement")
    if agree is None:
        diag_path = summary.get("_diagnosis_summary_path")
        agree = {}
        if diag_path and Path(diag_path).exists():
            agree = json.loads(Path(diag_path).read_text(encoding="utf-8")).get("weak_label_agreement", {})
    agree = agree or {}
    labels = ["Top-1 any", "Top-2 any", "Top-1 primary"]
    vals = [
        agree.get("weak_true_positive", {}).get("top1_any", np.nan),
        agree.get("weak_true_positive", {}).get("top2_any", np.nan),
        agree.get("weak_true_positive", {}).get("top1_primary", np.nan),
    ]
    bars = axes[2].bar(
        np.arange(len(labels)),
        vals,
        color=[PALETTE["teal"], PALETTE["blue_main"], PALETTE["gold"]],
        edgecolor="white",
        linewidth=0.6,
    )
    axes[2].set_xticks(np.arange(len(labels)))
    axes[2].set_xticklabels(labels, rotation=28, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Agreement")
    axes[2].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    n_weak_tp = agree.get("weak_true_positive", {}).get("n", 0)
    axes[2].text(0.02, 0.96, f"n={n_weak_tp}", transform=axes[2].transAxes, va="top", fontsize=8)
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.2f}", ha="center", fontsize=8)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    save_figure(fig, out_dir, "fig_11_fault_mode_diagnosis_overview")


def plot_event_diagnosis_matrix(module_df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    type_names = summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES
    display = summary.get("type_display_names", TYPE_DISPLAY_NAMES)
    tp_modules = module_df[module_df["module_hit"] == 1].sort_values("score", ascending=False).head(24)
    if tp_modules.empty:
        tp_modules = module_df[module_df["module_any_alert"] == 1].sort_values("score", ascending=False).head(24)
    if tp_modules.empty:
        return

    matrix = tp_modules[[f"prob_{name}" for name in type_names]].to_numpy(dtype=float)
    labels = [
        f"{i + 1:02d}" for i in range(len(tp_modules))
    ]
    scores = tp_modules["score"].to_numpy(dtype=float)
    margins = tp_modules["top1_margin"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 4.3), gridspec_kw={"width_ratios": [2.1, 0.6, 0.6]})
    im = axes[0].imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xticks(np.arange(len(type_names)))
    axes[0].set_xticklabels([display_mode(name, display) for name in type_names], rotation=35, ha="right")
    axes[0].set_ylabel("Alerted modules")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
    cbar.set_label("Fault-mode probability")

    y = np.arange(len(labels))
    axes[1].barh(y, scores, color=PALETTE["blue_main"], edgecolor="white", linewidth=0.5)
    axes[1].invert_yaxis()
    axes[1].set_yticks([])
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel("Risk")
    axes[1].grid(axis="x", color="#E0E0E0", linewidth=0.6)

    axes[2].barh(y, margins, color=PALETTE["gold"], edgecolor="white", linewidth=0.5)
    axes[2].invert_yaxis()
    axes[2].set_yticks([])
    axes[2].set_xlim(0, max(0.2, float(np.nanmax(margins)) * 1.15))
    axes[2].set_xlabel("Margin")
    axes[2].grid(axis="x", color="#E0E0E0", linewidth=0.6)
    fig.tight_layout(w_pad=0.9)
    save_figure(fig, out_dir, "fig_12_alert_module_fault_mode_matrix")


def choose_case_index(window_df: pd.DataFrame, requested: int | None = None) -> int | None:
    if requested is not None:
        if requested < 0 or requested >= len(window_df):
            raise ValueError(f"case_row_idx {requested} is outside [0, {len(window_df)})")
        return int(requested)
    candidates = window_df[window_df["status"] == "TP alert"].copy()
    if candidates.empty:
        candidates = window_df[window_df["alert"] == 1].copy()
    if candidates.empty:
        return None
    candidates["case_rank"] = candidates["score"] + 0.15 * candidates["top1_margin"]
    return int(candidates.sort_values("case_rank", ascending=False).iloc[0]["row_idx"])


def local_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    for col in range(arr.shape[1]):
        x = arr[:, col]
        finite = np.isfinite(x)
        if not finite.any():
            continue
        center = np.nanmedian(x)
        scale = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.nanstd(x)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        out[:, col] = (np.nan_to_num(x, nan=center) - center) / scale
    return out


def plot_case_report(exp_dir: Path, window_df: pd.DataFrame, summary: dict, out_dir: Path, case_idx: int | None) -> None:
    sensor_data = np.load(exp_dir / "results" / "sensor_weights.npz", allow_pickle=True)
    maps = np.load(exp_dir / "results" / "cross_attn_maps.npz", allow_pickle=True)
    test_frame = load_test_frame(summary)
    sensors = summary.get("sensors", base.SENSORS)
    type_names = summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES
    display = summary.get("type_display_names", TYPE_DISPLAY_NAMES)
    idx = choose_case_index(window_df, requested=case_idx)
    if idx is None:
        return

    row = window_df.iloc[idx]
    frame_row = test_frame.iloc[idx]
    type_probs = sensor_data["type_probs"][idx].astype(float)
    c2s = maps["concept_to_sensor"][idx].astype(float)
    s2t = maps["sensor_to_time"][idx].astype(float) if "sensor_to_time" in maps.files else None
    top_mode_idx = type_names.index(str(row["top1_mode"]))
    top_sensor_ids = np.argsort(-c2s[top_mode_idx])[:5]

    resampled, _, _ = cache_utils.load_resampled_module(str(row["file_name"]))
    start = int(frame_row["obs_start_idx"])
    end = int(frame_row["obs_end_idx"])
    obs = resampled.iloc[start : end + 1].copy()
    obs_values = obs[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    z = local_zscore(obs_values[:, top_sensor_ids])
    t_hours = (obs["timestamp"].to_numpy(dtype=float) - float(frame_row["obs_end_ts"])) / 3600.0

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.0, 1.15], hspace=0.40, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    mode_labels = [display_mode(name, display) for name in type_names]
    colors = [MODE_COLORS.get(name, PALETTE["neutral"]) for name in type_names]
    y_mode = np.arange(len(type_names))
    bars = ax0.barh(y_mode, type_probs, color=colors, edgecolor="white", linewidth=0.6)
    ax0.axvline(0.5, color=PALETTE["dark"], linewidth=0.8, linestyle="--")
    ax0.set_xlim(0, 1.12)
    ax0.set_xlabel("Probability")
    ax0.set_yticks(y_mode)
    ax0.set_yticklabels(mode_labels)
    ax0.invert_yaxis()
    ax0.text(
        0.02,
        0.96,
        f"risk={float(row['score']):.3f}, {row['status']}",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    for bar, val in zip(bars, type_probs):
        if val > 0.08:
            ax0.text(val + 0.018, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7)

    im1 = ax1.imshow(c2s, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.35, float(c2s.max())))
    ax1.set_yticks(np.arange(len(type_names)))
    ax1.set_yticklabels(mode_labels)
    ax1.set_xticks(np.arange(len(sensors)))
    ax1.set_xticklabels([short_sensor_name(s) for s in sensors], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Sensor")
    ax1.add_patch(
        plt.Rectangle(
            (-0.5, top_mode_idx - 0.5),
            len(sensors),
            1,
            fill=False,
            edgecolor=PALETTE["red"],
            linewidth=1.5,
        )
    )
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02)
    cbar1.set_label("Attention")

    if s2t is not None:
        s2t_sub = s2t[top_sensor_ids]
        im2 = ax2.imshow(s2t_sub, aspect="auto", cmap="YlOrRd")
        ax2.set_yticks(np.arange(len(top_sensor_ids)))
        ax2.set_yticklabels([short_sensor_name(sensors[i]) for i in top_sensor_ids])
        n_time = s2t_sub.shape[1]
        ticks = np.linspace(0, n_time - 1, 6, dtype=int)
        hour_labels = np.linspace(-24, 0, n_time)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{hour_labels[t]:.0f}" for t in ticks])
        ax2.set_xlabel("Observation-window time (h)")
        ax2.set_ylabel("Top sensors")
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02)
        cbar2.set_label("Attention")
    else:
        ax2.set_axis_off()

    for col, sensor_id in enumerate(top_sensor_ids):
        ax3.plot(t_hours, z[:, col], linewidth=1.5, label=short_sensor_name(sensors[sensor_id]))
    ax3.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax3.set_xlim(float(np.nanmin(t_hours)), 0.0)
    ax3.set_xlabel("Observation-window time (h)")
    ax3.set_ylabel("Local robust z-score")
    ax3.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax3.legend(ncol=2, loc="upper left", fontsize=8)

    save_figure(fig, out_dir, "fig_13_single_module_diagnosis_case")


def plot_faithfulness_digest(exp_dir: Path, out_dir: Path) -> None:
    path = exp_dir / "analysis" / "faithfulness_metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    method_occlusion = metrics.get("method_occlusion", {})
    if not method_occlusion:
        return
    methods = [
        ("attention", "Attention", PALETTE["neutral"]),
        ("gradient", "Gradient", PALETTE["teal"]),
        ("attention_x_gradient", "Attn*Grad", PALETTE["violet"]),
        ("attention_x_occlusion", "Attn*Occl.", PALETTE["blue_main"]),
        ("random", "Random", PALETTE["red_light"]),
    ]
    k_values = sorted(int(k) for k in next(iter(method_occlusion.values())).keys())
    x = np.arange(len(k_values))

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for method, label, color in methods:
        vals = []
        for k in k_values:
            if method == "random":
                vals.append(metrics["random_occlusion"][str(k)]["mean_risk_drop_true_positive"])
            else:
                vals.append(method_occlusion[method][str(k)]["top"]["mean_risk_drop_true_positive"])
        ax.plot(x, vals, marker="o", linewidth=2.0, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("Masked top-k sensors")
    ax.set_ylabel("Mean risk drop on TP")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.18), fontsize=8)
    save_figure(fig, out_dir, "fig_14_attribution_faithfulness_digest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--diagnosis_dir", type=Path, default=None)
    parser.add_argument("--case_row_idx", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    exp_dir = args.exp_dir
    out_dir = args.output_dir or (exp_dir / "figures")
    diagnosis_dir = args.diagnosis_dir or (exp_dir / "diagnosis")
    summary = load_summary(exp_dir)
    window_df, module_df, diag_summary = build_diagnosis_tables(exp_dir, summary, diagnosis_dir)
    summary["_diagnosis_summary_path"] = str(diagnosis_dir / "diagnosis_summary.json")
    summary["diagnosis_summary"] = diag_summary

    plot_diagnosis_overview(window_df, summary, out_dir)
    plot_event_diagnosis_matrix(module_df, summary, out_dir)
    plot_case_report(exp_dir, window_df, summary, out_dir, args.case_row_idx)
    plot_faithfulness_digest(exp_dir, out_dir)


if __name__ == "__main__":
    main()
