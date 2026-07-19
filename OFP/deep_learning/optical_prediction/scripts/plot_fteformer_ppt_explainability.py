"""Create PPT-oriented explainability figures for FTEformer.

The figures are intentionally more direct than the manuscript figures:
they show what an operator receives for one module, how post-hoc diagnostic
scores differ from raw expert probabilities, and whether the selected evidence
passes a simple occlusion faithfulness check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OFP.deep_learning.optical_prediction.common import cache_utils
from OFP.deep_learning.optical_prediction.deep_learning.fault_type_labels import TYPE_NAMES
from OFP.deep_learning.optical_prediction.scripts.generate_fteformer_diagnostic_report import (
    MODE_COLORS,
    PALETTE,
    apply_style,
    display_mode,
    load_summary,
    load_test_frame,
    local_zscore,
    save_figure,
    short_sensor_name,
)


MODE_SENSOR_GROUPS = {
    "thermal_anomaly": ["temperature"],
    "current_bias_anomaly": ["current"],
    "tx_power_anomaly": ["currentTXPower"],
    "rx_power_anomaly": ["currentRXPower"],
    "lane_imbalance": [
        "currentMultiTXPower1",
        "currentMultiTXPower2",
        "currentMultiTXPower3",
        "currentMultiTXPower4",
        "currentMultiRXPower1",
        "currentMultiRXPower2",
        "currentMultiRXPower3",
        "currentMultiRXPower4",
    ],
}


def load_faithfulness_by_sensor(exp_dir: Path, sensors: list[str]) -> dict[str, float]:
    path = exp_dir / "analysis" / "faithfulness_metrics.json"
    if not path.exists():
        return {sensor: 1.0 for sensor in sensors}
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows = metrics.get("single_sensor_ablation", [])
    drops = {
        row.get("sensor"): max(float(row.get("mean_risk_drop_true_positive", 0.0)), 0.0)
        for row in rows
    }
    max_drop = max([value for value in drops.values() if np.isfinite(value)] + [1e-6])
    return {sensor: float(drops.get(sensor, 0.0) / max_drop) for sensor in sensors}


def mode_faithfulness(exp_dir: Path, sensors: list[str], type_names: list[str]) -> np.ndarray:
    faith_by_sensor = load_faithfulness_by_sensor(exp_dir, sensors)
    vals = []
    for name in type_names:
        group = [sensor for sensor in MODE_SENSOR_GROUPS.get(name, []) if sensor in sensors]
        vals.append(np.mean([faith_by_sensor.get(sensor, 0.0) for sensor in group]) if group else 1.0)
    arr = np.asarray(vals, dtype=float)
    if np.nanmax(arr) > 0:
        arr = arr / np.nanmax(arr)
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)


def deviation_matrix(test_frame: pd.DataFrame, summary: dict, type_names: list[str]) -> np.ndarray:
    meta = summary.get("weak_type_label_meta", {})
    thresholds = meta.get("score_thresholds", {})
    cols = []
    for name in type_names:
        raw = pd.to_numeric(test_frame.get(f"weak_type_score_{name}", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
        threshold = float(thresholds.get(name, 1.0))
        threshold = threshold if np.isfinite(threshold) and threshold > 1e-8 else 1.0
        cols.append(raw / (raw + threshold))
    return np.clip(np.vstack(cols).T, 0.0, 1.0)


def diagnostic_scores(
    exp_dir: Path,
    summary: dict,
    test_frame: pd.DataFrame,
    type_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensors = list(summary.get("sensors", []))
    type_names = list(summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES)
    deviation = deviation_matrix(test_frame, summary, type_names)
    faith = mode_faithfulness(exp_dir, sensors, type_names)
    raw = type_probs * deviation * faith[None, :]
    row_max = np.maximum(np.nanmax(raw, axis=1, keepdims=True), 1e-8)
    return raw / row_max, deviation, faith


def choose_case(sensor_data: np.lib.npyio.NpzFile, scores_for_rank: np.ndarray | None = None) -> int:
    y_true = sensor_data["y_true"].astype(int)
    y_pred = sensor_data["y_pred"].astype(int)
    risk = sensor_data["scores"].astype(float)
    candidates = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(candidates) == 0:
        candidates = np.where(y_pred == 1)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(risk))
    if scores_for_rank is None:
        rank_score = risk[candidates]
    else:
        rank_score = risk[candidates] + 0.1 * np.nanmax(scores_for_rank[candidates], axis=1)
    return int(candidates[int(np.argmax(rank_score))])


def plot_output_flow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.54, 0.18, 0.26, "DOM telemetry\n24h window", PALETTE["blue_secondary"]),
        (0.30, 0.54, 0.18, 0.26, "FTEformer\nfault prediction", PALETTE["blue_main"]),
        (0.56, 0.62, 0.18, 0.18, "Fault risk\nalert / no alert", PALETTE["red"]),
        (0.56, 0.26, 0.18, 0.18, "Expert probability\nper fault mode", PALETTE["teal"]),
        (0.78, 0.26, 0.18, 0.18, "Diagnostic score\nTop-k candidates", PALETTE["gold"]),
        (0.78, 0.62, 0.18, 0.18, "Operator action\ntriage evidence", PALETTE["violet"]),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.10,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=PALETTE["dark"])

    arrows = [
        ((0.22, 0.67), (0.30, 0.67)),
        ((0.48, 0.67), (0.56, 0.71)),
        ((0.48, 0.62), (0.56, 0.35)),
        ((0.74, 0.35), (0.78, 0.35)),
        ((0.87, 0.44), (0.87, 0.62)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.1, color=PALETTE["dark"]))

    ax.text(
        0.67,
        0.13,
        "post-hoc diagnostic score = expert probability x sensor deviation x faithfulness",
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["dark"],
    )
    save_figure(fig, out_dir, "fig_16_ppt_fteformer_output_flow")


def plot_population_diagnosis(
    exp_dir: Path,
    summary: dict,
    test_frame: pd.DataFrame,
    sensor_data: np.lib.npyio.NpzFile,
    diag: np.ndarray,
    deviation: np.ndarray,
    faith: np.ndarray,
    out_dir: Path,
) -> None:
    type_names = list(summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES)
    display = summary.get("type_display_names", {})
    type_probs = sensor_data["type_probs"].astype(float)
    y_pred = sensor_data["y_pred"].astype(int)
    y_true = sensor_data["y_true"].astype(int)
    alert_mask = y_pred == 1
    tp_mask = (y_true == 1) & (y_pred == 1)
    use_mask = alert_mask if alert_mask.any() else np.ones(len(y_pred), dtype=bool)

    raw_top = np.argmax(type_probs[use_mask], axis=1)
    diag_top = np.argmax(diag[use_mask], axis=1)
    raw_counts = np.bincount(raw_top, minlength=len(type_names))
    diag_counts = np.bincount(diag_top, minlength=len(type_names))

    mean_prob = np.nanmean(type_probs[tp_mask if tp_mask.any() else use_mask], axis=0)
    mean_dev = np.nanmean(deviation[tp_mask if tp_mask.any() else use_mask], axis=0)
    mean_diag = np.nanmean(diag[tp_mask if tp_mask.any() else use_mask], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), gridspec_kw={"width_ratios": [1.0, 1.25]})
    labels = [display_mode(name, display) for name in type_names]
    x = np.arange(len(type_names))
    width = 0.34
    axes[0].bar(x - width / 2, raw_counts, width, color=PALETTE["blue_secondary"], label="Raw expert Top-1")
    axes[0].bar(x + width / 2, diag_counts, width, color=PALETTE["gold"], label="Diagnostic Top-1")
    axes[0].set_ylabel("Alert windows")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[0].legend(fontsize=8)

    w = 0.22
    axes[1].bar(x - w, mean_prob, w, color=PALETTE["blue_secondary"], label="Expert prob.")
    axes[1].bar(x, mean_dev, w, color=PALETTE["teal"], label="Deviation")
    axes[1].bar(x + w, mean_diag, w, color=[MODE_COLORS.get(n, PALETTE["neutral"]) for n in type_names], label="Diagnostic")
    axes[1].plot(x, faith, color=PALETTE["dark"], marker="o", linewidth=1.2, label="Faithfulness")
    axes[1].set_ylabel("Mean normalized score")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1].legend(ncol=2, fontsize=8, loc="upper left")

    fig.tight_layout(w_pad=1.2)
    save_figure(fig, out_dir, "fig_17_ppt_raw_vs_diagnostic_population")


def plot_case_dashboard(
    exp_dir: Path,
    summary: dict,
    test_frame: pd.DataFrame,
    sensor_data: np.lib.npyio.NpzFile,
    maps: np.lib.npyio.NpzFile,
    diag: np.ndarray,
    deviation: np.ndarray,
    faith: np.ndarray,
    out_dir: Path,
) -> None:
    sensors = list(summary.get("sensors", []))
    type_names = list(summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES)
    display = summary.get("type_display_names", {})
    idx = choose_case(sensor_data, scores_for_rank=diag)
    file_name = str(sensor_data["file_name"][idx])
    score = float(sensor_data["scores"][idx])
    threshold = float(summary.get("threshold", 0.5))
    y_true = int(sensor_data["y_true"][idx])
    y_pred = int(sensor_data["y_pred"][idx])
    status = "TP alert" if y_true == 1 and y_pred == 1 else ("FP alert" if y_pred == 1 else "non-alert")
    type_probs = sensor_data["type_probs"][idx].astype(float)
    top_mode_idx = int(np.argmax(diag[idx]))
    top_mode = type_names[top_mode_idx]

    files = sensor_data["file_name"].astype(str)
    module_rows = np.where(files == file_name)[0]
    module_times = test_frame.iloc[module_rows]["obs_end_ts"].to_numpy(dtype=float)
    order = np.argsort(module_times)
    module_rows = module_rows[order]
    module_times = module_times[order]
    module_scores = sensor_data["scores"][module_rows].astype(float)
    x_module = (module_times - float(test_frame.iloc[idx]["obs_end_ts"])) / 3600.0

    c2s = maps["concept_to_sensor"][idx].astype(float)
    resampled, _, _ = cache_utils.load_resampled_module(file_name)
    frame_row = test_frame.iloc[idx]
    obs = resampled.iloc[int(frame_row["obs_start_idx"]) : int(frame_row["obs_end_idx"]) + 1].copy()
    t_hours = (obs["timestamp"].to_numpy(dtype=float) - float(frame_row["obs_end_ts"])) / 3600.0

    group = [sensor for sensor in MODE_SENSOR_GROUPS.get(top_mode, []) if sensor in sensors]
    if not group:
        group = [sensors[i] for i in np.argsort(-c2s[top_mode_idx])[:4]]
    sensor_ids = [sensors.index(sensor) for sensor in group]
    if len(sensor_ids) > 5:
        attn = c2s[top_mode_idx, sensor_ids]
        sensor_ids = [sensor_ids[i] for i in np.argsort(-attn)[:5]]
    sensor_names = [sensors[i] for i in sensor_ids]
    z = local_zscore(obs[sensor_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))

    fig = plt.figure(figsize=(11.0, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2], width_ratios=[1.0, 1.35], hspace=0.38, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.plot(x_module, module_scores, color=PALETTE["blue_main"], marker="o", linewidth=1.8, markersize=3)
    ax0.axhline(threshold, color=PALETTE["dark"], linestyle="--", linewidth=0.9)
    ax0.axvline(0, color=PALETTE["red"], linewidth=1.1)
    ax0.set_ylim(-0.02, 1.04)
    ax0.set_ylabel("Fault risk")
    ax0.set_xlabel("Module timeline (h)")
    ax0.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax0.text(0.02, 0.96, f"risk={score:.3f}, {status}", transform=ax0.transAxes, va="top", fontsize=8)

    labels = [display_mode(name, display) for name in type_names]
    y = np.arange(len(type_names))
    ax1.barh(y, diag[idx], color=[MODE_COLORS.get(n, PALETTE["neutral"]) for n in type_names], edgecolor="white", linewidth=0.6)
    ax1.scatter(type_probs, y, color=PALETTE["blue_secondary"], s=28, label="Expert prob.")
    ax1.scatter(deviation[idx], y, color=PALETTE["gold"], s=28, label="Deviation")
    ax1.scatter(faith, y, color=PALETTE["dark"], s=28, label="Faithfulness")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.06)
    ax1.set_xlabel("Score")
    ax1.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    ax1.legend(ncol=3, fontsize=8, loc="lower right")

    for col, sensor in enumerate(sensor_names):
        ax2.plot(t_hours, z[:, col], linewidth=1.5, label=short_sensor_name(sensor))
    ax2.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax2.set_xlim(float(np.nanmin(t_hours)), 0.0)
    ax2.set_xlabel("Observation-window time (h)")
    ax2.set_ylabel("Local robust z-score")
    ax2.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax2.legend(ncol=2, fontsize=8, loc="upper left")

    s2t = maps["sensor_to_time"][idx].astype(float) if "sensor_to_time" in maps.files else None
    if s2t is not None:
        heat = s2t[sensor_ids]
        im = ax3.imshow(heat, aspect="auto", cmap="YlOrRd")
        ax3.set_yticks(np.arange(len(sensor_ids)))
        ax3.set_yticklabels([short_sensor_name(sensor) for sensor in sensor_names])
        ticks = np.linspace(0, heat.shape[1] - 1, 6, dtype=int)
        hour_labels = np.linspace(-24, 0, heat.shape[1])
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([f"{hour_labels[t]:.0f}" for t in ticks])
        ax3.set_xlabel("Observation-window time (h)")
        cbar = fig.colorbar(im, ax=ax3, fraction=0.035, pad=0.02)
        cbar.set_label("Sensor-time attention")
    else:
        ax3.axis("off")

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_18_ppt_single_module_diagnostic_dashboard")


def plot_faithfulness_check(exp_dir: Path, out_dir: Path) -> None:
    path = exp_dir / "analysis" / "faithfulness_metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    occlusion = metrics.get("occlusion", {})
    if not occlusion:
        return
    k_values = sorted(int(k) for k in occlusion.keys())
    x = np.arange(len(k_values))
    methods = [
        ("top", "Top evidence", PALETTE["blue_main"]),
        ("random", "Random", PALETTE["neutral"]),
        ("bottom", "Low evidence", PALETTE["red_light"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    width = 0.24
    for i, (key, label, color) in enumerate(methods):
        risk_drop = [float(occlusion[str(k)][key].get("mean_risk_drop_true_positive", 0.0)) for k in k_values]
        f1_drop = [float(occlusion[str(k)][key].get("f1_drop", 0.0)) for k in k_values]
        axes[0].bar(x + (i - 1) * width, risk_drop, width, label=label, color=color)
        axes[1].bar(x + (i - 1) * width, f1_drop, width, label=label, color=color)
    for ax, ylabel in zip(axes, ["Mean risk drop on TP", "F1 drop after occlusion"]):
        ax.set_xticks(x)
        ax.set_xticklabels([f"Top-{k}" for k in k_values])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, out_dir, "fig_19_ppt_occlusion_faithfulness_check")


def plot_sensor_ablation_ranking(exp_dir: Path, out_dir: Path) -> None:
    path = exp_dir / "analysis" / "faithfulness_metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows = metrics.get("single_sensor_ablation", [])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame["mean_risk_drop_true_positive"] = pd.to_numeric(
        frame["mean_risk_drop_true_positive"], errors="coerce"
    ).fillna(0.0)
    frame["f1_drop"] = pd.to_numeric(frame["f1_drop"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("mean_risk_drop_true_positive", ascending=False).head(8)
    frame = frame.iloc[::-1]

    labels = frame["label"].astype(str).tolist()
    risk_drop = frame["mean_risk_drop_true_positive"].to_numpy(float)
    f1_drop = frame["f1_drop"].to_numpy(float)
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.barh(y, risk_drop, color=PALETTE["blue_main"], edgecolor="white", linewidth=0.6)
    ax2 = ax.twiny()
    ax2.plot(f1_drop, y, color=PALETTE["gold"], marker="o", linewidth=1.5, label="F1 drop")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean risk drop on true-positive alerts")
    ax2.set_xlabel("F1 drop after single-sensor occlusion")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    ax2.grid(False)
    for bar, value in zip(bars, risk_drop):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_20_ppt_sensor_ablation_ranking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    exp_dir = args.exp_dir
    out_dir = args.out_dir or (exp_dir / "figures")
    summary = load_summary(exp_dir)
    test_frame = load_test_frame(summary)
    sensor_data = np.load(exp_dir / "results" / "sensor_weights.npz", allow_pickle=True)
    maps = np.load(exp_dir / "results" / "cross_attn_maps.npz", allow_pickle=True)
    diag, deviation, faith = diagnostic_scores(exp_dir, summary, test_frame, sensor_data["type_probs"].astype(float))

    plot_output_flow(out_dir)
    plot_population_diagnosis(exp_dir, summary, test_frame, sensor_data, diag, deviation, faith, out_dir)
    plot_case_dashboard(exp_dir, summary, test_frame, sensor_data, maps, diag, deviation, faith, out_dir)
    plot_faithfulness_check(exp_dir, out_dir)
    plot_sensor_ablation_ranking(exp_dir, out_dir)


if __name__ == "__main__":
    main()
