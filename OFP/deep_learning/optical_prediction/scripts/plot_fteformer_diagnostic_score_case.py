"""Plot a post-hoc diagnostic-score case study for FTEformer.

This figure does not change the binary fault prediction score.  It only
reranks fault-mode candidates after an alert by combining the learned expert
probability, weak physical-deviation evidence, and global faithfulness evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    drops = {row.get("sensor"): max(float(row.get("mean_risk_drop_true_positive", 0.0)), 0.0) for row in rows}
    max_drop = max([value for value in drops.values() if np.isfinite(value)] + [1e-6])
    return {sensor: float(drops.get(sensor, 0.0) / max_drop) for sensor in sensors}


def group_faithfulness(type_name: str, sensors: list[str], faith_by_sensor: dict[str, float]) -> float:
    group = [sensor for sensor in MODE_SENSOR_GROUPS.get(type_name, []) if sensor in sensors]
    if not group:
        return 1.0
    values = [faith_by_sensor.get(sensor, 0.0) for sensor in group]
    return float(np.mean(values))


def choose_case(sensor_data: np.lib.npyio.NpzFile, case_idx: int | None) -> int:
    if case_idx is not None:
        return int(case_idx)
    y_true = sensor_data["y_true"].astype(int)
    y_pred = sensor_data["y_pred"].astype(int)
    scores = sensor_data["scores"].astype(float)
    candidates = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(candidates) == 0:
        candidates = np.where(y_pred == 1)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(scores))
    return int(candidates[np.argmax(scores[candidates])])


def mode_deviation_scores(frame_row: pd.Series, summary: dict, type_names: list[str]) -> np.ndarray:
    meta = summary.get("weak_type_label_meta", {})
    thresholds = meta.get("score_thresholds", {})
    values = []
    for name in type_names:
        raw = float(frame_row.get(f"weak_type_score_{name}", 0.0))
        threshold = float(thresholds.get(name, 1.0))
        threshold = threshold if np.isfinite(threshold) and threshold > 1e-8 else 1.0
        values.append(raw / (raw + threshold))
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def mode_faithfulness_scores(exp_dir: Path, sensors: list[str], type_names: list[str]) -> np.ndarray:
    faith_by_sensor = load_faithfulness_by_sensor(exp_dir, sensors)
    values = np.asarray([group_faithfulness(name, sensors, faith_by_sensor) for name in type_names], dtype=float)
    if np.nanmax(values) > 0:
        values = values / np.nanmax(values)
    return np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)


def plot_case(exp_dir: Path, out_dir: Path, case_idx: int | None = None) -> Path:
    apply_style()
    summary = load_summary(exp_dir)
    sensor_data = np.load(exp_dir / "results" / "sensor_weights.npz", allow_pickle=True)
    maps = np.load(exp_dir / "results" / "cross_attn_maps.npz", allow_pickle=True)
    test_frame = load_test_frame(summary)

    sensors = list(summary.get("sensors", []))
    type_names = list(summary.get("fault_type_names") or summary.get("type_names_used") or TYPE_NAMES)
    display = summary.get("type_display_names", {})
    idx = choose_case(sensor_data, case_idx)

    frame_row = test_frame.iloc[idx]
    file_name = str(sensor_data["file_name"][idx])
    score = float(sensor_data["scores"][idx])
    threshold = float(summary.get("threshold", 0.5))
    y_true = int(sensor_data["y_true"][idx])
    y_pred = int(sensor_data["y_pred"][idx])
    status = "TP alert" if y_true == 1 and y_pred == 1 else ("FP alert" if y_pred == 1 else "non-alert")

    expert_prob = sensor_data["type_probs"][idx].astype(float)
    deviation = mode_deviation_scores(frame_row, summary, type_names)
    faithfulness = mode_faithfulness_scores(exp_dir, sensors, type_names)
    diagnostic_raw = expert_prob * deviation * faithfulness
    diagnostic_score = diagnostic_raw / max(float(np.nanmax(diagnostic_raw)), 1e-8)
    top_mode_idx = int(np.argmax(diagnostic_score))
    top_mode = type_names[top_mode_idx]

    files = sensor_data["file_name"].astype(str)
    module_rows = np.where(files == file_name)[0]
    module_times = test_frame.iloc[module_rows]["obs_end_ts"].to_numpy(dtype=float)
    order = np.argsort(module_times)
    module_rows = module_rows[order]
    module_times = module_times[order]
    module_scores = sensor_data["scores"][module_rows].astype(float)
    x_module = (module_times - float(frame_row["obs_end_ts"])) / 3600.0

    resampled, _, _ = cache_utils.load_resampled_module(file_name)
    obs = resampled.iloc[int(frame_row["obs_start_idx"]) : int(frame_row["obs_end_idx"]) + 1].copy()
    t_hours = (obs["timestamp"].to_numpy(dtype=float) - float(frame_row["obs_end_ts"])) / 3600.0

    c2s = maps["concept_to_sensor"][idx].astype(float)
    group_sensors = [sensor for sensor in MODE_SENSOR_GROUPS.get(top_mode, []) if sensor in sensors]
    if not group_sensors:
        group_sensors = [sensors[i] for i in np.argsort(-c2s[top_mode_idx])[:4]]
    sensor_ids = [sensors.index(sensor) for sensor in group_sensors if sensor in sensors]
    if len(sensor_ids) > 5:
        attn = c2s[top_mode_idx, sensor_ids]
        sensor_ids = [sensor_ids[i] for i in np.argsort(-attn)[:5]]

    obs_values = obs[[sensors[i] for i in sensor_ids]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    z_values = local_zscore(obs_values)
    s2t = maps["sensor_to_time"][idx].astype(float) if "sensor_to_time" in maps.files else None

    fig = plt.figure(figsize=(10.8, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.88, 1.12], width_ratios=[1.05, 1.35], hspace=0.42, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.plot(x_module, module_scores, color=PALETTE["blue_main"], linewidth=1.8, marker="o", markersize=3)
    ax0.axhline(threshold, color=PALETTE["dark"], linestyle="--", linewidth=0.9)
    ax0.axvline(0, color=PALETTE["red"], linewidth=1.1)
    ax0.set_ylim(-0.02, 1.04)
    ax0.set_xlabel("Module timeline relative to selected alert (h)")
    ax0.set_ylabel("Fault risk")
    ax0.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax0.text(0.02, 0.95, f"risk={score:.3f}, {status}", transform=ax0.transAxes, va="top", fontsize=8)

    mode_labels = [display_mode(name, display) for name in type_names]
    x = np.arange(len(type_names))
    width = 0.18
    ax1.bar(x - 1.5 * width, expert_prob, width, color=PALETTE["blue_secondary"], label="Expert prob.")
    ax1.bar(x - 0.5 * width, deviation, width, color=PALETTE["gold"], label="Deviation")
    ax1.bar(x + 0.5 * width, faithfulness, width, color=PALETTE["teal"], label="Faithfulness")
    ax1.bar(x + 1.5 * width, diagnostic_score, width, color=[MODE_COLORS.get(n, PALETTE["neutral"]) for n in type_names], label="Diagnostic")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Normalized score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels, rotation=28, ha="right")
    ax1.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax1.legend(ncol=2, fontsize=8, loc="upper left")

    for col, sensor_id in enumerate(sensor_ids):
        ax2.plot(t_hours, z_values[:, col], linewidth=1.5, label=short_sensor_name(sensors[sensor_id]))
    ax2.axvline(0, color=PALETTE["dark"], linewidth=0.8)
    ax2.set_xlim(float(np.nanmin(t_hours)), 0.0)
    ax2.set_xlabel("Observation-window time (h)")
    ax2.set_ylabel("Local robust z-score")
    ax2.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax2.legend(ncol=2, fontsize=8, loc="upper left")

    if s2t is not None and sensor_ids:
        heat = s2t[sensor_ids]
        im = ax3.imshow(heat, aspect="auto", cmap="YlOrRd")
        ax3.set_yticks(np.arange(len(sensor_ids)))
        ax3.set_yticklabels([short_sensor_name(sensors[i]) for i in sensor_ids])
        ticks = np.linspace(0, heat.shape[1] - 1, 6, dtype=int)
        hour_labels = np.linspace(-24, 0, heat.shape[1])
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([f"{hour_labels[t]:.0f}" for t in ticks])
        ax3.set_xlabel("Observation-window time (h)")
        cbar = fig.colorbar(im, ax=ax3, fraction=0.035, pad=0.02)
        cbar.set_label("Sensor-time attention")
    else:
        ax3.axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "fig_15_diagnostic_score_case_study")
    return out_dir / "fig_15_diagnostic_score_case_study.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--case_idx", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.exp_dir / "figures")
    path = plot_case(args.exp_dir, out_dir, args.case_idx)
    print(f"[case] {path}")


if __name__ == "__main__":
    main()
