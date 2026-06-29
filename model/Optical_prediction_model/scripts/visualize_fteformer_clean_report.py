"""Publication-style report figures for the clean FTEformer experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "red_light": "#E9A6A1",
    "neutral": "#CFCECE",
    "dark": "#263238",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#D59F00",
}


def apply_style():
    plt.rcParams.update({
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
    })


def save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[fig] {png}")
    print(f"[fig] {pdf}")


def load_summary(exp_dir: Path) -> dict | None:
    path = exp_dir / "results" / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_npz(exp_dir: Path, name: str):
    path = exp_dir / "results" / name
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def sensor_label(sensor: str) -> str:
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


def model_rows(fte_dir: Path):
    candidates = [
        ("iTransformer", EXP_ROOT / "dl_itransformer_R4", PALETTE["neutral"]),
        ("PatchTST", EXP_ROOT / "dl_patchtst_R4", PALETTE["neutral"]),
        ("ModernTCN", EXP_ROOT / "dl_moderntcn_R4", PALETTE["neutral"]),
        ("FITS", EXP_ROOT / "dl_fits_R4", PALETTE["neutral"]),
        ("FTEformer", fte_dir, PALETTE["blue_main"]),
    ]
    rows = []
    for name, path, color in candidates:
        summary = load_summary(path)
        if summary is None:
            continue
        wm = summary.get("window_metrics", {})
        rows.append({
            "name": name,
            "color": color,
            "precision": float(wm.get("precision", np.nan)),
            "recall": float(wm.get("recall", np.nan)),
            "f1": float(wm.get("f1", np.nan)),
            "roc_auc": float(wm.get("auc", wm.get("roc_auc", np.nan))),
            "pr_auc": float(wm.get("average_precision", wm.get("pr_auc", np.nan))),
        })
    return rows


def fig_metric_comparison(fte_dir: Path, out_dir: Path):
    rows = model_rows(fte_dir)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("pr_auc", "PR-AUC")]
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    width = 0.18
    colors = [PALETTE["teal"], PALETTE["green"], PALETTE["blue_main"], PALETTE["violet"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for idx, (key, label) in enumerate(metrics):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=label,
                      color=colors[idx], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.12)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.14), handlelength=1.2)
    save(fig, out_dir, "fig_01_metric_comparison")


def fig_training_curves(summary: dict, out_dir: Path):
    history = summary.get("history", [])
    if not history:
        return
    epochs = np.array([h["epoch"] for h in history])
    val_f1 = np.array([h["val_f1"] for h in history], dtype=float)
    val_auc = np.array([h["val_auc"] for h in history], dtype=float)
    train_fault = np.array([h["train_fault_loss"] for h in history], dtype=float)
    train_type = np.array([h["train_type_loss"] for h in history], dtype=float)
    train_mask = np.array([h.get("train_contrast_loss", np.nan) for h in history], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    axes[0].plot(epochs, val_f1, marker="o", color=PALETTE["blue_main"], label="Val F1")
    axes[0].plot(epochs, val_auc, marker="s", color=PALETTE["teal"], label="Val ROC-AUC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation score")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[0].legend(loc="lower right")

    axes[1].plot(epochs, train_fault, marker="o", color=PALETTE["red"], label="Fault loss")
    axes[1].plot(epochs, train_type, marker="s", color=PALETTE["violet"], label="Type loss")
    axes[1].plot(epochs, train_mask, marker="^", color=PALETTE["gold"], label="Mask loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1].legend(loc="upper right")
    save(fig, out_dir, "fig_02_training_curves")


def fig_sensor_attribution(summary: dict, out_dir: Path):
    ranking = summary.get("sensor_importance", {}).get("ranking", [])
    if not ranking:
        return
    labels = [sensor_label(item["sensor"]) for item in ranking]
    values = np.array([float(item["weight"]) for item in ranking])
    y = np.arange(len(labels))
    colors = [PALETTE["blue_main"] if i < 3 else PALETTE["blue_secondary"] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution weight")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    for idx, val in enumerate(values):
        ax.text(val + values.max() * 0.02, idx, f"{val:.3f}", va="center", fontsize=8)
    save(fig, out_dir, "fig_03_sensor_attribution")


def mean_tp_mask(sensor_data):
    y_true = np.asarray(sensor_data["y_true"]).astype(int)
    y_pred = np.asarray(sensor_data["y_pred"]).astype(int)
    mask = (y_true == 1) & (y_pred == 1)
    if not mask.any():
        mask = y_pred == 1
    if not mask.any():
        mask = y_true == 1
    return mask


def fig_fault_type_sensor_heatmap(summary: dict, maps, sensor_data, out_dir: Path):
    if maps is None or sensor_data is None or "concept_to_sensor" not in maps.files:
        return
    c2s = np.asarray(maps["concept_to_sensor"], dtype=float)
    mask = mean_tp_mask(sensor_data)
    matrix = c2s[mask].mean(axis=0)
    sensors = summary.get("sensors", [])
    type_names = summary.get("fault_type_names") or summary.get("type_names_used") or summary.get("model_cfg", {}).get("concept_names", [])
    display = summary.get("type_display_names", {})
    ylabels = [display.get(name, name) for name in type_names]

    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(sensors)))
    ax.set_xticklabels([sensor_label(s) for s in sensors], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Fault-type expert")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Attention")
    save(fig, out_dir, "fig_04_fault_type_sensor_heatmap")


def fig_dynamic_mask(summary: dict, maps, sensor_data, out_dir: Path):
    if maps is None or sensor_data is None or "dynamic_sensor_mask" not in maps.files:
        return
    sensor_mask = np.asarray(maps["dynamic_sensor_mask"], dtype=float)
    y_true = np.asarray(sensor_data["y_true"]).astype(int)
    y_pred = np.asarray(sensor_data["y_pred"]).astype(int)
    tp = (y_true == 1) & (y_pred == 1)
    neg = y_true == 0
    if not tp.any():
        tp = y_true == 1
    sensors = [sensor_label(s) for s in summary.get("sensors", [])]
    matrices = [sensor_mask[tp].mean(axis=0), sensor_mask[neg].mean(axis=0)]
    labels = ["TP warnings", "Negative windows"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), constrained_layout=True)
    vmax = max(float(np.nanmax(m)) for m in matrices)
    for ax, matrix, label in zip(axes, matrices, labels):
        im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(sensors)))
        ax.set_xticklabels(sensors, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(sensors)))
        ax.set_yticklabels(sensors, fontsize=7)
        ax.set_xlabel(label)
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
    cbar.set_label("Mask probability")
    save(fig, out_dir, "fig_05_dynamic_sensor_mask")


def fig_sensor_time_case(summary: dict, maps, sensor_data, out_dir: Path):
    if maps is None or sensor_data is None or "sensor_to_time" not in maps.files:
        return
    s2t = np.asarray(maps["sensor_to_time"], dtype=float)
    y_true = np.asarray(sensor_data["y_true"]).astype(int)
    y_pred = np.asarray(sensor_data["y_pred"]).astype(int)
    scores = np.asarray(sensor_data["scores"], dtype=float)
    candidates = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(candidates) == 0:
        candidates = np.where(y_pred == 1)[0]
    if len(candidates) == 0:
        return
    idx = int(candidates[np.argmax(scores[candidates])])
    matrix = s2t[idx]
    sensors = [sensor_label(s) for s in summary.get("sensors", [])]
    t_labels = np.linspace(-24, 0, matrix.shape[1])

    fig, ax = plt.subplots(figsize=(6.9, 3.25))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    step = max(1, matrix.shape[1] // 6)
    ticks = np.arange(0, matrix.shape[1], step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t_labels[t]:.0f}" for t in ticks])
    ax.set_yticks(np.arange(len(sensors)))
    ax.set_yticklabels(sensors)
    ax.set_xlabel("Observation-window time (h)")
    ax.set_ylabel("Sensor")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Attention")
    save(fig, out_dir, "fig_06_sensor_time_case")


def fig_occlusion(faithfulness_path: Path, out_dir: Path):
    if not faithfulness_path.exists():
        return
    with open(faithfulness_path, encoding="utf-8") as f:
        metrics = json.load(f)
    occlusion = metrics.get("occlusion", {})
    if not occlusion:
        return
    k_values = sorted(int(k) for k in occlusion)
    modes = ["top", "bottom", "random"]
    labels = ["Top-k", "Bottom-k", "Random-k"]
    colors = [PALETTE["red"], PALETTE["blue_secondary"], PALETTE["neutral"]]
    x = np.arange(len(k_values))
    width = 0.23

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for mode_idx, mode in enumerate(modes):
        drops = [occlusion[str(k)][mode]["mean_risk_drop_true_positive"] for k in k_values]
        f1drops = [occlusion[str(k)][mode]["f1_drop"] for k in k_values]
        axes[0].bar(x + (mode_idx - 1) * width, drops, width, color=colors[mode_idx],
                    edgecolor="white", label=labels[mode_idx])
        axes[1].bar(x + (mode_idx - 1) * width, f1drops, width, color=colors[mode_idx],
                    edgecolor="white", label=labels[mode_idx])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(k) for k in k_values])
    axes[0].set_xlabel("Masked sensors")
    axes[0].set_ylabel("Risk drop on TP")
    axes[0].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(k) for k in k_values])
    axes[1].set_xlabel("Masked sensors")
    axes[1].set_ylabel("F1 drop")
    axes[1].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(-0.08, 1.18))
    save(fig, out_dir, "fig_07_occlusion_faithfulness")


def fig_single_sensor_ablation(faithfulness_path: Path, out_dir: Path):
    if not faithfulness_path.exists():
        return
    with open(faithfulness_path, encoding="utf-8") as f:
        metrics = json.load(f)
    rows = metrics.get("single_sensor_ablation", [])
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["mean_risk_drop_true_positive"], reverse=True)
    labels = [r.get("label", sensor_label(r["sensor"])) for r in rows]
    risk_drop = np.array([float(r["mean_risk_drop_true_positive"]) for r in rows])
    f1_drop = np.array([float(r["f1_drop"]) for r in rows])
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.0), sharey=True)
    axes[0].barh(y, risk_drop, color=PALETTE["blue_main"], edgecolor="white", linewidth=0.5)
    axes[0].axvline(0, color=PALETTE["dark"], linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Risk drop on TP")
    axes[0].grid(axis="x", color="#E0E0E0", linewidth=0.6)

    axes[1].barh(y, f1_drop, color=PALETTE["teal"], edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color=PALETTE["dark"], linewidth=0.8)
    axes[1].set_xlabel("F1 drop")
    axes[1].grid(axis="x", color="#E0E0E0", linewidth=0.6)
    save(fig, out_dir, "fig_09_single_sensor_ablation")


def fig_attribution_method_faithfulness(faithfulness_path: Path, out_dir: Path):
    if not faithfulness_path.exists():
        return
    with open(faithfulness_path, encoding="utf-8") as f:
        metrics = json.load(f)
    method_occlusion = metrics.get("method_occlusion", {})
    if not method_occlusion:
        return

    method_specs = [
        ("attention", "Attention", PALETTE["red"]),
        ("gradient", "Gradient", PALETTE["teal"]),
        ("attention_x_gradient", "Attn*Grad", PALETTE["violet"]),
        ("attention_x_occlusion", "Attn*Occl.", PALETTE["blue_main"]),
        ("random", "Random", PALETTE["neutral"]),
    ]
    k_values = sorted(int(k) for k in next(iter(method_occlusion.values())).keys())
    x = np.arange(len(k_values))
    width = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for idx, (method, label, color) in enumerate(method_specs):
        if method == "random":
            risk_vals = [
                metrics["random_occlusion"][str(k)]["mean_risk_drop_true_positive"]
                for k in k_values
            ]
            f1_vals = [
                metrics["random_occlusion"][str(k)]["f1_drop"]
                for k in k_values
            ]
        else:
            risk_vals = [
                method_occlusion[method][str(k)]["top"]["mean_risk_drop_true_positive"]
                for k in k_values
            ]
            f1_vals = [
                method_occlusion[method][str(k)]["top"]["f1_drop"]
                for k in k_values
            ]
        offset = (idx - 2) * width
        axes[0].bar(x + offset, risk_vals, width, color=color, edgecolor="white", label=label)
        axes[1].bar(x + offset, f1_vals, width, color=color, edgecolor="white", label=label)

    for ax, ylabel in zip(axes, ["Risk drop on TP", "F1 drop"]):
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel("Masked top-k sensors")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(-0.10, 1.25))
    save(fig, out_dir, "fig_10_attribution_method_faithfulness")


def fig_event_summary(summary: dict, out_dir: Path):
    wm = summary.get("window_metrics", {})
    em = summary.get("event_metrics", {})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    conf_names = ["TP", "FP", "FN"]
    conf_vals = [wm.get("true_positive", 0), wm.get("false_positive", 0), wm.get("false_negative", 0)]
    axes[0].bar(conf_names, conf_vals, color=[PALETTE["blue_main"], PALETTE["red_light"], PALETTE["red"]],
                edgecolor="white")
    axes[0].set_ylabel("Windows")
    axes[0].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    for i, val in enumerate(conf_vals):
        axes[0].text(i, val + max(conf_vals) * 0.03, f"{int(val)}", ha="center", fontsize=9)

    event_names = ["All fault", "With windows", "Evaluable", "Hit"]
    event_vals = [
        em.get("all_fault_modules_total", 0),
        em.get("all_fault_modules_with_windows", 0),
        em.get("event_modules_evaluable", 0),
        em.get("event_modules_hit", 0),
    ]
    axes[1].bar(event_names, event_vals, color=[PALETTE["neutral"], PALETTE["teal"], PALETTE["gold"], PALETTE["blue_main"]],
                edgecolor="white")
    axes[1].set_ylabel("Modules")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", color="#E0E0E0", linewidth=0.6)
    for i, val in enumerate(event_vals):
        axes[1].text(i, val + max(event_vals) * 0.03, f"{int(val)}", ha="center", fontsize=9)
    save(fig, out_dir, "fig_08_event_summary")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--faithfulness_json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    apply_style()
    out_dir = args.output_dir or (args.exp_dir / "figures")
    summary = load_summary(args.exp_dir)
    if summary is None:
        raise FileNotFoundError(args.exp_dir / "results" / "summary.json")
    maps = load_npz(args.exp_dir, "cross_attn_maps.npz")
    sensor_data = load_npz(args.exp_dir, "sensor_weights.npz")
    faithfulness_json = args.faithfulness_json or (args.exp_dir / "analysis" / "faithfulness_metrics.json")

    fig_metric_comparison(args.exp_dir, out_dir)
    fig_training_curves(summary, out_dir)
    fig_sensor_attribution(summary, out_dir)
    fig_fault_type_sensor_heatmap(summary, maps, sensor_data, out_dir)
    fig_dynamic_mask(summary, maps, sensor_data, out_dir)
    fig_sensor_time_case(summary, maps, sensor_data, out_dir)
    fig_occlusion(faithfulness_json, out_dir)
    fig_event_summary(summary, out_dir)
    fig_single_sensor_ablation(faithfulness_json, out_dir)
    fig_attribution_method_faithfulness(faithfulness_json, out_dir)


if __name__ == "__main__":
    main()
