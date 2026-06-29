"""Publication-quality visualizations for MFP-Former model report.

Generates:
  1. Model comparison bar chart (F1 / Precision / Recall / PR-AUC)
  2. Concept→Sensor attention heatmap (with prior-aware bias)
  3. Concept gate activation distribution (Pos vs Neg)
  4. Per-sample sensor-time attribution case study (selected TP cases)
  5. Training convergence curve
  6. Architecture-level attribution flow diagram
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments"
FIG_DIR = EXP_ROOT / "dl_mfpformer_R4_full" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = [
    "temperature", "current", "currentTXPower", "currentRXPower",
    "currentMultiRXPower1", "currentMultiRXPower2",
    "currentMultiRXPower3", "currentMultiRXPower4",
    "currentMultiTXPower1", "currentMultiTXPower2",
    "currentMultiTXPower3", "currentMultiTXPower4",
]
SENSOR_SHORT = [
    "Temp", "Curr", "TXPwr", "RXPwr",
    "mRX1", "mRX2", "mRX3", "mRX4",
    "mTX1", "mTX2", "mTX3", "mTX4",
]
TYPE_NAMES = [
    "thermal_anomaly", "current_bias_anomaly", "tx_power_anomaly",
    "rx_power_anomaly", "lane_imbalance",
]
TYPE_DISPLAY = ["Thermal", "Bias current", "TX Power", "RX Power", "Lane Imbalance"]

# Colors
PALETTE = {
    "ml": "#95a5a6",        # grey for ML baselines
    "dl": "#5dade2",        # blue for DL baselines
    "ours": "#e74c3c",      # red for MFP-Former
    "ours_variant": "#f39c12",  # orange for iTransformer+MFP
}


# ─────────────────────────────────────────────────────────────────────
# Figure 1: Model Comparison
# ─────────────────────────────────────────────────────────────────────
def fig_model_comparison():
    """Grouped bar chart comparing all models."""
    models = [
        ("Random Forest",   0.373, 0.379, 0.367, 0.313, "ml"),
        ("XGBoost",         0.522, 0.750, 0.400, 0.392, "ml"),
        ("FITS",            0.836, 0.920, 0.767, 0.780, "dl"),
        ("iTransformer",    0.836, 0.920, 0.767, 0.798, "dl"),
        ("ModernTCN",       0.873, 0.960, 0.800, 0.800, "dl"),
        ("PatchTST",        0.889, 1.000, 0.800, 0.803, "dl"),
        ("iTrans.+MFP",     0.836, 0.920, 0.767, 0.780, "ours_variant"),
        ("MFP-Former",      0.852, 0.958, 0.767, 0.738, "ours"),
    ]
    names = [m[0] for m in models]
    metrics = {
        "F1": [m[1] for m in models],
        "Precision": [m[2] for m in models],
        "Recall": [m[3] for m in models],
        "PR-AUC": [m[4] for m in models],
    }
    colors_per_model = [PALETTE[m[5]] for m in models]

    n_models = len(names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 4.5))
    metric_colors = ["#2c3e50", "#2980b9", "#27ae60", "#8e44ad"]

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name,
                      color=metric_colors[i], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    # Highlight MFP-Former with background
    ax.axvspan(n_models - 1.5, n_models - 0.5, alpha=0.08, color="red")
    ax.axvspan(n_models - 2.5, n_models - 1.5, alpha=0.05, color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", ncol=4, framealpha=0.9)
    ax.set_title("Prediction Performance Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig1_model_comparison.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Figure 2: Concept→Sensor Heatmap
# ─────────────────────────────────────────────────────────────────────
def fig_concept_sensor_heatmap():
    """Heatmap showing which sensors each fault indicator attends to."""
    exp_dir = EXP_ROOT / "dl_mfpformer_R4_full"
    weights_data = np.load(exp_dir / "results" / "sensor_weights.npz")
    maps_data = np.load(exp_dir / "results" / "cross_attn_maps.npz")

    y_true = weights_data["y_true"]
    c2s = maps_data["concept_to_sensor"]
    pos_mask = y_true == 1
    avg_c2s = c2s[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else c2s.mean(axis=0)
    n_concepts = avg_c2s.shape[0]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(avg_c2s, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=max(0.5, avg_c2s.max()))

    ax.set_xticks(range(len(SENSORS)))
    ax.set_xticklabels(SENSOR_SHORT, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(TYPE_DISPLAY[:n_concepts], fontsize=10)

    # Draw prior boundaries
    prior_groups = [
        [0],                     # Thermal → Temp
        [1],                     # Current → Curr
        [2, 8, 9, 10, 11],      # TX Power
        [3, 4, 5, 6, 7],        # RX Power
        [4, 5, 6, 7, 8, 9, 10, 11],  # Lane Imbalance
    ]
    for i in range(min(n_concepts, len(prior_groups))):
        for j in prior_groups[i]:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 linewidth=1.8, edgecolor="#2c3e50", facecolor="none",
                                 linestyle="-")
            ax.add_patch(rect)

    for i in range(n_concepts):
        for j in range(len(SENSORS)):
            val = avg_c2s[i, j]
            color = "white" if val > avg_c2s.max() * 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=9)
    ax.set_title("Fault Indicator → Sensor Cross-Attention (Positive Samples)",
                 fontsize=12, fontweight="bold")

    # Legend annotation
    ax.annotate("■ = domain prior sensor group", xy=(0.01, -0.22),
                xycoords="axes fraction", fontsize=8, color="#2c3e50")

    plt.tight_layout()
    out = FIG_DIR / "fig2_concept_sensor_heatmap.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Figure 3: Concept Gate Distribution
# ─────────────────────────────────────────────────────────────────────
def fig_concept_gate_distribution():
    """Violin plot of concept gate values for positive vs negative samples."""
    exp_dir = EXP_ROOT / "dl_mfpformer_R4_full"
    weights_data = np.load(exp_dir / "results" / "sensor_weights.npz")
    y_true = weights_data["y_true"]
    type_probs = weights_data["type_probs"]
    n_concepts = type_probs.shape[1]

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    fig, axes = plt.subplots(1, n_concepts, figsize=(13, 3.2), sharey=True)
    for i, ax in enumerate(axes):
        neg_vals = type_probs[neg_mask, i]
        pos_vals = type_probs[pos_mask, i]

        parts_neg = ax.violinplot([neg_vals], positions=[0], showmedians=True,
                                  showextrema=False, widths=0.7)
        parts_pos = ax.violinplot([pos_vals], positions=[1], showmedians=True,
                                  showextrema=False, widths=0.7)

        for pc in parts_neg["bodies"]:
            pc.set_facecolor("#AED6F1")
            pc.set_alpha(0.8)
        parts_neg["cmedians"].set_color("#2c3e50")

        for pc in parts_pos["bodies"]:
            pc.set_facecolor("#F5B7B1")
            pc.set_alpha(0.8)
        parts_pos["cmedians"].set_color("#c0392b")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], fontsize=9)
        ax.set_title(TYPE_DISPLAY[i], fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Indicator Activation σ(logit)", fontsize=10)

    fig.suptitle("Fault Indicator Gate Distribution: Negative vs Positive Samples",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig3_concept_gate_distribution.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Figure 4: Training Convergence
# ─────────────────────────────────────────────────────────────────────
def fig_training_convergence():
    """Training loss and validation F1 curves."""
    summary_path = EXP_ROOT / "dl_mfpformer_R4_full" / "results" / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    history = summary["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    fault_loss = [h["train_fault_loss"] for h in history]
    type_loss = [h["train_type_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    # Left: training losses
    ax1.plot(epochs, train_loss, "o-", color="#2c3e50", label="Total Loss", linewidth=2, markersize=4)
    ax1.plot(epochs, fault_loss, "s--", color="#e74c3c", label="Fault BCE", linewidth=1.5, markersize=3)
    ax1.plot(epochs, type_loss, "^--", color="#3498db", label="Indicator Loss", linewidth=1.5, markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Convergence", fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: val metrics
    ax2.plot(epochs, val_f1, "o-", color="#e74c3c", label="Val F1", linewidth=2, markersize=5)
    ax2.plot(epochs, val_auc, "s-", color="#2980b9", label="Val AUC", linewidth=2, markersize=4)

    best_ep = max(range(len(val_f1)), key=lambda i: val_f1[i])
    ax2.axvline(epochs[best_ep], color="#e74c3c", alpha=0.3, linestyle="--")
    ax2.annotate(f"Best F1={val_f1[best_ep]:.3f}\n(ep{epochs[best_ep]})",
                 xy=(epochs[best_ep], val_f1[best_ep]),
                 xytext=(epochs[best_ep]+1.5, val_f1[best_ep]-0.08),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                 fontsize=9, color="#e74c3c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Performance", fontweight="bold")
    ax2.legend(framealpha=0.9)
    ax2.set_ylim(0.3, 1.02)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig4_training_convergence.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Figure 5: TP vs FN Case Study
# ─────────────────────────────────────────────────────────────────────
def fig_case_study():
    """Show TP cases (thermal-dominated + multi-indicator) vs FN cases."""
    exp_dir = EXP_ROOT / "dl_mfpformer_R4_full"
    weights_data = np.load(exp_dir / "results" / "sensor_weights.npz")
    maps_data = np.load(exp_dir / "results" / "cross_attn_maps.npz")

    y_true = weights_data["y_true"]
    scores = weights_data["scores"].flatten()
    w = weights_data["weights"]
    c2s = maps_data["concept_to_sensor"]
    type_probs = weights_data["type_probs"]
    file_names = weights_data["file_name"]
    weak_type = weights_data["weak_type_primary"]

    # ---- Deduplicate by file_name (best score per module) ----
    def dedup(idx_arr):
        seen = {}
        for idx in idx_arr[np.argsort(-scores[idx_arr])]:
            fn = file_names[idx]
            if fn not in seen:
                seen[fn] = idx
        return list(seen.values())

    tp_idx = dedup(np.where((y_true == 1) & (scores > 0.1))[0])
    fn_idx = dedup(np.where((y_true == 1) & (scores <= 0.1))[0])

    # Pick TP cases: one typical thermal + one multi-indicator (lowest temp weight)
    tp_by_temp = sorted(tp_idx, key=lambda i: w[i, 0])
    tp_picks = []
    if tp_by_temp:
        tp_picks.append(tp_by_temp[0])   # most diverse (multi-indicator)
    if len(tp_by_temp) >= 2:
        tp_picks.append(tp_by_temp[-1])   # most thermal-dominated
    # middle
    if len(tp_by_temp) >= 4:
        tp_picks.insert(1, tp_by_temp[len(tp_by_temp) // 2])

    # Pick FN cases: up to 2, prefer non-none weak_type
    fn_with_type = [i for i in fn_idx if weak_type[i] != "none"]
    fn_none = [i for i in fn_idx if weak_type[i] == "none"]
    fn_picks = (fn_with_type[:2] + fn_none[:2])[:2]

    cases = [(i, "TP") for i in tp_picks] + [(i, "FN") for i in fn_picks]
    if len(cases) == 0:
        print("  No cases to show, skipping.")
        return

    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 2.7 * n_rows),
                             gridspec_kw={"width_ratios": [3, 1.2, 2.5]})
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (idx, tag) in enumerate(cases):
        ax_bar = axes[row, 0]
        ax_gate = axes[row, 1]
        ax_hmap = axes[row, 2]

        sample_w = w[idx]
        sample_tp = type_probs[idx]
        sample_c2s = c2s[idx]
        fname = file_names[idx]
        wtype = weak_type[idx]
        score = scores[idx]

        tag_color = "#27ae60" if tag == "TP" else "#c0392b"
        tag_label = f"[{tag}]"

        # --- Column 1: sensor importance bar ---
        colors = ["#e74c3c" if v == sample_w.max() else "#5dade2" for v in sample_w]
        ax_bar.bar(range(len(SENSORS)), sample_w, color=colors,
                   edgecolor="white", linewidth=0.5)
        ax_bar.set_xticks(range(len(SENSORS)))
        ax_bar.set_xticklabels(SENSOR_SHORT, rotation=45, ha="right", fontsize=7.5)
        ax_bar.set_ylabel("Importance", fontsize=9)
        title = f"{tag_label} {fname}  score={score:.2f}  type={wtype}"
        ax_bar.set_title(title, fontsize=9.5, fontweight="bold", color=tag_color)
        ax_bar.set_ylim(0, min(1.0, sample_w.max() * 1.25 + 0.01))
        ax_bar.grid(axis="y", alpha=0.2)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        for bi, bv in enumerate(sample_w):
            if bv > 0.03:
                ax_bar.text(bi, bv + 0.003, f"{bv:.2f}", ha="center", fontsize=6.5)

        # --- Column 2: indicator gate ---
        n_c = len(sample_tp)
        bar_colors = ["#e74c3c" if v > 0.1 else "#bdc3c7" for v in sample_tp]
        ax_gate.barh(range(n_c), sample_tp, color=bar_colors, edgecolor="white")
        ax_gate.set_yticks(range(n_c))
        ax_gate.set_yticklabels(TYPE_DISPLAY[:n_c], fontsize=7.5)
        ax_gate.set_xlim(0, 1.05)
        ax_gate.set_xlabel("Gate σ", fontsize=8)
        ax_gate.set_title("Indicator Gate", fontsize=9)
        for j, v in enumerate(sample_tp):
            if v > 0.01:
                ax_gate.text(min(v + 0.02, 0.85), j, f"{v:.2f}",
                             va="center", fontsize=7)

        # --- Column 3: per-case concept→sensor heatmap ---
        im = ax_hmap.imshow(sample_c2s, aspect="auto", cmap="YlOrRd",
                            interpolation="nearest", vmin=0, vmax=0.5)
        ax_hmap.set_xticks(range(len(SENSORS)))
        ax_hmap.set_xticklabels(SENSOR_SHORT, rotation=45, ha="right", fontsize=6.5)
        ax_hmap.set_yticks(range(n_c))
        ax_hmap.set_yticklabels([t[:6] for t in TYPE_DISPLAY[:n_c]], fontsize=7)
        ax_hmap.set_title("Indicator→Sensor Attn", fontsize=9)
        for ci in range(n_c):
            for si in range(len(SENSORS)):
                val = sample_c2s[ci, si]
                if val > 0.05:
                    clr = "white" if val > 0.25 else "black"
                    ax_hmap.text(si, ci, f"{val:.2f}", ha="center", va="center",
                                 fontsize=5.5, color=clr)

    plt.tight_layout()
    out = FIG_DIR / "fig5_case_study.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Figure 6: Global Sensor Importance (grouped by sensor category)
# ─────────────────────────────────────────────────────────────────────
def fig_sensor_importance_grouped():
    """Grouped sensor importance with sensor categories."""
    exp_dir = EXP_ROOT / "dl_mfpformer_R4_full"
    weights_data = np.load(exp_dir / "results" / "sensor_weights.npz")
    y_true = weights_data["y_true"]
    w = weights_data["weights"]

    pos_mask = y_true == 1
    avg_w = w[pos_mask].mean(axis=0) if pos_mask.sum() > 0 else w.mean(axis=0)

    # Group sensors
    groups = {
        "Temperature": [0],
        "Bias Current": [1],
        "Aggregate TX": [2],
        "Aggregate RX": [3],
        "Multi-Lane RX": [4, 5, 6, 7],
        "Multi-Lane TX": [8, 9, 10, 11],
    }
    group_names = list(groups.keys())
    group_vals = [sum(avg_w[i] for i in idxs) for idxs in groups.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                    gridspec_kw={"width_ratios": [2, 1.3]})

    # Left: per-sensor bar
    colors_map = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71", 3: "#3498db",
                  4: "#3498db", 5: "#3498db", 6: "#3498db", 7: "#3498db",
                  8: "#2ecc71", 9: "#2ecc71", 10: "#2ecc71", 11: "#2ecc71"}
    colors = [colors_map[i] for i in range(12)]
    bars = ax1.bar(range(12), avg_w, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(SENSOR_SHORT, rotation=45, ha="right")
    ax1.set_ylabel("Importance Weight")
    ax1.set_title("Per-Sensor Importance (Positive Samples)", fontweight="bold")
    for bar, val in zip(bars, avg_w):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # Right: grouped pie/bar
    pie_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#3498db", "#2ecc71"]
    wedges, texts, autotexts = ax2.pie(
        group_vals, labels=group_names, autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        colors=pie_colors, startangle=90, textprops={"fontsize": 8},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax2.set_title("Sensor Group Attribution", fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "fig6_sensor_importance_grouped.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
def main():
    print("Generating report-quality figures...")
    fig_model_comparison()
    fig_concept_sensor_heatmap()
    fig_concept_gate_distribution()
    fig_training_convergence()
    fig_case_study()
    fig_sensor_importance_grouped()
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
