"""Full-picture attribution visualization for MFP-Former.

Addresses the issue that concept-path sensor_importance only shows ~34% of
the model's decision. This script adds gradient-based and ablation-based
attribution to reveal the full sensor reliance across both paths.

Figures:
  fig7: Multi-method sensor importance comparison (gradient vs concept-path)
  fig8: Dual-path decomposition (global path vs concept path)
  fig9: Gradient-based per-sensor importance (grouped bar)
"""
from __future__ import annotations

import sys, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from model.Optical_prediction_model.scripts.run_dl_l4_typeaware_R4 import (
    r4_task_cfg, TypeAwareFailureWindowDataset, collate_type, TYPE_NAMES,
)
from model.Optical_prediction_model.deep_learning.data import (
    WindowSliceCache, compute_norm_stats,
)
from model.Optical_prediction_model.deep_learning.fault_type_labels import annotate_fault_type_frames
from model.Optical_prediction_model.deep_learning.train import load_task_frames

EXP_DIR = PROJECT / "output" / "Optical_prediction_model" / "experiments" / "dl_mfpformer_R4_full"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = [
    "temperature", "current", "currentTXPower", "currentRXPower",
    "currentMultiRXPower1", "currentMultiRXPower2",
    "currentMultiRXPower3", "currentMultiRXPower4",
    "currentMultiTXPower1", "currentMultiTXPower2",
    "currentMultiTXPower3", "currentMultiTXPower4",
]
SHORT = ["Temp", "Curr", "TXPwr", "RXPwr",
         "mRX1", "mRX2", "mRX3", "mRX4",
         "mTX1", "mTX2", "mTX3", "mTX4"]
TYPE_DISPLAY = ["Thermal", "Bias current", "TX Power", "RX Power", "Lane Imbalance"]

# Sensor group colors
GROUP_COLORS = {
    "Temp": "#e74c3c",
    "Curr": "#f39c12",
    "TXPwr": "#2ecc71", "mTX1": "#2ecc71", "mTX2": "#2ecc71", "mTX3": "#2ecc71", "mTX4": "#2ecc71",
    "RXPwr": "#3498db", "mRX1": "#3498db", "mRX2": "#3498db", "mRX3": "#3498db", "mRX4": "#3498db",
}
SENSOR_COLORS = [GROUP_COLORS.get(s, "#95a5a6") for s in SHORT]

GROUP_MAP = {
    "Temperature": [0],
    "Bias Current": [1],
    "TX Power": [2, 8, 9, 10, 11],
    "RX Power": [3, 4, 5, 6, 7],
}
GROUP_PALETTE = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]


def load_model_and_data():
    from model.Optical_prediction_model.deep_learning.typeaware_transformer import (
        TypeAwareTransformerCfg, TypeAwareCrossAttnTransformer,
    )
    summary = json.load(open(EXP_DIR / "results" / "summary.json"))
    mcfg = summary["model_cfg"]
    cfg = TypeAwareTransformerCfg(**mcfg)
    model = TypeAwareCrossAttnTransformer(cfg)
    ckpt = torch.load(EXP_DIR / "models" / "model.pt", map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    task_cfg = r4_task_cfg()
    frames = load_task_frames(task_cfg=task_cfg, tps_lead_minutes=task_cfg.lead_minutes)
    frames, _ = annotate_fault_type_frames(frames)
    obs_steps = int(frames["train"].iloc[0]["obs_end_idx"] - frames["train"].iloc[0]["obs_start_idx"] + 1)
    slice_cache = WindowSliceCache()
    norm = compute_norm_stats(frames["train"], slice_cache, obs_steps, max_modules=400)
    test_ds = TypeAwareFailureWindowDataset(frames["test"], obs_steps, slice_cache, norm, TYPE_NAMES)
    threshold = summary["threshold"]
    return model, test_ds, device, threshold


def compute_gradient_importance(model, test_ds, device):
    """Input gradient importance on positive samples."""
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_type)
    grad_accum = torch.zeros(12, device=device)
    count = 0
    for batch in loader:
        x = batch[0].to(device).requires_grad_(True)
        mask = batch[1].to(device)
        y = batch[2].to(device)
        pos_mask = y == 1
        if pos_mask.sum() == 0:
            continue
        out = model(x, mask)
        out[0][pos_mask].sum().backward()
        g = x.grad[pos_mask].abs().mean(dim=1)
        grad_accum += g.sum(dim=0)
        count += pos_mask.sum().item()
        x.grad = None
    gi = (grad_accum / max(count, 1)).cpu().numpy()
    return gi / gi.sum()


def compute_concept_importance(test_ds, device):
    """Load pre-computed concept-path sensor importance."""
    data = np.load(EXP_DIR / "results" / "sensor_weights.npz")
    y_true = data["y_true"]
    w = data["weights"]
    pos = y_true == 1
    avg = w[pos].mean(axis=0) if pos.sum() > 0 else w.mean(axis=0)
    return avg / avg.sum()


def compute_ablation_scores(model, test_ds, device):
    """Score drop when zeroing each sensor (on positive samples only)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_type)

    def get_pos_scores(loader_fn):
        all_s, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                x, mask, y = loader_fn(batch, device)
                s = torch.sigmoid(model(x, mask)[0]).cpu().numpy()
                all_s.append(s)
                all_y.append(batch[2].numpy())
        return np.concatenate(all_s), np.concatenate(all_y)

    def baseline_fn(batch, dev):
        return batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)

    base_scores, all_y = get_pos_scores(baseline_fn)
    pos = all_y == 1
    base_pos_mean = base_scores[pos].mean()

    drops = np.zeros(12)
    for sid in range(12):
        def ablate_fn(batch, dev, _sid=sid):
            x = batch[0].to(dev).clone()
            m = batch[1].to(dev).clone()
            x[:, :, _sid] = 0.0
            m[:, :, _sid] = 0.0
            return x, m, batch[2].to(dev)
        abl_scores, _ = get_pos_scores(ablate_fn)
        drops[sid] = base_pos_mean - abl_scores[pos].mean()

    # Normalize: positive drop = sensor is important
    drops_pos = np.maximum(drops, 0)
    if drops_pos.sum() > 0:
        drops_pos = drops_pos / drops_pos.sum()
    return drops, drops_pos


# ─────────────────────────────────────────────────────────────────────
# Fig 7: Multi-method comparison
# ─────────────────────────────────────────────────────────────────────
def fig_multimethod_comparison(grad_imp, concept_imp, ablation_norm):
    """Side-by-side comparison of 3 attribution methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    methods = [
        ("Input Gradient\n(Full Model)", grad_imp, True),
        ("Concept-Path Attribution\n(Indicator × Attention)", concept_imp, True),
        ("Ablation Score Drop\n(Zero-out Each Sensor)", ablation_norm, True),
    ]

    for ax, (title, vals, do_pct) in zip(axes, methods):
        bars = ax.bar(range(12), vals, color=SENSOR_COLORS, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(12))
        ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if do_pct:
            ax.set_ylabel("Normalized Importance")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=6.5)

    # Add legend for sensor groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Temperature"),
        Patch(facecolor="#f39c12", label="Bias Current"),
        Patch(facecolor="#2ecc71", label="TX Power (agg + lanes)"),
        Patch(facecolor="#3498db", label="RX Power (agg + lanes)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "fig7_multimethod_attribution.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Fig 8: Sensor group importance — gradient vs concept path
# ─────────────────────────────────────────────────────────────────────
def fig_grouped_dual_path(grad_imp, concept_imp):
    """Show how sensor GROUP attribution differs between gradient and concept views."""
    group_names = list(GROUP_MAP.keys())

    grad_groups = [sum(grad_imp[i] for i in idxs) for idxs in GROUP_MAP.values()]
    concept_groups = [sum(concept_imp[i] for i in idxs) for idxs in GROUP_MAP.values()]

    x = np.arange(len(group_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width/2, grad_groups, width, label="Gradient-Based\n(full model reliance)",
                   color=[c + "CC" for c in GROUP_PALETTE], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, concept_groups, width, label="Concept-Path\n(indicator × attention)",
                   color=GROUP_PALETTE, edgecolor="white", linewidth=0.5,
                   hatch="//", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=11)
    ax.set_ylabel("Normalized Importance", fontsize=11)
    ax.set_title("Sensor Group Attribution: Full Model vs Concept Path",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "fig8_grouped_dual_path.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Fig 9: Architecture dual-path diagram with numbers
# ─────────────────────────────────────────────────────────────────────
def fig_path_decomposition():
    """Visualize the dual-path architecture contribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def box(x, y, w, h, text, color, fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="#2c3e50", linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white" if color != "#ecf0f1" else "#2c3e50")

    def arrow(x1, y1, x2, y2, text="", color="#2c3e50"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.15, my, text, fontsize=8, color=color)

    # Input
    box(0.2, 2.3, 2, 1.2, "12 Sensor\nTime Series\n(24h window)", "#7f8c8d", 10)

    # Temporal Encoder
    box(3, 2.5, 2, 0.8, "Temporal\nTransformer", "#2c3e50", 10)
    arrow(2.2, 2.9, 3.0, 2.9)

    # Global path (top)
    box(6, 4.2, 2.5, 1.0, "Global Path\n(pooling + bypass)\n66% contribution", "#2980b9", 9)
    arrow(5.0, 3.1, 6.0, 4.5, color="#2980b9")

    # Concept path (bottom)
    box(6, 0.8, 2.5, 1.0, "Concept Path\n(5 indicators × gate)\n34% contribution", "#e74c3c", 9)
    arrow(5.0, 2.7, 6.0, 1.5, color="#e74c3c")

    # Final fusion
    box(9.0, 2.3, 0.8, 1.2, "Risk\nScore", "#27ae60", 9)
    arrow(8.5, 4.7, 9.2, 3.5, color="#2980b9")
    arrow(8.5, 1.3, 9.2, 2.3, color="#e74c3c")

    # Annotations
    ax.text(6.1, 5.4, "Relies on: mRX (45%), mTX (44%), Curr (2%), Temp (1%)",
            fontsize=8.5, color="#2980b9", style="italic")
    ax.text(6.1, 0.4, "Relies on: Temp (96%), others (~1% each)",
            fontsize=8.5, color="#e74c3c", style="italic")

    ax.set_title("MFP-Former Dual-Path Decision Architecture",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    out = FIG_DIR / "fig9_path_decomposition.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Fig 10: Concept heatmap + gradient overlay
# ─────────────────────────────────────────────────────────────────────
def fig_concept_heatmap_with_gradient(grad_imp, concept_imp):
    """Concept attention heatmap with gradient importance overlay on top."""
    data = np.load(EXP_DIR / "results" / "sensor_weights.npz")
    maps = np.load(EXP_DIR / "results" / "cross_attn_maps.npz")
    y_true = data["y_true"]
    c2s = maps["concept_to_sensor"]
    pos = y_true == 1
    avg_c2s = c2s[pos].mean(axis=0) if pos.sum() > 0 else c2s.mean(axis=0)
    n_concepts = avg_c2s.shape[0]

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 3], hspace=0.15)

    # Top: gradient importance bar
    ax_top = fig.add_subplot(gs[0])
    ax_top.bar(range(12), grad_imp, color=SENSOR_COLORS, edgecolor="white", linewidth=0.5)
    ax_top.set_xticks(range(12))
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("Gradient\nImportance", fontsize=9)
    ax_top.set_title("Full-Model Sensor Reliance (gradient) vs Indicator Attention (heatmap)",
                     fontsize=12, fontweight="bold")
    ax_top.set_xlim(-0.5, 11.5)
    ax_top.grid(axis="y", alpha=0.2)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    for i, v in enumerate(grad_imp):
        if v > 0.02:
            ax_top.text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=7)

    # Bottom: concept attention heatmap
    ax_bot = fig.add_subplot(gs[1])
    im = ax_bot.imshow(avg_c2s, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest", vmin=0, vmax=max(0.5, avg_c2s.max()))
    ax_bot.set_xticks(range(12))
    ax_bot.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=9)
    ax_bot.set_yticks(range(n_concepts))
    ax_bot.set_yticklabels(TYPE_DISPLAY[:n_concepts], fontsize=10)

    # Prior boxes
    prior_groups = [
        [0], [1], [2, 8, 9, 10, 11], [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8, 9, 10, 11],
    ]
    for i in range(min(n_concepts, len(prior_groups))):
        for j in prior_groups[i]:
            ax_bot.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             linewidth=1.8, edgecolor="#2c3e50", facecolor="none"))

    for i in range(n_concepts):
        for j in range(12):
            val = avg_c2s[i, j]
            color = "white" if val > avg_c2s.max() * 0.55 else "black"
            ax_bot.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax_bot, fraction=0.02, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "fig10_gradient_plus_heatmap.png"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading model and data...")
    model, test_ds, device, threshold = load_model_and_data()

    print("Computing gradient importance...")
    grad_imp = compute_gradient_importance(model, test_ds, device)

    print("Loading concept-path importance...")
    concept_imp = compute_concept_importance(test_ds, device)

    print("Computing ablation scores...")
    ablation_raw, ablation_norm = compute_ablation_scores(model, test_ds, device)

    print("\nGenerating figures...")
    fig_multimethod_comparison(grad_imp, concept_imp, ablation_norm)
    fig_grouped_dual_path(grad_imp, concept_imp)
    fig_path_decomposition()
    fig_concept_heatmap_with_gradient(grad_imp, concept_imp)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("\n=== Summary ===")
    print("Gradient-based sensor GROUP importance:")
    for name, idxs in GROUP_MAP.items():
        print(f"  {name:15s}: {sum(grad_imp[i] for i in idxs):.1%}")
    print("Concept-path sensor GROUP importance:")
    for name, idxs in GROUP_MAP.items():
        print(f"  {name:15s}: {sum(concept_imp[i] for i in idxs):.1%}")


if __name__ == "__main__":
    main()
