"""Summarize FTEformer hierarchy-consistency ablation results.

This script reads existing experiment summaries and creates a compact report
figure for advisor updates. It does not change any training result.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments"
OUT_DIR = EXP_ROOT / "fteformer_hierarchy_ablation" / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

EXPERIMENTS = [
    ("Clean main", "dl_fteformer_R4_type_div_clean"),
    ("Current H=0", "dl_fteformer_R4_type_div_rerun"),
    ("H=0.005", "dl_fteformer_R4_type_div_hier0005"),
    ("H=0.02", "dl_fteformer_R4_type_div_hier002"),
]


def load_summary(exp_name: str) -> dict:
    path = EXP_ROOT / exp_name / "results" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows() -> list[dict]:
    rows = []
    for label, exp_name in EXPERIMENTS:
        summary = load_summary(exp_name)
        metrics = summary["window_metrics"]
        mask_metrics = summary.get("dynamic_sensor_mask_metrics") or {}
        rows.append(
            {
                "label": label,
                "experiment": exp_name,
                "hierarchy_weight": summary.get("model_cfg", {}).get("hierarchy_weight"),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics["auc"]),
                "pr_auc": float(metrics["average_precision"]),
                "tp": int(metrics["true_positive"]),
                "fp": int(metrics["false_positive"]),
                "fn": int(metrics["false_negative"]),
                "mask_density": float(mask_metrics.get("mean_offdiag_density", np.nan)),
                "tp_mask_density": float(mask_metrics.get("true_positive_windows_mean_offdiag_density", np.nan)),
                "expert_overlap": float(summary.get("concept_sensor_mean_offdiag_cosine", np.nan)),
            }
        )
    return rows


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_bars(ax, bars, fmt="{:.3f}", dy=0.006):
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + dy,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if fmt == "{:.3f}" else 0,
            )


def save_rows(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "hierarchy_ablation_metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    colors = [PALETTE["blue_main"], PALETTE["neutral"], PALETTE["red_strong"], PALETTE["teal"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.7, 1.0, 1.15]})

    metrics = ["precision", "recall", "f1", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1", "PR-AUC"]
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metrics))
    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        vals = [r[metric] for r in rows]
        bars = axes[0].bar(
            x + offsets[idx],
            vals,
            width=width,
            label=metric_label,
            color=[colors[j] for j in range(len(rows))],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.55 + idx * 0.12,
        )
        if metric == "f1":
            annotate_bars(axes[0], bars, fmt="{:.3f}", dy=0.004)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.70, 0.96)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=9)
    axes[0].grid(axis="y", alpha=0.18)

    count_metrics = ["tp", "fp", "fn"]
    count_labels = ["TP", "FP", "FN"]
    count_colors = [PALETTE["green_3"], PALETTE["red_strong"], PALETTE["violet"]]
    width2 = 0.23
    offsets2 = np.linspace(-width2, width2, len(count_metrics))
    for idx, (metric, metric_label) in enumerate(zip(count_metrics, count_labels)):
        vals = [r[metric] for r in rows]
        axes[1].bar(
            x + offsets2[idx],
            vals,
            width=width2,
            label=metric_label,
            color=count_colors[idx],
            edgecolor="black",
            linewidth=0.7,
        )
    axes[1].set_ylabel("Windows")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=9)
    axes[1].grid(axis="y", alpha=0.18)

    diag_metrics = ["mask_density", "expert_overlap"]
    diag_labels = ["Mask density", "Expert overlap"]
    width3 = 0.28
    offsets3 = [-width3 / 2, width3 / 2]
    for idx, (metric, metric_label) in enumerate(zip(diag_metrics, diag_labels)):
        vals = [r[metric] for r in rows]
        axes[2].bar(
            x + offsets3[idx],
            vals,
            width=width3,
            label=metric_label,
            color=PALETTE["blue_secondary"] if idx == 0 else PALETTE["teal"],
            edgecolor="black",
            linewidth=0.7,
        )
    axes[2].set_ylabel("Diagnostic value")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")
    axes[2].legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=9)
    axes[2].grid(axis="y", alpha=0.18)

    fig.tight_layout(w_pad=1.2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_fteformer_hierarchy_ablation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_faithfulness(exp_name: str) -> dict | None:
    path = EXP_ROOT / exp_name / "analysis" / "faithfulness_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_faithfulness_figure() -> None:
    panels = [
        ("Clean main", "dl_fteformer_R4_type_div_clean"),
        ("H=0.02", "dl_fteformer_R4_type_div_hier002"),
    ]
    methods = [
        ("attention", "Attention", PALETTE["neutral"]),
        ("gradient", "Gradient", PALETTE["red_strong"]),
        ("attention_x_gradient", "Attn x Grad", PALETTE["teal"]),
        ("attention_x_occlusion", "Attn x Occl", PALETTE["blue_main"]),
        ("random", "Random", PALETTE["violet"]),
    ]
    k_values = ["1", "2", "3"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    legend_handles = None
    legend_labels = None
    for ax, (panel_label, exp_name) in zip(axes, panels):
        faith = load_faithfulness(exp_name)
        if faith is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = np.asarray([1, 2, 3], dtype=float)
        for method_key, method_label, color in methods:
            vals = []
            for k in k_values:
                if method_key == "random":
                    vals.append(faith["random_occlusion"][k]["mean_risk_drop_true_positive"])
                else:
                    vals.append(faith["method_occlusion"][method_key][k]["top"]["mean_risk_drop_true_positive"])
            line = ax.plot(
                x,
                vals,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=color,
                label=method_label,
            )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_xlabel("Top-k masked sensors")
        ax.set_xticks(x)
        ax.set_title(panel_label, fontsize=12, pad=8)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Mean risk drop on TP windows")
    axes[0].set_ylim(-0.04, 0.46)
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=5,
            fontsize=9,
        )
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_fteformer_faithfulness_clean_vs_hier002.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    rows = collect_rows()
    save_rows(rows)
    make_figure(rows)
    make_faithfulness_figure()
    print(f"saved {OUT_DIR / 'hierarchy_ablation_metrics.csv'}")
    print(f"saved {OUT_DIR / 'fig_fteformer_hierarchy_ablation.png'}")
    print(f"saved {OUT_DIR / 'fig_fteformer_hierarchy_ablation.pdf'}")
    print(f"saved {OUT_DIR / 'fig_fteformer_faithfulness_clean_vs_hier002.png'}")
    print(f"saved {OUT_DIR / 'fig_fteformer_faithfulness_clean_vs_hier002.pdf'}")


if __name__ == "__main__":
    main()
