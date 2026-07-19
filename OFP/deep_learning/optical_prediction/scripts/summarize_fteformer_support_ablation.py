"""Summarize FTEformer task-support consistency experiments."""
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
OUT_DIR = EXP_ROOT / "fteformer_support_ablation" / "figures"

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
    ("Clean main", "dl_fteformer_R4_type_div_clean", "max"),
    ("Current H=0", "dl_fteformer_R4_type_div_rerun", "max"),
    ("NoisyOR S=.03", "dl_fteformer_R4_type_div_support003", "noisy_or"),
    ("Max S=.03", "dl_fteformer_R4_type_div_supportmax003", "max"),
    ("Max S=.10 N=.20", "dl_fteformer_R4_type_div_supportmax010_neg020", "max"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary(exp_name: str) -> dict:
    path = EXP_ROOT / exp_name / "results" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compute_support_from_npz(exp_name: str, mode: str = "max") -> dict:
    path = EXP_ROOT / exp_name / "results" / "sensor_weights.npz"
    data = np.load(path)
    scores = np.asarray(data["scores"], dtype=np.float32)
    type_probs = np.asarray(data["type_probs"], dtype=np.float32)
    y_true = np.asarray(data["y_true"], dtype=int)
    y_pred = np.asarray(data["y_pred"], dtype=int)
    noisy_or = 1.0 - np.prod(np.clip(1.0 - type_probs, 1e-6, 1.0), axis=-1)
    max_support = type_probs.max(axis=-1)
    top2_support = np.sort(type_probs, axis=-1)[:, -2:].mean(axis=-1)
    support = {"max": max_support, "top2": top2_support, "noisy_or": noisy_or}.get(mode, max_support)
    gap = scores - support
    out = {
        "support_mode": mode,
        "positive_mean_support": float(support[y_true == 1].mean()),
        "negative_mean_support": float(support[y_true == 0].mean()),
    }
    for name, mask in {
        "true_positive": (y_true == 1) & (y_pred == 1),
        "false_positive": (y_true == 0) & (y_pred == 1),
        "false_negative": (y_true == 1) & (y_pred == 0),
    }.items():
        out[f"{name}_count"] = int(mask.sum())
        if np.any(mask):
            out[f"{name}_mean_event_score"] = float(scores[mask].mean())
            out[f"{name}_mean_support_score"] = float(support[mask].mean())
            out[f"{name}_mean_event_minus_support"] = float(gap[mask].mean())
    return out


def collect_rows() -> list[dict]:
    rows = []
    for label, exp_name, mode_override in EXPERIMENTS:
        summary = load_summary(exp_name)
        metrics = summary["window_metrics"]
        support_cfg = summary.get("support_loss_cfg") or {}
        support = summary.get("support_metrics")
        support_mode = (
            (support or {}).get("support_mode")
            or support_cfg.get("support_mode")
            or mode_override
        )
        support = support or compute_support_from_npz(exp_name, mode=support_mode)
        rows.append(
            {
                "label": label,
                "experiment": exp_name,
                "support_weight": support_cfg.get("support_weight", 0.0),
                "support_mode": support_mode,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics["auc"]),
                "pr_auc": float(metrics["average_precision"]),
                "tp": int(metrics["true_positive"]),
                "fp": int(metrics["false_positive"]),
                "fn": int(metrics["false_negative"]),
                "tp_support": float(support.get("true_positive_mean_support_score", np.nan)),
                "fp_support": float(support.get("false_positive_mean_support_score", np.nan)),
                "fn_support": float(support.get("false_negative_mean_support_score", np.nan)),
                "positive_support": float(support.get("positive_mean_support", np.nan)),
                "negative_support": float(support.get("negative_mean_support", np.nan)),
                "expert_overlap": float(summary.get("concept_sensor_mean_offdiag_cosine", np.nan)),
            }
        )
    return rows


def load_faithfulness(exp_name: str) -> dict | None:
    path = EXP_ROOT / exp_name / "analysis" / "faithfulness_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "support_ablation_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary_figure(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    colors = [PALETTE["blue_main"], PALETTE["neutral"], PALETTE["teal"], PALETTE["violet"], PALETTE["red_strong"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]})

    metric_names = ["precision", "recall", "f1", "pr_auc"]
    metric_labels = ["P", "R", "F1", "PR-AUC"]
    width = 0.17
    for i, (metric, metric_label) in enumerate(zip(metric_names, metric_labels)):
        vals = [r[metric] for r in rows]
        axes[0].bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=metric_label,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.55 + i * 0.11,
        )
    axes[0].set_ylim(0.70, 0.96)
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=8)
    axes[0].grid(axis="y", alpha=0.18)

    count_names = ["tp", "fp", "fn"]
    count_labels = ["TP", "FP", "FN"]
    count_colors = [PALETTE["green_3"], PALETTE["red_strong"], PALETTE["violet"]]
    width2 = 0.22
    for i, (metric, metric_label) in enumerate(zip(count_names, count_labels)):
        axes[1].bar(
            x + (i - 1) * width2,
            [r[metric] for r in rows],
            width=width2,
            label=metric_label,
            color=count_colors[i],
            edgecolor="black",
            linewidth=0.6,
        )
    axes[1].set_ylabel("Windows")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=8)
    axes[1].grid(axis="y", alpha=0.18)

    support_names = ["tp_support", "fp_support", "negative_support"]
    support_labels = ["TP support", "FP support", "Neg support"]
    support_colors = [PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["neutral"]]
    width3 = 0.22
    for i, (metric, metric_label) in enumerate(zip(support_names, support_labels)):
        axes[2].bar(
            x + (i - 1) * width3,
            [r[metric] for r in rows],
            width=width3,
            label=metric_label,
            color=support_colors[i],
            edgecolor="black",
            linewidth=0.6,
        )
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("Fault-mode support")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")
    axes[2].legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=8)
    axes[2].grid(axis="y", alpha=0.18)

    fig.tight_layout(w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_fteformer_support_ablation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_faithfulness_figure() -> None:
    panels = [
        ("Clean main", "dl_fteformer_R4_type_div_clean"),
        ("Max S=.10 N=.20", "dl_fteformer_R4_type_div_supportmax010_neg020"),
    ]
    methods = [
        ("attention", "Attention", PALETTE["neutral"]),
        ("gradient", "Gradient", PALETTE["red_strong"]),
        ("attention_x_gradient", "Attn x Grad", PALETTE["teal"]),
        ("attention_x_occlusion", "Attn x Occl", PALETTE["blue_main"]),
        ("random", "Random", PALETTE["violet"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    legend_handles = legend_labels = None
    for ax, (panel_label, exp_name) in zip(axes, panels):
        faith = load_faithfulness(exp_name)
        if faith is None:
            ax.text(0.5, 0.5, "faithfulness not run", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = np.array([1, 2, 3])
        for key, label, color in methods:
            vals = []
            for k in ["1", "2", "3"]:
                if key == "random":
                    vals.append(faith["random_occlusion"][k]["mean_risk_drop_true_positive"])
                else:
                    vals.append(faith["method_occlusion"][key][k]["top"]["mean_risk_drop_true_positive"])
            ax.plot(x, vals, marker="o", linewidth=2.0, color=color, label=label)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(panel_label, pad=8)
        ax.set_xlabel("Top-k masked sensors")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Mean risk drop on TP windows")
    axes[0].set_ylim(-0.04, 0.98)
    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=9)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_fteformer_support_faithfulness.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    rows = collect_rows()
    save_rows(rows)
    make_summary_figure(rows)
    make_faithfulness_figure()
    print(f"saved {OUT_DIR / 'support_ablation_metrics.csv'}")
    print(f"saved {OUT_DIR / 'fig_fteformer_support_ablation.png'}")
    print(f"saved {OUT_DIR / 'fig_fteformer_support_faithfulness.png'}")


if __name__ == "__main__":
    main()
