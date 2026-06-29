"""Create a report figure summarizing the teacher OFP code pipeline."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "output" / "OFP_code_summary" / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box(ax, xy, text, color, width=2.20, height=0.68, fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=color,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.92,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def make_figure() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(15.5, 6.9))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.25, 6.55, "OFP Teacher Code Pipeline", fontsize=16, weight="bold", ha="left")
    ax.text(0.25, 6.18, "Two tracks: compact XGBoost baseline and fuller feature-engineering / model-fusion pipeline", fontsize=10, ha="left")

    # model1 track
    ax.text(0.35, 5.55, "model1", fontsize=12, weight="bold", color=PALETTE["blue_main"])
    box(ax, (0.35, 4.78), "CSV traces\n(test/train)", PALETTE["neutral"])
    box(ax, (3.10, 4.78), "Original DOM\nfeatures", PALETTE["blue_secondary"])
    box(ax, (5.85, 4.78), "XGBoost\nbinary classifier", PALETTE["green"])
    box(ax, (8.60, 4.78), "timestamp + predict\nper module file", PALETTE["teal"])
    box(ax, (11.35, 4.78), "Submission-style\nresult CSVs", PALETTE["neutral"])
    arrow(ax, (2.55, 5.12), (3.10, 5.12))
    arrow(ax, (5.30, 5.12), (5.85, 5.12))
    arrow(ax, (8.05, 5.12), (8.60, 5.12))
    arrow(ax, (10.80, 5.12), (11.35, 5.12))

    # model2 track
    ax.text(0.35, 3.75, "model2", fontsize=12, weight="bold", color=PALETTE["blue_main"])
    box(ax, (0.35, 2.95), "Raw CSV\npos / neg / test", PALETTE["neutral"])
    box(ax, (3.10, 2.95), "Clean + split\ncross-fold files", PALETTE["blue_secondary"])
    box(ax, (5.85, 2.84), "FeatureExtractor\nmin/max/diff/std\nskew/kurt/corr\nlane gaps", PALETTE["teal"], height=0.90, fontsize=8.3)
    box(ax, (8.60, 3.95), "RuleModel\nexpert thresholds", PALETTE["violet"])
    box(ax, (8.60, 2.95), "TreeModel\nRF / XGB / LGBM", PALETTE["green"])
    box(ax, (8.60, 1.95), "MutantModel\nunsupervised anomaly", PALETTE["red"])
    box(ax, (11.35, 2.95), "ModelRunner\ncross-fold train\npredict + evaluate", PALETTE["blue_main"], height=0.90, fontsize=8.4)
    box(ax, (14.05, 2.95), "ResultMerger\nensemble outputs", PALETTE["neutral"])
    arrow(ax, (2.55, 3.29), (3.10, 3.29))
    arrow(ax, (5.30, 3.29), (5.85, 3.29))
    arrow(ax, (8.05, 3.45), (8.60, 4.29))
    arrow(ax, (8.05, 3.29), (8.60, 3.29))
    arrow(ax, (8.05, 3.12), (8.60, 2.29))
    arrow(ax, (10.80, 4.29), (11.35, 3.68))
    arrow(ax, (10.80, 3.29), (11.35, 3.35))
    arrow(ax, (10.80, 2.29), (11.35, 3.02))
    arrow(ax, (13.55, 3.35), (14.05, 3.29))

    # relation to FTEformer
    ax.text(0.35, 1.02, "Relation to current FTEformer work", fontsize=12, weight="bold", color=PALETTE["blue_main"])
    relation = (
        "Teacher code provides feature-engineering, rules, event-level scoring, and result-fusion baselines. "
        "FTEformer replaces hand-crafted temporal aggregation with end-to-end window modeling and adds fault-mode / sensor-time attribution."
    )
    ax.text(0.35, 0.58, textwrap.fill(relation, width=150), fontsize=9.4, ha="left", va="center")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_ofp_teacher_pipeline.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print(f"saved {OUT_DIR / 'fig_ofp_teacher_pipeline.png'}")
