from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PAPER_METRICS = [
    "precision",
    "recall",
    "f1_score",
    "false_alarm_rate",
    "false_alarms_per_1000_normal",
    "balanced_accuracy",
    "warnable_recall",
    "first_warning_upper_bound",
    "median_lead_hour",
    "avg_lead_hour",
    "final_score",
]
COST_METRICS = [
    "scoring_ms_per_window",
    "scoring_rows_per_second",
    "encoder_model_mb",
    "decision_model_mb",
    "train_rows",
]

PALETTE = {
    "blue": "#0F4D92",
    "blue_light": "#3775BA",
    "green": "#5A9E6F",
    "red": "#B64342",
    "gray": "#9A9A9A",
    "dark": "#272727",
}

CANONICAL_EXPERIMENT_ID = "main_htsf"
CANONICAL_TABLE_ROWS = {
    "fusion": ("fusion_htsf", "HTSF", "Complete HTSF; reused from the main result."),
    "sampling": ("sampling_module_hybrid", "Hybrid sampling", "Complete HTSF with the proposed hybrid sampling."),
    "feature_groups": ("feature_full", "All expert-statistical", "Complete expert-statistical feature set."),
    "feature_selection": ("selection_all", "All features", "Complete feature set without learned selection."),
    "replacement": ("backbone_patchtst", "HTSF", "Default PatchTST encoder and XGBoost decision layer."),
}


def load_manifest(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / "suite_manifest.json"
    if not path.exists():
        return {}, {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = {str(item["experiment_id"]): item for item in manifest.get("experiments", [])}
    return manifest, entries


def load_fold_metrics(root: Path, entries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_id, entry in entries.items():
        path = Path(entry.get("experiment_root", root / "experiments" / experiment_id)) / "fold_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["experiment_id"] = experiment_id
        frame["suite"] = str(entry.get("suite", ""))
        frame["paper_method"] = str(entry.get("method_label", frame.get("method_label", "HTSF")))
        frame["claim"] = str(entry.get("claim", ""))
        frames.append(frame)
    if not frames:
        for path in sorted((root / "experiments").glob("*/fold_metrics.csv")):
            frame = pd.read_csv(path)
            frame["experiment_id"] = path.parent.name
            frame["suite"] = "unknown"
            frame["paper_method"] = frame.get("method_label", "HTSF")
            frame["claim"] = ""
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates().reset_index(drop=True)


def inject_canonical_table_rows(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    """Reuse one trained HTSF result as the full-model row in every ablation table."""
    if frame.empty or "experiment_id" not in frame.columns:
        return frame
    canonical = frame.loc[frame["experiment_id"].astype(str) == CANONICAL_EXPERIMENT_ID].copy()
    if canonical.empty:
        return frame

    alias_ids = {experiment_id for experiment_id, _, _ in CANONICAL_TABLE_ROWS.values()}
    base = frame.loc[~frame["experiment_id"].astype(str).isin(alias_ids)].copy()
    if "source_experiment_id" not in base.columns:
        base["source_experiment_id"] = base["experiment_id"].astype(str)

    requested = {str(value) for value in manifest.get("suites", [])}
    if "all" in requested:
        requested = set(CANONICAL_TABLE_ROWS) | {"main"}
    aliases: list[pd.DataFrame] = []
    for suite, (experiment_id, method_label, claim) in CANONICAL_TABLE_ROWS.items():
        if suite not in requested:
            continue
        alias = canonical.copy()
        alias["experiment_id"] = experiment_id
        alias["suite"] = suite
        alias["paper_method"] = method_label
        alias["claim"] = claim
        alias["source_experiment_id"] = CANONICAL_EXPERIMENT_ID
        aliases.append(alias)
    if not aliases:
        return base.reset_index(drop=True)
    return pd.concat([base, *aliases], ignore_index=True, sort=False)


def canonical_result_links(manifest: dict[str, Any]) -> pd.DataFrame:
    requested = {str(value) for value in manifest.get("suites", [])}
    if "all" in requested:
        requested = set(CANONICAL_TABLE_ROWS) | {"main"}
    rows = [
        {
            "suite": suite,
            "table_experiment_id": experiment_id,
            "source_experiment_id": CANONICAL_EXPERIMENT_ID,
            "paper_method": method_label,
        }
        for suite, (experiment_id, method_label, _claim) in CANONICAL_TABLE_ROWS.items()
        if suite in requested
    ]
    return pd.DataFrame(rows)


def ordered_experiment_ids(entries: dict[str, dict[str, Any]]) -> list[str]:
    return list(entries)


def summarize_metrics(frame: pd.DataFrame, entries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if frame.empty:
        return frame
    group_cols = [
        col
        for col in [
            "experiment_id",
            "suite",
            "paper_method",
            "temporal_encoder",
            "decision_layer",
            "fusion_mode",
            "sampling_mode",
            "sample_selection",
            "sample_topk_fraction",
            "stat_feature_groups",
            "exclude_stat_feature_groups",
            "stat_selector",
            "stat_select_k",
        ]
        if col in frame.columns
    ]
    metrics = [metric for metric in [*PAPER_METRICS, *COST_METRICS] if metric in frame.columns]
    extra = [name for name in ["stat_feature_count", "selected_feature_count", "threshold"] if name in frame.columns]
    summary = frame.groupby(group_cols, dropna=False)[metrics + extra].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index(inplace=True)
    order = {name: index for index, name in enumerate(ordered_experiment_ids(entries))}
    summary["_order"] = summary["experiment_id"].map(order).fillna(len(order))
    summary.sort_values(["_order", "experiment_id"], inplace=True)
    summary.drop(columns=["_order"], inplace=True)
    return summary


def paper_value(mean: Any, std: Any, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def latex_escape(value: Any) -> str:
    text = str(value)
    for source, target in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ]:
        text = text.replace(source, target)
    return text


def latex_value(mean: Any, std: Any, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"${float(mean):.{digits}f}$"
    return f"${float(mean):.{digits}f}\\pm{float(std):.{digits}f}$"


def write_suite_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    for suite, block in summary.groupby("suite", dropna=False, sort=False):
        safe_suite = str(suite).replace(" ", "_")
        block.to_csv(out_dir / f"table_{safe_suite}_raw.csv", index=False)
        paper = pd.DataFrame()
        for name in ["experiment_id", "paper_method", "temporal_encoder", "decision_layer"]:
            if name in block.columns:
                paper[name] = block[name]
        for metric in PAPER_METRICS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in block.columns:
                paper[metric] = [
                    paper_value(mean, std)
                    for mean, std in zip(block[mean_col], block.get(std_col, pd.Series(np.nan, index=block.index)))
                ]
        for metric in COST_METRICS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in block.columns:
                paper[metric] = [
                    paper_value(mean, std)
                    for mean, std in zip(block[mean_col], block.get(std_col, pd.Series(np.nan, index=block.index)))
                ]
        paper.to_csv(out_dir / f"table_{safe_suite}_paper.csv", index=False)
        latex_metrics = [
            metric
            for metric in ["precision", "recall", "f1_score", "false_alarm_rate", "median_lead_hour"]
            if f"{metric}_mean" in block.columns
        ]
        latex_rows: list[str] = []
        for _, row in block.iterrows():
            cells = [
                latex_escape(row.get("paper_method", "HTSF")),
                latex_escape(row.get("temporal_encoder", "")),
                latex_escape(row.get("decision_layer", "")),
            ]
            for metric in latex_metrics:
                cells.append(latex_value(row.get(f"{metric}_mean"), row.get(f"{metric}_std")))
            latex_rows.append(" & ".join(cells) + r" \\")
        (out_dir / f"table_{safe_suite}_rows.tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def save_figure(fig: Any, base_path: Path, formats: Iterable[str]) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    for fmt in formats:
        fig.savefig(base_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight", pad_inches=0.03)


def short_label(value: str, limit: int = 24) -> str:
    replacements = {
        "fusion_": "",
        "sampling_": "",
        "feature_": "",
        "selection_": "",
        "without_": "w/o ",
        "module_": "",
        "expert_stat": "Exp.-stat.",
        "temporal_only": "Temporal",
        "latent_concat": "Concat.",
        "cross_attention": "Cross-attn.",
        "htsf": "HTSF",
    }
    text = str(value)
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("_", " ")
    return text if len(text) <= limit else text[: limit - 1] + "."


def plot_metric_bars(summary: pd.DataFrame, suite: str, out_dir: Path, formats: list[str]) -> None:
    block = summary.loc[summary["suite"].astype(str) == suite].copy()
    metrics = [metric for metric in ["precision", "recall", "f1_score"] if f"{metric}_mean" in block.columns]
    if block.empty or not metrics:
        return
    plt = configure_matplotlib()
    width = max(3.35, 0.55 * len(block) + 1.2)
    fig, ax = plt.subplots(figsize=(width, 2.35))
    x = np.arange(len(block), dtype=float)
    bar_width = 0.22
    colors = [PALETTE["gray"], PALETTE["green"], PALETTE["blue"]]
    hatches = ["//", "..", ""]
    for index, metric in enumerate(metrics):
        means = block[f"{metric}_mean"].to_numpy(dtype=float)
        stds = block.get(f"{metric}_std", pd.Series(np.zeros(len(block)))).fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x + (index - (len(metrics) - 1) / 2.0) * bar_width,
            means,
            width=bar_width,
            yerr=stds,
            capsize=2,
            label={"precision": "Precision", "recall": "Recall", "f1_score": "F1"}[metric],
            color=colors[index],
            edgecolor="black",
            linewidth=0.6,
            hatch=hatches[index],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(value) for value in block["experiment_id"]], rotation=25, ha="right")
    ax.set_ylabel("Trace-level score")
    ax.set_ylim(0.0, max(1.0, float(block[[f"{m}_mean" for m in metrics]].max().max()) * 1.12))
    ax.legend(ncol=min(3, len(metrics)), loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save_figure(fig, out_dir / f"fig_{suite}_metrics", formats)
    plt.close(fig)


def load_lead_time(root: Path, entries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_id, entry in entries.items():
        path = Path(entry.get("experiment_root", root / "experiments" / experiment_id)) / "lead_time_sweep.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["experiment_id"] = experiment_id
        frame["suite"] = str(entry.get("suite", ""))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def plot_lead_time(frame: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if frame.empty or "min_hit_lead_hours" not in frame.columns or "test_recall" not in frame.columns:
        return
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    preferred = frame.loc[frame["experiment_id"].isin(["main_htsf", "lead_time_htsf"])]
    block = preferred if not preferred.empty else frame
    grouped = block.groupby("min_hit_lead_hours", dropna=False)["test_recall"].agg(["mean", "std"]).reset_index()
    x = grouped["min_hit_lead_hours"].to_numpy(dtype=float)
    y = grouped["mean"].to_numpy(dtype=float)
    std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(x, y, color=PALETTE["blue"], marker="o", markersize=3.5, linewidth=1.6, label="HTSF")
    ax.fill_between(x, np.maximum(y - std, 0.0), np.minimum(y + std, 1.0), color=PALETTE["blue_light"], alpha=0.18)
    ax.set_xlabel("Minimum lead time (h)")
    ax.set_ylabel("First-warning recall")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    save_figure(fig, out_dir / "fig_lead_time_sensitivity", formats)
    plt.close(fig)


def plot_feature_selection(summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    block = summary.loc[summary["suite"].astype(str) == "feature_selection"].copy()
    if block.empty or "f1_score_mean" not in block.columns:
        return
    x_col = "stat_feature_count_mean" if "stat_feature_count_mean" in block.columns else "stat_select_k"
    if x_col not in block.columns:
        return
    block[x_col] = pd.to_numeric(block[x_col], errors="coerce")
    block = block.dropna(subset=[x_col]).sort_values(x_col)
    if block.empty:
        return
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    x = block[x_col].to_numpy(dtype=float)
    y = block["f1_score_mean"].to_numpy(dtype=float)
    std = block.get("f1_score_std", pd.Series(np.zeros(len(block)))).fillna(0.0).to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=std, color=PALETTE["blue"], marker="o", markersize=4, linewidth=1.5, capsize=2)
    for x_value, y_value, label in zip(x, y, block["paper_method"]):
        ax.annotate(str(label), (x_value, y_value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6)
    ax.set_xlabel("Expert-statistical features")
    ax.set_ylabel("Trace-level F1")
    ax.set_ylim(bottom=0.0)
    save_figure(fig, out_dir / "fig_feature_selection", formats)
    plt.close(fig)


def aggregate_feature_importance(root: Path, entries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    preferred = ["main_htsf", "feature_full"]
    for experiment_id in [*preferred, *entries.keys()]:
        if experiment_id not in entries:
            continue
        experiment_root = Path(entries[experiment_id].get("experiment_root", root / "experiments" / experiment_id))
        sources = [
            ("stat_feature_importance.csv", "importance"),
            ("stat_feature_attribution.csv", "gradient_x_input"),
        ]
        for file_name, value_col in sources:
            for path in experiment_root.glob(f"**/explainability/{file_name}"):
                frame = pd.read_csv(path)
                if {"feature", value_col} <= set(frame.columns):
                    frame["experiment_id"] = experiment_id
                    frame["importance"] = pd.to_numeric(frame[value_col], errors="coerce")
                    frames.append(frame[["experiment_id", "feature", "importance"]].dropna())
            if frames:
                break
        if frames:
            break
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("feature", as_index=False)["importance"].agg(["mean", "std", "count"]).reset_index()


def abbreviate_feature(name: str) -> str:
    return (
        str(name)
        .replace("TemporalSummary_", "TS-")
        .replace("RuleLike", "Rule")
        .replace("FeCo", "Corr-")
        .replace("Fe", "")
        .replace("Ru", "Rule-")
    )


def plot_feature_importance(frame: pd.DataFrame, out_dir: Path, formats: list[str], top_k: int) -> None:
    if frame.empty:
        return
    top = frame.sort_values("mean", ascending=False).head(int(top_k)).sort_values("mean")
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, max(2.35, 0.16 * len(top) + 0.7)))
    ax.barh(
        np.arange(len(top)),
        top["mean"].to_numpy(dtype=float),
        xerr=top["std"].fillna(0.0).to_numpy(dtype=float),
        color=PALETTE["blue_light"],
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
    )
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([abbreviate_feature(value) for value in top["feature"]])
    ax.set_xlabel("Feature importance")
    save_figure(fig, out_dir / "fig_feature_importance", formats)
    plt.close(fig)


def aggregate_attention(root: Path, entries: dict[str, dict[str, Any]]) -> np.ndarray | None:
    matrices: list[np.ndarray] = []
    for experiment_id in ["main_htsf", "fusion_htsf"]:
        entry = entries.get(experiment_id)
        if not entry:
            continue
        experiment_root = Path(entry.get("experiment_root", root / "experiments" / experiment_id))
        for path in experiment_root.glob("**/explainability/cross_attention_matrix.csv"):
            frame = pd.read_csv(path, index_col=0)
            if frame.shape == (2, 2):
                matrices.append(frame.to_numpy(dtype=float))
    return np.mean(matrices, axis=0) if matrices else None


def plot_attention(matrix: np.ndarray | None, out_dir: Path, formats: list[str]) -> None:
    if matrix is None:
        return
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(2.4, 2.05))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(1.0, float(np.nanmax(matrix))))
    labels = ["Temporal", "Expert-stat."]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Key view")
    ax.set_ylabel("Query view")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_dir / "fig_cross_attention", formats)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HTSF experiments and generate paper-ready tables and figures.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = root / "paper_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, entries = load_manifest(root)
    folds = load_fold_metrics(root, entries)
    if folds.empty:
        print(f"[summary] no fold_metrics.csv found under {root}")
        return
    folds = inject_canonical_table_rows(folds, manifest)
    folds.to_csv(out_dir / "all_fold_metrics.csv", index=False)
    links = canonical_result_links(manifest)
    if not links.empty:
        links.to_csv(out_dir / "canonical_result_links.csv", index=False)
    summary = summarize_metrics(folds, entries)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    write_suite_tables(summary, out_dir)

    formats = [value.strip() for value in str(args.formats).split(",") if value.strip()]
    for suite in ["main", "fusion", "sampling", "feature_groups", "feature_selection", "replacement"]:
        plot_metric_bars(summary, suite, out_dir, formats)
    plot_feature_selection(summary, out_dir, formats)
    lead = load_lead_time(root, entries)
    if not lead.empty:
        lead.to_csv(out_dir / "lead_time_all_folds.csv", index=False)
    plot_lead_time(lead, out_dir, formats)
    importance = aggregate_feature_importance(root, entries)
    if not importance.empty:
        importance.sort_values("mean", ascending=False).to_csv(out_dir / "feature_importance_aggregated.csv", index=False)
    plot_feature_importance(importance, out_dir, formats, args.top_k)
    plot_attention(aggregate_attention(root, entries), out_dir, formats)
    print(f"[summary] rows={len(folds)} experiments={folds['experiment_id'].nunique()} out={out_dir}")


if __name__ == "__main__":
    main()
