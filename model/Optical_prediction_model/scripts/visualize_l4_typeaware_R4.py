"""PPT-ready figures for the L4 type-aware ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments"
FIG_DIR = EXP_ROOT / "dl_interp_L4_typeaware_R4_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


MODEL_DIRS = {
    "L4": EXP_ROOT / "dl_interp_L4_crossattn_R4",
    "Type": EXP_ROOT / "dl_interp_L4_typeaware_R4_type",
    "Type+Div": EXP_ROOT / "dl_interp_L4_typeaware_R4_type_div",
    "Type+Prior": EXP_ROOT / "dl_interp_L4_typeaware_R4_type_prior",
    "Full": EXP_ROOT / "dl_interp_L4_typeaware_R4_full",
}

COLORS = {
    "L4": "#EF5350",
    "Type": "#42A5F5",
    "Type+Div": "#7E57C2",
    "Type+Prior": "#26A69A",
    "Full": "#F59E0B",
}


def load_summary(path: Path) -> dict | None:
    try:
        with open(path / "results" / "summary.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_npz(path: Path, name: str):
    p = path / "results" / name
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def sensor_label(sensor: str) -> str:
    mapping = {
        "temperature": "Temp",
        "current": "Current",
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


def all_fault_hit(summary: dict) -> float:
    event = summary.get("event_metrics", {})
    return float(event.get("all_fault_module_hit_rate", 0.0))


def concept_similarity(model_dir: Path) -> float:
    attn_data = load_npz(model_dir, "cross_attn_maps.npz")
    sensor_data = load_npz(model_dir, "sensor_weights.npz")
    if attn_data is None:
        return float("nan")
    c2s = np.asarray(attn_data["concept_to_sensor"], dtype=float)
    if sensor_data is not None and "y_pred" in sensor_data.files:
        mask = np.asarray(sensor_data["y_pred"]).astype(int) == 1
        if mask.any():
            c2s = c2s[mask]
    if len(c2s) == 0:
        return float("nan")
    norm = c2s / np.clip(np.linalg.norm(c2s, axis=-1, keepdims=True), 1e-8, None)
    sim = np.matmul(norm, np.swapaxes(norm, 1, 2))
    k = sim.shape[1]
    return float(sim[:, ~np.eye(k, dtype=bool)].mean())


def top_sensors(summary: dict, n: int = 3) -> str:
    ranking = summary.get("sensor_importance", {}).get("ranking", [])
    return ", ".join(sensor_label(item["sensor"]) for item in ranking[:n]) if ranking else "-"


def collect_rows():
    rows = []
    for name, path in MODEL_DIRS.items():
        summary = load_summary(path)
        if not summary:
            continue
        wm = summary.get("window_metrics", {})
        em = summary.get("event_metrics", {})
        sim = summary.get("concept_sensor_mean_offdiag_cosine")
        if sim is None:
            sim = concept_similarity(path)
        rows.append({
            "name": name,
            "summary": summary,
            "f1": float(wm.get("f1", 0.0)),
            "precision": float(wm.get("precision", 0.0)),
            "recall": float(wm.get("recall", 0.0)),
            "hit_eval": float(em.get("event_hit_rate_evaluable", 0.0)),
            "hit_all": all_fault_hit(summary),
            "concept_sim": float(sim),
            "top": top_sensors(summary),
        })
    return rows


def fig_ablation_table(rows):
    col_labels = ["F1", "Precision", "Recall", "Hit@Eval", "Hit@All", "Concept sim\n(lower better)", "Top-3 sensors"]
    cell_text = []
    row_labels = []
    for row in rows:
        row_labels.append(row["name"])
        cell_text.append([
            f"{row['f1']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['hit_eval']:.4f}",
            f"{row['hit_all']:.4f}",
            f"{row['concept_sim']:.4f}",
            row["top"],
        ])

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.axis("off")
    ax.set_title("L4 Type-aware Ablation", fontsize=18, fontweight="bold", pad=12)
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.11, 0.12, 0.11, 0.12, 0.12, 0.14, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#C7CDD1")
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", weight="bold")
        if c == -1 and r > 0:
            label = row_labels[r - 1]
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F7F7F7")
            if label == "Full":
                cell.set_facecolor("#FFF3D6")
        if r > 0 and row_labels[r - 1] == "Full":
            cell.set_facecolor("#FFF3D6")

    fig.tight_layout()
    out = FIG_DIR / "01_l4_typeaware_ablation_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_metric_tradeoff(rows):
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    f1 = [r["f1"] for r in rows]
    dissimilarity = [1.0 - r["concept_sim"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, vals, title, ylabel in [
        (axes[0], f1, "Fault prediction", "F1"),
        (axes[1], dissimilarity, "Concept separation", "1 - mean cosine similarity"),
    ]:
        bars = ax.bar(x, vals, color=[COLORS[n] for n in names], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.03,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle("L4 Type-aware Trade-off", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "02_l4_typeaware_tradeoff.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_full_heatmap():
    model_dir = MODEL_DIRS["Full"]
    summary = load_summary(model_dir)
    attn = load_npz(model_dir, "cross_attn_maps.npz")
    sensor_data = load_npz(model_dir, "sensor_weights.npz")
    if not summary or attn is None or sensor_data is None:
        return
    c2s = np.asarray(attn["concept_to_sensor"], dtype=float)
    y_pred = np.asarray(sensor_data["y_pred"]).astype(int)
    y_true = np.asarray(sensor_data["y_true"]).astype(int)
    mask = (y_pred == 1) & (y_true == 1)
    if not mask.any():
        mask = y_pred == 1
    matrix = c2s[mask].mean(axis=0)

    sensors = summary.get("sensors", [])
    concept_names = summary.get("model_cfg", {}).get("concept_names", [])
    display = summary.get("type_display_names", {})
    concept_labels = [display.get(name, name) for name in concept_names]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(sensors)))
    ax.set_xticklabels([sensor_label(s) for s in sensors], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(concept_labels)))
    ax.set_yticklabels(concept_labels)
    ax.set_xlabel("sensor")
    ax.set_ylabel("weak fault type concept")
    ax.set_title("L4-Full: Concept-to-Sensor Attention on True Positives", fontsize=15, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= np.quantile(matrix, 0.85):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="#1f2933", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("attention weight")
    fig.tight_layout()
    out = FIG_DIR / "03_l4_full_concept_sensor_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_full_case():
    model_dir = MODEL_DIRS["Full"]
    summary = load_summary(model_dir)
    attn = load_npz(model_dir, "cross_attn_maps.npz")
    sensor_data = load_npz(model_dir, "sensor_weights.npz")
    if not summary or attn is None or sensor_data is None:
        return
    c2s = np.asarray(attn["concept_to_sensor"], dtype=float)
    weights = np.asarray(sensor_data["weights"], dtype=float)
    scores = np.asarray(sensor_data["scores"], dtype=float)
    y_pred = np.asarray(sensor_data["y_pred"]).astype(int)
    y_true = np.asarray(sensor_data["y_true"]).astype(int)
    type_probs = np.asarray(sensor_data["type_probs"], dtype=float)
    candidates = np.where((y_pred == 1) & (y_true == 1))[0]
    if len(candidates) == 0:
        candidates = np.where(y_pred == 1)[0]
    if len(candidates) == 0:
        return

    sensors = summary.get("sensors", [])
    short = [sensor_label(s) for s in sensors]
    concept_names = summary.get("model_cfg", {}).get("concept_names", [])
    display = summary.get("type_display_names", {})
    concept_labels = [display.get(name, name) for name in concept_names]

    groups = {
        "thermal_anomaly": ["temperature"],
        "current_bias_anomaly": ["current"],
        "tx_power_anomaly": ["currentTXPower"] + [f"currentMultiTXPower{i}" for i in range(1, 5)],
        "rx_power_anomaly": ["currentRXPower"] + [f"currentMultiRXPower{i}" for i in range(1, 5)],
        "lane_imbalance": [f"currentMultiTXPower{i}" for i in range(1, 5)]
                          + [f"currentMultiRXPower{i}" for i in range(1, 5)],
    }
    group_indices = {
        name: [i for i, sensor in enumerate(sensors) if sensor in set(groups.get(name, []))]
        for name in concept_names
    }
    best_idx, best_score = int(candidates[0]), -1.0
    for row in candidates:
        top_type = int(np.argmax(type_probs[row]))
        type_name = concept_names[top_type]
        idxs = group_indices.get(type_name, [])
        group_mass = float(weights[row, idxs].sum()) if idxs else 0.0
        selection_score = float(scores[row] + type_probs[row, top_type] + group_mass)
        if group_mass >= 0.35 and selection_score > best_score:
            best_idx, best_score = int(row), selection_score
    if best_score < 0:
        best_idx = int(candidates[np.argmax(scores[candidates])])
    idx = best_idx
    order = np.argsort(weights[idx])[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.4]})
    ax = axes[0]
    bars = ax.barh(np.arange(len(order)), weights[idx][order], color=COLORS["Full"], edgecolor="white")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([short[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("sensor attribution")
    ax.set_title("Sensor evidence", fontsize=12, fontweight="bold")
    for bar, val in zip(bars[:5], weights[idx][order][:5]):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    ax = axes[1]
    probs = type_probs[idx]
    y = np.arange(len(concept_labels))
    ax.barh(y, probs, color="#F59E0B", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(concept_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("type probability")
    ax.set_title("Predicted fault type", fontsize=12, fontweight="bold")
    for row, val in enumerate(probs):
        ax.text(min(val + 0.03, 0.98), row, f"{val:.2f}", va="center", fontsize=8)

    ax = axes[2]
    matrix = c2s[idx]
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(short)))
    ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(concept_labels)))
    ax.set_yticklabels(concept_labels)
    ax.set_title("Concept-to-sensor attention", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.78)

    fig.suptitle(f"L4-Full Case Explanation | saved index={idx} | score={scores[idx]:.3f}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "04_l4_full_case_explanation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    rows = collect_rows()
    if not rows:
        print("[skip] no L4 type-aware results found")
        return
    fig_ablation_table(rows)
    fig_metric_tradeoff(rows)
    fig_full_heatmap()
    fig_full_case()


if __name__ == "__main__":
    main()
