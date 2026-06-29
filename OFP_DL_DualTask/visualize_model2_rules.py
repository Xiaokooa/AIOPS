"""Visualize OFP model2 rule-model triggers from raw module CSV files.

The original OFP model2 RuleModel operates on expanding statistics such as
minimum, range, standard deviation, and kurtosis. This script reproduces those
transparent rule traces directly from raw training CSV files and exports
publication-ready figures plus CSV summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TRAINING_DIR = PROJECT_ROOT / "dataset" / "training"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "OFP_DL_DualTask" / "rule_visualization"


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "orange": "#C06C2E",
    "neutral": "#CFCECE",
}


RAW_LABELS = {
    "temperature": "Temperature",
    "current": "Current",
    "currentTXPower": "TxP0",
    "currentRXPower": "RxP0",
    "currentMultiRXPower1": "RxP1",
    "currentMultiRXPower2": "RxP2",
    "currentMultiRXPower3": "RxP3",
    "currentMultiRXPower4": "RxP4",
    "currentMultiTXPower1": "TxP1",
    "currentMultiTXPower2": "TxP2",
    "currentMultiTXPower3": "TxP3",
    "currentMultiTXPower4": "TxP4",
}


@dataclass(frozen=True)
class RuleTerm:
    sensor: str
    stat: str
    op: str
    threshold: float


@dataclass(frozen=True)
class RuleSpec:
    name: str
    terms: tuple[RuleTerm, ...]
    description: str


def _term(sensor: str, stat: str, op: str, threshold: float) -> RuleTerm:
    return RuleTerm(sensor=sensor, stat=stat, op=op, threshold=float(threshold))


def _one(name: str, sensor: str, stat: str, op: str, threshold: float) -> RuleSpec:
    label = RAW_LABELS.get(sensor, sensor)
    return RuleSpec(
        name=name,
        terms=(_term(sensor, stat, op, threshold),),
        description=f"{label} expanding {stat} {op} {threshold:g}",
    )


def _both(name: str, terms: tuple[RuleTerm, ...]) -> RuleSpec:
    desc = " and ".join(
        f"{RAW_LABELS.get(t.sensor, t.sensor)} expanding {t.stat} {t.op} {t.threshold:g}"
        for t in terms
    )
    return RuleSpec(name=name, terms=terms, description=desc)


RULE_SPECS: tuple[RuleSpec, ...] = (
    _one("RuTempMin", "temperature", "min", "<", 0),
    _one("RuCurrMin", "current", "min", "<", 5000),
    _one("RuTempDiff", "temperature", "diff", ">", 100),
    _one("RuCurrDiff", "current", "diff", ">", 6000),
    _one("RuTempStd", "temperature", "std", ">", 10),
    _one("RuCurrStd", "current", "std", ">", 1500),
    _one("RuTempKurt", "temperature", "kurt", ">", 500),
    _one("RuTxP0Min", "currentTXPower", "min", "<", 0),
    _one("RuRxP0Min", "currentRXPower", "min", "<", 0),
    _one("RuTxP0Diff", "currentTXPower", "diff", ">", 1000),
    _one("RuRxP0Diff", "currentRXPower", "diff", ">", 1000),
    _one("RuTxP0Std", "currentTXPower", "std", ">", 500),
    _one("RuRxP0Std", "currentRXPower", "std", ">", 500),
    _both(
        "RuTxRxP0Kurt",
        (
            _term("currentTXPower", "kurt", ">", 500),
            _term("currentRXPower", "kurt", ">", 500),
        ),
    ),
    _one("RuRxP1Min", "currentMultiRXPower1", "min", "<", 0),
    _one("RuRxP2Min", "currentMultiRXPower2", "min", "<", 0),
    _one("RuRxP1Diff", "currentMultiRXPower1", "diff", ">", 1000),
    _one("RuRxP2Diff", "currentMultiRXPower2", "diff", ">", 1000),
    _one("RuRxP1Std", "currentMultiRXPower1", "std", ">", 200),
    _one("RuRxP2Std", "currentMultiRXPower2", "std", ">", 200),
    _one("RuRxP3Min", "currentMultiRXPower3", "min", "<", 0),
    _one("RuRxP4Min", "currentMultiRXPower4", "min", "<", 0),
    _one("RuRxP3Diff", "currentMultiRXPower3", "diff", ">", 1000),
    _one("RuRxP4Diff", "currentMultiRXPower4", "diff", ">", 1000),
    _one("RuRxP3Std", "currentMultiRXPower3", "std", ">", 200),
    _one("RuRxP4Std", "currentMultiRXPower4", "std", ">", 200),
    _one("RuTxP1Min", "currentMultiTXPower1", "min", "<", 0),
    _one("RuTxP2Min", "currentMultiTXPower2", "min", "<", 0),
    _one("RuTxP1Diff", "currentMultiTXPower1", "diff", ">", 1000),
    _one("RuTxP2Diff", "currentMultiTXPower2", "diff", ">", 1000),
    _one("RuTxP1Std", "currentMultiTXPower1", "std", ">", 100),
    _one("RuTxP2Std", "currentMultiTXPower2", "std", ">", 100),
    _one("RuTxP3Min", "currentMultiTXPower3", "min", "<", 0),
    _one("RuTxP4Min", "currentMultiTXPower4", "min", "<", 0),
    _one("RuTxP3Diff", "currentMultiTXPower3", "diff", ">", 1000),
    _one("RuTxP4Diff", "currentMultiTXPower4", "diff", ">", 1000),
    _one("RuTxP3Std", "currentMultiTXPower3", "std", ">", 100),
    _one("RuTxP4Std", "currentMultiTXPower4", "std", ">", 100),
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.25,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_input_file(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = TRAINING_DIR / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"cannot find module CSV: {value}")


def load_raw_module(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} does not contain timestamp")
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in RAW_LABELS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "anomaly" in df.columns:
        df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0.0)
    else:
        df["anomaly"] = 0.0
    return df


def expanding_stat(series: pd.Series, stat: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    exp = values.expanding(min_periods=1)
    if stat == "min":
        return exp.min()
    if stat == "max":
        return exp.max()
    if stat == "diff":
        return exp.max() - exp.min()
    if stat == "std":
        return exp.std().fillna(0.0)
    if stat == "skew":
        return exp.skew().fillna(-999.0)
    if stat == "kurt":
        return exp.kurt().fillna(-999.0)
    raise ValueError(f"unknown stat={stat!r}")


def compare(values: pd.Series, op: str, threshold: float) -> pd.Series:
    if op == "<":
        return values < threshold
    if op == ">":
        return values > threshold
    raise ValueError(f"unknown op={op!r}")


def compute_rule_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    feature_cache: dict[str, pd.Series] = {}
    hits = pd.DataFrame({"timestamp": df["timestamp"].astype("int64")})
    for spec in RULE_SPECS:
        term_hits = []
        for term in spec.terms:
            if term.sensor not in df.columns:
                term_hits.append(pd.Series(False, index=df.index))
                continue
            key = f"{term.sensor}:{term.stat}"
            if key not in feature_cache:
                feature_cache[key] = expanding_stat(df[term.sensor], term.stat)
            term_hits.append(compare(feature_cache[key], term.op, term.threshold))
        rule_hit = term_hits[0].copy()
        for hit in term_hits[1:]:
            rule_hit = rule_hit & hit
        hits[spec.name] = rule_hit.astype(int)
    rule_cols = [spec.name for spec in RULE_SPECS]
    hits["RuleMatchCnt"] = hits[rule_cols].sum(axis=1)
    hits["predict"] = (hits["RuleMatchCnt"] > 0).astype(int)

    first_anomaly_ts = first_positive_timestamp(df, "anomaly")
    summary_rows = []
    for spec in RULE_SPECS:
        active = hits[spec.name] > 0
        first_hit_ts = int(hits.loc[active, "timestamp"].iloc[0]) if active.any() else -1
        lead_hours = (
            float((first_anomaly_ts - first_hit_ts) / 3600.0)
            if first_anomaly_ts is not None and first_hit_ts >= 0
            else np.nan
        )
        summary_rows.append(
            {
                "rule": spec.name,
                "description": spec.description,
                "hit_count": int(active.sum()),
                "first_hit_ts": first_hit_ts,
                "lead_hours_to_first_anomaly": lead_hours,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["hit_count", "rule"], ascending=[False, True])
    return hits, summary, feature_cache


def first_positive_timestamp(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns or "timestamp" not in df.columns:
        return None
    mask = pd.to_numeric(df[column], errors="coerce").fillna(0.0) > 0.0
    if not mask.any():
        return None
    return int(pd.to_numeric(df.loc[mask, "timestamp"], errors="coerce").dropna().iloc[0])


def hours_from_start(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    return (ts - ts[0]) / 3600.0


def save_figure(fig: plt.Figure, out_base: Path, formats: list[str]) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_rule_summary(summary: pd.DataFrame, out_dir: Path, stem: str, formats: list[str], top_k: int) -> list[Path]:
    top = summary.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, max(3.6, 0.34 * len(top))), constrained_layout=True)
    ax.barh(top["rule"], top["hit_count"], color=PALETTE["blue_main"], edgecolor="black", linewidth=0.7)
    ax.set_xlabel("Triggered time points")
    ax.set_ylabel("Rule")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    return save_figure(fig, out_dir / f"{stem}_rule_hit_summary", formats)


def plot_rule_heatmap(
    hits: pd.DataFrame,
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    out_dir: Path,
    stem: str,
    formats: list[str],
    top_k: int,
) -> list[Path]:
    selected = summary.head(top_k)["rule"].tolist()
    matrix = hits[selected].to_numpy(dtype=float).T if selected else np.zeros((0, len(hits)))
    x = hours_from_start(raw)
    first_anomaly_ts = first_positive_timestamp(raw, "anomaly")
    fig, ax = plt.subplots(figsize=(10.5, max(3.8, 0.28 * len(selected))), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected)
    ticks = np.linspace(0, max(len(x) - 1, 0), min(6, len(x)), dtype=int) if len(x) else []
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x[i]:.1f}" for i in ticks])
    ax.set_xlabel("Hours from module start")
    ax.set_ylabel("Rule")
    if first_anomaly_ts is not None:
        anomaly_idx = int(np.searchsorted(raw["timestamp"].to_numpy(dtype=np.int64), first_anomaly_ts))
        ax.axvline(anomaly_idx, color=PALETTE["red_strong"], linewidth=1.6, linestyle="--", label="First anomaly")
        ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Rule hit")
    return save_figure(fig, out_dir / f"{stem}_rule_hit_heatmap", formats)


def plot_feature_panels(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    feature_cache: dict[str, pd.Series],
    out_dir: Path,
    stem: str,
    formats: list[str],
    top_k: int,
) -> list[Path]:
    spec_map = {spec.name: spec for spec in RULE_SPECS}
    selected = [rule for rule in summary.head(top_k)["rule"].tolist() if rule in spec_map]
    selected = selected[: min(6, len(selected))]
    if not selected:
        return []
    x = hours_from_start(raw)
    first_anomaly_ts = first_positive_timestamp(raw, "anomaly")
    first_anomaly_hour = None
    if first_anomaly_ts is not None:
        first_anomaly_hour = (first_anomaly_ts - int(raw["timestamp"].iloc[0])) / 3600.0
    fig, axes = plt.subplots(len(selected), 1, figsize=(10.5, max(3.0, 1.95 * len(selected))), sharex=True)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for ax, rule in zip(axes, selected):
        spec = spec_map[rule]
        for idx, term in enumerate(spec.terms):
            key = f"{term.sensor}:{term.stat}"
            if key not in feature_cache:
                continue
            label = f"{RAW_LABELS.get(term.sensor, term.sensor)} {term.stat}"
            color = PALETTE["blue_main"] if idx == 0 else PALETTE["teal"]
            ax.plot(x, feature_cache[key].to_numpy(dtype=float), color=color, linewidth=1.8, label=label)
            ax.axhline(term.threshold, color=PALETTE["red_strong"], linewidth=1.3, linestyle="--")
        if first_anomaly_hour is not None:
            ax.axvline(first_anomaly_hour, color=PALETTE["red_strong"], linewidth=1.1, linestyle=":")
        ax.set_ylabel(rule)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        ax.legend(loc="upper left", fontsize=8.5)
    axes[-1].set_xlabel("Hours from module start")
    return save_figure(fig, out_dir / f"{stem}_rule_feature_panels", formats)


def plot_raw_signals(raw: pd.DataFrame, out_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    sensors = ["temperature", "current", "currentTXPower", "currentRXPower"]
    sensors = [sensor for sensor in sensors if sensor in raw.columns]
    if not sensors:
        return []
    x = hours_from_start(raw)
    first_anomaly_ts = first_positive_timestamp(raw, "anomaly")
    first_anomaly_hour = None
    if first_anomaly_ts is not None:
        first_anomaly_hour = (first_anomaly_ts - int(raw["timestamp"].iloc[0])) / 3600.0
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    colors = [PALETTE["blue_main"], PALETTE["orange"], PALETTE["teal"], PALETTE["violet"]]
    for sensor, color in zip(sensors, colors):
        values = pd.to_numeric(raw[sensor], errors="coerce").astype(float)
        std = float(values.std(skipna=True))
        if not np.isfinite(std) or std <= 1e-8:
            normed = values - values.mean(skipna=True)
        else:
            normed = (values - values.mean(skipna=True)) / std
        ax.plot(x, normed, linewidth=1.7, color=color, label=RAW_LABELS.get(sensor, sensor))
    if first_anomaly_hour is not None:
        ax.axvline(first_anomaly_hour, color=PALETTE["red_strong"], linewidth=1.5, linestyle="--", label="First anomaly")
    ax.set_xlabel("Hours from module start")
    ax.set_ylabel("Z-normalized raw value")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.legend(loc="upper left", ncol=2)
    return save_figure(fig, out_dir / f"{stem}_raw_signal_overview", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OFP model2 expert-rule triggers.")
    parser.add_argument("--file", required=True, help="Module CSV path or file name under dataset/training.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k_rules", type=int, default=12)
    parser.add_argument("--formats", default="png,pdf,svg", help="Comma-separated output formats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    module_path = resolve_input_file(args.file)
    raw = load_raw_module(module_path)
    hits, summary, feature_cache = compute_rule_outputs(raw)
    out_dir = Path(args.output_dir) / module_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [part.strip().lower() for part in str(args.formats).split(",") if part.strip()]
    stem = module_path.stem

    hits.to_csv(out_dir / f"{stem}_rule_hits.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_rule_summary.csv", index=False)
    rule_specs = pd.DataFrame(
        {
            "rule": spec.name,
            "description": spec.description,
            "terms": "; ".join(
                f"{term.sensor}:{term.stat}{term.op}{term.threshold:g}" for term in spec.terms
            ),
        }
        for spec in RULE_SPECS
    )
    rule_specs.to_csv(out_dir / "model2_rule_specs.csv", index=False)
    metadata = {
        "module_file": str(module_path),
        "rows": int(len(raw)),
        "first_anomaly_ts": first_positive_timestamp(raw, "anomaly"),
        "rule_count": int(len(RULE_SPECS)),
        "predict_points": int(hits["predict"].sum()),
        "output_dir": str(out_dir),
    }
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    saved = []
    saved += plot_rule_summary(summary, out_dir, stem, formats, max(1, int(args.top_k_rules)))
    saved += plot_rule_heatmap(hits, summary, raw, out_dir, stem, formats, max(1, int(args.top_k_rules)))
    saved += plot_feature_panels(raw, summary, feature_cache, out_dir, stem, formats, max(1, int(args.top_k_rules)))
    saved += plot_raw_signals(raw, out_dir, stem, formats)
    print(json.dumps({"output_dir": str(out_dir), "saved": [str(path) for path in saved]}, indent=2))


if __name__ == "__main__":
    main()
