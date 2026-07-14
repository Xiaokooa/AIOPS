"""Compact, protocol-aligned module ablations for FORT.

The ablation table is deliberately rebuilt from pooled out-of-fold module
decisions.  This prevents fold averaging and keeps every comparison on the
same ``legacy_inclusive_v1`` protocol used by the main N2 result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .evaluation import PROTOCOL_NAME, metrics_from_module_decisions
from .static_xgb import STATIC_FEATURE_COUNT
from .threefold import (
    EXPECTED_FULL_FAULTY,
    EXPECTED_FULL_MODULES,
    EXPECTED_FULL_NORMAL,
    OUTER_FOLDS,
)


LEAD_HORIZONS_HOUR = (0, 6, 12, 24, 72)


@dataclass(frozen=True)
class ModuleAblationSpec:
    """One learned three-fold run in the compact ablation suite."""

    experiment_id: str
    display_name: str
    architecture: str
    positive_weight_mode: str
    temporal_encoder: bool
    rule_guidance: bool
    pkd: bool = True


LEARNED_SPECS = (
    ModuleAblationSpec(
        experiment_id="M2_temporal_only",
        display_name="Temporal only",
        architecture="causal_depthwise_tcn",
        positive_weight_mode="module_normalized_auto",
        temporal_encoder=True,
        rule_guidance=False,
    ),
    ModuleAblationSpec(
        experiment_id="M3_fort_no_class_weight",
        display_name="FORT core w/o class weight",
        architecture="rule_guided_residual_tcn",
        positive_weight_mode="none",
        temporal_encoder=True,
        rule_guidance=True,
    ),
    ModuleAblationSpec(
        experiment_id="M4_fort",
        display_name="FORT core",
        architecture="rule_guided_residual_tcn",
        positive_weight_mode="module_normalized_auto",
        temporal_encoder=True,
        rule_guidance=True,
    ),
)


_CORE_METRICS = (
    "final_score",
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "avg_lead_hour",
    "normal_module_false_alarm_rate",
)

_EFFECT_METRICS = (
    "final_score",
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "avg_lead_hour",
    "early_hit_rate_at_24h",
    "normal_module_false_alarm_rate",
)


def _legacy_row(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    protocol_column = (
        "evaluation_protocol" if "evaluation_protocol" in frame else "protocol"
    )
    if protocol_column not in frame:
        raise ValueError(f"protocol column missing from {path}")
    selected = frame.loc[frame[protocol_column].astype(str) == PROTOCOL_NAME]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {PROTOCOL_NAME} row in {path}, got {len(selected)}"
        )
    return selected.iloc[0].to_dict()


def _lead_metrics(detail: pd.DataFrame) -> dict[str, Any]:
    outcomes = detail["outcome"].astype(str)
    tp_mask = outcomes == "TP"
    leads = pd.to_numeric(detail.loc[tp_mask, "lead_hour"], errors="raise")
    faulty = int(tp_mask.sum() + (outcomes == "FN").sum())
    tp = int(tp_mask.sum())
    lead_sum = float(leads.sum()) if tp else 0.0
    result: dict[str, Any] = {
        "mean_lead_hour_among_hits": float(lead_sum / tp) if tp else 0.0,
        "strict_early_hit_count": int((leads > 0.0).sum()),
        "strict_early_recall": (
            float((leads > 0.0).sum() / faulty) if faulty else 0.0
        ),
    }
    for horizon in LEAD_HORIZONS_HOUR[1:]:
        count = int((leads >= float(horizon)).sum())
        result[f"early_hit_count_at_{horizon}h"] = count
        result[f"early_hit_rate_at_{horizon}h"] = (
            float(count / faulty) if faulty else 0.0
        )
    return result


def _all_metrics(detail: pd.DataFrame) -> dict[str, Any]:
    return {
        **metrics_from_module_decisions(detail, protocol=PROTOCOL_NAME),
        **_lead_metrics(detail),
    }


def _assert_metric_consistency(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    source: Path,
) -> None:
    for name in _CORE_METRICS:
        if name not in recorded:
            raise ValueError(f"{name} missing from {source}")
        actual = float(recorded[name])
        expected = float(recomputed[name])
        if not np.isfinite(actual) or not np.isclose(
            actual, expected, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"metric/detail mismatch in {source}: "
                f"{name}={actual!r}, recomputed={expected!r}"
            )


def _metadata(
    experiment_id: str,
    display_name: str,
    *,
    model_family: str,
    temporal_encoder: bool,
    rule_guidance: bool,
    pkd: bool,
    class_weight: bool,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "display_name": display_name,
        "changed_axis": "module_ablation",
        "model_family": model_family,
        "temporal_encoder": bool(temporal_encoder),
        "rule_guidance": bool(rule_guidance),
        "pkd": bool(pkd),
        "class_weight": bool(class_weight),
        "safe_rule_fallback": False,
    }


def _validate_pooled_coverage(
    detail: pd.DataFrame,
    index: pd.DataFrame,
    *,
    formal: bool,
    source: Path,
) -> None:
    if "file_name" not in detail or "outer_fold" not in detail:
        raise ValueError(f"file_name/outer_fold missing from {source}")
    names = detail["file_name"].astype(str)
    if names.duplicated().any():
        raise ValueError(f"duplicate module decisions in {source}")
    if not formal:
        return
    expected_names = set(index["file_name"].astype(str))
    if set(names) != expected_names:
        raise ValueError(f"formal pooled decisions do not cover the index: {source}")
    labels = pd.to_numeric(index["Label"], errors="raise") > 0
    if (
        len(detail) != EXPECTED_FULL_MODULES
        or int(labels.sum()) != EXPECTED_FULL_FAULTY
        or int((~labels).sum()) != EXPECTED_FULL_NORMAL
    ):
        raise ValueError(f"formal OFP module universe mismatch in {source}")


def _fold_rows_from_pooled(
    source_rows: pd.DataFrame,
    pooled: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    protocol_column = "protocol" if "protocol" in source_rows else None
    if protocol_column is not None:
        source_rows = source_rows.loc[
            source_rows[protocol_column].astype(str) == PROTOCOL_NAME
        ]
    rows: list[dict[str, Any]] = []
    for fold in sorted(pd.to_numeric(pooled["outer_fold"], errors="raise").unique()):
        fold = int(fold)
        detail = pooled.loc[
            pd.to_numeric(pooled["outer_fold"], errors="raise") == fold
        ].drop(columns="outer_fold")
        metrics = _all_metrics(detail)
        candidates = source_rows.loc[
            pd.to_numeric(source_rows["outer_fold"], errors="raise") == fold
        ]
        recorded = candidates.iloc[0].to_dict() if len(candidates) == 1 else {}
        if recorded:
            _assert_metric_consistency(
                recorded, metrics, source=Path(f"fold_metrics[{fold}]")
            )
        rows.append(
            {
                **recorded,
                **metadata,
                "outer_fold": fold,
                "protocol": PROTOCOL_NAME,
                **metrics,
            }
        )
    return rows


def _load_threefold_family(
    directory: Path,
    metadata: Mapping[str, Any],
    index: pd.DataFrame,
    *,
    formal: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    comparison_path = directory / "comparison.csv"
    fold_path = directory / "fold_metrics.csv"
    pooled_path = directory / "pooled" / "test_module_decisions_legacy_inclusive.csv"
    for required in (comparison_path, fold_path, pooled_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    recorded = _legacy_row(comparison_path)
    pooled = pd.read_csv(pooled_path)
    _validate_pooled_coverage(pooled, index, formal=formal, source=pooled_path)
    metrics = _all_metrics(pooled.drop(columns="outer_fold"))
    _assert_metric_consistency(recorded, metrics, source=comparison_path)
    summary = {
        **recorded,
        "source_experiment_id": recorded.get("experiment_id", ""),
        **metadata,
        "evaluation_protocol": PROTOCOL_NAME,
        **metrics,
    }
    fold_rows = _fold_rows_from_pooled(
        pd.read_csv(fold_path), pooled, metadata
    )
    return summary, fold_rows, pooled


def _load_rule_only(
    fort_directory: Path,
    index: pd.DataFrame,
    *,
    formal: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    metadata = _metadata(
        "R0_rule_only",
        "RuleModel only",
        model_family="deterministic_rule_model",
        temporal_encoder=False,
        rule_guidance=True,
        pkd=False,
        class_weight=False,
    )
    pooled_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        path = fort_directory / f"fold_{fold}" / "test_rule_only_module_decisions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        detail = pd.read_csv(path)
        if formal:
            expected = set(
                index.loc[
                    pd.to_numeric(index["folder_index"], errors="raise") == fold,
                    "file_name",
                ].astype(str)
            )
            if set(detail["file_name"].astype(str)) != expected:
                raise ValueError(f"RuleModel fold {fold} does not cover its test fold")
        metrics = _all_metrics(detail)
        fold_rows.append(
            {
                **metadata,
                "outer_fold": fold,
                "protocol": PROTOCOL_NAME,
                "decision_source": "rule_only",
                "threshold_policy": "deterministic_rule_model",
                **metrics,
            }
        )
        detail = detail.copy()
        detail.insert(0, "outer_fold", fold)
        pooled_parts.append(detail)
    pooled = pd.concat(pooled_parts, ignore_index=True)
    _validate_pooled_coverage(
        pooled,
        index,
        formal=formal,
        source=fort_directory / "fold_*/test_rule_only_module_decisions.csv",
    )
    metrics = _all_metrics(pooled.drop(columns="outer_fold"))
    summary = {
        **metadata,
        "variant": "rule_only",
        "evaluation_protocol": PROTOCOL_NAME,
        "outer_folds": "1,2,3",
        "training_runs": 0,
        "formal_full_oof": bool(formal),
        "decision_source": "rule_only",
        "threshold_policy": "deterministic_rule_model",
        "student_parameters": 0,
        **metrics,
    }
    return summary, fold_rows, pooled


def _validate_learned_design(directory: Path, spec: ModuleAblationSpec) -> None:
    """Reject resumed artifacts that do not implement the declared ablation."""

    for fold in OUTER_FOLDS:
        path = directory / f"fold_{fold}" / "effective_config.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        architecture = str(config.get("model", {}).get("architecture", ""))
        weight_mode = str(
            config.get("training", {}).get("positive_weight_mode", "")
        )
        fgl_alpha = float(config.get("training", {}).get("fgl_alpha", 0.0))
        fallback = str(config.get("decision", {}).get("fallback_policy", ""))
        if architecture != spec.architecture:
            raise ValueError(
                f"{spec.experiment_id} fold {fold} architecture={architecture!r}; "
                f"expected {spec.architecture!r}"
            )
        if weight_mode != spec.positive_weight_mode:
            raise ValueError(
                f"{spec.experiment_id} fold {fold} positive_weight_mode="
                f"{weight_mode!r}; expected {spec.positive_weight_mode!r}"
            )
        # ``fgl_alpha`` is the CE mixture weight.  The legacy CLI disables
        # PKD/FGL by setting it to exactly 1.0, so an enabled distillation run
        # must remain strictly inside (0, 1).
        if not np.isfinite(fgl_alpha) or not 0.0 < fgl_alpha < 1.0:
            raise ValueError(
                f"{spec.experiment_id} must keep PKD/FGL enabled "
                f"(expected 0 < fgl_alpha < 1, got {fgl_alpha!r})"
            )
        if fallback != "none":
            raise ValueError(
                f"{spec.experiment_id} fold {fold} must disable safe fallback"
            )


def _effect_table(summary: pd.DataFrame) -> pd.DataFrame:
    planned = (
        (
            "temporal_vs_static_ml",
            "M2_temporal_only",
            "S0_static_xgb",
            "Does native temporal encoding improve over protocol-aligned XGB?",
        ),
        (
            "temporal_vs_rule_only",
            "M2_temporal_only",
            "R0_rule_only",
            "Does the temporal branch improve over deterministic rules?",
        ),
        (
            "rule_guidance",
            "M4_fort",
            "M2_temporal_only",
            "Does RuleModel guidance improve the temporal model?",
        ),
        (
            "class_weight",
            "M4_fort",
            "M3_fort_no_class_weight",
            "Does positive class weighting help the complete model?",
        ),
    )
    by_id = summary.set_index("experiment_id", drop=False)
    rows: list[dict[str, Any]] = []
    for effect_id, treatment_id, reference_id, question in planned:
        available = treatment_id in by_id.index and reference_id in by_id.index
        row: dict[str, Any] = {
            "effect_id": effect_id,
            "treatment_id": treatment_id,
            "reference_id": reference_id,
            "question": question,
            "available": bool(available),
        }
        for metric in _EFFECT_METRICS:
            row[f"delta_{metric}"] = (
                float(by_id.loc[treatment_id, metric])
                - float(by_id.loc[reference_id, metric])
                if available
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _lead_profile(
    summaries: Iterable[Mapping[str, Any]],
    details: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = {
        str(row["experiment_id"]): str(row.get("display_name", row["experiment_id"]))
        for row in summaries
    }
    for experiment_id, detail in details.items():
        outcomes = detail["outcome"].astype(str)
        leads = pd.to_numeric(
            detail.loc[outcomes == "TP", "lead_hour"], errors="raise"
        )
        faulty = int((outcomes == "TP").sum() + (outcomes == "FN").sum())
        for horizon in LEAD_HORIZONS_HOUR:
            hit_count = (
                int((outcomes == "TP").sum())
                if horizon == 0
                else int((leads >= float(horizon)).sum())
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "display_name": names[experiment_id],
                    "lead_threshold_hour": horizon,
                    "hit_count": hit_count,
                    "faulty_module_count": faulty,
                    "lead_recall": float(hit_count / faulty) if faulty else 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_module_ablation_outputs(
    output_dir: Path | str,
    index_path: Path | str,
    *,
    formal: bool = True,
) -> pd.DataFrame:
    """Audit completed runs and write the four paper-facing CSV artifacts."""

    output_dir = Path(output_dir)
    index = pd.read_csv(index_path)
    required = {"file_name", "folder_index", "Label"}
    if required - set(index.columns):
        raise ValueError(f"index missing columns: {sorted(required - set(index.columns))}")

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    details: dict[str, pd.DataFrame] = {}

    static_dir = output_dir / "static_xgb"
    if (static_dir / "comparison.csv").is_file():
        static_recorded = _legacy_row(static_dir / "comparison.csv")
        expected_static = {
            "decision_feature_count": STATIC_FEATURE_COUNT,
            "model_architecture": "xgboost_current_row",
            "positive_weight_mode": "module_normalized_auto",
        }
        for field, expected in expected_static.items():
            if field not in static_recorded or static_recorded[field] != expected:
                raise ValueError(
                    f"S0_static_xgb {field}={static_recorded.get(field)!r}; "
                    f"expected {expected!r}"
                )
        for field in ("sequence_to_sequence", "safe_rule_fallback"):
            if field not in static_recorded:
                raise ValueError(f"S0_static_xgb is missing {field}")
            value = static_recorded[field]
            if str(value).strip().lower() not in {"false", "0"}:
                raise ValueError(f"S0_static_xgb must set {field}=False")
        metadata = _metadata(
            "S0_static_xgb",
            "Static XGB",
            model_family="static_xgb",
            temporal_encoder=False,
            rule_guidance=False,
            pkd=False,
            class_weight=True,
        )
        row, folds, pooled = _load_threefold_family(
            static_dir, metadata, index, formal=formal
        )
        summary_rows.append(row)
        fold_rows.extend(folds)
        details["S0_static_xgb"] = pooled.drop(columns="outer_fold")

    for spec in LEARNED_SPECS:
        _validate_learned_design(output_dir / spec.experiment_id, spec)
        metadata = _metadata(
            spec.experiment_id,
            spec.display_name,
            model_family=(
                "causal_temporal_model"
                if not spec.rule_guidance
                else "rule_guided_temporal_model"
            ),
            temporal_encoder=spec.temporal_encoder,
            rule_guidance=spec.rule_guidance,
            pkd=spec.pkd,
            class_weight=spec.positive_weight_mode != "none",
        )
        row, folds, pooled = _load_threefold_family(
            output_dir / spec.experiment_id,
            metadata,
            index,
            formal=formal,
        )
        summary_rows.append(row)
        fold_rows.extend(folds)
        details[spec.experiment_id] = pooled.drop(columns="outer_fold")

    rule_row, rule_folds, rule_pooled = _load_rule_only(
        output_dir / "M4_fort", index, formal=formal
    )
    summary_rows.append(rule_row)
    fold_rows.extend(rule_folds)
    details["R0_rule_only"] = rule_pooled.drop(columns="outer_fold")

    reference_names = None
    for experiment_id, detail in details.items():
        names = set(detail["file_name"].astype(str))
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError(
                f"module universe differs for {experiment_id}; ablation is not comparable"
            )

    order = {
        name: rank
        for rank, name in enumerate(
            ("S0_static_xgb", "R0_rule_only", *(s.experiment_id for s in LEARNED_SPECS))
        )
    }
    summary = pd.DataFrame(summary_rows)
    summary["__order"] = summary["experiment_id"].map(order)
    summary = summary.sort_values("__order", kind="mergesort").drop(columns="__order")
    folds = pd.DataFrame(fold_rows)
    folds["__order"] = folds["experiment_id"].map(order)
    folds = folds.sort_values(["__order", "outer_fold"], kind="mergesort").drop(
        columns="__order"
    )
    effects = _effect_table(summary)
    lead = _lead_profile(summary_rows, details)
    lead["__order"] = lead["experiment_id"].map(order)
    lead = lead.sort_values(
        ["__order", "lead_threshold_hour"], kind="mergesort"
    ).drop(columns="__order")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "module_ablation.csv", index=False)
    folds.to_csv(output_dir / "module_ablation_fold_metrics.csv", index=False)
    effects.to_csv(output_dir / "module_effects.csv", index=False)
    lead.to_csv(output_dir / "lead_time_profile.csv", index=False)
    return summary


__all__ = [
    "LEAD_HORIZONS_HOUR",
    "LEARNED_SPECS",
    "ModuleAblationSpec",
    "write_module_ablation_outputs",
]
