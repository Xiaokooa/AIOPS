"""Original-OFP-compatible module evaluation and validation-only decisions.

The primary protocol in this module deliberately mirrors ``OFP/EvaluateResult.py``:

* the first positive anomaly is the failure timestamp;
* the first positive decision is the alarm timestamp;
* a faulty module is a hit only when the alarm is strictly earlier than failure;
* any alarm on a normal module is a false positive;
* missed faulty modules contribute zero to the OFP average lead time.

Additional point-level, lead-time, and branch diagnostics never change the primary
confusion matrix or ``final_score``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_TS_SECONDS = "__drfp_timestamp_seconds"
_ANOMALY = "__drfp_anomaly"


@dataclass(frozen=True)
class OFPEvaluationResult:
    """Scalar metrics plus one row of protocol detail per module."""

    metrics: dict[str, Any]
    detail: pd.DataFrame

    def __iter__(self) -> Iterator[Any]:
        """Allow ``metrics, detail = evaluate_ofp(...)``."""

        yield self.metrics
        yield self.detail


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Best validation threshold and an auditable candidate table."""

    threshold: float
    metrics: dict[str, Any]
    detail: pd.DataFrame
    table: pd.DataFrame


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _required_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _timestamps_as_seconds(values: pd.Series) -> pd.Series:
    """Convert numeric-second or datetime-like timestamps to floating seconds."""

    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError("timestamp contains missing or invalid values")
        return pd.Series(parsed.astype("int64") / 1_000_000_000.0, index=values.index)

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        numeric = numeric.astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("timestamp contains non-finite values")
        return numeric

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("timestamp must be numeric seconds or consistently datetime-like")
    return pd.Series(parsed.astype("int64") / 1_000_000_000.0, index=values.index)


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    module_col: str,
    timestamp_col: str,
    anomaly_col: str,
) -> pd.DataFrame:
    _required_columns(frame, (module_col, timestamp_col, anomaly_col))
    work = frame.copy().reset_index(drop=True)
    if work[module_col].isna().any():
        raise ValueError(f"{module_col!r} contains missing module identifiers")
    # Preserve compact integer/categorical module ids for multi-million-row
    # formal inference.  Converting them to Python strings here multiplies
    # memory without changing grouping semantics.
    if pd.api.types.is_object_dtype(work[module_col]) or pd.api.types.is_string_dtype(
        work[module_col]
    ):
        if (work[module_col].astype(str).str.len() == 0).any():
            raise ValueError(f"{module_col!r} contains an empty module identifier")
    work[_TS_SECONDS] = _timestamps_as_seconds(work[timestamp_col])
    anomaly = pd.to_numeric(work[anomaly_col], errors="coerce").fillna(0.0)
    work[_ANOMALY] = (anomaly > 0).astype(np.int8)
    return work


def _probability_series(frame: pd.DataFrame, column: str) -> pd.Series:
    _required_columns(frame, (column,))
    scores = pd.to_numeric(frame[column], errors="coerce").astype(float)
    finite = scores.notna()
    if finite.any():
        values = scores.loc[finite].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{column!r} contains non-finite scores")
        if (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{column!r} must contain probabilities in [0, 1]")
    return scores


def _decision_series(frame: pd.DataFrame, column: str) -> pd.Series:
    _required_columns(frame, (column,))
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return (values > 0).astype(bool)


def _resolve_main_decisions(
    work: pd.DataFrame,
    *,
    threshold: float,
    score_col: Optional[str],
    predict_col: Optional[str],
) -> Tuple[pd.Series, pd.Series, str]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    if predict_col is not None:
        decisions = _decision_series(work, predict_col)
        if score_col is not None and score_col in work.columns:
            scores = _probability_series(work, score_col)
        else:
            scores = decisions.astype(float)
        return decisions, scores, predict_col

    if score_col is not None and score_col in work.columns:
        scores = _probability_series(work, score_col)
        decisions = scores.notna() & (scores >= threshold)
        return decisions.astype(bool), scores, score_col

    if "predict" in work.columns:
        decisions = _decision_series(work, "predict")
        return decisions, decisions.astype(float), "predict"

    requested = score_col if score_col is not None else "score"
    raise ValueError(
        f"no decision input found: provide {requested!r} scores or a predict column"
    )


def _original_timestamp_at(
    group: pd.DataFrame, mask: pd.Series, timestamp_col: str
) -> Any:
    candidates = group.loc[mask]
    if candidates.empty:
        return None
    row_index = candidates[_TS_SECONDS].idxmin()
    return group.loc[row_index, timestamp_col]


def _evaluate_module_decisions(
    work: pd.DataFrame,
    decisions: pd.Series,
    *,
    module_col: str,
    timestamp_col: str,
) -> Tuple[dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    decisions = decisions.reindex(work.index, fill_value=False).astype(bool)

    for module_id, group in work.groupby(module_col, sort=False):
        group_decisions = decisions.loc[group.index]
        failure_mask = group[_ANOMALY] > 0
        is_faulty = bool(failure_mask.any())
        has_any_alarm = bool(group_decisions.any())

        failure_seconds: Optional[float]
        alarm_seconds: Optional[float]
        first_failure_timestamp: Any
        first_alarm_timestamp: Any

        if is_faulty:
            failure_seconds = float(group.loc[failure_mask, _TS_SECONDS].min())
            first_failure_timestamp = _original_timestamp_at(
                group, failure_mask, timestamp_col
            )
        else:
            failure_seconds = None
            first_failure_timestamp = None

        if has_any_alarm:
            alarm_seconds = float(group.loc[group_decisions, _TS_SECONDS].min())
            first_alarm_timestamp = _original_timestamp_at(
                group, group_decisions, timestamp_col
            )
        else:
            alarm_seconds = None
            first_alarm_timestamp = None

        alarm_before_failure = bool(
            is_faulty
            and has_any_alarm
            and alarm_seconds is not None
            and failure_seconds is not None
            and alarm_seconds < failure_seconds
        )

        if is_faulty:
            outcome = "TP" if alarm_before_failure else "FN"
            module_prediction = int(alarm_before_failure)
        else:
            outcome = "FP" if has_any_alarm else "TN"
            module_prediction = int(has_any_alarm)

        lead_hour = (
            float((failure_seconds - alarm_seconds) / 3600.0)
            if alarm_before_failure
            and failure_seconds is not None
            and alarm_seconds is not None
            else np.nan
        )
        if is_faulty and failure_seconds is not None:
            valid_prefault_alarm_count = int(
                (
                    group_decisions
                    & (group[_TS_SECONDS] < failure_seconds)
                ).sum()
            )
        else:
            valid_prefault_alarm_count = int(group_decisions.sum())

        records.append(
            {
                module_col: module_id,
                "is_faulty": int(is_faulty),
                "module_prediction": module_prediction,
                "outcome": outcome,
                "first_failure_timestamp": first_failure_timestamp,
                "first_alarm_timestamp": first_alarm_timestamp,
                "has_any_alarm": int(has_any_alarm),
                "alarm_strictly_before_failure": int(alarm_before_failure),
                "alarm_row_count": int(group_decisions.sum()),
                "valid_prefault_alarm_row_count": valid_prefault_alarm_count,
                "lead_hour": lead_hour,
            }
        )

    detail = pd.DataFrame.from_records(records)
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                module_col,
                "is_faulty",
                "module_prediction",
                "outcome",
                "first_failure_timestamp",
                "first_alarm_timestamp",
                "has_any_alarm",
                "alarm_strictly_before_failure",
                "alarm_row_count",
                "valid_prefault_alarm_row_count",
                "lead_hour",
            ]
        )

    outcomes = detail["outcome"] if "outcome" in detail else pd.Series(dtype=str)
    tp = int((outcomes == "TP").sum())
    fp = int((outcomes == "FP").sum())
    fn = int((outcomes == "FN").sum())
    tn = int((outcomes == "TN").sum())
    module_count = len(detail)
    faulty_count = tp + fn
    normal_count = fp + tn
    predicted_positive_count = tp + fp

    accuracy = _safe_ratio(tp + tn, module_count)
    precision = _safe_ratio(tp, predicted_positive_count)
    recall = _safe_ratio(tp, faulty_count)
    f1_score = _safe_ratio(2.0 * precision * recall, precision + recall)

    hit_leads = pd.to_numeric(
        detail.loc[outcomes == "TP", "lead_hour"], errors="coerce"
    ).dropna()
    lead_sum_hour = float(hit_leads.sum()) if len(hit_leads) else 0.0
    # This denominator is intentionally all faulty modules, not only hits.
    avg_lead_hour = _safe_ratio(lead_sum_hour, faulty_count)
    min_lead_hour = float(hit_leads.min()) if len(hit_leads) else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1_score + accuracy + avg_lead_score + min_lead_score

    metrics: dict[str, Any] = {
        "final_score": float(final_score),
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "avg_lead_hour": float(avg_lead_hour),
        "min_lead_hour": float(min_lead_hour),
        "avg_lead_score": float(avg_lead_score),
        "min_lead_score": float(min_lead_score),
        "lead_sum_hour": float(lead_sum_hour),
        "all_hit_cnt": tp,
        "all_predict_pos_cnt": predicted_positive_count,
        "all_true_pos_cnt": faulty_count,
        "lead_pred_cnt": tp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "module_count": module_count,
        "faulty_module_count": faulty_count,
        "normal_module_count": normal_count,
        "normal_module_false_alarm_rate": _safe_ratio(fp, normal_count),
    }
    return metrics, detail


def _lead_diagnostics(detail: pd.DataFrame) -> dict[str, Any]:
    hit_leads = pd.to_numeric(
        detail.loc[detail["outcome"] == "TP", "lead_hour"], errors="coerce"
    ).dropna()
    diagnostics: dict[str, Any] = {
        "hit_only_mean_lead_hour": float(hit_leads.mean()) if len(hit_leads) else 0.0,
        "hit_only_median_lead_hour": float(hit_leads.median()) if len(hit_leads) else 0.0,
        "hit_only_q25_lead_hour": float(hit_leads.quantile(0.25)) if len(hit_leads) else 0.0,
        "hit_only_q75_lead_hour": float(hit_leads.quantile(0.75)) if len(hit_leads) else 0.0,
        "hit_only_q90_lead_hour": float(hit_leads.quantile(0.90)) if len(hit_leads) else 0.0,
    }
    buckets: Sequence[Tuple[str, float, float]] = (
        ("0_16", 0.0, 16.0),
        ("16_24", 16.0, 24.0),
        ("24_72", 24.0, 72.0),
        ("72_120", 72.0, 120.0),
        ("gt_120", 120.0, math.inf),
    )
    hit_count = len(hit_leads)
    for name, lower, upper in buckets:
        if math.isinf(upper):
            count = int((hit_leads > lower).sum())
        else:
            count = int(((hit_leads > lower) & (hit_leads <= upper)).sum())
        diagnostics[f"lead_bucket_{name}_count"] = count
        diagnostics[f"lead_bucket_{name}_fraction_of_hits"] = _safe_ratio(
            count, hit_count
        )
    return diagnostics


def _prefault_targets(
    work: pd.DataFrame,
    *,
    module_col: str,
    horizon_hours: float,
) -> Tuple[pd.Series, pd.Series]:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    valid = pd.Series(False, index=work.index, dtype=bool)
    targets = pd.Series(0, index=work.index, dtype=np.int8)
    horizon_seconds = float(horizon_hours) * 3600.0

    for _, group in work.groupby(module_col, sort=False):
        failure_rows = group[_ANOMALY] > 0
        if failure_rows.any():
            failure_seconds = float(group.loc[failure_rows, _TS_SECONDS].min())
            group_valid = group[_TS_SECONDS] < failure_seconds
            valid.loc[group.index] = group_valid
            lead_seconds = failure_seconds - group[_TS_SECONDS]
            group_positive = group_valid & (lead_seconds <= horizon_seconds)
            targets.loc[group.index] = group_positive.astype(np.int8)
        else:
            valid.loc[group.index] = True
    return valid, targets


def _average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(targets.sum())
    if len(targets) == 0 or positive_count == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order].astype(int)
    cumulative_tp = np.cumsum(sorted_targets)
    cumulative_fp = np.cumsum(1 - sorted_targets)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp_at_threshold = cumulative_tp[ends]
    fp_at_threshold = cumulative_fp[ends]
    precision = tp_at_threshold / np.maximum(tp_at_threshold + fp_at_threshold, 1)
    recall = tp_at_threshold / positive_count
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def _expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    if bins < 2:
        raise ValueError("ece_bins must be at least 2")
    if len(targets) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.searchsorted(edges[1:-1], probabilities, side="right")
    ece = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        frequency = float(targets[mask].mean())
        ece += float(mask.mean()) * abs(confidence - frequency)
    return float(ece)


def _timestamp_diagnostics(
    work: pd.DataFrame,
    scores: pd.Series,
    *,
    module_col: str,
    horizon_hours: float,
    ece_bins: int,
) -> Tuple[dict[str, Any], pd.Series]:
    valid, targets = _prefault_targets(
        work, module_col=module_col, horizon_hours=horizon_hours
    )
    scored = valid & scores.notna()
    y = targets.loc[scored].to_numpy(dtype=int)
    probability = scores.loc[scored].to_numpy(dtype=float)
    diagnostics: dict[str, Any] = {
        "timestamp_valid_prefault_rows": int(valid.sum()),
        "timestamp_scored_valid_prefault_rows": int(scored.sum()),
        "timestamp_positive_rows_120h": int(y.sum()),
        "timestamp_pr_auc_120h": _average_precision(y, probability),
        "timestamp_brier_120h": (
            float(np.mean((probability - y) ** 2)) if len(y) else 0.0
        ),
        "timestamp_ece_120h": _expected_calibration_error(y, probability, ece_bins),
    }
    return diagnostics, valid


def multi_horizon_timestamp_diagnostics(
    frame: pd.DataFrame,
    score_columns: Mapping[float, str],
    *,
    module_col: str = "file_name",
    timestamp_col: str = "timestamp",
    anomaly_col: str = "anomaly",
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Supplemental PR-AUC/Brier/ECE for every supervised failure horizon."""

    work = _prepare_frame(
        frame,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
    )
    result: dict[str, Any] = {}
    for horizon, column in sorted(score_columns.items(), key=lambda item: float(item[0])):
        scores = _probability_series(work, str(column))
        valid, targets = _prefault_targets(
            work, module_col=module_col, horizon_hours=float(horizon)
        )
        scored = valid & scores.notna()
        y = targets.loc[scored].to_numpy(dtype=int)
        probability = scores.loc[scored].to_numpy(dtype=float)
        suffix = f"{float(horizon):g}h"
        result[f"timestamp_valid_rows_{suffix}"] = int(scored.sum())
        result[f"timestamp_positive_rows_{suffix}"] = int(y.sum())
        result[f"timestamp_pr_auc_{suffix}"] = _average_precision(y, probability)
        result[f"timestamp_brier_{suffix}"] = (
            float(np.mean((probability - y) ** 2)) if len(y) else 0.0
        )
        result[f"timestamp_ece_{suffix}"] = _expected_calibration_error(
            y, probability, ece_bins
        )
    return result


def _first_present(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _branch_decisions(
    work: pd.DataFrame,
    *,
    threshold: float,
    main_decisions: pd.Series,
    branch_score_cols: Optional[Mapping[str, str]],
    branch_predict_cols: Optional[Mapping[str, str]],
    branch_thresholds: Optional[Mapping[str, float]],
) -> dict[str, pd.Series]:
    resolved: dict[str, pd.Series] = {}
    explicit_scores = dict(branch_score_cols or {})
    explicit_predict = dict(branch_predict_cols or {})
    explicit_thresholds = {
        str(name): float(value) for name, value in dict(branch_thresholds or {}).items()
    }
    names = set(explicit_scores) | set(explicit_predict)

    defaults = {
        "deep": {
            "score": ("deep_score", "deep_probability", "p_deep"),
            "predict": ("deep_predict", "deep_decision"),
        },
        "rule": {
            "score": ("rule_score", "rule_probability", "p_rule"),
            "predict": ("rule_predict", "rule_decision"),
        },
        "fusion": {
            "score": ("fusion_score", "fusion_probability", "p_fusion"),
            "predict": ("fusion_predict", "fusion_decision"),
        },
    }
    names.update(defaults)

    for name in sorted(names):
        predict_column = explicit_predict.get(name)
        score_column = explicit_scores.get(name)
        if name in defaults:
            if predict_column is None:
                predict_column = _first_present(work, defaults[name]["predict"])
            if score_column is None:
                score_column = _first_present(work, defaults[name]["score"])
        if predict_column is not None:
            resolved[name] = _decision_series(work, predict_column)
        elif score_column is not None:
            branch_scores = _probability_series(work, score_column)
            branch_threshold = explicit_thresholds.get(name, float(threshold))
            if not 0.0 <= branch_threshold <= 1.0:
                raise ValueError(f"branch threshold for {name!r} must be in [0, 1]")
            resolved[name] = branch_scores.notna() & (branch_scores >= branch_threshold)

    # When component branches are available, the caller's main output is the
    # authoritative fusion decision unless an explicit fusion column exists.
    if "fusion" not in resolved and ("deep" in resolved or "rule" in resolved):
        resolved["fusion"] = main_decisions.astype(bool)
    return resolved


def _branch_diagnostics(
    work: pd.DataFrame,
    branch_decisions: Mapping[str, pd.Series],
    valid_prefault: pd.Series,
    *,
    module_col: str,
    timestamp_col: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    valid_count = int(valid_prefault.sum())
    for name, decisions in branch_decisions.items():
        branch_valid = decisions.reindex(work.index, fill_value=False) & valid_prefault
        positive_count = int(branch_valid.sum())
        diagnostics[f"branch_{name}_positive_rows"] = positive_count
        diagnostics[f"branch_{name}_positive_rate"] = _safe_ratio(
            positive_count, valid_count
        )
        branch_metrics, _ = _evaluate_module_decisions(
            work,
            decisions,
            module_col=module_col,
            timestamp_col=timestamp_col,
        )
        for metric_name in (
            "tp",
            "fp",
            "fn",
            "tn",
            "f1_score",
            "precision",
            "recall",
            "accuracy",
            "final_score",
        ):
            diagnostics[f"branch_{name}_{metric_name}"] = branch_metrics[metric_name]

    deep = branch_decisions.get("deep")
    rule = branch_decisions.get("rule")
    fusion = branch_decisions.get("fusion")
    if deep is not None and rule is not None:
        deep_valid = deep.reindex(work.index, fill_value=False) & valid_prefault
        rule_valid = rule.reindex(work.index, fill_value=False) & valid_prefault
        disagreement = (deep != rule).reindex(work.index, fill_value=False) & valid_prefault
        disagreement_count = int(disagreement.sum())
        diagnostics["deep_rule_disagreement_rows"] = disagreement_count
        diagnostics["deep_rule_disagreement_rate"] = _safe_ratio(
            disagreement_count, valid_count
        )
        diagnostics["deep_only_positive_rows"] = int((deep_valid & ~rule_valid).sum())
        diagnostics["rule_only_positive_rows"] = int((rule_valid & ~deep_valid).sum())
        diagnostics["deep_rule_both_positive_rows"] = int((deep_valid & rule_valid).sum())

    if fusion is not None and deep is not None:
        fusion_valid = fusion.reindex(work.index, fill_value=False) & valid_prefault
        deep_valid = deep.reindex(work.index, fill_value=False) & valid_prefault
        diagnostics["fusion_added_vs_deep_rows"] = int(
            (fusion_valid & ~deep_valid).sum()
        )
        diagnostics["fusion_suppressed_vs_deep_rows"] = int(
            (deep_valid & ~fusion_valid).sum()
        )
        diagnostics["fusion_changed_vs_deep_rows"] = int(
            ((fusion != deep).reindex(work.index, fill_value=False) & valid_prefault).sum()
        )
        if rule is not None:
            rule_valid = rule.reindex(work.index, fill_value=False) & valid_prefault
            diagnostics["fusion_positive_rule_only_support_rows"] = int(
                (fusion_valid & ~deep_valid & rule_valid).sum()
            )
            diagnostics["fusion_positive_deep_only_support_rows"] = int(
                (fusion_valid & deep_valid & ~rule_valid).sum()
            )
            diagnostics["fusion_positive_both_support_rows"] = int(
                (fusion_valid & deep_valid & rule_valid).sum()
            )
    return diagnostics


def evaluate_ofp(
    frame: pd.DataFrame,
    threshold: float = 0.5,
    *,
    module_col: str = "file_name",
    timestamp_col: str = "timestamp",
    anomaly_col: str = "anomaly",
    score_col: Optional[str] = "score",
    predict_col: Optional[str] = None,
    horizon_hours: float = 120.0,
    ece_bins: int = 10,
    branch_score_cols: Optional[Mapping[str, str]] = None,
    branch_predict_cols: Optional[Mapping[str, str]] = None,
    branch_thresholds: Optional[Mapping[str, float]] = None,
    include_diagnostics: bool = True,
) -> OFPEvaluationResult:
    """Evaluate a long-form prediction frame with the frozen original OFP protocol.

    ``predict_col`` takes precedence when explicitly provided. Otherwise ``score_col``
    is thresholded with ``score >= threshold``; if no score exists, a conventional
    ``predict`` column is accepted. Missing scores are conservative negative decisions.
    """

    work = _prepare_frame(
        frame,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
    )
    decisions, scores, decision_source = _resolve_main_decisions(
        work,
        threshold=threshold,
        score_col=score_col,
        predict_col=predict_col,
    )
    metrics, detail = _evaluate_module_decisions(
        work,
        decisions,
        module_col=module_col,
        timestamp_col=timestamp_col,
    )
    metrics["decision_threshold"] = float(threshold)
    metrics["decision_source"] = decision_source

    if include_diagnostics:
        metrics.update(_lead_diagnostics(detail))
        timestamp_metrics, valid_prefault = _timestamp_diagnostics(
            work,
            scores,
            module_col=module_col,
            horizon_hours=horizon_hours,
            ece_bins=ece_bins,
        )
        metrics.update(timestamp_metrics)
        branches = _branch_decisions(
            work,
            threshold=threshold,
            main_decisions=decisions,
            branch_score_cols=branch_score_cols,
            branch_predict_cols=branch_predict_cols,
            branch_thresholds=branch_thresholds,
        )
        metrics.update(
            _branch_diagnostics(
                work,
                branches,
                valid_prefault,
                module_col=module_col,
                timestamp_col=timestamp_col,
            )
        )
    return OFPEvaluationResult(metrics=metrics, detail=detail)


def threshold_candidates(
    scores: Sequence[float],
    *,
    fixed_threshold: float = 0.5,
    grid_size: int = 201,
    quantile_count: int = 201,
) -> np.ndarray:
    """Return deterministic candidates containing 0.5, a grid, and score quantiles."""

    if grid_size < 2 or quantile_count < 2:
        raise ValueError("grid_size and quantile_count must be at least 2")
    if not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("fixed_threshold must be in [0, 1]")
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) and ((values < 0.0).any() or (values > 1.0).any()):
        raise ValueError("threshold search scores must be in [0, 1]")
    candidates = [np.linspace(0.0, 1.0, grid_size), np.asarray([0.5, fixed_threshold])]
    if len(values):
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, quantile_count))
        candidates.append(np.asarray(quantiles, dtype=float))
    combined = np.concatenate(candidates)
    # Rounding removes floating duplicates while retaining stable, auditable values.
    return np.asarray(sorted(set(np.round(combined, 12).tolist())), dtype=float)


def search_validation_threshold(
    validation_frame: pd.DataFrame,
    *,
    score_col: str = "score",
    fixed_threshold: float = 0.5,
    grid_size: int = 201,
    quantile_count: int = 201,
    module_col: str = "file_name",
    timestamp_col: str = "timestamp",
    anomaly_col: str = "anomaly",
    horizon_hours: float = 120.0,
    ece_bins: int = 10,
    branch_score_cols: Optional[Mapping[str, str]] = None,
    branch_predict_cols: Optional[Mapping[str, str]] = None,
) -> ThresholdSearchResult:
    """Select a threshold using validation data only.

    Ranking is descending by original OFP ``final_score``, then F1, precision,
    and finally the higher threshold. The returned table records every candidate.
    """

    work = _prepare_frame(
        validation_frame,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
    )
    scores = _probability_series(work, score_col)
    valid_prefault, _ = _prefault_targets(
        work, module_col=module_col, horizon_hours=horizon_hours
    )
    candidates = threshold_candidates(
        scores.loc[valid_prefault & scores.notna()].to_numpy(),
        fixed_threshold=fixed_threshold,
        grid_size=grid_size,
        quantile_count=quantile_count,
    )
    # Evaluate every threshold in one scan per module.  Re-running a pandas
    # groupby for each of ~400 candidates is prohibitively slow on the formal
    # validation set (millions of timestamp rows).
    descending = candidates[::-1]
    module_predictions: list[np.ndarray] = []
    module_leads: list[np.ndarray] = []
    module_faulty: list[bool] = []
    for _, group in work.groupby(module_col, sort=False):
        failure_rows = group[_ANOMALY] > 0
        is_faulty = bool(failure_rows.any())
        failure_seconds = (
            float(group.loc[failure_rows, _TS_SECONDS].min()) if is_faulty else None
        )
        group_scores = scores.loc[group.index].to_numpy(dtype=float)
        group_times = group[_TS_SECONDS].to_numpy(dtype=float)
        finite = np.isfinite(group_scores)
        if is_faulty and failure_seconds is not None:
            finite &= group_times < failure_seconds
        score_values = group_scores[finite]
        time_values = group_times[finite]
        order = np.argsort(-score_values, kind="mergesort")
        ordered_scores = score_values[order]
        ordered_times = time_values[order]
        predictions = np.zeros(len(descending), dtype=bool)
        leads = np.zeros(len(descending), dtype=np.float64)
        pointer = 0
        earliest = math.inf
        for candidate_index, candidate in enumerate(descending):
            while pointer < len(ordered_scores) and ordered_scores[pointer] >= candidate:
                earliest = min(earliest, float(ordered_times[pointer]))
                pointer += 1
            if math.isfinite(earliest):
                predictions[candidate_index] = True
                if is_faulty and failure_seconds is not None:
                    leads[candidate_index] = (failure_seconds - earliest) / 3600.0
        module_predictions.append(predictions[::-1])
        module_leads.append(leads[::-1])
        module_faulty.append(is_faulty)

    prediction_matrix = np.stack(module_predictions, axis=0)
    lead_matrix = np.stack(module_leads, axis=0)
    faulty = np.asarray(module_faulty, dtype=bool)[:, None]
    tp = (prediction_matrix & faulty).sum(axis=0).astype(float)
    fp = (prediction_matrix & ~faulty).sum(axis=0).astype(float)
    fn = (~prediction_matrix & faulty).sum(axis=0).astype(float)
    tn = (~prediction_matrix & ~faulty).sum(axis=0).astype(float)
    predicted_positive = tp + fp
    faulty_count = tp + fn
    module_count = float(len(module_faulty))
    precision = np.divide(tp, predicted_positive, out=np.zeros_like(tp), where=predicted_positive > 0)
    recall = np.divide(tp, faulty_count, out=np.zeros_like(tp), where=faulty_count > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    accuracy = (tp + tn) / module_count
    hit_mask = prediction_matrix & faulty
    hit_leads = np.where(hit_mask, lead_matrix, 0.0)
    lead_sum = hit_leads.sum(axis=0)
    avg_lead = np.divide(
        lead_sum, faulty_count, out=np.zeros_like(lead_sum), where=faulty_count > 0
    )
    min_lead = np.min(np.where(hit_mask, lead_matrix, np.inf), axis=0)
    min_lead = np.where(np.isfinite(min_lead), min_lead, 0.0)
    final_score = f1 + accuracy + np.tanh(avg_lead) + np.tanh(min_lead)
    rows = [
        {
            "threshold": float(candidates[index]),
            "final_score": float(final_score[index]),
            "f1_score": float(f1[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "accuracy": float(accuracy[index]),
            "avg_lead_hour": float(avg_lead[index]),
            "min_lead_hour": float(min_lead[index]),
        }
        for index in range(len(candidates))
    ]

    table = pd.DataFrame.from_records(rows)
    table = table.sort_values(
        ["final_score", "f1_score", "precision", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=int))
    table["selected"] = False
    table.loc[0, "selected"] = True
    best_threshold = float(table.loc[0, "threshold"])

    best = evaluate_ofp(
        validation_frame,
        threshold=best_threshold,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
        score_col=score_col,
        horizon_hours=horizon_hours,
        ece_bins=ece_bins,
        branch_score_cols=branch_score_cols,
        branch_predict_cols=branch_predict_cols,
        branch_thresholds=None,
        include_diagnostics=True,
    )
    return ThresholdSearchResult(
        threshold=best_threshold,
        metrics=best.metrics,
        detail=best.detail,
        table=table,
    )


def write_ofp_module_csvs(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    threshold: float = 0.5,
    module_col: str = "file_name",
    timestamp_col: str = "timestamp",
    anomaly_col: str = "anomaly",
    score_col: Optional[str] = "score",
    predict_col: Optional[str] = None,
    include_score: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Write one original-OFP-compatible ``timestamp,predict`` CSV per module."""

    work = _prepare_frame(
        frame,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
    )
    decisions, scores, _ = _resolve_main_decisions(
        work,
        threshold=threshold,
        score_col=score_col,
        predict_col=predict_col,
    )
    output = Path(output_dir)
    targets: list[Tuple[Path, pd.DataFrame]] = []
    seen: set[Path] = set()
    for module_id, group in work.groupby(module_col, sort=False):
        base_name = Path(str(module_id)).name
        if not base_name:
            raise ValueError(f"invalid module file name: {module_id!r}")
        if Path(base_name).suffix.lower() != ".csv":
            base_name = f"{base_name}.csv"
        target = output / base_name
        if target in seen:
            raise ValueError(f"module identifiers collide at output file {target.name!r}")
        seen.add(target)
        ordered = group.sort_values(_TS_SECONDS, kind="mergesort")
        result = pd.DataFrame(
            {
                "timestamp": ordered[timestamp_col].to_numpy(),
                "predict": decisions.loc[ordered.index].astype(np.int8).to_numpy(),
            }
        )
        if include_score:
            result["score"] = scores.loc[ordered.index].to_numpy(dtype=float)
        targets.append((target, result))

    existing = [target for target, _ in targets if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing OFP CSV: {existing[0]}")
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for target, result in targets:
        result.to_csv(target, index=False)
        written.append(target)
    return written


# Descriptive alias for callers that prefer the input layout in the function name.
evaluate_ofp_long_frame = evaluate_ofp


__all__ = [
    "OFPEvaluationResult",
    "ThresholdSearchResult",
    "evaluate_ofp",
    "evaluate_ofp_long_frame",
    "search_validation_threshold",
    "threshold_candidates",
    "write_ofp_module_csvs",
]
