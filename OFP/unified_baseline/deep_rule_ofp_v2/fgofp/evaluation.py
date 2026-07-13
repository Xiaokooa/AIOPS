"""Leakage-safe module evaluation for the frozen Legacy-Inclusive protocol.

``legacy_inclusive_v1`` intentionally counts an alarm at the first failure
timestamp as a hit.  Timestamps are represented as exact ``int64`` seconds so
that this inclusion can never be produced by float32 rounding or backdating.

Thresholds are selected on validation data only.  The optimized search below
uses the same full-module, first-crossing decision as :func:`evaluate_legacy_inclusive`;
in particular it does not discard the first failure row while selecting a
threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


PROTOCOL_NAME = "legacy_inclusive_v1"
_TIMESTAMP_SECONDS = "__fgofp_timestamp_seconds"
_ANOMALY = "__fgofp_anomaly"


@dataclass(frozen=True)
class LegacyEvaluationResult:
    """Scalar protocol metrics and one auditable record per module."""

    metrics: dict[str, Any]
    detail: pd.DataFrame

    def __iter__(self) -> Iterator[Any]:
        yield self.metrics
        yield self.detail


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Selected validation threshold and the complete candidate audit table."""

    threshold: float
    metrics: dict[str, Any]
    detail: pd.DataFrame
    table: pd.DataFrame


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _timestamps_to_int64_seconds(values: pd.Series) -> pd.Series:
    """Return exact seconds, rejecting fractional or out-of-range timestamps."""

    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError("timestamp contains missing or invalid values")
        nanoseconds = parsed.astype("int64")
        if (nanoseconds % 1_000_000_000 != 0).any():
            raise ValueError("timestamp must have whole-second resolution")
        return pd.Series(
            (nanoseconds // 1_000_000_000).astype(np.int64), index=values.index
        )

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        # A consistently datetime-like text column is also accepted, but it is
        # still converted to exact whole seconds rather than floating seconds.
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError("timestamp must contain integer seconds or datetimes")
        nanoseconds = parsed.astype("int64")
        if (nanoseconds % 1_000_000_000 != 0).any():
            raise ValueError("timestamp must have whole-second resolution")
        return pd.Series(
            (nanoseconds // 1_000_000_000).astype(np.int64), index=values.index
        )

    if pd.api.types.is_integer_dtype(numeric.dtype):
        try:
            return numeric.astype(np.int64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("timestamp is outside the int64 range") from exc

    array = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("timestamp contains non-finite values")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError("timestamp must contain whole seconds")
    int64_info = np.iinfo(np.int64)
    if (rounded < int64_info.min).any() or (rounded > int64_info.max).any():
        raise ValueError("timestamp is outside the int64 range")
    return pd.Series(rounded.astype(np.int64), index=values.index)


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    module_col: str,
    timestamp_col: str,
    anomaly_col: str,
) -> pd.DataFrame:
    _require_columns(frame, (module_col, timestamp_col, anomaly_col))
    work = frame.copy().reset_index(drop=True)
    if work[module_col].isna().any():
        raise ValueError(f"{module_col!r} contains a missing module identifier")
    work[_TIMESTAMP_SECONDS] = _timestamps_to_int64_seconds(work[timestamp_col])
    anomaly = pd.to_numeric(work[anomaly_col], errors="coerce").fillna(0.0)
    work[_ANOMALY] = (anomaly > 0).astype(np.int8)
    for module_id, group in work.groupby(module_col, sort=False, observed=True):
        times = group[_TIMESTAMP_SECONDS].to_numpy(dtype=np.int64)
        if len(times) > 1 and np.any(np.diff(times) < 0):
            raise ValueError(
                f"timestamps must be non-decreasing within module {module_id!r}"
            )
    return work


def _legacy_decision_row_mask(work: pd.DataFrame, *, module_col: str) -> pd.Series:
    """Rows eligible for a module decision under Legacy-Inclusive.

    Healthy modules retain every row. Faulty modules retain the chronological
    prefix through the positional first anomaly row, including that row. This
    prevents later rows that repeat the same timestamp from masquerading as a
    zero-lead hit and keeps validation threshold candidates post-fault-free.
    """

    eligible = pd.Series(False, index=work.index, dtype=bool)
    for _, group in work.groupby(module_col, sort=False, observed=True):
        failure_positions = np.flatnonzero(
            group[_ANOMALY].to_numpy(dtype=np.int8) > 0
        )
        stop = int(failure_positions[0]) + 1 if len(failure_positions) else len(group)
        eligible.loc[group.index[:stop]] = True
    return eligible


def _probabilities(frame: pd.DataFrame, score_col: str) -> pd.Series:
    _require_columns(frame, (score_col,))
    scores = pd.to_numeric(frame[score_col], errors="coerce").astype(np.float64)
    finite = scores.notna()
    values = scores.loc[finite].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{score_col!r} contains non-finite values")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(f"{score_col!r} must contain probabilities in [0, 1]")
    return scores


def _decisions(
    work: pd.DataFrame,
    *,
    threshold: float,
    score_col: str,
    predict_col: str | None,
) -> tuple[pd.Series, pd.Series, str]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    scores = _probabilities(work, score_col)
    if predict_col is None:
        decisions = scores.notna() & (scores >= threshold)
        return decisions.astype(bool), scores, score_col
    _require_columns(work, (predict_col,))
    values = pd.to_numeric(work[predict_col], errors="coerce").fillna(0.0)
    return (values > 0).astype(bool), scores, predict_col


def _module_metrics(
    work: pd.DataFrame,
    decisions: pd.Series,
    *,
    module_col: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    decisions = decisions.reindex(work.index, fill_value=False).astype(bool)
    records: list[dict[str, Any]] = []

    for module_id, group in work.groupby(module_col, sort=False, observed=True):
        group_decisions = decisions.loc[group.index]
        failure_mask = group[_ANOMALY] > 0
        is_faulty = bool(failure_mask.any())
        if is_faulty:
            first_fault_position = int(
                np.flatnonzero(failure_mask.to_numpy(dtype=bool))[0]
            )
            decision_prefix = group.index[: first_fault_position + 1]
            failure_seconds = int(
                group.iloc[first_fault_position][_TIMESTAMP_SECONDS]
            )
        else:
            first_fault_position = None
            decision_prefix = group.index
            failure_seconds = None
        eligible_decisions = group_decisions.loc[decision_prefix]
        has_any_alarm = bool(group_decisions.any())
        has_eligible_alarm = bool(eligible_decisions.any())
        first_any_alarm_seconds = (
            int(group.loc[group_decisions, _TIMESTAMP_SECONDS].min())
            if has_any_alarm
            else None
        )
        alarm_seconds = (
            int(group.loc[eligible_decisions[eligible_decisions].index, _TIMESTAMP_SECONDS].min())
            if has_eligible_alarm
            else None
        )

        # Frozen Legacy-Inclusive rule: equality is a valid hit.  An alarm that
        # exists only after failure remains an FN rather than an FP or TP.
        is_hit = bool(
            is_faulty
            and has_eligible_alarm
            and alarm_seconds is not None
            and failure_seconds is not None
            and alarm_seconds <= failure_seconds
        )
        if is_faulty:
            outcome = "TP" if is_hit else "FN"
            module_prediction = int(is_hit)
        else:
            outcome = "FP" if has_eligible_alarm else "TN"
            module_prediction = int(has_eligible_alarm)

        lead_hour = (
            float(failure_seconds - alarm_seconds) / 3600.0
            if is_hit and failure_seconds is not None and alarm_seconds is not None
            else np.nan
        )
        alarm_at_failure = bool(
            is_hit
            and failure_seconds is not None
            and alarm_seconds == failure_seconds
        )
        postfault_only = bool(
            is_faulty
            and has_any_alarm
            and not has_eligible_alarm
        )
        records.append(
            {
                module_col: module_id,
                "is_faulty": int(is_faulty),
                "module_prediction": module_prediction,
                "outcome": outcome,
                "first_failure_timestamp": failure_seconds,
                "first_alarm_timestamp": alarm_seconds,
                "first_any_alarm_timestamp": first_any_alarm_seconds,
                "has_any_alarm": int(has_any_alarm),
                "has_decision_eligible_alarm": int(has_eligible_alarm),
                "alarm_at_or_before_failure": int(is_hit),
                "alarm_at_failure": int(alarm_at_failure),
                "postfault_only_alarm": int(postfault_only),
                "alarm_row_count": int(group_decisions.sum()),
                "decision_eligible_alarm_row_count": int(eligible_decisions.sum()),
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
                "first_any_alarm_timestamp",
                "has_any_alarm",
                "has_decision_eligible_alarm",
                "alarm_at_or_before_failure",
                "alarm_at_failure",
                "postfault_only_alarm",
                "alarm_row_count",
                "decision_eligible_alarm_row_count",
                "lead_hour",
            ]
        )
    outcomes = detail["outcome"]
    tp = int((outcomes == "TP").sum())
    fp = int((outcomes == "FP").sum())
    fn = int((outcomes == "FN").sum())
    tn = int((outcomes == "TN").sum())
    faulty_count = tp + fn
    normal_count = fp + tn
    module_count = tp + fp + fn + tn
    predicted_positive = tp + fp

    precision = _safe_ratio(tp, predicted_positive)
    recall = _safe_ratio(tp, faulty_count)
    f1_score = _safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = _safe_ratio(tp + tn, module_count)
    hit_leads = pd.to_numeric(
        detail.loc[outcomes == "TP", "lead_hour"], errors="coerce"
    ).dropna()
    lead_sum_hour = float(hit_leads.sum()) if len(hit_leads) else 0.0
    avg_lead_hour = _safe_ratio(lead_sum_hour, faulty_count)
    # Equality hits contribute zero and therefore correctly force the minimum
    # hit lead to zero when present.
    min_lead_hour = float(hit_leads.min()) if len(hit_leads) else 0.0
    avg_lead_score = math.tanh(avg_lead_hour)
    min_lead_score = math.tanh(min_lead_hour)
    final_score = f1_score + accuracy + avg_lead_score + min_lead_score

    metrics: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "all_hit_cnt": tp,
        "all_predict_pos_cnt": predicted_positive,
        "all_true_pos_cnt": faulty_count,
        "lead_pred_cnt": tp,
        "module_count": module_count,
        "faulty_module_count": faulty_count,
        "normal_module_count": normal_count,
        "normal_module_false_alarm_rate": _safe_ratio(fp, normal_count),
        "at_failure_hit_count": int(detail["alarm_at_failure"].sum()),
        "postfault_only_alarm_count": int(detail["postfault_only_alarm"].sum()),
    }
    return metrics, detail


def evaluate_legacy_inclusive(
    frame: pd.DataFrame,
    threshold: float = 0.5,
    *,
    module_col: str = "file_name",
    timestamp_col: str = "timestamp",
    anomaly_col: str = "anomaly",
    score_col: str = "score",
    predict_col: str | None = None,
) -> LegacyEvaluationResult:
    """Evaluate full-module first crossings under ``legacy_inclusive_v1``."""

    work = _prepare_frame(
        frame,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
    )
    decisions, _, source = _decisions(
        work,
        threshold=threshold,
        score_col=score_col,
        predict_col=predict_col,
    )
    metrics, detail = _module_metrics(work, decisions, module_col=module_col)
    metrics["decision_threshold"] = float(threshold)
    metrics["decision_source"] = source
    return LegacyEvaluationResult(metrics=metrics, detail=detail)


def threshold_candidates(
    scores: Sequence[float],
    *,
    fixed_threshold: float = 0.5,
    grid_size: int = 201,
    quantile_count: int = 201,
) -> np.ndarray:
    """Build deterministic validation candidates from every scored row."""

    if grid_size < 2 or quantile_count < 2:
        raise ValueError("grid_size and quantile_count must be at least 2")
    if not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("fixed_threshold must be in [0, 1]")
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) and ((values < 0.0).any() or (values > 1.0).any()):
        raise ValueError("threshold search scores must be in [0, 1]")
    parts = [
        np.linspace(0.0, 1.0, grid_size, dtype=np.float64),
        np.asarray([0.5, fixed_threshold], dtype=np.float64),
    ]
    if len(values):
        parts.append(
            np.quantile(values, np.linspace(0.0, 1.0, quantile_count))
        )
    return np.unique(np.concatenate(parts))


def _candidate_table(
    work: pd.DataFrame,
    scores: pd.Series,
    candidates: np.ndarray,
    *,
    module_col: str,
) -> pd.DataFrame:
    """Evaluate all candidates in one module scan, exactly as the direct path."""

    descending = candidates[::-1]
    module_positive: list[np.ndarray] = []
    module_lead: list[np.ndarray] = []
    module_faulty: list[bool] = []

    for _, group in work.groupby(module_col, sort=False, observed=True):
        failure_mask = group[_ANOMALY] > 0
        is_faulty = bool(failure_mask.any())
        first_fault_position = (
            int(np.flatnonzero(failure_mask.to_numpy(dtype=bool))[0])
            if is_faulty
            else None
        )
        failure_seconds = (
            int(group.iloc[first_fault_position][_TIMESTAMP_SECONDS])
            if first_fault_position is not None
            else None
        )
        group_scores = scores.loc[group.index].to_numpy(dtype=np.float64)
        group_times = group[_TIMESTAMP_SECONDS].to_numpy(dtype=np.int64)
        finite = np.isfinite(group_scores)
        if first_fault_position is not None:
            finite[first_fault_position + 1 :] = False
        score_values = group_scores[finite]
        time_values = group_times[finite]
        order = np.argsort(-score_values, kind="mergesort")
        ordered_scores = score_values[order]
        ordered_times = time_values[order]

        has_alarm = np.zeros(len(descending), dtype=bool)
        first_alarm = np.full(len(descending), np.iinfo(np.int64).max, dtype=np.int64)
        pointer = 0
        earliest = np.iinfo(np.int64).max
        for index, threshold in enumerate(descending):
            while pointer < len(ordered_scores) and ordered_scores[pointer] >= threshold:
                earliest = min(earliest, int(ordered_times[pointer]))
                pointer += 1
            if earliest != np.iinfo(np.int64).max:
                has_alarm[index] = True
                first_alarm[index] = earliest

        if is_faulty and failure_seconds is not None:
            positive = has_alarm & (first_alarm <= failure_seconds)
            lead = np.where(
                positive,
                (failure_seconds - first_alarm).astype(np.float64) / 3600.0,
                0.0,
            )
        else:
            positive = has_alarm
            lead = np.zeros(len(descending), dtype=np.float64)
        module_positive.append(positive[::-1])
        module_lead.append(lead[::-1])
        module_faulty.append(is_faulty)

    if not module_positive:
        raise ValueError("validation frame must contain at least one module")
    prediction = np.stack(module_positive, axis=0)
    leads = np.stack(module_lead, axis=0)
    faulty = np.asarray(module_faulty, dtype=bool)[:, None]
    tp = (prediction & faulty).sum(axis=0).astype(np.float64)
    fp = (prediction & ~faulty).sum(axis=0).astype(np.float64)
    fn = (~prediction & faulty).sum(axis=0).astype(np.float64)
    tn = (~prediction & ~faulty).sum(axis=0).astype(np.float64)
    predicted_positive = tp + fp
    faulty_count = tp + fn
    precision = np.divide(
        tp, predicted_positive, out=np.zeros_like(tp), where=predicted_positive > 0
    )
    recall = np.divide(tp, faulty_count, out=np.zeros_like(tp), where=faulty_count > 0)
    f1_score = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    module_count = float(len(module_faulty))
    accuracy = (tp + tn) / module_count
    hit_mask = prediction & faulty
    hit_leads = np.where(hit_mask, leads, 0.0)
    lead_sum = hit_leads.sum(axis=0)
    avg_lead = np.divide(
        lead_sum,
        faulty_count,
        out=np.zeros_like(lead_sum),
        where=faulty_count > 0,
    )
    min_lead = np.min(np.where(hit_mask, leads, np.inf), axis=0)
    min_lead = np.where(np.isfinite(min_lead), min_lead, 0.0)
    final_score = f1_score + accuracy + np.tanh(avg_lead) + np.tanh(min_lead)
    return pd.DataFrame(
        {
            "threshold": candidates,
            "final_score": final_score,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "avg_lead_hour": avg_lead,
            "min_lead_hour": min_lead,
        }
    )


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
    split_name: str = "validation",
) -> ThresholdSearchResult:
    """Select a threshold from validation only, including first-failure rows."""

    if split_name != "validation":
        raise ValueError("threshold selection is validation-only")
    work = _prepare_frame(
        validation_frame,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
    )
    scores = _probabilities(work, score_col)
    # Candidate values follow the same positional prefix as module decisions:
    # include the first failure row, exclude every later faulty-module row.
    decision_rows = _legacy_decision_row_mask(work, module_col=module_col)
    candidates = threshold_candidates(
        scores.loc[decision_rows & scores.notna()].to_numpy(dtype=np.float64),
        fixed_threshold=fixed_threshold,
        grid_size=grid_size,
        quantile_count=quantile_count,
    )
    table = _candidate_table(
        work, scores, candidates, module_col=module_col
    ).sort_values(
        ["final_score", "f1_score", "precision", "threshold"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=np.int64))
    table["selected"] = False
    table.loc[0, "selected"] = True
    table["selection_split"] = "validation"
    best_threshold = float(table.loc[0, "threshold"])
    best = evaluate_legacy_inclusive(
        validation_frame,
        threshold=best_threshold,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
        score_col=score_col,
    )
    return ThresholdSearchResult(
        threshold=best_threshold,
        metrics=best.metrics,
        detail=best.detail,
        table=table,
    )


def write_module_predictions(
    frame: pd.DataFrame,
    output_dir: Path | str,
    *,
    threshold: float = 0.5,
    module_col: str = "file_name",
    timestamp_col: str = "timestamp",
    anomaly_col: str = "anomaly",
    score_col: str = "score",
    predict_col: str | None = None,
    overwrite: bool,
) -> list[Path]:
    """Write exact ``timestamp,score,predict`` CSVs, one per module.

    ``overwrite`` is intentionally explicit.  When true, stale CSVs directly
    inside ``output_dir`` are removed after every new target has been validated.
    """

    work = _prepare_frame(
        frame,
        module_col=module_col,
        timestamp_col=timestamp_col,
        anomaly_col=anomaly_col,
    )
    decisions, scores, _ = _decisions(
        work,
        threshold=threshold,
        score_col=score_col,
        predict_col=predict_col,
    )
    output = Path(output_dir)
    targets: dict[str, Path] = {}
    seen: set[Path] = set()
    for module_id in work[module_col].drop_duplicates():
        base_name = Path(str(module_id)).name
        if not base_name:
            raise ValueError(f"invalid module identifier: {module_id!r}")
        if Path(base_name).suffix.lower() != ".csv":
            base_name += ".csv"
        target = output / base_name
        if target in seen:
            raise ValueError(f"module identifiers collide at {base_name!r}")
        seen.add(target)
        targets[str(module_id)] = target

    existing = list(output.glob("*.csv")) if output.exists() else []
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing CSV: {existing[0]}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale in existing:
            stale.unlink()
    written: list[Path] = []
    for module_id, group in work.groupby(module_col, sort=False, observed=True):
        target = targets[str(module_id)]
        ordered = group.sort_values(_TIMESTAMP_SECONDS, kind="mergesort")
        result = pd.DataFrame(
            {
                "timestamp": ordered[_TIMESTAMP_SECONDS].to_numpy(dtype=np.int64),
                "score": scores.loc[ordered.index].to_numpy(dtype=np.float64),
                "predict": decisions.loc[ordered.index].to_numpy(dtype=np.int8),
            }
        )
        result.to_csv(target, index=False)
        written.append(target)
    return written


# Narrow compatibility aliases for callers shared with the v1 baseline.
OFPEvaluationResult = LegacyEvaluationResult
evaluate_ofp = evaluate_legacy_inclusive
evaluate_ofp_long_frame = evaluate_legacy_inclusive
write_ofp_module_csvs = write_module_predictions


__all__ = [
    "PROTOCOL_NAME",
    "LegacyEvaluationResult",
    "OFPEvaluationResult",
    "ThresholdSearchResult",
    "evaluate_legacy_inclusive",
    "evaluate_ofp",
    "evaluate_ofp_long_frame",
    "search_validation_threshold",
    "threshold_candidates",
    "write_module_predictions",
    "write_ofp_module_csvs",
]
