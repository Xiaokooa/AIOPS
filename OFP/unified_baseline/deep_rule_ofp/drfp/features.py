"""Causal temporal and predictive-rule features for DRFP.

The raw/statistical registries and rule thresholds are intentionally reused
from the unified OFP baseline.  This module only changes how the legacy hard
rules are represented: every rule becomes a signed, normalized distance to
its decision boundary.  Positive values mean the rule is satisfied, negative
values mean it is not yet satisfied, and the complete history remains usable
for learning approach and persistence patterns.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ofp_unified.features import (
    CORRELATION_SPECS,
    CORRELATION_WINDOW,
    RAW_ALIASES,
    RAW_FEATURES,
    RULE_FEATURES,
    RULE_SPECS,
    STATISTICAL_FEATURES,
    STATISTICAL_OPERATIONS,
)


RULE_SIGNAL_COMPONENTS = (
    "current_margin",
    "approach_delta_12h",
    "positive_persistence_6h",
    "coverage_6h",
)
RULE_SUMMARY_NAMES = (
    "rule_summary_max_margin",
    "rule_summary_positive_fraction",
    "rule_summary_max_approach",
    "rule_summary_mean_coverage",
)
RULE_SIGNAL_NAMES = tuple(
    f"{rule_name}__{component}"
    for rule_name in RULE_FEATURES
    for component in RULE_SIGNAL_COMPONENTS
) + RULE_SUMMARY_NAMES
RULE_SIGNAL_DIM = len(RULE_SIGNAL_NAMES)


class RuleHistory(Protocol):
    timestamps: np.ndarray
    rule_margin: np.ndarray
    rule_valid: np.ndarray


class RawHistory(Protocol):
    timestamps: np.ndarray
    raw: np.ndarray
    raw_valid: np.ndarray


def _window_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Return fixed-width trailing sums without consulting future rows."""

    prefix = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64))
    )
    result = np.zeros(len(values), dtype=np.float64)
    if len(values) >= window:
        result[window - 1 :] = prefix[window:] - prefix[:-window]
    return result


def _expanding_channel_statistics(
    values: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the six Model2 expanding statistics from prefix sums only.

    Pandas' expanding skew/kurtosis implementation can change an already
    emitted prefix when an extreme future value is appended.  The formulas
    below use explicit cumulative raw moments, so output row ``i`` is a pure
    function of rows ``[:i+1]``.  Small-sample skew/kurtosis and standard
    deviation follow the unified baseline policy: they are zero once at
    least one observation exists but the statistic is not yet defined.
    """

    x = np.asarray(values, dtype=np.float64)
    observed = np.asarray(valid, dtype=bool) & np.isfinite(x)
    safe = np.where(observed, x, 0.0)
    count = np.cumsum(observed, dtype=np.int64)
    has_value = count > 0
    count_float = count.astype(np.float64)

    sum1 = np.cumsum(safe, dtype=np.float64)
    sum2 = np.cumsum(safe * safe, dtype=np.float64)
    sum3 = np.cumsum(safe * safe * safe, dtype=np.float64)
    sum4 = np.cumsum((safe * safe) * (safe * safe), dtype=np.float64)
    mean = np.divide(
        sum1,
        count_float,
        out=np.zeros_like(sum1),
        where=has_value,
    )

    minimum = np.minimum.accumulate(np.where(observed, x, np.inf))
    maximum = np.maximum.accumulate(np.where(observed, x, -np.inf))
    minimum = np.where(has_value, minimum, np.nan)
    maximum = np.where(has_value, maximum, np.nan)
    difference = maximum - minimum

    # Central moment sums expressed through causal raw-moment prefixes.
    moment2 = sum2 - sum1 * mean
    scale2 = np.abs(sum2) + np.abs(sum1 * mean)
    tolerance2 = np.finfo(np.float64).eps * np.maximum(scale2, 1.0) * 64.0
    moment2 = np.where(
        (moment2 < 0.0) & (moment2 >= -tolerance2), 0.0, moment2
    )
    moment2 = np.maximum(moment2, 0.0)
    moment3 = sum3 - 3.0 * mean * sum2 + 2.0 * count_float * mean**3
    moment4 = (
        sum4
        - 4.0 * mean * sum3
        + 6.0 * mean**2 * sum2
        - 3.0 * count_float * mean**4
    )

    std = np.zeros_like(mean)
    std_defined = count > 1
    std[std_defined] = np.sqrt(
        moment2[std_defined] / (count_float[std_defined] - 1.0)
    )

    skew = np.zeros_like(mean)
    skew_defined = (count > 2) & (moment2 > tolerance2)
    if skew_defined.any():
        n = count_float[skew_defined]
        skew[skew_defined] = (
            n * np.sqrt(n - 1.0)
            / (n - 2.0)
            * moment3[skew_defined]
            / np.power(moment2[skew_defined], 1.5)
        )

    kurt = np.zeros_like(mean)
    kurt_defined = (count > 3) & (moment2 > tolerance2)
    if kurt_defined.any():
        n = count_float[kurt_defined]
        fisher = (
            n * moment4[kurt_defined] / np.square(moment2[kurt_defined]) - 3.0
        )
        kurt[kurt_defined] = (
            (n - 1.0)
            / ((n - 2.0) * (n - 3.0))
            * ((n + 1.0) * fisher + 6.0)
        )

    # Match the original helper: once a channel has been observed, undefined
    # or numerically non-finite higher moments become zero rather than NaN.
    for higher_moment in (std, skew, kurt):
        higher_moment[has_value & ~np.isfinite(higher_moment)] = 0.0
        higher_moment[~has_value] = np.nan
    return {
        "Min": minimum,
        "Max": maximum,
        "Diff": difference,
        "Std": std,
        "Skew": skew,
        "Kurt": kurt,
    }


def build_causal_statistics(
    raw: np.ndarray,
    raw_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the locked 76-D statistical registry with strict causality.

    The first 72 dimensions are expanding Min/Max/Diff/Std/Skew/Kurt for the
    twelve raw channels.  The final four are five-row trailing Pearson
    correlations.  Missing rows do not enter expanding moments, while a
    correlation is valid only when both channels are present in all five
    rows.  The returned column order is exactly ``STATISTICAL_FEATURES``.
    """

    values = np.asarray(raw, dtype=np.float64)
    valid = np.asarray(raw_valid, dtype=bool)
    expected_shape = (len(values), len(RAW_FEATURES))
    if values.ndim != 2 or values.shape != expected_shape:
        raise ValueError(
            f"raw must have shape [time, {len(RAW_FEATURES)}]"
        )
    if valid.shape != values.shape:
        raise ValueError("raw_valid must have the same shape as raw")
    valid &= np.isfinite(values)

    by_name: dict[str, np.ndarray] = {}
    raw_index = {name: index for index, name in enumerate(RAW_FEATURES)}
    for raw_name in RAW_FEATURES:
        column = raw_index[raw_name]
        channel = _expanding_channel_statistics(
            values[:, column], valid[:, column]
        )
        alias = RAW_ALIASES[raw_name]
        for operation in STATISTICAL_OPERATIONS:
            by_name[f"Fe{alias}{operation}"] = channel[operation]

    window = int(CORRELATION_WINDOW)
    for name, left_name, right_name in CORRELATION_SPECS:
        left_column = raw_index[left_name]
        right_column = raw_index[right_name]
        pair_valid = valid[:, left_column] & valid[:, right_column]
        left = np.where(pair_valid, values[:, left_column], 0.0)
        right = np.where(pair_valid, values[:, right_column], 0.0)
        pair_count = _window_sum(pair_valid.astype(np.float64), window)
        left_sum = _window_sum(left, window)
        right_sum = _window_sum(right, window)
        left2_sum = _window_sum(left * left, window)
        right2_sum = _window_sum(right * right, window)
        cross_sum = _window_sum(left * right, window)
        covariance = cross_sum - left_sum * right_sum / float(window)
        left_variance = left2_sum - np.square(left_sum) / float(window)
        right_variance = right2_sum - np.square(right_sum) / float(window)
        variance_scale = np.maximum(
            np.abs(left2_sum) + np.square(left_sum) / float(window),
            np.abs(right2_sum) + np.square(right_sum) / float(window),
        )
        tolerance = (
            np.finfo(np.float64).eps * np.maximum(variance_scale, 1.0) * 64.0
        )
        correlation_valid = (
            (pair_count == float(window))
            & (left_variance > tolerance)
            & (right_variance > tolerance)
        )
        correlation = np.full(len(values), np.nan, dtype=np.float64)
        denominator = np.sqrt(
            left_variance[correlation_valid]
            * right_variance[correlation_valid]
        )
        correlation[correlation_valid] = (
            covariance[correlation_valid] / denominator
        )
        correlation[correlation_valid] = np.clip(
            correlation[correlation_valid], -1.0, 1.0
        )
        by_name[name] = correlation

    if tuple(by_name) != tuple(STATISTICAL_FEATURES):
        raise RuntimeError("statistical registry and causal builder disagree")
    stacked = np.column_stack([by_name[name] for name in STATISTICAL_FEATURES])
    with np.errstate(over="ignore", invalid="ignore"):
        statistics = stacked.astype(np.float32)
    stat_valid = np.isfinite(statistics)
    return statistics, stat_valid


def _rule_scale(inputs: tuple[str, ...], threshold: float) -> float:
    """Return a fixed physical scale without looking at dataset outcomes."""

    threshold = float(threshold)
    if threshold != 0.0:
        return abs(threshold)
    joined = " ".join(inputs)
    if "Temp" in joined:
        return 10.0
    if "Curr" in joined:
        return 1000.0
    return 100.0


def build_rule_margins(
    statistics: np.ndarray,
    stat_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert all legacy OFP rules to signed normalized margins.

    For an ``all_gt`` rule the margin is ``min(input - threshold) / scale``;
    for an ``all_lt`` rule it is ``min(threshold - input) / scale``.  A
    multi-input rule is therefore positive only when every predicate holds,
    exactly matching the legacy hard-rule semantics.  Extreme finite values
    are clipped to ``[-10, 10]`` so they cannot dominate the rule branch.
    """

    values = np.asarray(statistics, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(STATISTICAL_FEATURES):
        raise ValueError(
            "statistics must have shape [time, "
            f"{len(STATISTICAL_FEATURES)}]"
        )
    if stat_valid is None:
        valid_values = np.isfinite(values)
    else:
        valid_values = np.asarray(stat_valid, dtype=bool) & np.isfinite(values)
        if valid_values.shape != values.shape:
            raise ValueError("stat_valid must have the same shape as statistics")

    feature_index = {name: index for index, name in enumerate(STATISTICAL_FEATURES)}
    margins = np.zeros((len(values), len(RULE_SPECS)), dtype=np.float32)
    margin_valid = np.zeros_like(margins, dtype=bool)
    for rule_index, (_, inputs, comparison, threshold) in enumerate(RULE_SPECS):
        try:
            columns = np.asarray([feature_index[name] for name in inputs], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(f"rule references unknown statistic {error.args[0]!r}") from error
        selected = values[:, columns]
        selected_valid = valid_values[:, columns].all(axis=1)
        if comparison == "all_gt":
            signed = selected - float(threshold)
        elif comparison == "all_lt":
            signed = float(threshold) - selected
        else:
            raise RuntimeError(f"unknown rule comparison={comparison!r}")
        scale = _rule_scale(tuple(inputs), float(threshold))
        rule_margin = np.min(signed, axis=1) / scale
        finite = selected_valid & np.isfinite(rule_margin)
        margins[finite, rule_index] = np.clip(
            rule_margin[finite], -10.0, 10.0
        ).astype(np.float32)
        margin_valid[:, rule_index] = finite
    return margins, margin_valid


def _latest_valid_at_or_before(
    timestamps: np.ndarray,
    valid: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Find one causal reference row per rule, or ``-1`` when unavailable."""

    last_position = int(np.searchsorted(timestamps, cutoff, side="right") - 1)
    references = np.full(valid.shape[1], -1, dtype=np.int64)
    if last_position < 0:
        return references
    prefix = valid[: last_position + 1]
    for rule_index in range(prefix.shape[1]):
        positions = np.flatnonzero(prefix[:, rule_index])
        if len(positions):
            references[rule_index] = int(positions[-1])
    return references


def endpoint_rule_signals(
    record: RuleHistory,
    endpoint_index: int,
    *,
    approach_hours: float = 12.0,
    persistence_hours: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the causal 156-D rule representation at one endpoint.

    Each rule contributes current margin, its change from the latest valid
    value at or before ``t-12h``, positive persistence in ``[t-6h, t]``, and
    valid-data coverage in that same recent interval.  Four cross-rule
    summaries are appended.  No value after the endpoint can be accessed.
    """

    timestamps = np.asarray(record.timestamps, dtype=np.float64)
    margins = np.asarray(record.rule_margin, dtype=np.float32)
    valid = np.asarray(record.rule_valid, dtype=bool) & np.isfinite(margins)
    endpoint_index = int(endpoint_index)
    if timestamps.ndim != 1 or len(timestamps) != len(margins):
        raise ValueError("rule history timestamps and margins are not aligned")
    if margins.shape != (len(timestamps), len(RULE_SPECS)) or valid.shape != margins.shape:
        raise ValueError("rule history has an unexpected shape")
    if not 0 <= endpoint_index < len(timestamps):
        raise IndexError("endpoint_index is outside the module")
    if approach_hours <= 0 or persistence_hours <= 0:
        raise ValueError("rule history durations must be positive")
    prefix_timestamps = timestamps[: endpoint_index + 1]
    if len(prefix_timestamps) > 1 and np.any(np.diff(prefix_timestamps) < 0):
        raise ValueError("timestamps must be non-decreasing")
    prefix_margins = margins[: endpoint_index + 1]
    prefix_valid = valid[: endpoint_index + 1]
    endpoint_ts = float(prefix_timestamps[-1])

    current = np.zeros(len(RULE_SPECS), dtype=np.float32)
    current_valid = prefix_valid[-1].copy()
    current[current_valid] = prefix_margins[-1, current_valid]

    approach = np.zeros(len(RULE_SPECS), dtype=np.float32)
    approach_valid = np.zeros(len(RULE_SPECS), dtype=bool)
    reference_rows = _latest_valid_at_or_before(
        prefix_timestamps,
        prefix_valid,
        endpoint_ts - float(approach_hours) * 3600.0,
    )
    for rule_index, reference_row in enumerate(reference_rows):
        if reference_row >= 0 and current_valid[rule_index]:
            approach[rule_index] = (
                current[rule_index] - prefix_margins[reference_row, rule_index]
            )
            approach_valid[rule_index] = True

    recent_start = endpoint_ts - float(persistence_hours) * 3600.0
    recent_first = int(np.searchsorted(prefix_timestamps, recent_start, side="left"))
    recent_margins = prefix_margins[recent_first:]
    recent_valid = prefix_valid[recent_first:]
    total_rows = len(recent_valid)
    valid_counts = recent_valid.sum(axis=0, dtype=np.int64)
    positive_counts = (recent_valid & (recent_margins > 0.0)).sum(
        axis=0, dtype=np.int64
    )
    persistence = np.divide(
        positive_counts,
        valid_counts,
        out=np.zeros(len(RULE_SPECS), dtype=np.float64),
        where=valid_counts > 0,
    ).astype(np.float32)
    persistence_valid = valid_counts > 0
    coverage = (
        valid_counts.astype(np.float32) / float(total_rows)
        if total_rows
        else np.zeros(len(RULE_SPECS), dtype=np.float32)
    )
    coverage_valid = np.full(len(RULE_SPECS), total_rows > 0, dtype=bool)

    per_rule = np.stack((current, approach, persistence, coverage), axis=1).reshape(-1)
    per_rule_valid = np.stack(
        (current_valid, approach_valid, persistence_valid, coverage_valid), axis=1
    ).reshape(-1)

    any_current = bool(current_valid.any())
    any_approach = bool(approach_valid.any())
    any_coverage = bool(coverage_valid.any())
    summaries = np.asarray(
        [
            float(np.max(current[current_valid])) if any_current else 0.0,
            float(np.mean(current[current_valid] > 0.0)) if any_current else 0.0,
            float(np.max(approach[approach_valid])) if any_approach else 0.0,
            float(np.mean(coverage[coverage_valid])) if any_coverage else 0.0,
        ],
        dtype=np.float32,
    )
    summary_valid = np.asarray(
        [any_current, any_current, any_approach, any_coverage], dtype=bool
    )
    signals = np.concatenate((per_rule.astype(np.float32), summaries))
    signal_valid = np.concatenate((per_rule_valid, summary_valid))
    if len(signals) != RULE_SIGNAL_DIM:
        raise RuntimeError("rule signal registry and implementation disagree")
    return signals, signal_valid


def causal_hourly_window(
    record: RawHistory,
    endpoint_index: int,
    *,
    history_hours: int = 168,
    grid_hours: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a module to a regular causal grid ending at the endpoint.

    At every grid timestamp, the most recent *row* at or before that timestamp
    is selected.  Invalid raw channels remain masked and are numerically zeroed.
    ``delta_hours`` is the age of the selected row with shape ``[steps, 1]``;
    a fully missing prefix receives ``history_hours + grid_hours``.
    """

    timestamps = np.asarray(record.timestamps, dtype=np.float64)
    raw = np.asarray(record.raw, dtype=np.float32)
    raw_valid = np.asarray(record.raw_valid, dtype=bool) & np.isfinite(raw)
    endpoint_index = int(endpoint_index)
    if timestamps.ndim != 1 or raw.shape != (len(timestamps), len(RAW_FEATURES)):
        raise ValueError("raw history has an unexpected shape")
    if raw_valid.shape != raw.shape:
        raise ValueError("raw_valid must have the same shape as raw")
    if not 0 <= endpoint_index < len(timestamps):
        raise IndexError("endpoint_index is outside the module")
    if history_hours <= 0 or grid_hours <= 0 or history_hours % grid_hours:
        raise ValueError("history_hours must be divisible by a positive grid_hours")
    prefix_timestamps = timestamps[: endpoint_index + 1]
    if len(prefix_timestamps) > 1 and np.any(np.diff(prefix_timestamps) < 0):
        raise ValueError("timestamps must be non-decreasing")

    steps = int(history_hours // grid_hours)
    grid_seconds = float(grid_hours) * 3600.0
    endpoint_ts = float(prefix_timestamps[-1])
    grid_timestamps = endpoint_ts - np.arange(steps - 1, -1, -1) * grid_seconds
    source_rows = np.searchsorted(prefix_timestamps, grid_timestamps, side="right") - 1

    values = np.zeros((steps, len(RAW_FEATURES)), dtype=np.float32)
    masks = np.zeros_like(values, dtype=bool)
    delta = np.full(
        (steps, 1),
        float(history_hours + grid_hours),
        dtype=np.float32,
    )
    available = source_rows >= 0
    if available.any():
        selected_rows = source_rows[available].astype(np.int64)
        selected_values = raw[selected_rows]
        selected_valid = raw_valid[selected_rows]
        values[available] = np.where(selected_valid, selected_values, 0.0)
        masks[available] = selected_valid
        ages = (
            grid_timestamps[available] - prefix_timestamps[selected_rows]
        ) / 3600.0
        delta[available, 0] = np.maximum(ages, 0.0).astype(np.float32)
    return values, masks, delta


__all__ = [
    "RAW_FEATURES",
    "STATISTICAL_FEATURES",
    "RULE_SPECS",
    "RULE_FEATURES",
    "RULE_SIGNAL_COMPONENTS",
    "RULE_SUMMARY_NAMES",
    "RULE_SIGNAL_NAMES",
    "RULE_SIGNAL_DIM",
    "build_causal_statistics",
    "build_rule_margins",
    "endpoint_rule_signals",
    "causal_hourly_window",
]
