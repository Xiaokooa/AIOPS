"""Self-contained causal implementation of the canonical OFP hard rules.

The registry contains the 38 enabled rules from ``model2/Config.py``.  The
four disabled skew rules are intentionally excluded.  Every statistic at row
``t`` is computed from the current module's rows ``[:t + 1]`` only; no label,
future row, or dataset-level fitted quantity is used.

The original Model2 feature pipeline has historical missing-value and row
filtering quirks.  V2 instead uses the canonical native-row interpretation:
invalid observations do not enter expanding statistics and every original
row is retained.  The thresholds and hard OR semantics remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import RAW_FEATURES


RULE_SCHEMA_VERSION = "canonical-legacy-38-causal-v1"

RAW_ALIASES = {
    "temperature": "Temp",
    "current": "Curr",
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

# (rule name, statistic names, comparison, physical threshold).  Strict
# comparisons and the enabled set match Model2's RULE_SET.
RULE_SPECS = (
    ("RuTempMin", ("FeTempMin",), "all_lt", 0.0),
    ("RuCurrMin", ("FeCurrMin",), "all_lt", 5000.0),
    ("RuTempDiff", ("FeTempDiff",), "all_gt", 100.0),
    ("RuCurrDiff", ("FeCurrDiff",), "all_gt", 6000.0),
    ("RuTempStd", ("FeTempStd",), "all_gt", 10.0),
    ("RuCurrStd", ("FeCurrStd",), "all_gt", 1500.0),
    ("RuTempKurt", ("FeTempKurt",), "all_gt", 500.0),
    ("RuTxP0Min", ("FeTxP0Min",), "all_lt", 0.0),
    ("RuRxP0Min", ("FeRxP0Min",), "all_lt", 0.0),
    ("RuTxP0Diff", ("FeTxP0Diff",), "all_gt", 1000.0),
    ("RuRxP0Diff", ("FeRxP0Diff",), "all_gt", 1000.0),
    ("RuTxP0Std", ("FeTxP0Std",), "all_gt", 500.0),
    ("RuRxP0Std", ("FeRxP0Std",), "all_gt", 500.0),
    ("RuTxRxP0Kurt", ("FeTxP0Kurt", "FeRxP0Kurt"), "all_gt", 500.0),
    ("RuRxP1Min", ("FeRxP1Min",), "all_lt", 0.0),
    ("RuRxP2Min", ("FeRxP2Min",), "all_lt", 0.0),
    ("RuRxP1Diff", ("FeRxP1Diff",), "all_gt", 1000.0),
    ("RuRxP2Diff", ("FeRxP2Diff",), "all_gt", 1000.0),
    ("RuRxP1Std", ("FeRxP1Std",), "all_gt", 200.0),
    ("RuRxP2Std", ("FeRxP2Std",), "all_gt", 200.0),
    ("RuRxP3Min", ("FeRxP3Min",), "all_lt", 0.0),
    ("RuRxP4Min", ("FeRxP4Min",), "all_lt", 0.0),
    ("RuRxP3Diff", ("FeRxP3Diff",), "all_gt", 1000.0),
    ("RuRxP4Diff", ("FeRxP4Diff",), "all_gt", 1000.0),
    ("RuRxP3Std", ("FeRxP3Std",), "all_gt", 200.0),
    ("RuRxP4Std", ("FeRxP4Std",), "all_gt", 200.0),
    ("RuTxP1Min", ("FeTxP1Min",), "all_lt", 0.0),
    ("RuTxP2Min", ("FeTxP2Min",), "all_lt", 0.0),
    ("RuTxP1Diff", ("FeTxP1Diff",), "all_gt", 1000.0),
    ("RuTxP2Diff", ("FeTxP2Diff",), "all_gt", 1000.0),
    ("RuTxP1Std", ("FeTxP1Std",), "all_gt", 100.0),
    ("RuTxP2Std", ("FeTxP2Std",), "all_gt", 100.0),
    ("RuTxP3Min", ("FeTxP3Min",), "all_lt", 0.0),
    ("RuTxP4Min", ("FeTxP4Min",), "all_lt", 0.0),
    ("RuTxP3Diff", ("FeTxP3Diff",), "all_gt", 1000.0),
    ("RuTxP4Diff", ("FeTxP4Diff",), "all_gt", 1000.0),
    ("RuTxP3Std", ("FeTxP3Std",), "all_gt", 100.0),
    ("RuTxP4Std", ("FeTxP4Std",), "all_gt", 100.0),
)
RULE_FEATURES = tuple(spec[0] for spec in RULE_SPECS)
RULE_FEATURE_COUNT = len(RULE_FEATURES)


@dataclass(frozen=True)
class CausalRuleHistory:
    """Per-row continuous margins and exact deterministic rule decisions."""

    margin: np.ndarray
    valid: np.ndarray
    per_rule_hard: np.ndarray
    hard_or: np.ndarray


def _expanding_statistics(values: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Return stable causal Model2 expanding statistics.

    Pandas' expanding kernels use stable online moments.  This intentionally
    avoids subtracting large raw fourth-power prefix sums, which can create
    enormous false kurtosis on long, nearly constant telemetry sequences.
    """

    x = np.asarray(values, dtype=np.float64)
    observed = np.asarray(valid, dtype=bool) & np.isfinite(x)
    series = pd.Series(np.where(observed, x, np.nan), dtype=np.float64)
    expanding = series.expanding(min_periods=1)
    count = expanding.count().to_numpy(dtype=np.int64)
    has_value = count > 0
    minimum = expanding.min().to_numpy(dtype=np.float64)
    maximum = expanding.max().to_numpy(dtype=np.float64)
    std = expanding.std(ddof=1).to_numpy(dtype=np.float64)
    kurt = expanding.kurt().to_numpy(dtype=np.float64)
    for statistic in (std, kurt):
        statistic[has_value & ~np.isfinite(statistic)] = 0.0
        statistic[~has_value] = np.nan
    return {
        "Min": minimum,
        "Max": maximum,
        "Diff": maximum - minimum,
        "Std": std,
        "Kurt": kurt,
    }


def _rule_scale(inputs: tuple[str, ...], threshold: float) -> float:
    if float(threshold) != 0.0:
        return abs(float(threshold))
    joined = " ".join(inputs)
    if "Temp" in joined:
        return 10.0
    if "Curr" in joined:
        return 1000.0
    return 100.0


def build_causal_rule_history(
    raw: np.ndarray,
    raw_valid: np.ndarray,
) -> CausalRuleHistory:
    """Build 38 signed margins and the exact hard-rule OR at every row.

    Positive margins indicate a satisfied rule.  Invalid rules retain a
    numerical zero in ``margin`` and are distinguished by ``valid``.  The hard
    decision is calculated directly from strict predicates before margin
    clipping, so float clipping can never alter the RuleModel fallback.
    """

    values = np.asarray(raw, dtype=np.float32)
    observed = np.asarray(raw_valid, dtype=bool).copy()
    expected = (len(values), len(RAW_FEATURES))
    if values.ndim != 2 or values.shape != expected:
        raise ValueError(f"raw must have shape [time, {len(RAW_FEATURES)}]")
    if observed.shape != values.shape:
        raise ValueError("raw_valid must have the same shape as raw")
    if len(values) == 0:
        raise ValueError("rule history cannot be empty")
    observed &= np.isfinite(values)

    statistics: dict[str, np.ndarray] = {}
    for column, raw_name in enumerate(RAW_FEATURES):
        alias = RAW_ALIASES[raw_name]
        channel = _expanding_statistics(values[:, column], observed[:, column])
        for operation in ("Min", "Max", "Diff", "Std", "Kurt"):
            statistics[f"Fe{alias}{operation}"] = channel[operation].astype(
                np.float32
            )

    margins = np.zeros((len(values), RULE_FEATURE_COUNT), dtype=np.float32)
    valid = np.zeros_like(margins, dtype=bool)
    per_rule_hard = np.zeros_like(margins, dtype=bool)
    for rule_index, (_, inputs, comparison, threshold) in enumerate(RULE_SPECS):
        selected = np.column_stack([statistics[name] for name in inputs])
        selected_valid = np.isfinite(selected).all(axis=1)
        if comparison == "all_gt":
            predicate = selected > float(threshold)
            signed = selected - float(threshold)
        elif comparison == "all_lt":
            predicate = selected < float(threshold)
            signed = float(threshold) - selected
        else:  # pragma: no cover - protected by the frozen registry
            raise RuntimeError(f"unknown rule comparison={comparison!r}")
        hard = selected_valid & predicate.all(axis=1)
        scale = _rule_scale(tuple(inputs), float(threshold))
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = np.min(signed, axis=1) / scale
        finite = selected_valid & np.isfinite(normalized)
        margins[finite, rule_index] = np.clip(
            normalized[finite], -10.0, 10.0
        ).astype(np.float32)
        valid[:, rule_index] = finite
        per_rule_hard[:, rule_index] = hard

    return CausalRuleHistory(
        margin=margins,
        valid=valid,
        per_rule_hard=per_rule_hard,
        hard_or=per_rule_hard.any(axis=1),
    )


__all__ = [
    "CausalRuleHistory",
    "RULE_FEATURE_COUNT",
    "RULE_FEATURES",
    "RULE_SCHEMA_VERSION",
    "RULE_SPECS",
    "build_causal_rule_history",
]
