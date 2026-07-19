"""Causal feature registry and builders for the strict OFP ablations.

Every model input belongs to exactly one of raw/statistical/expert.  Feature
engineering is reset for every module and uses only the current row and its
prefix, never future rows or labels.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "strict-causal-v1"
FEATURE_GROUPS = ("raw", "statistical", "expert")

RAW_FEATURES = (
    "temperature",
    "current",
    "currentTXPower",
    "currentRXPower",
    "currentMultiRXPower1",
    "currentMultiRXPower2",
    "currentMultiRXPower3",
    "currentMultiRXPower4",
    "currentMultiTXPower1",
    "currentMultiTXPower2",
    "currentMultiTXPower3",
    "currentMultiTXPower4",
)

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

# Model2 semantics: Diff is the expanding range Max-Min, not a first
# difference.  The operation order is part of the serialized feature schema.
STATISTICAL_OPERATIONS = ("Min", "Max", "Diff", "Std", "Skew", "Kurt")
UNIVARIATE_STATISTICAL_FEATURES = tuple(
    f"Fe{RAW_ALIASES[raw]}{operation}"
    for raw in RAW_FEATURES
    for operation in STATISTICAL_OPERATIONS
)

# These are trailing five-observation Pearson correlations from Model2.  They
# are statistical features because the formula is generic; only the pair list
# is domain-selected.
CORRELATION_WINDOW = 5
CORRELATION_SPECS = (
    ("FeCoCurrTemp", "current", "temperature"),
    ("FeCoCurrTxP0", "current", "currentTXPower"),
    ("FeCoCurrRxP0", "current", "currentRXPower"),
    ("FeCoTxP0RxP0", "currentTXPower", "currentRXPower"),
)
CORRELATION_FEATURES = tuple(name for name, _, _ in CORRELATION_SPECS)
STATISTICAL_FEATURES = UNIVARIATE_STATISTICAL_FEATURES + CORRELATION_FEATURES

# Four contemporaneous physical consistency relations used by Model2.
LANE_RELATION_SPECS = (
    (
        "FeTxP0-Max",
        "currentTXPower",
        (
            "currentMultiTXPower1",
            "currentMultiTXPower2",
            "currentMultiTXPower3",
            "currentMultiTXPower4",
        ),
        "max",
    ),
    (
        "FeTxP0-Min",
        "currentTXPower",
        (
            "currentMultiTXPower1",
            "currentMultiTXPower2",
            "currentMultiTXPower3",
            "currentMultiTXPower4",
        ),
        "min",
    ),
    (
        "FeRxP0-Max",
        "currentRXPower",
        (
            "currentMultiRXPower1",
            "currentMultiRXPower2",
            "currentMultiRXPower3",
            "currentMultiRXPower4",
        ),
        "max",
    ),
    (
        "FeRxP0-Min",
        "currentRXPower",
        (
            "currentMultiRXPower1",
            "currentMultiRXPower2",
            "currentMultiRXPower3",
            "currentMultiRXPower4",
        ),
        "min",
    ),
)
LANE_RELATION_FEATURES = tuple(name for name, _, _, _ in LANE_RELATION_SPECS)

# (output name, input statistic names, comparison, threshold).  Comparisons
# are strict, matching Model2 RuleModel.  Missing statistics never trigger.
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
RULE_FEATURES = tuple(name for name, _, _, _ in RULE_SPECS)
EXPERT_FEATURES = LANE_RELATION_FEATURES + RULE_FEATURES

FEATURE_SET_GROUPS = {
    "raw": ("raw",),
    "raw_statistical": ("raw", "statistical"),
    "raw_statistical_expert": ("raw", "statistical", "expert"),
}
FEATURE_SET_ALIASES = {
    "b0": "raw",
    "b1": "raw_statistical",
    "b2": "raw_statistical_expert",
}

# Preserve the 12 raw inputs exactly.  These sentinels are treated as missing
# only in the derived-feature view so they cannot poison expanding statistics.
MISSING_SENTINEL = -999.0
TEMPERATURE_OUTLIER_SENTINEL = -255.0


def normalize_feature_set(feature_set: str) -> str:
    normalized = FEATURE_SET_ALIASES.get(str(feature_set), str(feature_set))
    if normalized not in FEATURE_SET_GROUPS:
        raise ValueError(
            f"unknown feature_set={feature_set!r}; expected one of "
            f"{sorted(FEATURE_SET_GROUPS)}"
        )
    return normalized


def feature_names(feature_set: str) -> tuple[str, ...]:
    normalized = normalize_feature_set(feature_set)
    groups = FEATURE_SET_GROUPS[normalized]
    names: list[str] = []
    if "raw" in groups:
        names.extend(RAW_FEATURES)
    if "statistical" in groups:
        names.extend(STATISTICAL_FEATURES)
    if "expert" in groups:
        names.extend(EXPERT_FEATURES)
    if len(names) != len(set(names)):
        raise RuntimeError("feature schema contains duplicate names")
    return tuple(names)


def _numeric_raw(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"feature frame missing columns: {sorted(missing)}")
    return frame.loc[:, RAW_FEATURES].apply(pd.to_numeric, errors="coerce")


def _engineering_view(raw: pd.DataFrame) -> pd.DataFrame:
    clean = raw.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    clean = clean.mask(clean == MISSING_SENTINEL)
    clean.loc[:, "temperature"] = clean["temperature"].mask(
        clean["temperature"] == TEMPERATURE_OUTLIER_SENTINEL
    )
    return clean


def _finite_or_zero_when_observed(values: pd.Series, count: pd.Series) -> pd.Series:
    result = values.replace([np.inf, -np.inf], np.nan)
    return result.mask(result.isna() & count.gt(0), 0.0)


def _build_statistical_features(clean: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for raw_name in RAW_FEATURES:
        alias = RAW_ALIASES[raw_name]
        series = clean[raw_name]
        expanding = series.expanding(min_periods=1)
        count = expanding.count()
        minimum = expanding.min()
        maximum = expanding.max()
        std = _finite_or_zero_when_observed(expanding.std(ddof=1), count)
        skew = _finite_or_zero_when_observed(expanding.skew(), count)
        kurt = _finite_or_zero_when_observed(expanding.kurt(), count)
        channel_values = {
            "Min": minimum,
            "Max": maximum,
            "Diff": maximum - minimum,
            "Std": std,
            "Skew": skew,
            "Kurt": kurt,
        }
        for operation in STATISTICAL_OPERATIONS:
            values[f"Fe{alias}{operation}"] = channel_values[operation]

    for name, left, right in CORRELATION_SPECS:
        corr = clean[left].rolling(
            window=CORRELATION_WINDOW,
            min_periods=CORRELATION_WINDOW,
        ).corr(clean[right])
        values[name] = corr.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(values, index=clean.index).loc[:, STATISTICAL_FEATURES]


def _build_expert_features(clean: pd.DataFrame, statistics: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for name, center_name, lane_names, reduction in LANE_RELATION_SPECS:
        lanes = clean.loc[:, list(lane_names)]
        all_valid = clean[center_name].notna() & lanes.notna().all(axis=1)
        lane_reference = lanes.max(axis=1) if reduction == "max" else lanes.min(axis=1)
        values[name] = (clean[center_name] - lane_reference).where(all_valid)

    for name, inputs, comparison, threshold in RULE_SPECS:
        selected = statistics.loc[:, list(inputs)]
        valid = selected.notna().all(axis=1)
        if comparison == "all_lt":
            triggered = selected.lt(float(threshold)).all(axis=1)
        elif comparison == "all_gt":
            triggered = selected.gt(float(threshold)).all(axis=1)
        else:
            raise RuntimeError(f"unknown rule comparison={comparison!r}")
        values[name] = (valid & triggered).astype(np.float32)
    return pd.DataFrame(values, index=clean.index).loc[:, EXPERT_FEATURES]


def build_feature_frame(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Build one module's causal model matrix with a fixed column order."""

    normalized = normalize_feature_set(feature_set)
    groups = FEATURE_SET_GROUPS[normalized]
    raw = _numeric_raw(frame)
    parts = [raw]
    statistics: pd.DataFrame | None = None
    if "statistical" in groups or "expert" in groups:
        clean = _engineering_view(raw)
        statistics = _build_statistical_features(clean)
        if "statistical" in groups:
            parts.append(statistics)
        if "expert" in groups:
            parts.append(_build_expert_features(clean, statistics))
    result = pd.concat(parts, axis=1).loc[:, feature_names(normalized)]
    return result.astype(np.float32)


def _feature_metadata() -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {
        name: {"name": name, "group": "raw", "causal": True, "source": name}
        for name in RAW_FEATURES
    }
    for raw_name in RAW_FEATURES:
        alias = RAW_ALIASES[raw_name]
        for operation in STATISTICAL_OPERATIONS:
            name = f"Fe{alias}{operation}"
            formula = "expanding_max - expanding_min" if operation == "Diff" else f"expanding_{operation.lower()}"
            metadata[name] = {
                "name": name,
                "group": "statistical",
                "causal": True,
                "source": [raw_name],
                "formula": formula,
            }
    for name, left, right in CORRELATION_SPECS:
        metadata[name] = {
            "name": name,
            "group": "statistical",
            "causal": True,
            "source": [left, right],
            "formula": f"trailing_pearson_correlation(window={CORRELATION_WINDOW})",
        }
    for name, center, lanes, reduction in LANE_RELATION_SPECS:
        metadata[name] = {
            "name": name,
            "group": "expert",
            "causal": True,
            "source": [center, *lanes],
            "formula": f"{center} - lane_{reduction}",
        }
    for name, inputs, comparison, threshold in RULE_SPECS:
        symbol = "<" if comparison == "all_lt" else ">"
        metadata[name] = {
            "name": name,
            "group": "expert",
            "causal": True,
            "source": list(inputs),
            "formula": f"all(inputs {symbol} {threshold:g})",
        }
    return metadata


def feature_manifest(feature_set: str = "raw") -> dict[str, object]:
    normalized = normalize_feature_set(feature_set)
    names = feature_names(normalized)
    metadata = _feature_metadata()
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_set": normalized,
        "allowed_groups": list(FEATURE_GROUPS),
        "active_groups": list(FEATURE_SET_GROUPS[normalized]),
        "features": [metadata[name] for name in names],
        "group_counts": {
            group: sum(metadata[name]["group"] == group for name in names)
            for group in FEATURE_GROUPS
        },
        "feature_count": len(names),
        "invalid_value_policy": {
            "raw": "preserve numeric values exactly",
            "derived": {
                "all_channels": [MISSING_SENTINEL],
                "temperature_only": [TEMPERATURE_OUTLIER_SENTINEL],
                "replacement": "NaN handled by XGBoost missing branch",
            },
        },
        "statistical_operations": list(STATISTICAL_OPERATIONS),
        "correlation_window_rows": CORRELATION_WINDOW,
    }


def validate_feature_schema_version(configured_version: str | None) -> None:
    if configured_version is not None and configured_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"feature schema mismatch: config={configured_version!r}, "
            f"code={FEATURE_SCHEMA_VERSION!r}"
        )


def groups_for_features(names: Sequence[str]) -> tuple[str, ...]:
    """Return groups for tests/audits while rejecting unknown names."""

    metadata = _feature_metadata()
    unknown = [name for name in names if name not in metadata]
    if unknown:
        raise ValueError(f"unknown feature names: {unknown[:3]}")
    return tuple(str(metadata[name]["group"]) for name in names)
