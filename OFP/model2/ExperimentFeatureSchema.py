from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


NA_DEFAULT = -999.0

RAW_DDM_FEATURES = [
    "Temp",
    "Curr",
    "TxP0",
    "RxP0",
    "RxP1",
    "RxP2",
    "RxP3",
    "RxP4",
    "TxP1",
    "TxP2",
    "TxP3",
    "TxP4",
]
TIME_BASE_FEATURES = ["Ts", "TsDelta"]
PREFIX_CHANNELS = ["Ts", *RAW_DDM_FEATURES]
PREFIX_OPERATIONS = ["Min", "Diff", "Max", "Skew", "Kurt", "Std"]

CORRELATION_FEATURES = [
    "FeCoCurrTemp",
    "FeCoCurrTxP0",
    "FeCoCurrRxP0",
    "FeCoTxP0RxP0",
]
LANE_CONSISTENCY_FEATURES = [
    "FeTxP0-Max",
    "FeTxP0-Min",
    "FeRxP0-Max",
    "FeRxP0-Min",
]


def _prefix_feature(channel: str, operation: str) -> str:
    return f"Fe{channel}{operation}"


PREFIX_FEATURES = [
    _prefix_feature(channel, operation)
    for channel in PREFIX_CHANNELS
    for operation in PREFIX_OPERATIONS
]
TIME_PREFIX_FEATURES = [
    _prefix_feature("Ts", operation)
    for operation in PREFIX_OPERATIONS
]
DDM_PREFIX_FEATURES = [
    _prefix_feature(channel, operation)
    for channel in RAW_DDM_FEATURES
    for operation in PREFIX_OPERATIONS
]

RULE_FEATURES = [
    "RuTempMin",
    "RuCurrMin",
    "RuTempDiff",
    "RuCurrDiff",
    "RuTempStd",
    "RuCurrStd",
    "RuTempKurt",
    "RuTxP0Min",
    "RuRxP0Min",
    "RuTxP0Diff",
    "RuRxP0Diff",
    "RuTxP0Std",
    "RuRxP0Std",
    "RuTxRxP0Kurt",
    "RuRxP1Min",
    "RuRxP2Min",
    "RuRxP1Diff",
    "RuRxP2Diff",
    "RuRxP1Std",
    "RuRxP2Std",
    "RuRxP3Min",
    "RuRxP4Min",
    "RuRxP3Diff",
    "RuRxP4Diff",
    "RuRxP3Std",
    "RuRxP4Std",
    "RuTxP1Min",
    "RuTxP2Min",
    "RuTxP1Diff",
    "RuTxP2Diff",
    "RuTxP1Std",
    "RuTxP2Std",
    "RuTxP3Min",
    "RuTxP4Min",
    "RuTxP3Diff",
    "RuTxP4Diff",
    "RuTxP3Std",
    "RuTxP4Std",
]


def deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


OFP_ENGINEERED_FEATURES = deduplicate(
    [
        *CORRELATION_FEATURES,
        *LANE_CONSISTENCY_FEATURES,
        "TsDelta",
        *PREFIX_FEATURES,
        *RULE_FEATURES,
    ]
)


def _numeric(frame: pd.DataFrame, name: str, default: float = NA_DEFAULT) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).astype(float)


def _rolling_corr(left: pd.Series, right: pd.Series, window: int, fill: float) -> pd.Series:
    corr = left.rolling(window=window, min_periods=window).corr(right)
    if len(corr):
        corr.iloc[: max(window - 1, 0)] = fill
    return corr.replace([np.inf, -np.inf], np.nan).fillna(NA_DEFAULT).astype(float)


def add_ofp_expert_stat_features(
    frame: pd.DataFrame,
    correlation_window: int = 5,
    na_default: float = NA_DEFAULT,
) -> pd.DataFrame:
    """Add the cumulative statistics, OFP relations, and deterministic rule triggers.

    The input uses the normalized OFP column names (Ts, Temp, Curr, TxP0/RxP0,
    and lane-level Tx/Rx power). Features are computed before normalization so
    the deterministic thresholds retain their original physical scale.
    """

    out = frame.copy()
    for name in ["Ts", *RAW_DDM_FEATURES]:
        if name not in out.columns:
            out[name] = float(na_default)
        out[name] = pd.to_numeric(out[name], errors="coerce").fillna(float(na_default))

    if "TsDelta" not in out.columns:
        out["TsDelta"] = out["Ts"].diff().fillna(0.0)

    engineered: dict[str, pd.Series] = {}
    if "FeCoCurrTemp" not in out.columns:
        engineered["FeCoCurrTemp"] = _rolling_corr(out["Curr"], out["Temp"], correlation_window, 99.0)
    if "FeCoCurrTxP0" not in out.columns:
        engineered["FeCoCurrTxP0"] = _rolling_corr(out["Curr"], out["TxP0"], correlation_window, -99.0)
    if "FeCoCurrRxP0" not in out.columns:
        engineered["FeCoCurrRxP0"] = _rolling_corr(out["Curr"], out["RxP0"], correlation_window, -99.0)
    if "FeCoTxP0RxP0" not in out.columns:
        engineered["FeCoTxP0RxP0"] = _rolling_corr(out["TxP0"], out["RxP0"], correlation_window, 99.0)

    tx_lanes = ["TxP1", "TxP2", "TxP3", "TxP4"]
    rx_lanes = ["RxP1", "RxP2", "RxP3", "RxP4"]
    if "FeTxP0-Max" not in out.columns:
        engineered["FeTxP0-Max"] = out["TxP0"] - out[tx_lanes].max(axis=1)
    if "FeTxP0-Min" not in out.columns:
        engineered["FeTxP0-Min"] = out["TxP0"] - out[tx_lanes].min(axis=1)
    if "FeRxP0-Max" not in out.columns:
        engineered["FeRxP0-Max"] = out["RxP0"] - out[rx_lanes].max(axis=1)
    if "FeRxP0-Min" not in out.columns:
        engineered["FeRxP0-Min"] = out["RxP0"] - out[rx_lanes].min(axis=1)

    for channel in PREFIX_CHANNELS:
        series = _numeric(out, channel, float(na_default))
        expanding = series.expanding(min_periods=1)
        minimum = expanding.min()
        maximum = expanding.max()
        std = expanding.std().fillna(0.0)
        skew = expanding.skew().replace([np.inf, -np.inf], np.nan).fillna(float(na_default))
        kurt = expanding.kurt().replace([np.inf, -np.inf], np.nan).fillna(float(na_default))
        engineered[_prefix_feature(channel, "Min")] = minimum
        engineered[_prefix_feature(channel, "Diff")] = maximum - minimum
        engineered[_prefix_feature(channel, "Max")] = maximum
        engineered[_prefix_feature(channel, "Skew")] = skew
        engineered[_prefix_feature(channel, "Kurt")] = kurt
        engineered[_prefix_feature(channel, "Std")] = std

    if engineered:
        out = pd.concat([out, pd.DataFrame(engineered, index=out.index)], axis=1)

    rule_columns: dict[str, pd.Series] = {
        "RuTempMin": out["FeTempMin"] < 0.0,
        "RuCurrMin": out["FeCurrMin"] < 5000.0,
        "RuTempDiff": out["FeTempDiff"] > 100.0,
        "RuCurrDiff": out["FeCurrDiff"] > 6000.0,
        "RuTempStd": out["FeTempStd"] > 10.0,
        "RuCurrStd": out["FeCurrStd"] > 1500.0,
        "RuTempKurt": out["FeTempKurt"] > 500.0,
        "RuTxP0Min": out["FeTxP0Min"] < 0.0,
        "RuRxP0Min": out["FeRxP0Min"] < 0.0,
        "RuTxP0Diff": out["FeTxP0Diff"] > 1000.0,
        "RuRxP0Diff": out["FeRxP0Diff"] > 1000.0,
        "RuTxP0Std": out["FeTxP0Std"] > 500.0,
        "RuRxP0Std": out["FeRxP0Std"] > 500.0,
        "RuTxRxP0Kurt": (out["FeTxP0Kurt"] > 500.0) & (out["FeRxP0Kurt"] > 500.0),
    }
    for channel in ["RxP1", "RxP2", "RxP3", "RxP4"]:
        rule_columns[f"Ru{channel}Min"] = out[f"Fe{channel}Min"] < 0.0
        rule_columns[f"Ru{channel}Diff"] = out[f"Fe{channel}Diff"] > 1000.0
        rule_columns[f"Ru{channel}Std"] = out[f"Fe{channel}Std"] > 200.0
    for channel in ["TxP1", "TxP2", "TxP3", "TxP4"]:
        rule_columns[f"Ru{channel}Min"] = out[f"Fe{channel}Min"] < 0.0
        rule_columns[f"Ru{channel}Diff"] = out[f"Fe{channel}Diff"] > 1000.0
        rule_columns[f"Ru{channel}Std"] = out[f"Fe{channel}Std"] > 100.0
    rule_frame = pd.DataFrame(
        {
            name: rule_columns[name].fillna(False).astype(np.float32)
            for name in RULE_FEATURES
        },
        index=out.index,
    )
    out = pd.concat([out, rule_frame], axis=1)

    return out.replace([np.inf, -np.inf], np.nan).fillna(float(na_default))


def classify_feature(name: str) -> str:
    if name in RAW_DDM_FEATURES:
        return "raw_ddm"
    if name in TIME_BASE_FEATURES or name in TIME_PREFIX_FEATURES:
        return "time"
    if name in DDM_PREFIX_FEATURES:
        return "stat_prefix"
    if name in CORRELATION_FEATURES:
        return "stat_correlation"
    if name in LANE_CONSISTENCY_FEATURES:
        return "expert_lane"
    if name in RULE_FEATURES or name.startswith("Ru"):
        return "expert_rule"
    if name.endswith("InvalidFlag") or "RuleLike" in name or "Storm" in name:
        return "expert_pattern"
    if any(token in name for token in ("LaneRange", "LaneStd", "P0MinusLaneMean")):
        return "expert_lane_extended"
    if any(token in name for token in ("_r", "_exp_range", "_d1", "ElapsedHours", "DeltaSeconds")):
        return "stat_rolling"
    return "other"


GROUP_ALIASES = {
    "statistical": {"stat_prefix", "stat_correlation", "stat_rolling"},
    "expert": {"expert_lane", "expert_lane_extended", "expert_rule", "expert_pattern"},
    "expert_statistical": {
        "stat_prefix",
        "stat_correlation",
        "stat_rolling",
        "expert_lane",
        "expert_lane_extended",
        "expert_rule",
        "expert_pattern",
    },
    "all_engineered": {
        "time",
        "stat_prefix",
        "stat_correlation",
        "stat_rolling",
        "expert_lane",
        "expert_lane_extended",
        "expert_rule",
        "expert_pattern",
        "other",
    },
    "all": {
        "raw_ddm",
        "time",
        "stat_prefix",
        "stat_correlation",
        "stat_rolling",
        "expert_lane",
        "expert_lane_extended",
        "expert_rule",
        "expert_pattern",
        "other",
    },
}


def parse_group_list(groups: str | Sequence[str] | None) -> list[str]:
    if groups is None:
        return []
    if isinstance(groups, str):
        values = groups.replace(";", ",").split(",")
    else:
        values = list(groups)
    return [str(value).strip() for value in values if str(value).strip()]


def expand_groups(groups: str | Sequence[str] | None) -> set[str]:
    expanded: set[str] = set()
    for group in parse_group_list(groups):
        expanded.update(GROUP_ALIASES.get(group, {group}))
    return expanded


def feature_groups(feature_names: Sequence[str]) -> OrderedDict[str, list[str]]:
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for name in feature_names:
        group = classify_feature(str(name))
        grouped.setdefault(group, []).append(str(name))
    return grouped


def select_feature_names(
    feature_names: Sequence[str],
    include_groups: str | Sequence[str] | None,
    exclude_groups: str | Sequence[str] | None = None,
) -> list[str]:
    include = expand_groups(include_groups)
    exclude = expand_groups(exclude_groups)
    if not include:
        include = set(GROUP_ALIASES["expert_statistical"])
    selected = [
        str(name)
        for name in feature_names
        if classify_feature(str(name)) in include and classify_feature(str(name)) not in exclude
    ]
    return deduplicate(selected)


def feature_group_manifest(feature_names: Sequence[str]) -> dict[str, object]:
    grouped = feature_groups(feature_names)
    return {
        "groups": dict(grouped),
        "counts": {name: len(values) for name, values in grouped.items()},
        "total": len(feature_names),
    }
