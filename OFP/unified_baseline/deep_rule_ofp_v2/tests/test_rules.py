from __future__ import annotations

import numpy as np

from fgofp.features import RAW_FEATURES
from fgofp.rules import (
    RULE_FEATURE_COUNT,
    RULE_FEATURES,
    RULE_SCHEMA_VERSION,
    RULE_SPECS,
    build_causal_rule_history,
)


def _healthy_raw(rows: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.full((rows, len(RAW_FEATURES)), 100.0, dtype=np.float32)
    raw[:, RAW_FEATURES.index("temperature")] = 20.0
    raw[:, RAW_FEATURES.index("current")] = 6000.0
    return raw, np.ones_like(raw, dtype=bool)


def test_canonical_registry_contains_exactly_38_enabled_rules() -> None:
    assert RULE_SCHEMA_VERSION == "canonical-legacy-38-causal-v1"
    assert RULE_FEATURE_COUNT == 38
    assert len(RULE_FEATURES) == len(set(RULE_FEATURES))
    assert not any("Skew" in name for name in RULE_FEATURES)


def test_canonical_registry_is_an_exact_frozen_snapshot() -> None:
    assert RULE_SPECS == (
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
        (
            "RuTxRxP0Kurt",
            ("FeTxP0Kurt", "FeRxP0Kurt"),
            "all_gt",
            500.0,
        ),
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


def test_strict_current_min_rule_and_hard_or_match_threshold_semantics() -> None:
    raw, mask = _healthy_raw(3)
    current = RAW_FEATURES.index("current")
    raw[:, current] = [6000.0, 5000.0, 4999.0]

    history = build_causal_rule_history(raw, mask)
    rule = RULE_FEATURES.index("RuCurrMin")

    # Equality is deliberately not a hit; the first value below 5000 is.
    assert history.per_rule_hard[:, rule].tolist() == [False, False, True]
    assert history.hard_or.tolist() == [False, False, True]
    assert history.margin[0, rule] < 0.0
    assert history.margin[1, rule] == 0.0
    assert history.margin[2, rule] > 0.0


def test_missing_observation_does_not_enter_expanding_rule_statistics() -> None:
    raw, mask = _healthy_raw(2)
    current = RAW_FEATURES.index("current")
    raw[:, current] = [6000.0, 1000.0]
    mask[1, current] = False

    history = build_causal_rule_history(raw, mask)
    rule = RULE_FEATURES.index("RuCurrMin")

    assert history.valid[:, rule].tolist() == [True, True]
    assert history.per_rule_hard[:, rule].tolist() == [False, False]
    assert history.margin[1, rule] == history.margin[0, rule]


def test_appending_future_rows_cannot_change_emitted_rule_prefix() -> None:
    raw, mask = _healthy_raw(8)
    prefix = build_causal_rule_history(raw[:6], mask[:6])
    raw[6:, :] = -10000.0
    extended = build_causal_rule_history(raw, mask)

    assert np.array_equal(prefix.margin, extended.margin[:6])
    assert np.array_equal(prefix.valid, extended.valid[:6])
    assert np.array_equal(prefix.per_rule_hard, extended.per_rule_hard[:6])
    assert np.array_equal(prefix.hard_or, extended.hard_or[:6])


def test_long_low_variance_sequence_cannot_create_false_kurtosis_alarm() -> None:
    rows = 20_000
    raw, mask = _healthy_raw(rows)
    rng = np.random.default_rng(2026)
    for name, center in (
        ("temperature", 25.0),
        ("current", 6000.0),
        ("currentTXPower", 1000.0),
        ("currentRXPower", 1000.0),
    ):
        raw[:, RAW_FEATURES.index(name)] = center + rng.normal(0.0, 0.01, rows)

    history = build_causal_rule_history(raw, mask)
    temperature_kurt = RULE_FEATURES.index("RuTempKurt")
    tx_rx_kurt = RULE_FEATURES.index("RuTxRxP0Kurt")

    assert not history.per_rule_hard[:, temperature_kurt].any()
    assert not history.per_rule_hard[:, tx_rx_kurt].any()
    assert not history.hard_or.any()
