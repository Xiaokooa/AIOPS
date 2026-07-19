from __future__ import annotations

import math

import pandas as pd
import pytest

from drfp.evaluation import (
    evaluate_ofp,
    multi_horizon_timestamp_diagnostics,
    search_validation_threshold,
    write_ofp_module_csvs,
)


def _row(
    file_name: str,
    hour: float,
    anomaly: int,
    *,
    score: float = 0.0,
    predict: int = 0,
    deep: float = 0.0,
    rule: float = 0.0,
) -> dict[str, object]:
    return {
        "file_name": file_name,
        "timestamp": hour * 3600.0,
        "anomaly": anomaly,
        "score": score,
        "predict": predict,
        "deep_score": deep,
        "rule_score": rule,
    }


def test_original_ofp_strict_before_and_all_faulty_lead_denominator() -> None:
    frame = pd.DataFrame(
        [
            _row("fault_hit.csv", 40, 0, predict=1),
            _row("fault_hit.csv", 100, 1),
            _row("fault_same.csv", 50, 0),
            _row("fault_same.csv", 100, 1, predict=1),
            _row("fault_miss.csv", 50, 0),
            _row("fault_miss.csv", 100, 1),
            _row("normal_alarm.csv", 20, 0, predict=1),
            _row("normal_alarm.csv", 30, 0),
            _row("normal_clear.csv", 20, 0),
        ]
    )

    result = evaluate_ofp(frame, predict_col="predict")
    metrics = result.metrics
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (
        1,
        1,
        2,
        1,
    )
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0 / 3.0)
    assert metrics["f1_score"] == pytest.approx(0.4)
    assert metrics["accuracy"] == pytest.approx(0.4)
    assert metrics["lead_sum_hour"] == pytest.approx(60.0)
    # The original protocol divides by all three faulty modules, not one hit.
    assert metrics["avg_lead_hour"] == pytest.approx(20.0)
    assert metrics["min_lead_hour"] == pytest.approx(60.0)
    assert metrics["hit_only_mean_lead_hour"] == pytest.approx(60.0)
    assert metrics["lead_bucket_24_72_count"] == 1
    assert metrics["final_score"] == pytest.approx(
        0.4 + 0.4 + math.tanh(20.0) + math.tanh(60.0)
    )

    by_module = result.detail.set_index("file_name")
    assert by_module.loc["fault_hit.csv", "outcome"] == "TP"
    # Alarm exactly at the first anomaly is not strictly early.
    assert by_module.loc["fault_same.csv", "outcome"] == "FN"
    assert by_module.loc["normal_alarm.csv", "outcome"] == "FP"


def test_faulty_module_with_only_post_failure_alarm_is_fn_not_fp() -> None:
    frame = pd.DataFrame(
        [
            _row("fault.csv", 10, 1),
            _row("fault.csv", 11, 0, predict=1),
            _row("normal.csv", 10, 0),
        ]
    )
    metrics, detail = evaluate_ofp(frame, predict_col="predict")
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (
        0,
        0,
        1,
        1,
    )
    assert detail.set_index("file_name").loc["fault.csv", "outcome"] == "FN"


def test_zero_positive_and_zero_prediction_cases_are_safe() -> None:
    frame = pd.DataFrame(
        [
            _row("n1.csv", 1, 0, score=0.1),
            _row("n2.csv", 1, 0, score=0.2),
        ]
    )
    metrics, _ = evaluate_ofp(frame, threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["avg_lead_hour"] == 0.0
    assert metrics["min_lead_hour"] == 0.0
    assert metrics["timestamp_pr_auc_120h"] == 0.0
    assert metrics["final_score"] == 1.0


def test_timestamp_diagnostics_exclude_failure_and_post_failure_rows() -> None:
    frame = pd.DataFrame(
        [
            _row("fault.csv", 0, 0, score=0.7),
            _row("fault.csv", 90, 0, score=0.8),
            _row("fault.csv", 100, 1, score=1.0),
            _row("fault.csv", 110, 0, score=1.0),
            _row("normal.csv", 0, 0, score=0.2),
        ]
    )
    metrics, _ = evaluate_ofp(frame, threshold=0.5)
    assert metrics["timestamp_valid_prefault_rows"] == 3
    assert metrics["timestamp_scored_valid_prefault_rows"] == 3
    assert metrics["timestamp_positive_rows_120h"] == 2
    assert metrics["timestamp_pr_auc_120h"] == pytest.approx(1.0)


def test_threshold_search_uses_highest_threshold_as_last_tie_break() -> None:
    frame = pd.DataFrame(
        [
            _row("n1.csv", 0, 0, score=0.1),
            _row("n2.csv", 0, 0, score=0.2),
        ]
    )
    search = search_validation_threshold(
        frame, grid_size=5, quantile_count=3
    )
    assert search.threshold == pytest.approx(1.0)
    assert search.table.loc[0, "selected"]
    assert 0.5 in set(search.table["threshold"])


def test_vectorized_threshold_search_matches_direct_evaluator() -> None:
    frame = pd.DataFrame(
        [
            _row("f1.csv", 0, 0, score=0.2),
            _row("f1.csv", 5, 0, score=0.8),
            _row("f1.csv", 10, 1, score=0.9),
            _row("f2.csv", 0, 0, score=0.4),
            _row("f2.csv", 10, 1, score=0.95),  # post-failure-only alarm at high t
            _row("n.csv", 0, 0, score=0.6),
        ]
    )
    search = search_validation_threshold(frame, grid_size=5, quantile_count=4)
    for row in search.table.itertuples(index=False):
        direct = evaluate_ofp(
            frame, threshold=float(row.threshold), include_diagnostics=False
        ).metrics
        for metric in ("final_score", "f1_score", "precision", "recall", "accuracy"):
            assert getattr(row, metric) == pytest.approx(direct[metric])


def test_branch_diagnostics_measure_rule_participation_and_disagreement() -> None:
    frame = pd.DataFrame(
        [
            _row("fault.csv", 0, 0, score=0.9, deep=0.1, rule=0.9),
            _row("fault.csv", 10, 1, score=0.0, deep=0.0, rule=0.0),
            _row("normal.csv", 0, 0, score=0.0, deep=0.8, rule=0.1),
        ]
    )
    metrics, _ = evaluate_ofp(frame, threshold=0.5)
    assert metrics["deep_rule_disagreement_rows"] == 2
    assert metrics["rule_only_positive_rows"] == 1
    assert metrics["deep_only_positive_rows"] == 1
    assert metrics["fusion_added_vs_deep_rows"] == 1
    assert metrics["fusion_positive_rule_only_support_rows"] == 1


def test_branch_diagnostics_accept_branch_specific_thresholds() -> None:
    frame = pd.DataFrame(
        [
            _row("fault.csv", 0, 0, score=0.8, deep=0.6, rule=0.8),
            _row("fault.csv", 10, 1),
            _row("normal.csv", 0, 0, score=0.0, deep=0.6, rule=0.8),
        ]
    )
    metrics, _ = evaluate_ofp(
        frame,
        threshold=0.5,
        branch_thresholds={"deep": 0.7, "rule": 0.9, "fusion": 0.5},
    )
    assert metrics["branch_deep_positive_rows"] == 0
    assert metrics["branch_rule_positive_rows"] == 0
    assert metrics["branch_fusion_positive_rows"] == 1


def test_multi_horizon_diagnostics_use_each_horizon_label() -> None:
    frame = pd.DataFrame(
        [
            {**_row("fault.csv", 0, 0), "p16": 0.1, "p120": 0.9},
            {**_row("fault.csv", 100, 1), "p16": 0.9, "p120": 0.9},
            {**_row("normal.csv", 0, 0), "p16": 0.2, "p120": 0.1},
        ]
    )
    metrics = multi_horizon_timestamp_diagnostics(
        frame, {16.0: "p16", 120.0: "p120"}
    )
    assert metrics["timestamp_positive_rows_16h"] == 0
    assert metrics["timestamp_positive_rows_120h"] == 1


def test_write_ofp_module_csvs_uses_threshold_and_compatible_columns(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            _row("a.csv", 2, 0, score=0.7),
            _row("a.csv", 1, 0, score=0.2),
            _row("b", 1, 0, score=0.5),
        ]
    )
    written = write_ofp_module_csvs(frame, tmp_path, threshold=0.5)
    assert {path.name for path in written} == {"a.csv", "b.csv"}
    a = pd.read_csv(tmp_path / "a.csv")
    b = pd.read_csv(tmp_path / "b.csv")
    assert list(a.columns) == ["timestamp", "predict"]
    assert a["timestamp"].tolist() == [3600.0, 7200.0]
    assert a["predict"].tolist() == [0, 1]
    assert b["predict"].tolist() == [1]
