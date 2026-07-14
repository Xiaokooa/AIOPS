from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgofp.evaluation import (  # noqa: E402
    OFP_ORIGINAL_PROTOCOL_NAME,
    evaluate_legacy_inclusive,
    evaluate_ofp_original_strict,
    metrics_from_module_decisions,
    search_validation_threshold,
    write_module_predictions,
)


def _protocol_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(module: str, values: list[tuple[int, int, float]]) -> None:
        for timestamp, anomaly, score in values:
            rows.append(
                {
                    "file_name": module,
                    "timestamp": np.int64(timestamp),
                    "anomaly": anomaly,
                    "score": score,
                }
            )

    base = 1_700_000_000
    add("same.csv", [(base, 0, 0.1), (base + 20, 1, 0.8)])
    add("pre.csv", [(base + 11, 0, 0.8), (base + 20, 1, 0.1)])
    add("post.csv", [(base + 20, 1, 0.1), (base + 30, 1, 0.8)])
    add("miss.csv", [(base, 0, 0.1), (base + 20, 1, 0.1)])
    add("normal_fp.csv", [(base, 0, 0.8), (base + 20, 0, 0.1)])
    add("normal_tn.csv", [(base, 0, 0.1), (base + 20, 0, 0.1)])
    return pd.DataFrame(rows)


def test_legacy_inclusive_full_module_outcomes_and_score() -> None:
    result = evaluate_legacy_inclusive(_protocol_frame(), threshold=0.8)
    metrics = result.metrics
    detail = result.detail.set_index("file_name")

    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (
        2,
        1,
        2,
        1,
    )
    assert detail.loc["same.csv", "outcome"] == "TP"
    assert detail.loc["same.csv", "lead_hour"] == 0.0
    assert detail.loc["same.csv", "alarm_at_failure"] == 1
    assert detail.loc["post.csv", "outcome"] == "FN"
    assert detail.loc["post.csv", "postfault_only_alarm"] == 1
    assert detail.loc["normal_fp.csv", "outcome"] == "FP"

    expected_lead_sum = 9.0 / 3600.0
    expected_avg = expected_lead_sum / 4.0
    assert metrics["lead_sum_hour"] == pytest.approx(expected_lead_sum)
    assert metrics["avg_lead_hour"] == pytest.approx(expected_avg)
    assert metrics["min_lead_hour"] == 0.0
    expected_final = metrics["f1_score"] + metrics["accuracy"] + math.tanh(
        expected_avg
    )
    assert metrics["final_score"] == pytest.approx(expected_final)


def test_original_ofp_strict_requires_alarm_before_failure_timestamp() -> None:
    strict = evaluate_ofp_original_strict(_protocol_frame(), threshold=0.8)
    metrics = strict.metrics
    detail = strict.detail.set_index("file_name")

    assert metrics["protocol"] == OFP_ORIGINAL_PROTOCOL_NAME
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (
        1,
        1,
        3,
        1,
    )
    assert detail.loc["pre.csv", "outcome"] == "TP"
    assert detail.loc["pre.csv", "lead_hour"] == pytest.approx(9.0 / 3600.0)
    assert detail.loc["same.csv", "outcome"] == "FN"
    assert pd.isna(detail.loc["same.csv", "first_alarm_timestamp"])
    assert detail.loc["same.csv", "first_any_alarm_timestamp"] == 1_700_000_020
    assert detail.loc["same.csv", "postfault_only_alarm"] == 1
    assert detail.loc["post.csv", "outcome"] == "FN"
    assert detail.loc["normal_fp.csv", "outcome"] == "FP"
    assert metrics["at_failure_hit_count"] == 0
    assert metrics["postfault_only_alarm_count"] == 2

    # The frozen comparison convention divides total hit lead by every faulty
    # module, not only by the single hit module.
    expected_lead_sum = 9.0 / 3600.0
    expected_avg = expected_lead_sum / 4.0
    assert metrics["lead_sum_hour"] == pytest.approx(expected_lead_sum)
    assert metrics["avg_lead_hour"] == pytest.approx(expected_avg)
    assert metrics["min_lead_hour"] == pytest.approx(expected_lead_sum)
    expected_final = (
        metrics["f1_score"]
        + metrics["accuracy"]
        + math.tanh(expected_avg)
        + math.tanh(expected_lead_sum)
    )
    assert metrics["final_score"] == pytest.approx(expected_final)

    # Legacy-Inclusive intentionally differs only at equality: same.csv is a
    # valid zero-lead hit there and remains covered by its existing evaluator.
    inclusive = evaluate_legacy_inclusive(_protocol_frame(), threshold=0.8)
    assert inclusive.detail.set_index("file_name").loc["same.csv", "outcome"] == "TP"
    assert inclusive.metrics["tp"] == 2


def test_pooled_metrics_are_recomputed_from_concatenated_strict_decisions() -> None:
    frame = _protocol_frame()
    fold_modules = (
        {"same.csv", "pre.csv", "normal_fp.csv"},
        {"post.csv", "normal_tn.csv"},
        {"miss.csv"},
    )
    fold_results = [
        evaluate_ofp_original_strict(
            frame.loc[frame["file_name"].isin(modules)].copy(), threshold=0.8
        )
        for modules in fold_modules
    ]
    pooled_detail = pd.DataFrame.from_records(
        [
            {**record, "fold": fold}
            for fold, result in enumerate(fold_results, start=1)
            for record in result.detail.to_dict(orient="records")
        ]
    )

    pooled = metrics_from_module_decisions(
        pooled_detail, protocol=OFP_ORIGINAL_PROTOCOL_NAME
    )
    direct = evaluate_ofp_original_strict(frame, threshold=0.8).metrics
    for name in (
        "final_score",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
        "avg_lead_hour",
        "min_lead_hour",
        "lead_sum_hour",
        "tp",
        "fp",
        "fn",
        "tn",
        "module_count",
    ):
        assert pooled[name] == pytest.approx(direct[name])

    fold_mean_f1 = float(
        np.mean([result.metrics["f1_score"] for result in fold_results])
    )
    assert pooled["f1_score"] != pytest.approx(fold_mean_f1)
    assert pooled["module_count"] == 6


def test_pooled_metric_input_rejects_protocol_inconsistent_decisions() -> None:
    inclusive = evaluate_legacy_inclusive(_protocol_frame(), threshold=0.8)
    with pytest.raises(ValueError, match="positive lead_hour"):
        metrics_from_module_decisions(
            inclusive.detail, protocol=OFP_ORIGINAL_PROTOCOL_NAME
        )
    with pytest.raises(ValueError, match="unsupported evaluation protocol"):
        metrics_from_module_decisions(inclusive.detail, protocol="unknown")

    malformed = inclusive.detail.copy()
    malformed.loc[malformed["outcome"] == "FN", "lead_hour"] = 1.0
    with pytest.raises(ValueError, match="non-TP"):
        metrics_from_module_decisions(malformed)


def test_timestamp_is_exact_int64_and_fractional_seconds_are_rejected() -> None:
    large = np.int64(4_294_967_301)
    frame = pd.DataFrame(
        {
            "file_name": ["m.csv"],
            "timestamp": [large],
            "anomaly": [1],
            "score": [1.0],
        }
    )
    result = evaluate_legacy_inclusive(frame, threshold=1.0)
    assert result.detail.loc[0, "first_failure_timestamp"] == int(large)
    assert result.detail.loc[0, "first_alarm_timestamp"] == int(large)

    bad = frame.copy()
    bad["timestamp"] = [1.25]
    with pytest.raises(ValueError, match="whole seconds"):
        evaluate_legacy_inclusive(bad)


def test_postfault_duplicate_timestamp_cannot_be_an_inclusive_hit_or_candidate() -> None:
    frame = pd.DataFrame(
        {
            "file_name": ["fault.csv", "fault.csv", "fault.csv", "normal.csv"],
            "timestamp": [100, 200, 200, 100],
            "anomaly": [0, 1, 1, 0],
            "score": [0.1, 0.2, 0.73, 0.1],
        }
    )
    result = evaluate_legacy_inclusive(frame, threshold=0.73)
    detail = result.detail.set_index("file_name")
    assert detail.loc["fault.csv", "outcome"] == "FN"
    assert detail.loc["fault.csv", "postfault_only_alarm"] == 1
    assert pd.isna(detail.loc["fault.csv", "first_alarm_timestamp"])
    assert detail.loc["fault.csv", "first_any_alarm_timestamp"] == 200

    search = search_validation_threshold(
        frame, grid_size=2, quantile_count=2, fixed_threshold=0.5
    )
    assert 0.73 not in set(np.round(search.table["threshold"], 12))


def test_module_timestamps_must_remain_chronological() -> None:
    frame = pd.DataFrame(
        {
            "file_name": ["m.csv", "m.csv"],
            "timestamp": [200, 100],
            "anomaly": [0, 1],
            "score": [0.1, 0.9],
        }
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        evaluate_legacy_inclusive(frame)


def test_threshold_search_includes_failure_row_and_is_validation_only() -> None:
    frame = pd.DataFrame(
        {
            "file_name": ["fault.csv", "fault.csv", "normal.csv"],
            "timestamp": [100, 200, 100],
            "anomaly": [0, 1, 0],
            "score": [0.1, 0.9, 0.8],
        }
    )
    selected = search_validation_threshold(
        frame, grid_size=2, quantile_count=2, fixed_threshold=0.5
    )
    assert selected.threshold == pytest.approx(0.9)
    assert selected.metrics["at_failure_hit_count"] == 1
    assert selected.table.loc[0, "selection_split"] == "validation"
    with pytest.raises(ValueError, match="validation-only"):
        search_validation_threshold(frame, split_name="test")


def test_optimized_threshold_table_matches_direct_evaluator() -> None:
    frame = _protocol_frame()
    search = search_validation_threshold(
        frame, grid_size=7, quantile_count=7, fixed_threshold=0.73
    )
    metric_names = (
        "final_score",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
        "avg_lead_hour",
        "min_lead_hour",
    )
    for row in search.table.itertuples(index=False):
        direct = evaluate_legacy_inclusive(frame, threshold=row.threshold).metrics
        for name in metric_names:
            assert getattr(row, name) == pytest.approx(direct[name], abs=1e-14)


def test_writer_preserves_int64_score_predict_and_cleans_stale_csv(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "file_name": ["m.csv", "m.csv"],
            "timestamp": np.asarray([4_294_967_301, 4_294_967_306], dtype=np.int64),
            "anomaly": [0, 1],
            "score": [0.25, 0.75],
        }
    )
    output = tmp_path / "predictions"
    output.mkdir()
    (output / "stale.csv").write_text("timestamp,score,predict\n1,0,0\n")

    written = write_module_predictions(
        frame, output, threshold=0.5, overwrite=True
    )
    assert [path.name for path in written] == ["m.csv"]
    assert not (output / "stale.csv").exists()
    loaded = pd.read_csv(output / "m.csv")
    assert loaded.columns.tolist() == ["timestamp", "score", "predict"]
    assert loaded["timestamp"].dtype == np.int64
    assert loaded["timestamp"].tolist() == [4_294_967_301, 4_294_967_306]
    assert loaded["score"].tolist() == [0.25, 0.75]
    assert loaded["predict"].tolist() == [0, 1]

    with pytest.raises(FileExistsError):
        write_module_predictions(frame, output, overwrite=False)
