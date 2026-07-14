from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from fgofp.evaluation import (
    OFP_ORIGINAL_PROTOCOL_NAME,
    PROTOCOL_NAME,
    evaluate_legacy_inclusive,
    evaluate_ofp_original_strict,
    metrics_from_module_decisions,
)
from fgofp.threefold import aggregate_three_fold_results
from run_threefold import build_fold_command


def _fold_frame(fold: int) -> pd.DataFrame:
    if fold == 1:
        faulty_scores, normal_scores = (0.9, 0.1), (0.1, 0.1)
    elif fold == 2:
        faulty_scores, normal_scores = (0.1, 0.9), (0.8, 0.1)
    else:
        faulty_scores, normal_scores = (0.1, 0.1), (0.1, 0.1)
    return pd.DataFrame(
        {
            "file_name": [
                f"f{fold}_fault.csv",
                f"f{fold}_fault.csv",
                f"f{fold}_normal.csv",
                f"f{fold}_normal.csv",
            ],
            "timestamp": [100, 200, 100, 200],
            "anomaly": [0, 1, 0, 0],
            "score": [*faulty_scores, *normal_scores],
        }
    )


def _write_fold_artifacts(root: Path, fold: int, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = root / f"fold_{fold}"
    fold_dir.mkdir(parents=True)
    frame = _fold_frame(fold)
    legacy = evaluate_legacy_inclusive(frame, threshold=threshold)
    original = evaluate_ofp_original_strict(frame, threshold=0.5)
    legacy.detail.to_csv(fold_dir / "test_module_decisions.csv", index=False)
    original.detail.to_csv(
        fold_dir / "test_module_decisions_ofp_original.csv", index=False
    )
    pd.DataFrame(
        {
            "file_name": [f"f{fold}_fault.csv", f"f{fold}_normal.csv"],
            "folder_index": [fold, fold],
            "Label": [1, 0],
            "split": ["test", "test"],
        }
    ).to_csv(fold_dir / "split_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected": True,
                "selection_split": "validation",
                "threshold": threshold,
            }
        ]
    ).to_csv(fold_dir / "validation_threshold_search.csv", index=False)
    common = {
        "experiment_id": f"N1_fold{fold}",
        "variant": "native_s2s_ce_fgl",
        "test_fold": fold,
        "native_rows": True,
        "sequence_to_sequence": True,
        "teacher_used_at_inference": False,
        "decision_feature_count": 25,
        "student_parameters": 100,
        "teacher_training_parameters": 100,
        "seconds_total": float(fold),
    }
    pd.DataFrame(
        [
            {
                **common,
                "decision_threshold": threshold,
                "threshold_policy": "validation_selected",
                **legacy.metrics,
            }
        ]
    ).to_csv(fold_dir / "result.csv", index=False)
    pd.DataFrame(
        [
            {
                **common,
                "decision_threshold": 0.5,
                "threshold_policy": "fixed_0.5_model1_ofp_compatibility",
                **original.metrics,
            }
        ]
    ).to_csv(fold_dir / "result_ofp_original.csv", index=False)
    (fold_dir / "effective_config.json").write_text(
        json.dumps(
            {
                "experiment_name": f"N1_fold{fold}",
                "split": {"test_fold": fold, "seed": 42},
                "model": {"architecture": "unit"},
            }
        ),
        encoding="utf-8",
    )
    return legacy.detail, original.detail


def _write_complete_fixture(root: Path) -> Path:
    index_rows: list[dict[str, object]] = []
    thresholds = {1: 0.2, 2: 0.5, 3: 0.8}
    for fold in (1, 2, 3):
        _write_fold_artifacts(root, fold, thresholds[fold])
        index_rows.extend(
            (
                {"file_name": f"f{fold}_fault.csv", "folder_index": fold, "Label": 1},
                {"file_name": f"f{fold}_normal.csv", "folder_index": fold, "Label": 0},
            )
        )
    index_path = root / "index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return index_path


def test_threefold_aggregation_recomputes_exact_pooled_metrics(tmp_path: Path) -> None:
    index_rows: list[dict[str, object]] = []
    legacy_details: list[pd.DataFrame] = []
    original_details: list[pd.DataFrame] = []
    thresholds = {1: 0.2, 2: 0.5, 3: 0.8}
    for fold in (1, 2, 3):
        legacy, original = _write_fold_artifacts(tmp_path, fold, thresholds[fold])
        legacy_details.append(legacy)
        original_details.append(original)
        index_rows.extend(
            (
                {"file_name": f"f{fold}_fault.csv", "folder_index": fold, "Label": 1},
                {"file_name": f"f{fold}_normal.csv", "folder_index": fold, "Label": 0},
            )
        )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    comparison = aggregate_three_fold_results(
        tmp_path, index_path, experiment_id="unit", formal=False
    ).set_index("evaluation_protocol")
    legacy_pooled = pd.concat(
        [detail.dropna(axis=1, how="all") for detail in legacy_details],
        ignore_index=True,
    )
    original_pooled = pd.concat(
        [detail.dropna(axis=1, how="all") for detail in original_details],
        ignore_index=True,
    )
    expected_legacy = metrics_from_module_decisions(
        legacy_pooled, protocol=PROTOCOL_NAME
    )
    expected_original = metrics_from_module_decisions(
        original_pooled,
        protocol=OFP_ORIGINAL_PROTOCOL_NAME,
    )
    for name in ("final_score", "f1_score", "precision", "recall", "accuracy"):
        assert comparison.loc[PROTOCOL_NAME, name] == pytest.approx(
            expected_legacy[name]
        )
        assert comparison.loc[OFP_ORIGINAL_PROTOCOL_NAME, name] == pytest.approx(
            expected_original[name]
        )
    assert comparison.loc[PROTOCOL_NAME, "module_count"] == 6
    assert comparison.loc[PROTOCOL_NAME, "threshold_policy"] == "per_fold_validation_selected"
    assert comparison.loc[OFP_ORIGINAL_PROTOCOL_NAME, "decision_threshold"] == 0.5
    assert (tmp_path / "pooled" / "test_module_decisions_legacy_inclusive.csv").is_file()
    assert (tmp_path / "comparison.csv").is_file()


def test_threefold_accepts_per_fold_validation_safe_rule_sources(
    tmp_path: Path,
) -> None:
    index_path = _write_complete_fixture(tmp_path)
    expected_sources = {1: "rule_only", 2: "rule_guided_residual", 3: "rule_only"}
    for fold, source in expected_sources.items():
        fold_dir = tmp_path / f"fold_{fold}"
        result_path = fold_dir / "result.csv"
        result = pd.read_csv(result_path)
        residual_threshold = float(result.loc[0, "decision_threshold"])
        result.loc[0, "decision_source"] = source
        result.loc[0, "fallback_policy"] = "validation_safe_rule"
        if source == "rule_only":
            result.loc[0, "decision_threshold"] = 0.5
            result.loc[0, "threshold_policy"] = "validation_safe_rule_fallback"
        result.to_csv(result_path, index=False)
        pd.DataFrame(
            [
                {
                    "selection_split": "validation",
                    "candidate": "rule_only",
                    "decision_threshold": 0.5,
                    "selected": source == "rule_only",
                },
                {
                    "selection_split": "validation",
                    "candidate": "rule_guided_residual",
                    "decision_threshold": residual_threshold,
                    "selected": source == "rule_guided_residual",
                },
            ]
        ).to_csv(fold_dir / "validation_decision_selection.csv", index=False)
        legacy_selected = pd.read_csv(fold_dir / "test_module_decisions.csv")
        legacy_selected.to_csv(
            fold_dir
            / (
                "test_rule_only_module_decisions.csv"
                if source == "rule_only"
                else "test_residual_module_decisions.csv"
            ),
            index=False,
        )

        original_path = fold_dir / "result_ofp_original.csv"
        original = pd.read_csv(original_path)
        original.loc[0, "decision_source"] = source
        original.loc[0, "fallback_policy"] = "validation_safe_rule"
        if source == "rule_only":
            original.loc[0, "threshold_policy"] = "validation_safe_rule_fallback"
        original.to_csv(original_path, index=False)
        pd.DataFrame(
            [
                {
                    "selection_split": "validation",
                    "candidate": "rule_only",
                    "decision_threshold": 0.5,
                    "selected": source == "rule_only",
                },
                {
                    "selection_split": "validation",
                    "candidate": "rule_guided_residual",
                    "decision_threshold": 0.5,
                    "selected": source == "rule_guided_residual",
                },
            ]
        ).to_csv(
            fold_dir / "validation_decision_selection_ofp_original.csv",
            index=False,
        )
        original_selected = pd.read_csv(
            fold_dir / "test_module_decisions_ofp_original.csv"
        )
        original_selected.to_csv(
            fold_dir
            / (
                "test_rule_only_module_decisions_ofp_original.csv"
                if source == "rule_only"
                else "test_residual_module_decisions_ofp_original.csv"
            ),
            index=False,
        )

    comparison = aggregate_three_fold_results(
        tmp_path, index_path, experiment_id="safe", formal=False
    ).set_index("evaluation_protocol")

    assert comparison.loc[PROTOCOL_NAME, "threshold_policy"] == (
        "per_fold_validation_safe_rule"
    )
    for fold, source in expected_sources.items():
        assert comparison.loc[
            PROTOCOL_NAME, f"fold_{fold}_decision_source"
        ] == source
        assert comparison.loc[
            OFP_ORIGINAL_PROTOCOL_NAME, f"fold_{fold}_decision_source"
        ] == source

    tampered_path = tmp_path / "fold_1" / "test_rule_only_module_decisions.csv"
    tampered = pd.read_csv(tampered_path)
    tampered.loc[0, "outcome"] = "TN" if tampered.loc[0, "outcome"] != "TN" else "FP"
    tampered.to_csv(tampered_path, index=False)
    with pytest.raises(ValueError, match="declared candidate"):
        aggregate_three_fold_results(
            tmp_path, index_path, experiment_id="safe_tampered", formal=False
        )


def test_threefold_preserves_temporal_source_when_safe_fallback_is_disabled(
    tmp_path: Path,
) -> None:
    index_path = _write_complete_fixture(tmp_path)
    for fold in (1, 2, 3):
        for name in ("result.csv", "result_ofp_original.csv"):
            path = tmp_path / f"fold_{fold}" / name
            result = pd.read_csv(path)
            result.loc[0, "decision_source"] = "temporal_model"
            result.loc[0, "fallback_policy"] = "none"
            result.to_csv(path, index=False)

    comparison = aggregate_three_fold_results(
        tmp_path, index_path, experiment_id="temporal", formal=False
    )

    for row in comparison.itertuples(index=False):
        assert row.fold_1_decision_source == "temporal_model"
        assert row.fold_2_decision_source == "temporal_model"
        assert row.fold_3_decision_source == "temporal_model"


def test_threefold_rejects_model_source_that_conflicts_with_architecture(
    tmp_path: Path,
) -> None:
    index_path = _write_complete_fixture(tmp_path)
    config_path = tmp_path / "fold_1" / "effective_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["architecture"] = "causal_depthwise_tcn"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result_path = tmp_path / "fold_1" / "result.csv"
    result = pd.read_csv(result_path)
    result.loc[0, "decision_source"] = "rule_guided_residual"
    result.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="decision_source/model architecture mismatch"):
        aggregate_three_fold_results(
            tmp_path, index_path, experiment_id="mislabeled", formal=False
        )


def test_fold_command_is_n1_and_targets_one_outer_fold(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_dir=tmp_path / "out",
        config=tmp_path / "config.json",
        data_dir=tmp_path / "training",
        index_path=tmp_path / "index.csv",
        device="cuda",
        experiment_name="N1_full",
        batch_size=4,
        inference_batch_size=8,
        no_mixed_precision=False,
        smoke=False,
        overwrite=False,
    )
    command = build_fold_command(args, 2)
    assert command[command.index("--test-fold") + 1] == "2"
    assert command[command.index("--experiment-name") + 1] == "N1_full_fold2"
    assert str(tmp_path / "out" / "fold_2") in command
    assert "--disable-fgl" not in command
    assert "--smoke" not in command


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("test_fold", 1, "result/test_fold mismatch"),
        ("protocol", "legacy_inclusive_v1", "result/protocol mismatch"),
        (
            "decision_threshold",
            0.6,
            "result/decision_threshold mismatch",
        ),
        (
            "threshold_policy",
            "validation_selected",
            "result/threshold_policy mismatch",
        ),
    ),
)
def test_threefold_rejects_mislabeled_original_result(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    index_path = _write_complete_fixture(tmp_path)
    result_path = tmp_path / "fold_2" / "result_ofp_original.csv"
    result = pd.read_csv(result_path)
    result.loc[0, field] = invalid
    result.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match=message):
        aggregate_three_fold_results(tmp_path, index_path, formal=False)


@pytest.mark.parametrize(
    "result_name",
    ("result.csv", "result_ofp_original.csv"),
)
def test_threefold_rejects_fold_result_detail_metric_mismatch(
    tmp_path: Path,
    result_name: str,
) -> None:
    index_path = _write_complete_fixture(tmp_path)
    result_path = tmp_path / "fold_1" / result_name
    result = pd.read_csv(result_path)
    result.loc[0, "f1_score"] = float(result.loc[0, "f1_score"]) + 0.125
    result.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="result/detail metric mismatch"):
        aggregate_three_fold_results(tmp_path, index_path, formal=False)
