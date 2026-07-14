"""Auditable aggregation for the three outer OFP folds.

Each fold owns its model and decision threshold.  The pooled result is rebuilt
from the 13,372 out-of-fold module decisions; fold scalar metrics are never
averaged.  A second decision stream follows the root ``OFP/EvaluateResult.py``
compatibility protocol (strictly pre-failure, fixed threshold 0.5).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .evaluation import (
    OFP_ORIGINAL_PROTOCOL_NAME,
    PROTOCOL_NAME,
    metrics_from_module_decisions,
)


OUTER_FOLDS = (1, 2, 3)
EXPECTED_FULL_MODULES = 13_372
EXPECTED_FULL_FAULTY = 4_102
EXPECTED_FULL_NORMAL = 9_270
OFP_ORIGINAL_THRESHOLD = 0.5
OFP_ORIGINAL_THRESHOLD_POLICY = "fixed_0.5_model1_ofp_compatibility"
LEGACY_THRESHOLD_POLICY = "validation_selected"
SAFE_RULE_THRESHOLD_POLICY = "validation_safe_rule_fallback"
_AUDITED_METRICS = (
    "final_score",
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "avg_lead_hour",
    "min_lead_hour",
    "avg_lead_score",
    "min_lead_score",
    "lead_sum_hour",
    "tp",
    "fp",
    "fn",
    "tn",
    "all_hit_cnt",
    "all_predict_pos_cnt",
    "all_true_pos_cnt",
    "lead_pred_cnt",
    "module_count",
    "faulty_module_count",
    "normal_module_count",
    "normal_module_false_alarm_rate",
    "at_failure_hit_count",
    "postfault_only_alarm_count",
)


def _one_row(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"expected exactly one row in {path}, got {len(frame)}")
    return frame.iloc[0].to_dict()


def _selected_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def _normalized_fold_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.pop("experiment_name", None)
    split = dict(config.get("split", {}))
    split.pop("test_fold", None)
    config["split"] = split
    return config


def _validate_decision_source_architecture(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    source: Path,
) -> None:
    """Ensure a non-fallback source names the model that was actually built."""

    if "decision_source" not in result:
        return
    architecture = str(config.get("model", {}).get("architecture", ""))
    expected_by_architecture = {
        "causal_depthwise_tcn": "temporal_model",
        "rule_guided_residual_tcn": "rule_guided_residual",
    }
    expected = expected_by_architecture.get(architecture)
    decision_source = str(result["decision_source"])
    if expected is not None and decision_source not in {expected, "rule_only"}:
        raise ValueError(
            f"decision_source/model architecture mismatch in {source}: "
            f"architecture={architecture!r}, source={decision_source!r}"
        )


def _validate_threshold_provenance(fold_dir: Path, result: Mapping[str, Any]) -> None:
    source = str(result.get("decision_source", "rule_guided_residual"))
    selection_path = fold_dir / "validation_decision_selection.csv"
    if selection_path.is_file():
        selection = pd.read_csv(selection_path)
        if "selected" not in selection or "selection_split" not in selection:
            raise ValueError(f"decision selection audit columns missing in {fold_dir}")
        chosen = selection.loc[_selected_mask(selection["selected"])]
        if len(chosen) != 1 or str(chosen.iloc[0]["selection_split"]) != "validation":
            raise ValueError(f"invalid validation decision selection in {fold_dir}")
        row = chosen.iloc[0]
        if str(row["candidate"]) != source:
            raise ValueError(f"selected decision source/result mismatch in {fold_dir}")
        if not np.isclose(
            float(row["decision_threshold"]),
            float(result["decision_threshold"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"selected decision threshold/result mismatch in {fold_dir}")
        if source == "rule_only":
            return
    table = pd.read_csv(fold_dir / "validation_threshold_search.csv")
    if "selected" not in table or "selection_split" not in table:
        raise ValueError(f"threshold audit columns missing in {fold_dir}")
    selected = table.loc[_selected_mask(table["selected"])]
    if len(selected) != 1:
        raise ValueError(f"expected one selected validation threshold in {fold_dir}")
    row = selected.iloc[0]
    if str(row["selection_split"]) != "validation":
        raise ValueError(f"threshold in {fold_dir} was not selected on validation")
    if not np.isclose(
        float(row["threshold"]),
        float(result["decision_threshold"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"selected threshold/result mismatch in {fold_dir}")


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    source: Path,
    fold: int,
    protocol: str,
    threshold_policy: str | tuple[str, ...],
    threshold: float | None = None,
) -> None:
    if int(result.get("test_fold", -1)) != int(fold):
        raise ValueError(f"result/test_fold mismatch in {source}")
    if str(result.get("protocol", "")) != protocol:
        raise ValueError(f"result/protocol mismatch in {source}")
    expected_policies = (
        (threshold_policy,)
        if isinstance(threshold_policy, str)
        else tuple(threshold_policy)
    )
    if str(result.get("threshold_policy", "")) not in expected_policies:
        raise ValueError(f"result/threshold_policy mismatch in {source}")
    if threshold is not None and not np.isclose(
        float(result.get("decision_threshold", np.nan)),
        float(threshold),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"result/decision_threshold mismatch in {source}")


def _validate_result_metrics(
    result: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    source: Path,
) -> None:
    missing = [name for name in _AUDITED_METRICS if name not in result]
    if missing:
        raise ValueError(f"audited metrics missing in {source}: {missing}")
    for name in _AUDITED_METRICS:
        expected = float(recomputed[name])
        actual = float(result[name])
        if not np.isfinite(actual) or not np.isclose(
            actual,
            expected,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"result/detail metric mismatch in {source}: "
                f"{name}={actual!r}, recomputed={expected!r}"
            )


def _validate_detail_names(
    detail: pd.DataFrame,
    expected: set[str],
    *,
    source: Path,
) -> None:
    if "file_name" not in detail:
        raise ValueError(f"file_name missing from {source}")
    names = detail["file_name"].astype(str)
    if names.duplicated().any():
        raise ValueError(f"duplicate module decisions in {source}")
    actual = set(names)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"module coverage mismatch in {source}: "
            f"missing={missing[:1]} extra={extra[:1]}"
        )


def _validate_selected_detail_matches_candidate(
    selected: pd.DataFrame,
    candidate_path: Path,
    *,
    source: Path,
) -> None:
    """Prevent a declared decision source from pooling another branch's rows."""

    if not candidate_path.is_file():
        raise ValueError(f"selected candidate detail missing: {candidate_path}")
    candidate = pd.read_csv(candidate_path)
    if set(selected.columns) != set(candidate.columns):
        raise ValueError(f"selected/candidate detail columns differ in {source}")
    left = selected.sort_values("file_name", kind="mergesort").reset_index(drop=True)
    right = candidate.loc[:, left.columns].sort_values(
        "file_name", kind="mergesort"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(
            f"selected module decisions do not match declared candidate in {source}"
        ) from error


def _common_result_fields(fold_results: pd.DataFrame) -> dict[str, Any]:
    fields = (
        "variant",
        "native_rows",
        "sequence_to_sequence",
        "teacher_used_at_inference",
        "decision_feature_count",
        "student_parameters",
        "teacher_training_parameters",
    )
    common: dict[str, Any] = {}
    for name in fields:
        if name not in fold_results:
            continue
        values = fold_results[name].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"fold results disagree on {name}")
        common[name] = values.iloc[0]
    return common


def _comparison_row(
    *,
    experiment_id: str,
    protocol: str,
    metrics: Mapping[str, Any],
    fold_results: pd.DataFrame,
    thresholds: Mapping[int, float],
    formal: bool,
    decision_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    original = protocol == OFP_ORIGINAL_PROTOCOL_NAME
    decision_frame = fold_results if decision_rows is None else decision_rows
    safe_sources = {"rule_guided_residual", "temporal_model", "rule_only"}
    has_declared_sources = (
        "decision_source" in decision_frame.columns
        and set(decision_frame["decision_source"].astype(str)) <= safe_sources
    )
    safe_selection = (
        has_declared_sources
        and "fallback_policy" in decision_frame.columns
        and set(decision_frame["fallback_policy"].astype(str))
        == {"validation_safe_rule"}
    )
    sources = (
        decision_frame.set_index("outer_fold")["decision_source"].astype(str).to_dict()
        if has_declared_sources
        else {fold: "rule_guided_residual" for fold in OUTER_FOLDS}
    )
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "variant": "native_s2s_ce_fgl_3fold_pooled",
        "changed_axis": "evaluation_protocol",
        "training_protocol": PROTOCOL_NAME,
        "evaluation_protocol": protocol,
        "outer_folds": "1,2,3",
        "training_runs": 3,
        "formal_full_oof": bool(formal),
        "decision_threshold": 0.5 if original else np.nan,
        "threshold_policy": (
            "per_fold_validation_safe_rule"
            if safe_selection
            else (
                "fixed_0.5_model1_ofp_compatibility"
                if original
                else "per_fold_validation_selected"
            )
        ),
        "fold_1_threshold": 0.5 if original else thresholds[1],
        "fold_2_threshold": 0.5 if original else thresholds[2],
        "fold_3_threshold": 0.5 if original else thresholds[3],
        "fold_1_decision_source": sources[1],
        "fold_2_decision_source": sources[2],
        "fold_3_decision_source": sources[3],
        "seconds_total": float(
            pd.to_numeric(fold_results["seconds_total"], errors="raise").sum()
        ),
        **_common_result_fields(fold_results),
        **dict(metrics),
    }
    return row


def aggregate_three_fold_results(
    output_dir: Path | str,
    index_path: Path | str,
    *,
    experiment_id: str = "N1_native_s2s_ce_fgl_3fold",
    formal: bool = True,
) -> pd.DataFrame:
    """Validate three fold artifacts and write exact pooled module metrics."""

    output_dir = Path(output_dir)
    index = pd.read_csv(index_path)
    required = {"file_name", "folder_index", "Label"}
    missing_columns = required - set(index.columns)
    if missing_columns:
        raise ValueError(f"index missing columns: {sorted(missing_columns)}")
    index = index.copy()
    index["file_name"] = index["file_name"].astype(str)
    index["folder_index"] = pd.to_numeric(
        index["folder_index"], errors="raise"
    ).astype(int)
    if index["file_name"].duplicated().any():
        raise ValueError("index contains duplicate module names")
    if set(index["folder_index"]) != set(OUTER_FOLDS):
        raise ValueError("three-fold aggregation requires index folds 1, 2, 3")

    legacy_details: list[pd.DataFrame] = []
    original_details: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    original_fold_rows: list[dict[str, Any]] = []
    thresholds: dict[int, float] = {}
    normalized_configs: list[dict[str, Any]] = []

    for fold in OUTER_FOLDS:
        fold_dir = output_dir / f"fold_{fold}"
        fold_config = _normalized_fold_config(
            fold_dir / "effective_config.json"
        )
        expected = set(
            index.loc[index["folder_index"] == fold, "file_name"].astype(str)
        )
        split = pd.read_csv(fold_dir / "split_manifest.csv")
        split_test = split.loc[split["split"].astype(str) == "test", "file_name"]
        split_names = set(split_test.astype(str))
        if formal and split_names != expected:
            raise ValueError(f"fold {fold} split manifest is not the full outer test fold")

        result_path = fold_dir / "result.csv"
        result = _one_row(result_path)
        _validate_decision_source_architecture(
            result, fold_config, source=result_path
        )
        _validate_result_identity(
            result,
            source=result_path,
            fold=fold,
            protocol=PROTOCOL_NAME,
            threshold_policy=(
                LEGACY_THRESHOLD_POLICY,
                "fixed",
                SAFE_RULE_THRESHOLD_POLICY,
            ),
        )
        _validate_threshold_provenance(fold_dir, result)
        thresholds[fold] = float(result["decision_threshold"])
        result["outer_fold"] = fold
        fold_rows.append(result)

        original_result_path = fold_dir / "result_ofp_original.csv"
        original_result = _one_row(original_result_path)
        _validate_decision_source_architecture(
            original_result, fold_config, source=original_result_path
        )
        _validate_result_identity(
            original_result,
            source=original_result_path,
            fold=fold,
            protocol=OFP_ORIGINAL_PROTOCOL_NAME,
            threshold_policy=(
                OFP_ORIGINAL_THRESHOLD_POLICY,
                SAFE_RULE_THRESHOLD_POLICY,
            ),
            threshold=OFP_ORIGINAL_THRESHOLD,
        )
        original_selection_path = (
            fold_dir / "validation_decision_selection_ofp_original.csv"
        )
        if original_selection_path.is_file():
            selection = pd.read_csv(original_selection_path)
            selected = selection.loc[_selected_mask(selection["selected"])]
            if len(selected) != 1:
                raise ValueError(
                    f"expected one selected original decision in {fold_dir}"
                )
            selected_row = selected.iloc[0]
            source = str(
                original_result.get("decision_source", "rule_guided_residual")
            )
            if (
                str(selected_row["selection_split"]) != "validation"
                or str(selected_row["candidate"]) != source
                or not np.isclose(
                    float(selected_row["decision_threshold"]),
                    float(original_result["decision_threshold"]),
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    f"original decision selection/result mismatch in {fold_dir}"
                )
        original_result["outer_fold"] = fold

        legacy_path = fold_dir / "test_module_decisions.csv"
        legacy = pd.read_csv(legacy_path)
        original_path = fold_dir / "test_module_decisions_ofp_original.csv"
        original = pd.read_csv(original_path)
        if (fold_dir / "validation_decision_selection.csv").is_file():
            legacy_source = str(
                result.get("decision_source", "rule_guided_residual")
            )
            legacy_candidate = fold_dir / (
                "test_rule_only_module_decisions.csv"
                if legacy_source == "rule_only"
                else "test_residual_module_decisions.csv"
            )
            _validate_selected_detail_matches_candidate(
                legacy, legacy_candidate, source=legacy_path
            )
        if (
            fold_dir / "validation_decision_selection_ofp_original.csv"
        ).is_file():
            original_source = str(
                original_result.get("decision_source", "rule_guided_residual")
            )
            original_candidate = fold_dir / (
                "test_rule_only_module_decisions_ofp_original.csv"
                if original_source == "rule_only"
                else "test_residual_module_decisions_ofp_original.csv"
            )
            _validate_selected_detail_matches_candidate(
                original, original_candidate, source=original_path
            )
        expected_for_run = expected if formal else split_names
        _validate_detail_names(legacy, expected_for_run, source=legacy_path)
        _validate_detail_names(original, expected_for_run, source=original_path)
        legacy_metrics = metrics_from_module_decisions(
            legacy, protocol=PROTOCOL_NAME
        )
        original_metrics = metrics_from_module_decisions(
            original, protocol=OFP_ORIGINAL_PROTOCOL_NAME
        )
        _validate_result_metrics(result, legacy_metrics, source=result_path)
        _validate_result_metrics(
            original_result,
            original_metrics,
            source=original_result_path,
        )
        original_fold_rows.append(original_result)
        legacy.insert(0, "outer_fold", fold)
        original.insert(0, "outer_fold", fold)
        legacy_details.append(legacy)
        original_details.append(original)
        normalized_configs.append(fold_config)

    if any(config != normalized_configs[0] for config in normalized_configs[1:]):
        raise ValueError("fold configurations differ beyond test_fold/experiment_name")

    legacy_pooled = pd.concat(legacy_details, ignore_index=True)
    original_pooled = pd.concat(original_details, ignore_index=True)
    if legacy_pooled["file_name"].astype(str).duplicated().any():
        raise ValueError("outer test folds overlap in pooled Legacy decisions")
    if original_pooled["file_name"].astype(str).duplicated().any():
        raise ValueError("outer test folds overlap in pooled OFP decisions")

    if formal:
        all_names = set(index["file_name"].astype(str))
        if set(legacy_pooled["file_name"].astype(str)) != all_names:
            raise ValueError("pooled Legacy decisions do not cover the full index")
        if set(original_pooled["file_name"].astype(str)) != all_names:
            raise ValueError("pooled OFP decisions do not cover the full index")
        faulty = int((pd.to_numeric(index["Label"], errors="raise") > 0).sum())
        if (
            len(index) != EXPECTED_FULL_MODULES
            or faulty != EXPECTED_FULL_FAULTY
            or len(index) - faulty != EXPECTED_FULL_NORMAL
        ):
            raise ValueError("formal index does not match the frozen OFP module universe")

    legacy_metrics = metrics_from_module_decisions(
        legacy_pooled, protocol=PROTOCOL_NAME
    )
    original_metrics = metrics_from_module_decisions(
        original_pooled, protocol=OFP_ORIGINAL_PROTOCOL_NAME
    )
    fold_results = pd.DataFrame(fold_rows).sort_values("outer_fold")
    original_fold_results = pd.DataFrame(original_fold_rows).sort_values("outer_fold")
    comparison = pd.DataFrame(
        [
            _comparison_row(
                experiment_id=f"{experiment_id}_legacy_inclusive",
                protocol=PROTOCOL_NAME,
                metrics=legacy_metrics,
                fold_results=fold_results,
                thresholds=thresholds,
                formal=formal,
            ),
            _comparison_row(
                experiment_id=f"{experiment_id}_ofp_original_compatibility",
                protocol=OFP_ORIGINAL_PROTOCOL_NAME,
                metrics=original_metrics,
                fold_results=fold_results,
                thresholds=thresholds,
                formal=formal,
                decision_rows=original_fold_results,
            ),
        ]
    )

    pooled_dir = output_dir / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    legacy_pooled.to_csv(
        pooled_dir / "test_module_decisions_legacy_inclusive.csv", index=False
    )
    original_pooled.to_csv(
        pooled_dir / "test_module_decisions_ofp_original.csv", index=False
    )
    fold_results.to_csv(output_dir / "fold_metrics.csv", index=False)
    original_fold_results.to_csv(
        output_dir / "fold_metrics_ofp_original.csv", index=False
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    index_payload = index.loc[:, ["file_name", "folder_index", "Label"]].sort_values(
        "file_name"
    ).to_csv(index=False)
    manifest = {
        "experiment_id": experiment_id,
        "formal_full_oof": bool(formal),
        "outer_folds": list(OUTER_FOLDS),
        "index_sha256": hashlib.sha256(index_payload.encode("utf-8")).hexdigest(),
        "module_count": int(len(legacy_pooled)),
        "faulty_module_count": int(legacy_metrics["faulty_module_count"]),
        "normal_module_count": int(legacy_metrics["normal_module_count"]),
        "threshold_by_fold": {str(key): value for key, value in thresholds.items()},
        "decision_source_by_fold": {
            str(int(row.outer_fold)): str(
                getattr(row, "decision_source", "rule_guided_residual")
                if getattr(row, "decision_source", "rule_guided_residual")
                in {"rule_guided_residual", "temporal_model", "rule_only"}
                else "rule_guided_residual"
            )
            for row in fold_results.itertuples(index=False)
        },
        "ofp_original_decision_source_by_fold": {
            str(int(row.outer_fold)): str(
                getattr(row, "decision_source", "rule_guided_residual")
                if getattr(row, "decision_source", "rule_guided_residual")
                in {"rule_guided_residual", "temporal_model", "rule_only"}
                else "rule_guided_residual"
            )
            for row in original_fold_results.itertuples(index=False)
        },
        "legacy_inclusive_result": dict(legacy_metrics),
        "ofp_original_compatibility_result": dict(original_metrics),
        "aggregation": "concatenate_oof_module_decisions_then_recompute",
        "fold_metrics_averaged": False,
        "ofp_historical_reference_warning": (
            "The repository snapshot does not fully identify the historical README "
            "XGB training run; this row freezes the root EvaluateResult.py semantics."
        ),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return comparison


__all__ = [
    "EXPECTED_FULL_FAULTY",
    "EXPECTED_FULL_MODULES",
    "EXPECTED_FULL_NORMAL",
    "OUTER_FOLDS",
    "aggregate_three_fold_results",
]
