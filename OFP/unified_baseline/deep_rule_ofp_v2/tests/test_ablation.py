from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest

from fgofp.ablation import LEARNED_SPECS, write_module_ablation_outputs
from fgofp.evaluation import PROTOCOL_NAME, metrics_from_module_decisions
from run_module_ablation import (
    _run_or_resume,
    build_learned_command,
    build_static_command,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        config=tmp_path / "default.json",
        data_dir=tmp_path / "data",
        index_path=tmp_path / "index.csv",
        output_dir=tmp_path / "suite",
        device="cuda",
        batch_size=4,
        inference_batch_size=8,
        no_mixed_precision=True,
        smoke=True,
        overwrite=True,
    )


def test_ablation_commands_freeze_pkd_and_disable_fallback(tmp_path: Path) -> None:
    args = _args(tmp_path)
    commands = {
        spec.experiment_id: build_learned_command(args, spec)
        for spec in LEARNED_SPECS
    }
    assert "--disable-fgl" not in {part for command in commands.values() for part in command}
    for command in commands.values():
        assert "--disable-safe-rule-fallback" in command
        assert "--smoke" in command
        assert "--overwrite" in command
        assert command[command.index("--batch-size") + 1] == "4"
    temporal = commands["M2_temporal_only"]
    assert temporal[temporal.index("--architecture") + 1] == "causal_depthwise_tcn"
    assert temporal[temporal.index("--positive-weight-mode") + 1] == (
        "module_normalized_auto"
    )
    no_weight = commands["M3_fort_no_class_weight"]
    assert no_weight[no_weight.index("--positive-weight-mode") + 1] == "none"
    fort = commands["M4_fort"]
    assert fort[fort.index("--architecture") + 1] == "rule_guided_residual_tcn"
    assert fort[fort.index("--positive-weight-mode") + 1] == (
        "module_normalized_auto"
    )
    static = build_static_command(args)
    assert static[static.index("--experiment-name") + 1] == "S0_static_xgb"
    assert "--smoke" in static


def test_resume_restarts_only_an_incomplete_subrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "partial"
    output.mkdir()
    (output / "fold_1.tmp").write_text("partial", encoding="utf-8")
    captured: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> None:
        captured.append(command)

    monkeypatch.setattr("run_module_ablation.subprocess.run", fake_run)
    _run_or_resume(["python", "runner.py"], output, resume=True, label="M2")
    assert captured == [["python", "runner.py", "--overwrite"]]


def _decision_rows(
    *,
    lead_hours: tuple[float | None, float | None, float | None],
    normal_outcome: str = "TN",
) -> pd.DataFrame:
    rows = []
    for fold, lead in enumerate(lead_hours, start=1):
        faulty_tp = lead is not None
        rows.append(
            {
                "outer_fold": fold,
                "file_name": f"fault_{fold}",
                "outcome": "TP" if faulty_tp else "FN",
                "lead_hour": lead,
                "alarm_at_failure": int(faulty_tp and float(lead) == 0.0),
                "postfault_only_alarm": 0,
            }
        )
        rows.append(
            {
                "outer_fold": fold,
                "file_name": f"normal_{fold}",
                "outcome": normal_outcome,
                "lead_hour": None,
                "alarm_at_failure": 0,
                "postfault_only_alarm": 0,
            }
        )
    return pd.DataFrame(rows)


def _write_family(directory: Path, pooled: pd.DataFrame) -> None:
    directory.mkdir(parents=True)
    (directory / "pooled").mkdir()
    pooled.to_csv(
        directory / "pooled" / "test_module_decisions_legacy_inclusive.csv",
        index=False,
    )
    metrics = metrics_from_module_decisions(
        pooled.drop(columns="outer_fold"), protocol=PROTOCOL_NAME
    )
    static_metadata = (
        {
            "decision_feature_count": 25,
            "model_architecture": "xgboost_current_row",
            "positive_weight_mode": "module_normalized_auto",
            "sequence_to_sequence": False,
            "safe_rule_fallback": False,
        }
        if directory.name == "static_xgb"
        else {}
    )
    pd.DataFrame(
        [
            {
                "experiment_id": directory.name,
                "evaluation_protocol": PROTOCOL_NAME,
                **static_metadata,
                **metrics,
            }
        ]
    ).to_csv(directory / "comparison.csv", index=False)
    fold_rows = []
    for fold in (1, 2, 3):
        detail = pooled.loc[pooled["outer_fold"] == fold].drop(columns="outer_fold")
        fold_rows.append(
            {
                "outer_fold": fold,
                "protocol": PROTOCOL_NAME,
                **metrics_from_module_decisions(detail, protocol=PROTOCOL_NAME),
            }
        )
    pd.DataFrame(fold_rows).to_csv(directory / "fold_metrics.csv", index=False)


def _write_learned_configs(directory: Path, *, architecture: str, weight: str) -> None:
    for fold in (1, 2, 3):
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        with (fold_dir / "effective_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": {"architecture": architecture},
                    "training": {"positive_weight_mode": weight, "fgl_alpha": 0.7},
                    "decision": {"fallback_policy": "none"},
                },
                handle,
            )


def test_ablation_summary_effects_and_lead_profile(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    index = pd.DataFrame(
        [
            {"file_name": f"fault_{fold}", "folder_index": fold, "Label": 1}
            for fold in (1, 2, 3)
        ]
        + [
            {"file_name": f"normal_{fold}", "folder_index": fold, "Label": 0}
            for fold in (1, 2, 3)
        ]
    )
    index_path = tmp_path / "index.csv"
    index.to_csv(index_path, index=False)

    _write_family(root / "static_xgb", _decision_rows(lead_hours=(12.0, 12.0, None)))
    _write_family(root / "M2_temporal_only", _decision_rows(lead_hours=(24.0, 24.0, 24.0)))
    _write_learned_configs(
        root / "M2_temporal_only",
        architecture="causal_depthwise_tcn",
        weight="module_normalized_auto",
    )
    _write_family(
        root / "M3_fort_no_class_weight",
        _decision_rows(lead_hours=(30.0, 30.0, None)),
    )
    _write_learned_configs(
        root / "M3_fort_no_class_weight",
        architecture="rule_guided_residual_tcn",
        weight="none",
    )
    m4 = _decision_rows(lead_hours=(30.0, 30.0, 30.0))
    _write_family(root / "M4_fort", m4)
    _write_learned_configs(
        root / "M4_fort",
        architecture="rule_guided_residual_tcn",
        weight="module_normalized_auto",
    )
    rule = _decision_rows(lead_hours=(0.0, 0.0, 0.0))
    for fold in (1, 2, 3):
        fold_dir = root / "M4_fort" / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        rule.loc[rule["outer_fold"] == fold].drop(columns="outer_fold").to_csv(
            fold_dir / "test_rule_only_module_decisions.csv", index=False
        )

    summary = write_module_ablation_outputs(root, index_path, formal=False)
    assert list(summary["experiment_id"]) == [
        "S0_static_xgb",
        "R0_rule_only",
        "M2_temporal_only",
        "M3_fort_no_class_weight",
        "M4_fort",
    ]
    assert set(summary["evaluation_protocol"]) == {PROTOCOL_NAME}
    assert (summary["safe_rule_fallback"] == False).all()  # noqa: E712
    assert summary.set_index("experiment_id").loc[
        "M4_fort", "early_hit_rate_at_24h"
    ] == 1.0

    effects = pd.read_csv(root / "module_effects.csv").set_index("effect_id")
    assert effects.loc["temporal_vs_static_ml", "available"]
    assert effects.loc["rule_guidance", "delta_avg_lead_hour"] == 6.0
    assert effects.loc["class_weight", "delta_recall"] > 0.0

    lead = pd.read_csv(root / "lead_time_profile.csv")
    assert set(lead["lead_threshold_hour"]) == {0, 6, 12, 24, 72}
    m2 = lead.loc[lead["experiment_id"] == "M2_temporal_only"].set_index(
        "lead_threshold_hour"
    )
    assert m2.loc[0, "lead_recall"] == 1.0
    assert m2.loc[24, "lead_recall"] == 1.0
    assert m2.loc[72, "lead_recall"] == 0.0
    assert (root / "module_ablation_fold_metrics.csv").is_file()

    # Resuming a CE-only artifact must not silently relabel it as the fixed-PKD
    # temporal branch (the legacy disable flag encodes CE-only as alpha=1).
    config_path = root / "M2_temporal_only" / "fold_1" / "effective_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["fgl_alpha"] = 1.0
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must keep PKD/FGL enabled"):
        write_module_ablation_outputs(root, index_path, formal=False)
