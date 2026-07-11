from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from OFP.model2.ExperimentFeatureSchema import (
    RULE_FEATURES,
    add_ofp_expert_stat_features,
    feature_group_manifest,
    select_feature_names,
)
from OFP_DL_model2_compat.run_model2_compat_deep import evaluate_prediction_frames
from OFP_DL_model2_compat.run_htsf_experiments import merge_manifest_entries, selected_specs
from OFP_DL_model2_compat.run_patchtst_stat_aligned_xgb import (
    FUSION_MODES,
    TemporalStatAligner,
    pre_fusion_latent_names,
    resolve_fusion_mode,
    select_pre_fusion_latent_features,
)
from OFP_DL_model2_compat.summarize_htsf_experiments import inject_canonical_table_rows


class FeatureSchemaTest(unittest.TestCase):
    def test_ofp_features_and_semantic_groups(self) -> None:
        rows = 8
        frame = pd.DataFrame(
            {
                "Ts": np.arange(rows) * 300,
                "Temp": [20, 21, 22, 23, 24, 25, 26, -255],
                "Curr": [6000, 6100, 6200, 6300, 6400, 6500, 6600, 4000],
                "TxP0": [100, 101, 102, 103, 104, 105, 106, -1],
                "RxP0": [90, 91, 92, 93, 94, 95, 96, -1],
                **{f"RxP{i}": np.linspace(80 + i, 90 + i, rows) for i in range(1, 5)},
                **{f"TxP{i}": np.linspace(95 + i, 105 + i, rows) for i in range(1, 5)},
            }
        )
        out = add_ofp_expert_stat_features(frame)
        self.assertTrue(set(RULE_FEATURES) <= set(out.columns))
        self.assertIn("FeTempDiff", out.columns)
        self.assertIn("FeCoCurrTemp", out.columns)
        self.assertEqual(int(out.iloc[-1]["RuTempMin"]), 1)
        manifest = feature_group_manifest(list(out.columns))
        self.assertGreater(manifest["counts"].get("stat_prefix", 0), 0)
        self.assertGreater(manifest["counts"].get("expert_rule", 0), 0)
        selected = select_feature_names(list(out.columns), "statistical,expert", "expert_rule")
        self.assertIn("FeTempDiff", selected)
        self.assertNotIn("RuTempMin", selected)

    def test_fusion_ablation_mapping(self) -> None:
        self.assertEqual(resolve_fusion_mode("gated_attn", "no_stat_branch"), "temporal_only")
        self.assertEqual(resolve_fusion_mode("gated_attn", "no_temporal_branch"), "stat_only")
        self.assertEqual(resolve_fusion_mode("gated_attn", "no_cross_attention"), "latent_concat")

    def test_all_fusion_modes_return_fixed_latent_shape(self) -> None:
        raw = torch.randn(2, 16, 12)
        mask = torch.ones_like(raw)
        stat = torch.randn(2, 20)
        for mode in FUSION_MODES:
            model = TemporalStatAligner(
                temporal_encoder="fits",
                seq_len=16,
                n_raw_features=12,
                n_stat_features=20,
                embedding_dim=32,
                latent_dim=16,
                stat_hidden=32,
                attn_heads=4,
                dropout=0.0,
                fusion_mode=mode,
            )
            output = model.encode(raw, mask, stat)
            self.assertEqual(tuple(output.shape), (2, 16), msg=mode)
            model.close()

    def test_pre_fusion_latent_selector_recovers_temporal_signal(self) -> None:
        rng = np.random.default_rng(17)
        rows = 600
        temporal = rng.normal(size=(rows, 4)).astype(np.float32)
        statistical = rng.normal(size=(rows, 4)).astype(np.float32)
        labels = (temporal[:, 0] > 0.0).astype(np.int8)
        features = np.concatenate([temporal, statistical], axis=1)
        args = SimpleNamespace(
            latent_probe_max_rows=0,
            latent_probe_estimators=60,
            seed=42,
            n_jobs=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            selected, meta = select_pre_fusion_latent_features(
                features,
                labels,
                pre_fusion_latent_names(4),
                2,
                args,
                fold=1,
                run_dir=Path(tmp),
            )
            self.assertIn(0, selected)
            self.assertGreaterEqual(int(meta["temporal_selected"]), 1)
            self.assertTrue(
                (Path(tmp) / "explainability" / "pre_fusion_latent_probe" / "latent_feature_importance.csv").exists()
            )


class FirstWarningMetricTest(unittest.TestCase):
    def test_warnable_ceiling_and_false_alarm_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_dir = Path(tmp)
            labels = {
                "positive_warnable.csv": ([0, 50, 100], [0, 0, 1]),
                "positive_non_warnable.csv": ([20, 30], [1, 0]),
                "normal_fp.csv": ([0, 10], [0, 0]),
                "normal_tn.csv": ([0, 10], [0, 0]),
            }
            for name, (timestamps, anomaly) in labels.items():
                pd.DataFrame({"timestamp": timestamps, "anomaly": anomaly}).to_csv(label_dir / name, index=False)
            predictions = {
                "positive_warnable.csv": pd.DataFrame({"timestamp": [0, 50, 90], "predict": [0, 1, 0]}),
                "positive_non_warnable.csv": pd.DataFrame({"timestamp": [20, 30], "predict": [0, 1]}),
                "normal_fp.csv": pd.DataFrame({"timestamp": [0, 10], "predict": [1, 0]}),
                "normal_tn.csv": pd.DataFrame({"timestamp": [0, 10], "predict": [0, 0]}),
            }
            summary, _detail = evaluate_prediction_frames(predictions, label_dir)
            metrics = {str(row.Item): float(row.Value) for row in summary.itertuples(index=False)}
            self.assertAlmostEqual(metrics["precision"], 0.5)
            self.assertAlmostEqual(metrics["recall"], 0.5)
            self.assertAlmostEqual(metrics["warnable_recall"], 1.0)
            self.assertAlmostEqual(metrics["first_warning_upper_bound"], 0.5)
            self.assertAlmostEqual(metrics["false_alarm_rate"], 0.5)
            self.assertAlmostEqual(metrics["false_alarms_per_1000_normal"], 500.0)


class CanonicalResultTest(unittest.TestCase):
    def test_complete_htsf_is_scheduled_once(self) -> None:
        specs = selected_specs(["fusion", "sampling", "feature_groups"], "")
        experiment_ids = [item.experiment_id for item in specs]
        self.assertEqual(experiment_ids.count("main_htsf"), 1)
        self.assertNotIn("fusion_htsf", experiment_ids)
        self.assertNotIn("sampling_module_hybrid", experiment_ids)
        self.assertNotIn("feature_full", experiment_ids)

    def test_manifest_merge_removes_deprecated_duplicate_runs(self) -> None:
        merged = merge_manifest_entries(
            [
                {"experiment_id": "main_htsf", "status": "completed"},
                {"experiment_id": "fusion_htsf", "status": "completed"},
            ],
            [{"experiment_id": "main_htsf", "status": "skipped_existing"}],
        )
        self.assertEqual(merged, [{"experiment_id": "main_htsf", "status": "skipped_existing"}])

    def test_ablation_full_rows_reuse_main_fold_metrics(self) -> None:
        folds = pd.DataFrame(
            {
                "experiment_id": ["main_htsf", "main_htsf", "fusion_temporal_only"],
                "suite": ["main", "main", "fusion"],
                "paper_method": ["HTSF", "HTSF", "Temporal view only"],
                "fold": [1, 2, 1],
                "precision": [0.71, 0.73, 0.20],
                "recall": [0.40, 0.42, 0.55],
            }
        )
        out = inject_canonical_table_rows(
            folds,
            {"suites": ["main", "fusion", "sampling", "feature_groups"]},
        )
        main = out.loc[out["experiment_id"] == "main_htsf"].sort_values("fold")
        for alias_id in ["fusion_htsf", "sampling_module_hybrid", "feature_full"]:
            alias = out.loc[out["experiment_id"] == alias_id].sort_values("fold")
            np.testing.assert_array_equal(alias["precision"].to_numpy(), main["precision"].to_numpy())
            np.testing.assert_array_equal(alias["recall"].to_numpy(), main["recall"].to_numpy())
            self.assertTrue((alias["source_experiment_id"] == "main_htsf").all())


if __name__ == "__main__":
    unittest.main()
