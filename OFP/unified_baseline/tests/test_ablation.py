import json
import tempfile
import unittest
from pathlib import Path

from ofp_unified.ablation import collect_ablation_results


METRICS = {
    "final_score": 1.0,
    "f1_score": 0.1,
    "precision": 0.2,
    "recall": 0.3,
    "accuracy": 0.4,
    "all_hit_cnt": 5.0,
    "all_predict_pos_cnt": 6.0,
    "avg_lead_hour": 7.0,
}


class AblationTests(unittest.TestCase):
    def _write_manifest(
        self,
        root: Path,
        variant: str,
        threshold: float = 0.5,
        legacy: bool = False,
    ) -> None:
        out = root / f"{variant}_strict_120h"
        out.mkdir(parents=True)
        feature_set = "raw" if variant == "b0" else "raw_statistical"
        feature_count = 12 if variant == "b0" else 88
        feature_manifest = {
            "feature_count": feature_count,
            "schema_version": "strict-causal-v1",
        }
        if variant != "b0":
            feature_manifest["feature_set"] = feature_set
        manifest = {
            "config": {
                "experiment_name": variant,
                "feature_set": feature_set,
                "threshold": threshold,
                "horizon_hours": 120.0,
                "folds": [1, 2, 3],
                "xgboost": {"seed": 42},
            },
            "feature_manifest": feature_manifest,
            "pooled_metrics": {**METRICS, "final_score": 1.0 + (variant == "b1")},
            "effective_device_by_fold": {"1": "cpu", "2": "cpu", "3": "cpu"},
            "data_split_fingerprint_by_fold": {
                "1": "split-1",
                "2": "split-2",
                "3": "split-3",
            },
            "dataset_metadata_fingerprint": "dataset-1",
        }
        if legacy:
            manifest["feature_manifest"].pop("schema_version")
            manifest.pop("effective_device_by_fold")
            manifest.pop("data_split_fingerprint_by_fold")
            manifest.pop("dataset_metadata_fingerprint")
        (out / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_b0_manifest_without_feature_set_key_and_b1_are_summarized(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self._write_manifest(root, "b0")
            self._write_manifest(root, "b1")
            table = collect_ablation_results(root)
            self.assertEqual(table["variant"].tolist(), ["b0", "b1"])
            self.assertEqual(table["feature_count"].tolist(), [12, 88])
            self.assertEqual(table["delta_final_score_vs_b0"].tolist(), [0.0, 1.0])

    def test_legacy_manifest_without_execution_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self._write_manifest(root, "b0", legacy=True)
            self._write_manifest(root, "b1")
            with self.assertRaisesRegex(ValueError, "lacks execution identity"):
                collect_ablation_results(root)

    def test_protocol_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self._write_manifest(root, "b0")
            self._write_manifest(root, "b1", threshold=0.6)
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                collect_ablation_results(root)


if __name__ == "__main__":
    unittest.main()
