import sys
import unittest
from pathlib import Path


HTSF_DIR = Path(__file__).resolve().parents[1]
UNIFIED_DIR = HTSF_DIR.parent
sys.path[:0] = [str(HTSF_DIR), str(UNIFIED_DIR)]

from ofp_htsf.bridge import validate_strict_b2_bridge


def manifests():
    xgboost = {"max_depth": 6, "num_boost_round": 100, "device": "auto"}
    splits = {"1": "s1", "2": "s2", "3": "s3"}
    devices = {"1": "cuda", "2": "cuda", "3": "cuda"}
    strict = {
        "config": {
            "feature_set": "raw_statistical_expert",
            "matrix_mode": "quantile",
            "horizon_hours": 120.0,
            "threshold": 0.5,
            "xgboost": xgboost,
        },
        "feature_manifest": {
            "feature_set": "raw_statistical_expert",
            "feature_count": 130,
            "schema_version": "strict-causal-v1",
        },
        "data_split_fingerprint_by_fold": splits,
        "dataset_metadata_fingerprint": "data",
        "effective_device_by_fold": devices,
        "pooled_metrics": {"evaluated_module_cnt": 13372},
    }
    sampled = {
        "variant": "sampled_b2_xgb",
        "decision_inputs": ["b2"],
        "folds": [1, 2, 3],
        "config": {
            "feature_schema_version": "strict-causal-v1",
            "task": {"horizon_hours": 120.0},
            "decision": {"threshold": 0.5, "xgboost": xgboost},
            "sampling": {"mode": "uniform_per_module", "windows_per_module": 32},
            "weighting": {"mode": "none"},
        },
        "split_fingerprints": splits,
        "dataset_fingerprints": {"1": "data", "2": "data", "3": "data"},
        "xgboost_devices": devices,
        "pooled_metrics": {"evaluated_module_cnt": 13372},
    }
    suite = {
        "suite_name": "bridge",
        "run_scope": "formal",
        "suite_complete": True,
        "canonical_definition": True,
        "experiments": [
            {"id": "H0_sampled_b2_xgb", "variant": "sampled_b2_xgb"},
            {"id": "H5_htsf", "variant": "htsf_fusion"},
            {"id": "H6_htsf_b2_skip_diagnostic", "variant": "htsf_fusion_b2_skip"},
        ],
    }
    return strict, sampled, suite


class BridgeTests(unittest.TestCase):
    def test_compatible_manifests_pass(self):
        strict, sampled, suite = manifests()
        self.assertTrue(validate_strict_b2_bridge(strict, sampled, suite)["compatible"])

    def test_protocol_difference_fails_closed(self):
        strict, sampled, suite = manifests()
        sampled["config"]["decision"]["threshold"] = 0.4
        with self.assertRaisesRegex(ValueError, "thresholds differ"):
            validate_strict_b2_bridge(strict, sampled, suite)


if __name__ == "__main__":
    unittest.main()
