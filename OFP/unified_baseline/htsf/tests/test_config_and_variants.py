import sys
import json
import unittest
from pathlib import Path


HTSF_DIR = Path(__file__).resolve().parents[1]
UNIFIED_DIR = HTSF_DIR.parent
sys.path[:0] = [str(HTSF_DIR), str(UNIFIED_DIR)]

from helpers import tiny_payload
from ofp_htsf.config import HTSFConfig, load_config
from ofp_htsf.data import ENGINEERED_FEATURES
from ofp_htsf.suite import audit_suite_configs
from ofp_htsf.suite import load_suite
from ofp_htsf.training import decision_feature_names
from ofp_htsf.variants import VARIANTS, get_variant
from ofp_unified.features import EXPERT_FEATURES, RAW_FEATURES, STATISTICAL_FEATURES
from ofp_unified.features import FEATURE_SCHEMA_VERSION
from run_htsf import (
    CANONICAL_INDEX_SHA256,
    CANONICAL_PROTOCOL_SHA256,
    CANONICAL_SUITE_SHA256,
    _file_sha256,
    _index_assignment_sha256,
)
from ofp_unified.b0 import read_index


class ConfigAndVariantTests(unittest.TestCase):
    def test_three_frozen_feature_groups_have_expected_dimensions(self):
        self.assertEqual(len(RAW_FEATURES), 12)
        self.assertEqual(len(STATISTICAL_FEATURES), 76)
        self.assertEqual(len(EXPERT_FEATURES), 42)
        self.assertEqual(len(ENGINEERED_FEATURES), 118)

    def test_protocol_fixes_task_decision_and_threshold(self):
        config = load_config(HTSF_DIR / "configs" / "protocol.json")
        self.assertEqual(config.task.target, "first_event")
        self.assertEqual(config.feature_schema_version, FEATURE_SCHEMA_VERSION)
        self.assertEqual(config.task.horizon_hours, 120.0)
        self.assertEqual(config.decision.model, "xgboost")
        self.assertEqual(config.decision.threshold_policy, "fixed")
        self.assertEqual(config.decision.threshold, 0.5)
        self.assertFalse(config.decision.use_sample_weight)

    def test_learned_main_variants_expose_equal_width_to_xgboost(self):
        latent = 128
        for name in (
            "temporal_only",
            "expert_stat_only",
            "direct_concat",
            "cross_attention",
            "htsf_fusion",
        ):
            self.assertEqual(len(decision_feature_names(get_variant(name), latent)), latent)
        self.assertEqual(len(decision_feature_names(get_variant("sampled_b2_xgb"), latent)), 130)
        self.assertEqual(len(decision_feature_names(get_variant("htsf_fusion_b2_skip"), latent)), 258)

    def test_architecture_suite_rejects_hidden_config_change(self):
        first = HTSFConfig.from_dict(tiny_payload())
        changed = tiny_payload()
        changed["decision"]["threshold"] = 0.6
        second = HTSFConfig.from_dict(changed)
        with self.assertRaisesRegex(ValueError, "outside"):
            audit_suite_configs("architecture", [first, second], ["temporal_only", "htsf_fusion"])

    def test_config_rejects_unknown_top_level_field(self):
        payload = tiny_payload()
        payload["thresholdd"] = 0.5
        with self.assertRaisesRegex(ValueError, "unknown top-level"):
            HTSFConfig.from_dict(payload)

    def test_suite_rejects_path_traversal_experiment_id(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "suite.json"
            path.write_text(
                json.dumps(
                    {
                        "suite_name": "custom",
                        "changed_axis": "architecture",
                        "experiments": [
                            {"id": "../escape", "variant": "htsf_fusion", "overrides": {}}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsafe experiment"):
                load_suite(path)

    def test_all_canonical_variants_are_named(self):
        self.assertEqual(len(VARIANTS), 7)
        for name, variant in VARIANTS.items():
            self.assertEqual(name, variant.name)

    def test_canonical_formal_config_hashes_match_packaged_json(self):
        self.assertEqual(
            _file_sha256(HTSF_DIR / "configs" / "protocol.json"),
            CANONICAL_PROTOCOL_SHA256,
        )
        for suite_name, expected in CANONICAL_SUITE_SHA256.items():
            self.assertEqual(
                _file_sha256(HTSF_DIR / "configs" / f"{suite_name}_suite.json"),
                expected,
            )
        index_path = UNIFIED_DIR.parents[1] / "dataset" / "train_test_set_index(in).csv"
        if index_path.is_file():
            self.assertEqual(
                _index_assignment_sha256(read_index(index_path)),
                CANONICAL_INDEX_SHA256,
            )

    def test_core_architecture_and_bridge_are_separate(self):
        architecture = json.loads(
            (HTSF_DIR / "configs" / "architecture_suite.json").read_text(encoding="utf-8")
        )
        bridge = json.loads(
            (HTSF_DIR / "configs" / "bridge_suite.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [item["variant"] for item in architecture["experiments"]],
            ["temporal_only", "expert_stat_only", "direct_concat", "cross_attention", "htsf_fusion"],
        )
        self.assertEqual(
            [item["variant"] for item in bridge["experiments"]],
            ["sampled_b2_xgb", "htsf_fusion", "htsf_fusion_b2_skip"],
        )


if __name__ == "__main__":
    unittest.main()
