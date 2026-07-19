import unittest

from ofp_unified.features import (
    EXPERT_FEATURES,
    FEATURE_GROUPS,
    RAW_FEATURES,
    STATISTICAL_FEATURES,
    feature_manifest,
    feature_names,
    groups_for_features,
)


class FeatureRegistryTests(unittest.TestCase):
    def test_only_three_groups_exist(self):
        self.assertEqual(FEATURE_GROUPS, ("raw", "statistical", "expert"))

    def test_feature_groups_are_disjoint(self):
        raw = set(RAW_FEATURES)
        statistical = set(STATISTICAL_FEATURES)
        expert = set(EXPERT_FEATURES)
        self.assertFalse(raw & statistical)
        self.assertFalse(raw & expert)
        self.assertFalse(statistical & expert)
        self.assertEqual(len(RAW_FEATURES), 12)
        self.assertEqual(len(STATISTICAL_FEATURES), 76)
        self.assertEqual(len(EXPERT_FEATURES), 42)

    def test_b0_b1_b2_dimensions_and_groups(self):
        expected = {
            "raw": (12, ["raw"]),
            "raw_statistical": (88, ["raw", "statistical"]),
            "raw_statistical_expert": (130, ["raw", "statistical", "expert"]),
        }
        for feature_set, (count, groups) in expected.items():
            manifest = feature_manifest(feature_set)
            self.assertEqual(manifest["active_groups"], groups)
            self.assertEqual(manifest["feature_count"], count)
            self.assertEqual(len(feature_names(feature_set)), count)
            self.assertEqual(
                list(groups_for_features(feature_names(feature_set))),
                [item["group"] for item in manifest["features"]],
            )

    def test_b0_is_exactly_raw12(self):
        manifest = feature_manifest("b0")
        self.assertEqual(tuple(feature_names("b0")), RAW_FEATURES)
        self.assertTrue(all(item["group"] == "raw" for item in manifest["features"]))


if __name__ == "__main__":
    unittest.main()
