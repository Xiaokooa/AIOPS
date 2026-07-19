import unittest

import numpy as np
import pandas as pd

from ofp_unified.features import (
    EXPERT_FEATURES,
    RAW_FEATURES,
    build_feature_frame,
    feature_names,
)


def make_frame(length: int) -> pd.DataFrame:
    index = np.arange(length, dtype=float)
    data = {
        name: (position + 1) * 10.0 + index + 0.1 * index**2
        for position, name in enumerate(RAW_FEATURES)
    }
    return pd.DataFrame(data)


class FeatureEngineeringTests(unittest.TestCase):
    def test_known_expanding_statistics(self):
        frame = make_frame(4)
        frame["temperature"] = [1.0, 2.0, 3.0, 4.0]
        result = build_feature_frame(frame, "b1")
        last = result.iloc[-1]
        self.assertAlmostEqual(float(last["FeTempMin"]), 1.0)
        self.assertAlmostEqual(float(last["FeTempMax"]), 4.0)
        self.assertAlmostEqual(float(last["FeTempDiff"]), 3.0)
        self.assertAlmostEqual(float(last["FeTempStd"]), np.std([1, 2, 3, 4], ddof=1), places=6)
        self.assertAlmostEqual(float(last["FeTempSkew"]), 0.0, places=6)
        self.assertAlmostEqual(float(last["FeTempKurt"]), -1.2, places=6)
        self.assertEqual(float(result.iloc[0]["FeTempStd"]), 0.0)
        self.assertEqual(float(result.iloc[0]["FeTempSkew"]), 0.0)
        self.assertEqual(float(result.iloc[0]["FeTempKurt"]), 0.0)

    def test_future_rows_cannot_change_prefix_features(self):
        full = make_frame(9)
        full.loc[8, "temperature"] = 1_000_000.0
        prefix = full.iloc[:7].copy()
        full_features = build_feature_frame(full, "b2").iloc[:7]
        prefix_features = build_feature_frame(prefix, "b2")
        np.testing.assert_allclose(
            full_features.to_numpy(),
            prefix_features.to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    def test_module_state_is_reset(self):
        first = make_frame(3)
        second = make_frame(3) + 1000.0
        first_features = build_feature_frame(first, "b1")
        second_features = build_feature_frame(second, "b1")
        self.assertEqual(float(first_features.iloc[0]["FeTempMin"]), float(first.iloc[0]["temperature"]))
        self.assertEqual(float(second_features.iloc[0]["FeTempMin"]), float(second.iloc[0]["temperature"]))

    def test_invalid_sentinels_do_not_poison_statistics_or_rules(self):
        frame = pd.DataFrame({name: [-999.0, 10.0, 20.0] for name in RAW_FEATURES})
        frame["temperature"] = [-255.0, 10.0, 20.0]
        result = build_feature_frame(frame, "b2")
        self.assertEqual(float(result.iloc[0]["temperature"]), -255.0)
        self.assertEqual(float(result.iloc[0]["current"]), -999.0)
        self.assertTrue(np.isnan(result.iloc[0]["FeTempMin"]))
        self.assertTrue(np.isnan(result.iloc[0]["FeCurrMin"]))
        self.assertEqual(float(result.iloc[1]["FeTempMin"]), 10.0)
        self.assertTrue((result.iloc[0][list(EXPERT_FEATURES)[4:]] == 0.0).all())

    def test_rule_boundaries_are_strict(self):
        frame = make_frame(3)
        frame["temperature"] = [0.0, 100.0, 100.1]
        frame["current"] = [5000.0, 4999.0, 4999.0]
        result = build_feature_frame(frame, "b2")
        self.assertEqual(result["RuTempDiff"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(result["RuCurrMin"].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(result["RuTempMin"].tolist(), [0.0, 0.0, 0.0])

    def test_correlation_and_lane_relations(self):
        frame = make_frame(5)
        frame["temperature"] = [1, 2, 3, 4, 5]
        frame["current"] = [2, 4, 6, 8, 10]
        frame["currentTXPower"] = 10.0
        frame["currentMultiTXPower1"] = 1.0
        frame["currentMultiTXPower2"] = 2.0
        frame["currentMultiTXPower3"] = 3.0
        frame["currentMultiTXPower4"] = 4.0
        result = build_feature_frame(frame, "b2")
        self.assertTrue(result["FeCoCurrTemp"].iloc[:4].isna().all())
        self.assertAlmostEqual(float(result.iloc[4]["FeCoCurrTemp"]), 1.0, places=6)
        self.assertEqual(float(result.iloc[4]["FeTxP0-Max"]), 6.0)
        self.assertEqual(float(result.iloc[4]["FeTxP0-Min"]), 9.0)

    def test_output_column_order_matches_registry(self):
        frame = make_frame(6)
        for feature_set in ("b0", "b1", "b2"):
            self.assertEqual(
                tuple(build_feature_frame(frame, feature_set).columns),
                feature_names(feature_set),
            )


if __name__ == "__main__":
    unittest.main()
