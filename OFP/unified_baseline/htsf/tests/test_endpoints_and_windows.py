import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HTSF_DIR = Path(__file__).resolve().parents[1]
UNIFIED_DIR = HTSF_DIR.parent
sys.path[:0] = [str(HTSF_DIR), str(UNIFIED_DIR)]

from helpers import tiny_payload, write_module
from ofp_htsf.config import HTSFConfig
from ofp_htsf.data import causal_windows, read_module_views
from ofp_htsf.endpoints import (
    apply_static_weights,
    build_endpoint_manifest,
    endpoint_fingerprint,
)
from ofp_htsf.training import apply_adaptive_negative_weights


class EndpointAndWindowTests(unittest.TestCase):
    def test_first_fault_and_later_rows_never_enter_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_module(root / "fault.csv", fault_row=8)
            payload = tiny_payload()
            payload["task"]["horizon_hours"] = 5.0
            config = HTSFConfig.from_dict(payload)
            manifest = build_endpoint_manifest(root, ["fault.csv"], config)
            self.assertTrue((manifest["row_index"] < 8).all())
            selected_positive = manifest.loc[manifest["target"] > 0, "row_index"]
            self.assertTrue(selected_positive.between(3, 7).all())
            self.assertTrue((manifest["lead_hours"] > 0).all())

    def test_manifest_is_deterministic_and_weights_do_not_change_endpoint_hash(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_module(root / "fault.csv", fault_row=9)
            write_module(root / "normal.csv", offset=100.0)
            config = HTSFConfig.from_dict(tiny_payload())
            first = build_endpoint_manifest(root, ["fault.csv", "normal.csv"], config)
            second = build_endpoint_manifest(root, ["fault.csv", "normal.csv"], config)
            self.assertEqual(endpoint_fingerprint(first), endpoint_fingerprint(second))
            weighted = apply_static_weights(first, config)
            self.assertEqual(endpoint_fingerprint(first), endpoint_fingerprint(weighted))
            self.assertFalse(weighted.duplicated(["file_name", "row_index"]).any())

    def test_sampling_comparison_has_matched_module_budgets(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_module(root / "fault.csv", length=20, fault_row=15)
            write_module(root / "normal.csv", length=20, offset=100.0)
            base = tiny_payload()
            base["sampling"].update(
                {
                    "positive_windows_per_faulty_module": 4,
                    "negative_windows_per_faulty_module": 2,
                    "normal_windows_per_module": 2,
                }
            )
            module_quota_payload = {**base, "sampling": {**base["sampling"], "mode": "module_quota"}}
            hss_payload = {**base, "sampling": {**base["sampling"], "mode": "hss"}}
            module_quota = build_endpoint_manifest(
                root,
                ["fault.csv", "normal.csv"],
                HTSFConfig.from_dict(module_quota_payload),
            )
            hss = build_endpoint_manifest(
                root,
                ["fault.csv", "normal.csv"],
                HTSFConfig.from_dict(hss_payload),
            )
            self.assertEqual(
                module_quota.groupby("file_name").size().to_dict(),
                hss.groupby("file_name").size().to_dict(),
            )
            self.assertGreater(int((hss.loc[hss["file_name"] == "fault.csv", "target"] > 0).sum()), 0)

    def test_tpw_only_increases_positive_rows_and_is_monotonic_in_proximity(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_module(root / "fault.csv", fault_row=10)
            payload = tiny_payload()
            payload["weighting"].update(
                {"mode": "tpw", "temporal_positive_max_extra": 2.0}
            )
            config = HTSFConfig.from_dict(payload)
            manifest = build_endpoint_manifest(root, ["fault.csv"], config)
            weighted = apply_static_weights(manifest, config)
            negative = weighted["target"] == 0
            self.assertTrue((weighted.loc[negative, "sample_weight"] == 1.0).all())
            positive = weighted.loc[weighted["target"] > 0].sort_values("lead_hours")
            self.assertGreater(float(positive.iloc[0]["sample_weight"]), float(positive.iloc[-1]["sample_weight"]))

    def test_anw_only_changes_negative_weights_after_scores_exist(self):
        manifest = pd.DataFrame(
            {
                "file_name": ["a.csv"] * 4,
                "row_index": [0, 1, 2, 3],
                "timestamp": [1, 2, 3, 4],
                "target": [0, 0, 1, 1],
                "module_fault": [1, 1, 1, 1],
                "lead_hours": [4.0, 3.0, 2.0, 1.0],
                "sample_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        scores = pd.DataFrame(
            {
                "file_name": ["a.csv"] * 4,
                "row_index": [0, 1, 2, 3],
                "auxiliary_score": [0.2, 0.8, 0.9, 0.1],
            }
        )
        weighted, meta = apply_adaptive_negative_weights(manifest, scores, 2.0)
        self.assertEqual(weighted.loc[2:, "sample_weight"].tolist(), [1.0, 1.0])
        self.assertEqual(float(weighted.loc[0, "sample_weight"]), 1.0)
        self.assertEqual(float(weighted.loc[1, "sample_weight"]), 3.0)
        self.assertEqual(meta["updated_negative_rows"], 2.0)

    def test_hss_backfills_faulty_trace_without_positive_endpoints(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_module(root / "fault.csv", length=20, fault_row=15)
            payload = tiny_payload()
            payload["task"]["horizon_hours"] = 0.5
            payload["sampling"].update(
                {
                    "mode": "hss",
                    "positive_windows_per_faulty_module": 4,
                    "negative_windows_per_faulty_module": 2,
                }
            )
            manifest = build_endpoint_manifest(root, ["fault.csv"], HTSFConfig.from_dict(payload))
            self.assertEqual(len(manifest), 6)
            self.assertEqual(int(manifest["target"].sum()), 0)

    def test_causal_window_left_padding_and_future_independence(self):
        raw = np.arange(20, dtype=np.float32).reshape(10, 2)
        valid = np.ones_like(raw, dtype=bool)
        window, mask = causal_windows(raw, valid, [1], 4)
        np.testing.assert_array_equal(window[0, -2:], raw[:2])
        self.assertTrue((window[0, :2] == 0).all())
        self.assertFalse(mask[0, :2].any())
        changed = raw.copy()
        changed[2:] = 99999
        future_window, future_mask = causal_windows(changed, valid, [1], 4)
        np.testing.assert_array_equal(window, future_window)
        np.testing.assert_array_equal(mask, future_mask)

    def test_raw_invalid_markers_have_explicit_temporal_mask(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "invalid.csv"
            write_module(path)
            frame = pd.read_csv(path)
            frame.loc[0, "temperature"] = -255.0
            frame.loc[0, "current"] = -999.0
            frame.to_csv(path, index=False)
            views = read_module_views(path)
            self.assertFalse(bool(views.raw_valid[0, 0]))
            self.assertFalse(bool(views.raw_valid[0, 1]))
            self.assertEqual(float(views.raw[0, 0]), -255.0)
            self.assertEqual(float(views.raw[0, 1]), -999.0)


if __name__ == "__main__":
    unittest.main()
