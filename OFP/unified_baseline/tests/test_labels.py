import unittest

import numpy as np
import pandas as pd

from ofp_unified.labels import first_event_target


class LabelTests(unittest.TestCase):
    def test_healthy_module_is_all_valid_negative(self):
        frame = pd.DataFrame({"timestamp": [0, 300, 600], "anomaly": [0, 0, 0]})
        target, valid, first_ts = first_event_target(frame, horizon_hours=1)
        np.testing.assert_array_equal(target, [0, 0, 0])
        np.testing.assert_array_equal(valid, [True, True, True])
        self.assertIsNone(first_ts)

    def test_only_pre_first_event_window_is_positive(self):
        frame = pd.DataFrame(
            {"timestamp": [0, 3600, 7200, 10800], "anomaly": [0, 0, 1, 1]}
        )
        target, valid, first_ts = first_event_target(frame, horizon_hours=1)
        np.testing.assert_array_equal(target, [0, 1, 0, 0])
        np.testing.assert_array_equal(valid, [True, True, False, False])
        self.assertEqual(first_ts, 7200.0)

    def test_unsorted_timestamps_fail(self):
        frame = pd.DataFrame({"timestamp": [300, 0], "anomaly": [0, 1]})
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            first_event_target(frame)

    def test_later_anomaly_rows_do_not_create_post_fault_targets(self):
        frame = pd.DataFrame(
            {
                "timestamp": [0, 3600, 7200, 10800, 14400],
                "anomaly": [0, 0, 1, 0, 1],
            }
        )
        target, valid, first_ts = first_event_target(frame, horizon_hours=2)
        np.testing.assert_array_equal(target, [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(valid, [True, True, False, False, False])
        self.assertEqual(first_ts, 7200.0)


if __name__ == "__main__":
    unittest.main()
