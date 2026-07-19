import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ofp_unified.b0 import (
    booster_effective_device,
    config_fingerprint,
    resolve_device,
    run_fingerprint,
)
from ofp_unified.features import FEATURE_SCHEMA_VERSION


BASE = Path(__file__).resolve().parents[1]
WORKSPACE = BASE.parents[1]


class ProtocolTests(unittest.TestCase):
    def test_model1_compatible_xgboost_defaults_are_explicit(self):
        configs = {
            name: json.loads((BASE / "configs" / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("b0", "b1", "b2")
        }
        self.assertEqual(configs["b1"]["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(configs["b2"]["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        for config in configs.values():
            xgb = config["xgboost"]
            self.assertEqual(config["horizon_hours"], 120.0)
            self.assertEqual(config["threshold"], 0.5)
            self.assertEqual(xgb["num_boost_round"], 100)
            self.assertEqual(xgb["max_depth"], 6)
            self.assertEqual(xgb["eta"], 0.3)
            self.assertEqual(xgb["base_score"], 0.5)
        self.assertEqual(configs["b0"]["xgboost"], configs["b1"]["xgboost"])
        self.assertEqual(configs["b1"]["xgboost"], configs["b2"]["xgboost"])

    def test_auto_device_falls_back_when_cuda_probe_fails(self):
        with patch("ofp_unified.b0._cuda_probe", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA was explicitly requested"):
                resolve_device("cuda")
        with patch("ofp_unified.b0._cuda_probe", return_value=True):
            self.assertEqual(resolve_device("auto"), "cuda")
            self.assertEqual(resolve_device("cuda"), "cuda")
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_effective_device_is_read_from_booster_config(self):
        class FakeBooster:
            def __init__(self, device):
                self.device = device

            def save_config(self):
                return json.dumps({"learner": {"generic_param": {"device": self.device}}})

        self.assertEqual(booster_effective_device(FakeBooster("cuda:0")), "cuda")
        self.assertEqual(booster_effective_device(FakeBooster("cpu")), "cpu")

    def test_supplied_module_folds_are_disjoint_and_complete(self):
        index_path = WORKSPACE / "dataset" / "train_test_set_index(in).csv"
        data_dir = WORKSPACE / "dataset" / "training"
        if not index_path.exists() or not data_dir.exists():
            self.skipTest("workspace OFP dataset is not available")
        frame = pd.read_csv(index_path)
        self.assertEqual(len(frame), 13372)
        self.assertEqual(frame["file_name"].nunique(), 13372)
        self.assertEqual(frame.groupby("folder_index").size().to_dict(), {1: 4457, 2: 4457, 3: 4458})
        positives = frame.loc[frame["Label"] > 0].groupby("folder_index").size().to_dict()
        self.assertEqual(positives, {1: 1367, 2: 1367, 3: 1368})
        self.assertTrue(all((data_dir / name).is_file() for name in frame["file_name"]))

    def test_config_fingerprint_is_order_stable_and_change_sensitive(self):
        self.assertEqual(config_fingerprint({"a": 1, "b": 2}), config_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(config_fingerprint({"a": 1}), config_fingerprint({"a": 2}))

    def test_run_fingerprint_includes_split_and_effective_device(self):
        common = {
            "config": {"a": 1},
            "fold": 1,
            "data_dir": WORKSPACE / "dataset" / "training",
            "train_files": ["a.csv"],
            "validation_files": ["b.csv"],
            "device": "cuda",
        }
        base = run_fingerprint(**common)
        changed = dict(common)
        changed["validation_files"] = ["c.csv"]
        self.assertNotEqual(base, run_fingerprint(**changed))
        changed = dict(common)
        changed["device"] = "cpu"
        self.assertNotEqual(base, run_fingerprint(**changed))
        changed = dict(common)
        changed["dataset_metadata_fingerprint"] = "different-data"
        self.assertNotEqual(base, run_fingerprint(**changed))


if __name__ == "__main__":
    unittest.main()
