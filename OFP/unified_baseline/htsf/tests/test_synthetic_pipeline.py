import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


HTSF_DIR = Path(__file__).resolve().parents[1]
UNIFIED_DIR = HTSF_DIR.parent
sys.path[:0] = [str(HTSF_DIR), str(UNIFIED_DIR)]

from helpers import tiny_payload, write_module
from ofp_htsf.config import HTSFConfig
from ofp_htsf.data import fit_standardizers, materialize_training_tensor_cache
from ofp_htsf.endpoints import (
    apply_static_weights,
    build_endpoint_manifest,
    endpoint_fingerprint,
)
from ofp_htsf.pipeline import run_fold_variant
from ofp_htsf.variants import get_variant


class SyntheticPipelineTests(unittest.TestCase):
    def test_two_stage_htsf_runs_end_to_end_with_one_xgboost(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data_dir = root / "data"
            data_dir.mkdir()
            for name, fault, offset in (
                ("train_p1.csv", 9, 0.0),
                ("train_p2.csv", 8, 5.0),
                ("train_n1.csv", None, 50.0),
                ("train_n2.csv", None, 55.0),
                ("val_p.csv", 9, 2.0),
                ("val_n.csv", None, 52.0),
            ):
                write_module(data_dir / name, fault_row=fault, offset=offset)
            config = HTSFConfig.from_dict(tiny_payload())
            train_files = ["train_p1.csv", "train_p2.csv", "train_n1.csv", "train_n2.csv"]
            manifest = build_endpoint_manifest(data_dir, train_files, config)
            weighted = apply_static_weights(manifest, config)
            raw_norm, engineered_norm = fit_standardizers(data_dir, manifest, config)
            tensor_cache = root / "tensor_cache"
            cache_meta = materialize_training_tensor_cache(
                tensor_cache,
                data_dir,
                manifest,
                config,
                raw_norm,
                engineered_norm,
            )
            endpoint_meta = {"endpoint_fingerprint": endpoint_fingerprint(manifest)}
            result = run_fold_variant(
                fold=1,
                variant=get_variant("htsf_fusion"),
                config=config,
                data_dir=data_dir,
                train_manifest=weighted,
                endpoint_meta=endpoint_meta,
                validation_files=["val_p.csv", "val_n.csv"],
                raw_standardizer=raw_norm,
                engineered_standardizer=engineered_norm,
                output_dir=root / "run",
                split_fingerprint="synthetic-split",
                dataset_fingerprint="synthetic-data",
                overwrite=True,
                tensor_cache_dir=tensor_cache,
            )
            self.assertEqual(result["variant"], "htsf_fusion")
            self.assertEqual(result["decision_feature_count"], 8)
            self.assertEqual(result["threshold"], 0.5)
            self.assertFalse(result["sample_weight_used_by_xgboost"])
            self.assertTrue((root / "run" / "representation.pt").is_file())
            self.assertTrue((root / "run" / "decision_xgb.ubj").is_file())
            self.assertTrue((root / "run" / "evaluation" / "evaluate_result.csv").is_file())
            self.assertEqual(len(pd.read_csv(root / "run" / "predictions" / "val_p.csv")), 12)
            self.assertEqual(cache_meta["rows"], len(manifest))
            self.assertTrue((tensor_cache / "sampled_feature_activity.csv").is_file())


if __name__ == "__main__":
    unittest.main()
