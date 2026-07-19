import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ofp_unified.evaluation import evaluate_prediction_dir


class EvaluationTests(unittest.TestCase):
    def test_strict_two_module_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = root / "labels"
            preds = root / "preds"
            labels.mkdir()
            preds.mkdir()
            pd.DataFrame(
                {"timestamp": [0, 3600, 7200], "anomaly": [0, 0, 1]}
            ).to_csv(labels / "fault.csv", index=False)
            pd.DataFrame(
                {"timestamp": [0, 3600, 7200], "predict": [0, 1, 0]}
            ).to_csv(preds / "fault.csv", index=False)
            pd.DataFrame(
                {"timestamp": [0, 3600], "anomaly": [0, 0]}
            ).to_csv(labels / "normal.csv", index=False)
            pd.DataFrame(
                {"timestamp": [0, 3600], "predict": [0, 0]}
            ).to_csv(preds / "normal.csv", index=False)
            metrics, detail = evaluate_prediction_dir(
                preds, labels, ["fault.csv", "normal.csv"]
            )
            self.assertEqual(len(detail), 2)
            self.assertEqual(metrics["f1_score"], 1.0)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["avg_lead_hour"], 1.0)

    def test_missing_prediction_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = root / "labels"
            preds = root / "preds"
            labels.mkdir()
            preds.mkdir()
            pd.DataFrame({"timestamp": [0], "anomaly": [0]}).to_csv(
                labels / "a.csv", index=False
            )
            with self.assertRaises(FileNotFoundError):
                evaluate_prediction_dir(preds, labels, ["a.csv"])

    def test_alert_at_or_after_fault_is_not_positive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = root / "labels"
            preds = root / "preds"
            labels.mkdir()
            preds.mkdir()
            pd.DataFrame(
                {"timestamp": [0, 3600, 7200], "anomaly": [0, 1, 1]}
            ).to_csv(labels / "fault.csv", index=False)
            pd.DataFrame(
                {"timestamp": [0, 3600, 7200], "predict": [0, 1, 1]}
            ).to_csv(preds / "fault.csv", index=False)
            metrics, detail = evaluate_prediction_dir(preds, labels, ["fault.csv"])
            self.assertEqual(metrics["all_predict_pos_cnt"], 0.0)
            self.assertEqual(metrics["recall"], 0.0)
            self.assertEqual(int(detail.iloc[0]["predict_label"]), 0)

    def test_alert_on_healthy_module_is_false_positive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = root / "labels"
            preds = root / "preds"
            labels.mkdir()
            preds.mkdir()
            pd.DataFrame(
                {"timestamp": [0, 3600], "anomaly": [0, 0]}
            ).to_csv(labels / "normal.csv", index=False)
            pd.DataFrame(
                {"timestamp": [0, 3600], "predict": [0, 1]}
            ).to_csv(preds / "normal.csv", index=False)
            metrics, detail = evaluate_prediction_dir(preds, labels, ["normal.csv"])
            self.assertEqual(metrics["fp"], 1.0)
            self.assertEqual(metrics["all_predict_pos_cnt"], 1.0)
            self.assertEqual(int(detail.iloc[0]["true_label"]), 0)
            self.assertEqual(int(detail.iloc[0]["predict_label"]), 1)

    def test_timestamp_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = root / "labels"
            preds = root / "preds"
            labels.mkdir()
            preds.mkdir()
            pd.DataFrame(
                {"timestamp": [0, 3600], "anomaly": [0, 0]}
            ).to_csv(labels / "a.csv", index=False)
            pd.DataFrame(
                {"timestamp": [0, 7200], "predict": [0, 0]}
            ).to_csv(preds / "a.csv", index=False)
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                evaluate_prediction_dir(preds, labels, ["a.csv"])

    def test_average_lead_uses_hit_modules_only(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = root / "labels"
            preds = root / "preds"
            labels.mkdir()
            preds.mkdir()
            cases = {
                "hit_2h.csv": ([0, 3600, 7200], [0, 0, 1], [1, 0, 0]),
                "hit_1h.csv": ([0, 3600, 7200], [0, 0, 1], [0, 1, 0]),
                "miss.csv": ([0, 3600, 7200], [0, 0, 1], [0, 0, 0]),
            }
            for name, (timestamp, anomaly, predict) in cases.items():
                pd.DataFrame(
                    {"timestamp": timestamp, "anomaly": anomaly}
                ).to_csv(labels / name, index=False)
                pd.DataFrame(
                    {"timestamp": timestamp, "predict": predict}
                ).to_csv(preds / name, index=False)
            metrics, _ = evaluate_prediction_dir(preds, labels, cases)
            self.assertEqual(metrics["all_hit_cnt"], 2.0)
            self.assertEqual(metrics["avg_lead_hour"], 1.5)


if __name__ == "__main__":
    unittest.main()
