import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from OFP.ssffn.model import SSFFN, STATISTIC_NAMES, VARIANTS, build_model, cached_statistic_names
from OFP.ssffn.report import metrics_from_decisions
from OFP.ssffn.splits import fixed_manifest, inner_partitions, validate_index
from OFP.ssffn._engine.modules import sensor_distribution_signal
from OFP.ssffn._engine import training as engine


class ModelContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_feature_partition_and_backbone(self):
        torch.manual_seed(43)
        model = build_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 133057)
        self.assertEqual(model.temporal_encoder_cfg['total_tokens'], 16)
        self.assertEqual(model.temporal_encoder_cfg['token_feature_dimensions'], [36,36,8])
        self.assertEqual(len(STATISTIC_NAMES), 80)
        self.assertEqual(len(set(STATISTIC_NAMES)), 80)
        self.assertEqual(len(model.backbone), 3)
        self.assertFalse(hasattr(model.sensor_input, 'history_projection'))

    def test_inactive_features_and_removals(self):
        raw, mask, stats = torch.randn(2,32,13), torch.ones(2,32,13), torch.randn(2,160)
        inactive = [i for i,n in enumerate(cached_statistic_names()) if not n.startswith('Fe')]
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                changed = stats.clone()
                changed[:, inactive] += 1000
                expected = model(raw, mask, stats)
                torch.testing.assert_close(expected, model(raw, mask, changed), rtol=0, atol=0)
                # The SMB uses the current measurement only.
                history = raw.clone()
                history[:, :-1, :] += 50
                history[:, -1, 12] += 500
                torch.testing.assert_close(expected, model(history, mask, stats), rtol=0, atol=0)
                if variant == 'no_statistics':
                    torch.testing.assert_close(expected, model(raw, mask, stats+300), rtol=0, atol=0)
                if variant == 'no_sensor':
                    torch.testing.assert_close(expected, model(raw+100, mask, stats), rtol=0, atol=0)
                model.train()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(raw, mask, stats), torch.tensor([0.,1.]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_tensor_api_matches_core(self):
        model = SSFFN().eval()
        sensor, stats = torch.randn(2,12), torch.randn(2,80)
        raw = torch.cat((sensor, torch.zeros(2,1)), 1)[:, None]
        cache = torch.zeros(2,160)
        cache[:, model.active_indices] = stats
        torch.testing.assert_close(model(sensor, stats), model.network(raw, torch.ones_like(raw), cache))

    def test_rule_free_sampling_signal(self):
        names = engine.compat_feature_names(engine.CompatCfg(feature_mode='ofp', preserve_timepoints=True))
        x = np.random.default_rng(0).normal(size=(30,len(names))).astype(np.float32)
        before = sensor_distribution_signal(x, np.zeros(30), names)
        raw, _, _, _ = engine.feature_indices(names, 'ofp_expert_stat', 'statistical,expert')
        changed = x.copy()
        changed[:, [i for i in range(len(names)) if i not in raw[:12]]] += 500
        np.testing.assert_array_equal(before, sensor_distribution_signal(changed, np.ones(30), names))

    def test_features_are_prefix_only_and_label_independent(self):
        from OFP.ssffn.smoke import generate_data
        from OFP.ssffn._engine.data import _read_feature_parts
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, _ = generate_data(Path(tmp)/'sample')
            name = 'synthetic_001.csv'
            raw = pd.read_csv(data_dir/name)
            cfg = engine.CompatCfg(feature_mode='ofp', preserve_timepoints=True, rule_mode='none', target_mode='pre_event')
            full = _read_feature_parts(data_dir, name, cfg, {name:1})[0]
            active = [*engine.RAW_SEQUENCE_FEATURES[:12], *STATISTIC_NAMES]
            raw.iloc[:25].to_csv(data_dir/'prefix.csv', index=False)
            prefix = _read_feature_parts(data_dir, 'prefix.csv', cfg, {'prefix.csv':0})[0]
            np.testing.assert_allclose(full[active].iloc[:25], prefix[active], rtol=0, atol=0)
            raw['anomaly'] = 1-raw['anomaly']
            raw.to_csv(data_dir/'labels_changed.csv', index=False)
            flipped = _read_feature_parts(data_dir, 'labels_changed.csv', cfg, {'labels_changed.csv':0})[0]
            np.testing.assert_array_equal(full[active], flipped[active])
            # Keep the explicitly specified float32 timestamp convention.
            np.testing.assert_array_equal(full.Ts.to_numpy(), raw.timestamp.to_numpy().astype(np.float32))


class EvaluationContract(unittest.TestCase):
    def test_holdout_isolation_and_reproducibility(self):
        index = pd.DataFrame(dict(file_name=[f'{i:03d}.csv' for i in range(60)],
                                  Label=[i%2 for i in range(60)], folder_index=[i%3+1 for i in range(60)]))
        frame = fixed_manifest(index)
        pd.testing.assert_frame_equal(frame, fixed_manifest(index.sample(frac=1, random_state=5)))
        self.assertEqual(int(frame.subset.eq('test').sum()), 12)
        test = set(frame.loc[frame.subset.eq('test'), 'file_name'])
        seen = set()
        for fold in (1,2,3):
            part = inner_partitions(frame, fold)
            self.assertFalse(set(part['train']) & set(part['validation']))
            self.assertFalse(test & (set(part['train']) | set(part['validation'])))
            self.assertEqual(part['test'], [])
            seen.update(part['validation'])
        self.assertEqual(len(seen), 48)

    def test_duplicate_and_unsafe_names_rejected(self):
        for names in (['same.csv','same.csv'], ['../file.csv','ok.csv']):
            with self.assertRaises(ValueError):
                validate_index(pd.DataFrame(dict(file_name=names, Label=[0,1], folder_index=[1,2])))

    def test_seven_metrics_and_no_minlead(self):
        detail = pd.DataFrame(dict(file_name=['a','b','c','d'], true_label=[1,1,0,0],
                                   valid_predict_positive=[1,0,1,0], hit=[1,0,0,0], lead_hour=[.25,np.nan,np.nan,np.nan]))
        result = metrics_from_decisions(detail)
        self.assertEqual(len(result), 7)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['F1'], .5)
        self.assertAlmostEqual(result['AFWS'], 1 + np.tanh(.25))
        with self.assertRaises(ValueError):
            metrics_from_decisions(pd.concat([detail, detail]))


if __name__ == '__main__':
    unittest.main()
