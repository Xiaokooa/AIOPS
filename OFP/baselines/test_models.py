import importlib
import sys
import unittest

import numpy as np
import pandas as pd
import torch

from .rules import predict_rules
from .trees import build_tree


class BaselineModels(unittest.TestCase):
    def test_deep_baselines_and_import_isolation(self):
        torch.set_num_threads(2)
        values = torch.randn(3, 64, 12)
        mask = torch.ones_like(values)
        specifications = (
            ('patchtst', 'PatchTSTCfg', 'PatchTSTClassifier'),
            ('itransformer', 'ITransformerCfg', 'ITransformerClassifier'),
            ('moderntcn', 'ModernTCNCfg', 'ModernTCNClassifier'),
            ('fits', 'FITSCfg', 'FITSClassifier'),
        )
        for name, config_name, model_name in specifications:
            with self.subTest(model=name):
                before = {k: v for k, v in sys.modules.items()
                          if k.split('.')[0] in {'layers', 'models', 'utils'}}
                module = importlib.import_module(f'OFP.baselines.{name}')
                after = {k: v for k, v in sys.modules.items()
                         if k.split('.')[0] in {'layers', 'models', 'utils'}}
                self.assertEqual(before, after)
                config = getattr(module, config_name)(seq_len=64, n_sensors=12)
                model = getattr(module, model_name)(config)
                logits = model(values, mask)
                if isinstance(logits, tuple):
                    logits = logits[0]
                self.assertEqual(tuple(logits.shape), (3,))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))

    def test_rule_baseline(self):
        from OFP.ssffn.model import STATISTIC_NAMES
        statistics = pd.DataFrame(0., index=range(2), columns=STATISTIC_NAMES)
        statistics['FeCurrMin'] = 6000.
        statistics.loc[1, 'FeTxP0Std'] = 501.
        np.testing.assert_array_equal(predict_rules(statistics).to_numpy(), [0, 1])

    def test_tree_constructors(self):
        features = np.random.default_rng(7).normal(size=(32, 4))
        labels = np.arange(32) % 2
        for name, package in [('random_forest', 'sklearn'), ('xgboost', 'xgboost'),
                              ('lightgbm', 'lightgbm'), ('catboost', 'catboost')]:
            with self.subTest(model=name):
                if importlib.util.find_spec(package) is None:
                    self.skipTest(f'Optional dependency not installed: {package}')
                parameters = dict(n_estimators=3, random_state=42)
                if name == 'catboost': parameters.update(verbose=False, allow_writing_files=False, thread_count=2)
                elif name == 'lightgbm': parameters.update(verbosity=-1, n_jobs=2)
                else: parameters.update(n_jobs=2)
                model = build_tree(name, **parameters).fit(features, labels)
                self.assertEqual(model.predict_proba(features).shape, (32, 2))


if __name__ == '__main__':
    unittest.main()
