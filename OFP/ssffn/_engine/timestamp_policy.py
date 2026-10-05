# Numerical core retained for compatibility with the archived experiments.
from __future__ import annotations


import numpy as np


from OFP.ssffn._engine import data as deep


ORIGINAL_PRESERVE = deep.make_preserved_htsf_features


def historical_preserve(raw_df, with_label=True):
    normal, extra = ORIGINAL_PRESERVE(raw_df, with_label)
    # This study explicitly retains the convention underlying the supplied table.
    normal['Ts'] = normal['Ts'].astype(np.float32)
    return normal, extra


def install_historical_policy():
    deep.make_preserved_htsf_features = historical_preserve
    deep.FEATURE_CACHE_VERSION = 'htsf_rule_margin_tokens_v3_transfer_float32'
