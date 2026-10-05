from __future__ import annotations
import numpy as np
from OFP.ssffn._engine import data as deep
ORIGINAL_PRESERVE = deep.make_features

def preserve_timestamps(raw_df, with_label=True):
    normal, extra = ORIGINAL_PRESERVE(raw_df, with_label)
    normal['Ts'] = normal['Ts'].astype(np.float32)
    return (normal, extra)

def install_timestamp_policy():
    deep.make_features = preserve_timestamps
    deep.FEATURE_CACHE_VERSION = 'ssffn_features_float32'
