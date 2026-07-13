"""Self-contained raw OFP input schema.

The v2 model deliberately consumes only the 12 original monitoring channels.
Keeping their order and missing-value sentinels in this package prevents the
standalone experiment from depending on an untracked sibling baseline.
"""
from __future__ import annotations


RAW_FEATURE_SCHEMA_VERSION = "ofp-native-raw-v1"

RAW_FEATURES = (
    "temperature",
    "current",
    "currentTXPower",
    "currentRXPower",
    "currentMultiRXPower1",
    "currentMultiRXPower2",
    "currentMultiRXPower3",
    "currentMultiRXPower4",
    "currentMultiTXPower1",
    "currentMultiTXPower2",
    "currentMultiTXPower3",
    "currentMultiTXPower4",
)

# These values are invalid observations in the original OFP data rather than
# physical measurements.  They are masked before train-only normalization.
MISSING_SENTINEL = -999.0
TEMPERATURE_OUTLIER_SENTINEL = -255.0


__all__ = [
    "MISSING_SENTINEL",
    "RAW_FEATURES",
    "RAW_FEATURE_SCHEMA_VERSION",
    "TEMPERATURE_OUTLIER_SENTINEL",
]
