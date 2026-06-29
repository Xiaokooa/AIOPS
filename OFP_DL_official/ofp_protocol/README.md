# OFP Protocol Layer

This directory contains new code for the final optical-module fault-prediction
task protocol.  It is separate from `OFP/` on purpose: `OFP/model1` and
`OFP/model2` are treated as teacher/baseline sources and should not be edited.

The protocol is module-level:

1. A model emits one prediction file per module.
2. Each prediction file contains `timestamp,predict`.
3. A module is counted as a valid predicted positive only when its first alert
   is earlier than its first anomaly, or when the module has no anomaly.
4. A true-positive hit is a faulty module whose first alert is earlier than the
   first anomaly.
5. The primary metric is F1 score.  Lead-time and final-score metrics are kept
   for consistency with `OFP/readme.md`.

## Current Files

- `evaluator.py`: unified evaluator compatible with OFP output files.
- `BASELINES.md`: map from original `OFP/model1` and `OFP/model2` code to the
  new protocol.

