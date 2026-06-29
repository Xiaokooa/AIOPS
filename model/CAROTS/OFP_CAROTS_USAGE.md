# Using CAROTS For OFP

CAROTS is designed for multivariate time-series anomaly detection through
causality-aware contrastive learning. It is a good candidate for OFP if we treat
it as a learned anomaly/near-anomaly scorer, then apply the OFP first-warning
alarm policy.

## What CAROTS Learns

The code builds a CAROTS model with:

- a sequence encoder: `lstm`, `gru`, `iTransformer`, `TimesNet`, or `GATv2`;
- a CUTS+ causal discoverer;
- positive and negative augmentors guided by the causal matrix;
- a contrastive loss over original/augmented windows;
- a predictor/scorer that turns learned representations into anomaly scores.

## First OFP Experiment

Recommended minimum experiment:

1. Convert each OFP module trace into CAROTS-style train/test arrays.
2. Train on normal modules or normal pre-event windows.
3. Score all test module windows.
4. For each module, emit at most the first alarm timestamp after validation
   thresholding.
5. Evaluate with the same OFP `EvaluateResult.py` first-warning metric.

## Why This May Help

CAROTS does not need dense 120h positive labels. That matters for OFP because
only a small fraction of faulty modules have genuine long pre-fault histories.
It can focus on whether the current multivariate state looks abnormal or
causally inconsistent, then the event evaluator handles "early enough" alarms.

## Caveats

- The upstream code assumes CUDA in several model components.
- There is no upstream `requirements.txt`; install dependencies from
  `REPRODUCTION_STATUS.md`.
- Direct use of upstream `best_f1` thresholding may leak if applied to the OFP
  test set. For OFP, tune thresholds only on validation folds.
- CAROTS output is row/window-level anomaly score; OFP output must be converted
  to `timestamp,predict` with first-warning post-processing.
