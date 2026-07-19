# Frozen DRFP–OFP Protocol

## Task

- Paper task: Optical Module Failure Prediction.
- Failure anchor: first row whose `anomaly > 0`.
- Supervised cumulative horizons: 16, 24, 72, 120 hours.
- Primary decision head: calibrated 120-hour failure risk.
- Rows at or after first failure are forbidden for training/calibration.

## Split

| Split | Source | Modules | Faulty | Normal |
| --- | --- | ---: | ---: | ---: |
| Train | folds 1+2, stratified development split | 8022 | 2460 | 5562 |
| Validation | folds 1+2, seed 42, 10% | 892 | 274 | 618 |
| Test | complete fold 3 | 4458 | 1368 | 3090 |

There is one development/test run, not 3-fold model pooling. Split membership is written to `split_manifest.csv` and included in the split fingerprint.

## Leakage boundaries

- Feature construction is reset for every module.
- Raw window, statistics, rule margins, trend and persistence are causal.
- Normalizers fit only train rows strictly before first failure.
- Temperature calibration uses only valid pre-failure validation rows.
- All branch thresholds are selected on validation and frozen before fold3 files are read.
- Test labels are used only by the frozen evaluator and supplemental diagnostics.

The same validation set is used for early stopping surrogate loss, temperature calibration and threshold selection. This is not test leakage, but it can make model selection optimistic; it must be disclosed as a limitation or separated in a later robustness experiment.

## Sampling and imbalance

- v1 sampling mode is `uniform_per_module`.
- Every training epoch draws the same endpoint quota from every module.
- Multi-horizon positive weights are computed only from sampled train targets and capped by configuration.
- HSS, TPW and ANW are not silently active in v1. They must be separate, one-axis-at-a-time ablations if introduced later.

## Original OFP evaluator

For each module:

1. `failure_ts` is first anomaly timestamp.
2. `alarm_ts` is first timestamp whose decision is positive.
3. Faulty module: TP iff `alarm_ts < failure_ts`; otherwise FN.
4. Normal module: any alarm is FP; otherwise TN.
5. `lead_i = failure_ts - alarm_ts` only for TP modules.

Metrics:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
accuracy  = (TP + TN) / all_modules
AvgLead   = sum(lead_i for TP) / all_faulty_modules
MinLead   = min(lead_i for TP), or 0 with no hit
final     = F1 + accuracy + tanh(AvgLead) + tanh(MinLead)
```

An alarm more than 120 hours before failure is still a valid OFP hit. This is intentionally retained even though its timestamp label is negative under the 120-hour training surrogate.

## Decision policies

- Fixed reference: threshold 0.5 after validation temperature calibration.
- Deployment comparison: threshold selected on validation by OFP final score, then F1, precision and the higher threshold as tie-breakers.

Both policies are reported. The README three-fold pooled result also differs in test membership and calibration; even the fixed-0.5 row is therefore historical context rather than a strict apples-to-apples comparison. Strict claims require rerunning every baseline on this exact split with the same calibration/threshold policy.
