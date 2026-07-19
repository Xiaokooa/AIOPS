# HTSF evidence plan

No formal experimental result has been generated here.  Every `TBD` cell must
be filled from a completed audited run with the matching protocol.

## Main comparison bridge

| Method | Training endpoints | XGBoost input | Final | F1 | Precision | Recall | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| B2-strict | all valid rows | Raw+Statistical+Expert | TBD | TBD | TBD | TBD | external unified baseline anchor |
| H0 sampled-B2 | frozen HTSF manifest | same 130 direct features | TBD | TBD | TBD | TBD | cost of endpoint sampling/scale |
| H5 clean HTSF | same frozen manifest | fused 128-D representation | TBD | TBD | TBD | TBD | representation effect versus H0 |
| H6 HTSF+B2 skip | same frozen manifest | direct B2 + fused representation | TBD | TBD | TBD | TBD | incremental information diagnostic |

## Architecture mechanism ablation

| Variant | Single changed mechanism | Final | F1 | Precision | Recall | Interpretation after results |
|---|---|---:|---:|---:|---:|---|
| H1 temporal-only | remove engineered branch | TBD | TBD | TBD | TBD | value of Raw temporal history |
| H2 expert-stat-only | remove temporal branch | TBD | TBD | TBD | TBD | value of domain/statistical view |
| H3 direct-concat | combine views without attention/gate | TBD | TBD | TBD | TBD | generic two-view combination |
| H4 cross-attention | add attention, gate fixed at 0.5 | TBD | TBD | TBD | TBD | attention mechanism contribution |
| H5 full HTSF | add learned gate | TBD | TBD | TBD | TBD | adaptive modality weighting contribution |

Only if H5 consistently exceeds both H2 and H3 under the same endpoints,
weights and threshold should the paper claim that temporal-statistical fusion
is effective.  H3 -> H4 -> H5 isolates attention and gate rather than removing
both in one row.

## Sampling and weighting tables

| Sampling strategy | Architecture | Weighting | Final | F1 | Precision | Recall |
|---|---|---|---:|---:|---:|---:|
| matched module quota, unstratified | H5 | none | TBD | TBD | TBD | TBD |
| matched HSS stratified quota | H5 | none | TBD | TBD | TBD | TBD |

| Representation weighting | Architecture | Sampling | XGB weighting | Final | F1 | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|
| none | H5 | HSS | off | TBD | TBD | TBD | TBD |
| TPW | H5 | HSS | off | TBD | TBD | TBD | TBD |
| ANW | H5 | HSS | off | TBD | TBD | TBD | TBD |
| TPW + ANW | H5 | HSS | off | TBD | TBD | TBD | TBD |

## Execution priority

| Priority | Experiment | Claim defended | Cost | Stop condition |
|---|---|---|---|---|
| P0 | fold-1 H1--H5 + separate H0 bridge | mechanism sanity | medium | stop if endpoints/hashes differ |
| P1 | 3-fold H1--H5 | main architecture claim | high | reconsider fusion if H5 does not beat H2/H3 |
| P2 | sampling suite | sampling contribution | high | keep result separate from architecture table |
| P3 | weighting suite | TPW/ANW contribution | high | drop gains that are unstable across folds |
| P4 | H0/H5/H6 bridge and sequence-length/seed robustness | incremental value/stability | high | report variance; do not cherry-pick |
