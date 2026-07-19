# HTSF clean protocol and migration audit

Schema: `htsf-clean-v1`.

## Independent axes

| Axis | Canonical value in architecture suite | Where defined | May change in |
|---|---|---|---|
| Task | first-event, 120h, pre-fault training only | `protocol.json/task` | no main ablation |
| Split | supplied 3-fold SN split | index CSV | robustness study only |
| Window | 32 rows, causal, masked left padding | `protocol.json/window` | sensitivity study |
| Feature view | Raw window; endpoint Statistical+Expert | frozen registry | feature-view study |
| Endpoint sampling | uniform per module | `protocol.json/sampling` | sampling suite |
| Loss weighting | none | `protocol.json/weighting` | weighting suite |
| Representation | equal-width H1--H5 | `architecture_suite.json` | architecture suite |
| Decision | one XGBoost, shared parameters | `protocol.json/decision` | no main ablation |
| Threshold | fixed 0.5 | `protocol.json/decision` | separate threshold study |
| Evaluator | first valid warning per SN | `ofp_unified/evaluation.py` | never |

The suite comparison fails closed: core architecture runs require a common
128-dimensional XGBoost width and reject direct B2 input; all architecture runs
require identical endpoint and weight hashes; weighting runs require identical endpoint hashes;
all suites require an identical signature for every axis not declared changed.

## Causality and missingness

For endpoint `t`, the temporal branch sees only rows
`max(0,t-sequence_length+1)..t`.  The statistical and rule features are
expanding functions of rows `0..t`.  Neither branch sees future rows.

Raw values remain unchanged in the H0/B2 decision view.  The neural temporal
view treats `-999` on every channel and temperature `-255` as invalid, replaces
the normalized value with zero and supplies a separate mask.  Engineered NaNs
are handled in the same masked manner.  Normalizers are fit only on the outer
training fold's selected endpoints and their causal windows.

## Why the old HTSF result cannot be compared directly

The old compatibility runner was audited before this rewrite.  Four protocol
differences are decisive:

1. Its CLI name `ahead120` calls a helper whose horizon constant is one hour,
   so it does not train the unified 120h task.
2. Its hook detaches the captured temporal embedding; the backbone is therefore
   a frozen random feature extractor even though its parameters are passed to
   the optimizer.
3. Its temporal builder already wraps the backbone with window statistics, so
   `temporal_only` is not a pure Raw temporal branch.
4. Sampling, signal Top-K, TPW, ANW, BCE class weighting, XGBoost class/sample
   weighting, validation threshold search and optional rule `OR` can change in
   the same runner.

The clean code reuses none of those data, cache or hook paths.  It retains only
the high-level two-view attention/gate idea and implements that idea under the
same label, features, threshold and evaluator as the unified baseline.

## Formal-result gate

A run can enter the paper's strict main table only if:

- all three folds and all indexed validation modules are present;
- no module appears in more than one validation fold;
- every architecture row shares endpoint and weight hashes per fold;
- threshold policy is fixed and equal across rows;
- result status is formal, not smoke or partial;
- the pooled official metrics are computed after concatenating fold decisions.

Selecting only part of a non-smoke suite forces the output scope to `partial`; it cannot
overwrite or be labeled as a complete formal suite.  Resume fingerprints also
include the implementation hash of every `ofp_htsf/*.py` file, the runner hash,
and the Python/NumPy/pandas/Torch/XGBoost runtime identity.
