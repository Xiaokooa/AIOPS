# Release validation

Validated on 2026-10-05 with Python 3.11.15, PyTorch 2.11.0+cu128,
NumPy 2.4.6, pandas 3.0.3 and scikit-learn 1.9.0. The source snapshot is recorded
in `provenance.json`. The public source does not import the private research
workspace.

## Parity with the original split3 implementation

The full model and all four ablations were compared against the source used by
the archived experiments. All state-dict keys and tensors, RNG state after
initialization, evaluation logits and seeded training-mode logits matched
exactly. No temporal encoder, rule model, or tree classifier was introduced.

| Variant | Trainable parameters |
|---|---:|
| Full | 133,057 |
| w/o SMB | 131,137 |
| w/o SFB | 127,489 |
| w/o SIT | 7,809 |
| w/o HSS | 133,057 |

A real module's feature arrays and exported timestamps also matched exactly.
The fixed 80:20 split matched the previously generated analysis split manifest
module by module. Original data and module identities are not published with
this validation record.

## Executable checks

`python -m unittest OFP.ssffn.tests.test_contract -v` checks:

- Token grouping and model dimensions.
- No effect from excluded rule/threshold features.
- Actual removal of SMB/SFB and no sensor-history residual.
- Finite ordinary-BCE gradients in all five variants.
- Agreement between the 12+80 tensor API and the compatibility input API.
- Rule-free HSS initial signal.
- Prefix-only active feature construction and independence from anomaly labels.
- Fixed-test isolation, deterministic splitting and invalid index rejection.
- Seven-metric reporting without MinLead and duplicate-module rejection.

The synthetic integration check runs all five variants for three epochs in
one legacy fold, then runs three inner folds and the final fixed-test prediction
for the full model. It covers HSS refresh after epoch two, checkpoint export and
reload, validation threshold selection, and all seven reported metrics. These
synthetic runs are explicitly marked `synthetic_smoke` and are not experimental
evidence for model quality. The formal `train` command retains 16 epochs.

Full production training and a fresh multi-seed benchmark were not rerun for
this release. The included result CSVs are the archived, previously completed
15-fold ablation evidence under the legacy protocol.
