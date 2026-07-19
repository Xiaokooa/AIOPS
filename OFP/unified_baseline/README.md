# OFP Unified Baselines

This directory is an independent, reproducible B0/B1/B2 ablation layer.  The
teacher code under `OFP/model1` and `OFP/model2` remains unchanged.

Only three feature groups are allowed:

1. `raw`: the 12 optical-module measurements, preserved in their original order.
2. `statistical`: causal expanding statistics and trailing correlations.
3. `expert`: physical lane relations and deterministic Model2 rule indicators.

| Variant | Active groups | Dimensions | Mechanism tested |
|---|---|---:|---|
| B0 | Raw | 12 | strict reproducible Model1-style XGB baseline |
| B1 | Raw + Statistical | 88 | whether causal statistical history helps |
| B2 | Raw + Statistical + Expert | 130 | whether domain relations/rule priors add value |

All formulas, thresholds, missing-value rules and feature names are frozen in
[`FEATURES.md`](FEATURES.md).  B0/B1/B2 use the same labels, folds, XGBoost
hyperparameters, threshold and evaluator; only the feature set changes.

## Shared strict protocol

- One CSV file is one module/SN.
- The target is positive only in `[first_fault - 120h, first_fault)`.
- Rows at or after the first fault are excluded from training.
- The supplied three-fold module split is used; rows from one SN never cross
  train and validation.
- XGBoost uses 100 boosting rounds and a fixed `0.5` output threshold.
- Fold decisions are pooled before F1/precision/recall/final score are
  calculated, matching the table aggregation in `OFP/readme.md`.
- The README teacher checkpoint remains a `Legacy Reference`, not a held-out
  feature-ablation baseline because its training SN boundary is unavailable.

## Environment and data

The verified environment is Python 3.12.3, NumPy 1.26.4, pandas 2.2.2,
scikit-learn 1.4.2 and XGBoost 3.1.2.  Python packages are pinned in
`requirements.txt`.

The repository ignores `dataset/`; GitHub therefore does **not** contain the
1.88 GB training data.  Before running, mount or copy these inputs, or pass
custom paths:

```text
dataset/training/*.csv
dataset/train_test_set_index(in).csv
```

`--device auto` performs a real CUDA training probe and falls back to CPU when
the XGBoost wheel supports CUDA but the runner has no usable GPU.  Use
`--device cuda` only to require GPU execution explicitly.

B2 has 130 columns and roughly 9.9 million training rows per fold.  Run all
three variants on the same machine/device.  A low-memory CPU runner can use
`--device cpu --matrix-mode external_memory`; CUDA uses `quantile` mode.
This repository does not define an automatic GitHub Actions workflow because
the ignored dataset must be supplied by the runner.

## Validation commands

Run unit tests:

```powershell
Push-Location OFP\unified_baseline
python -B -m unittest discover -s tests -v
Pop-Location
```

Run B1/B2 smoke tests on real files:

```powershell
python -B OFP\unified_baseline\run_experiments.py `
  --variants b1 b2 --smoke --folds 1 --overwrite
```

Run the complete fair B0/B1/B2 experiment on a clean checkout:

```powershell
python -B OFP\unified_baseline\run_experiments.py `
  --variants b0 b1 b2 --folds 1 2 3 --overwrite
```

Run only B1/B2 when a compatible B0 artifact already exists:

```powershell
python -B OFP\unified_baseline\run_experiments.py `
  --variants b1 b2 --folds 1 2 3 --overwrite
```

Compatibility is checked, not assumed: feature schema, actual device, file
metadata and all three split fingerprints must match.  A B0 artifact created
before these identity fields were added is rejected for automatic ablation
aggregation; rerun B0 with the current code.

Example with externally mounted data:

```powershell
python -B OFP\unified_baseline\run_experiments.py `
  --variants b0 b1 b2 `
  --data-dir D:\data\ofp\training `
  --index-path 'D:\data\ofp\train_test_set_index(in).csv' `
  --folds 1 2 3 --overwrite
```

The existing `run_b0.py` command remains available for backward compatibility.

## Outputs

Formal variant outputs are written under:

```text
artifacts/b0_strict_120h/
artifacts/b1_strict_120h/
artifacts/b2_strict_120h/
```

Each variant contains models, split/config fingerprints, per-SN predictions,
pooled metrics and feature importance.  When completed manifests are present,
`artifacts/feature_ablation.csv` is generated automatically.  Important files:

- `pooled/evaluate_result.csv`: pooled official metrics.
- `fold_metrics.csv`: three fold results and runtime.
- `feature_importance_by_feature.csv`: mean feature gain share across folds.
- `feature_importance_by_group.csv`: Raw/Statistical/Expert gain share.
- `readme_comparison.csv`: descriptive comparison with the legacy checkpoint.

Partial and smoke runs use separate directories and are never presented as
formal results.

## Result status

Only verified real values are recorded; B1/B2 remain `TBD` until the formal
commands complete.

| Variant | Final | F1 | Precision | Recall | Status |
|---|---:|---:|---:|---:|---|
| B0 | 1.912967 | 0.166574 | 0.345888 | 0.109703 | verified strict 3-fold run |
| B1 | TBD | TBD | TBD | TBD | not yet run formally |
| B2 | TBD | TBD | TBD | TBD | not yet run formally |

The historical README XGBoost value is Final `2.057151`, but it is reported
separately because it comes from a fixed teacher checkpoint with legacy/unknown
training provenance.

The supplied data also imposes an observable-recall ceiling: among 4,102
faulty modules, only 936 have any pre-fault observation, so unconditional
Recall cannot exceed `936 / 4102 = 22.818%` under the official module metric.

## Clean HTSF extension

The protocol-locked hybrid model now lives in [`htsf/`](htsf/README.md).  It
reuses this directory's first-event labels, Raw/Statistical/Expert registry and
official evaluator, while keeping architecture, endpoint sampling, loss
weighting and threshold policy in separate audited suites.  It also provides a
same-endpoint B2 control so a representation gain is not confused with the
change from all-row XGBoost training to sampled temporal windows.
