# Deep Backbones with Pure OFP Decision Adapter

This note defines the clean adapter used when reporting deep-learning baselines
such as iTransformer, PatchTST, ModernTCN, FITS, and FTEformer under the OFP
first-warning evaluator.

## What the Adapter Does

The pure OFP decision adapter only performs:

1. threshold selection on validation modules using the OFP first-warning metric;
2. conversion from deep model scores to binary predictions;
3. export of one `timestamp,predict` CSV file per module;
4. evaluation with the unchanged OFP evaluator.

It does not inject non-neural alarm evidence.

## What the Adapter Must Not Do

The pure adapter does not use:

- RuleModel prior;
- RF/XGBoost predictions;
- hand-written threshold rules;
- legacy `float32` timestamp casting;
- any post-processing that can create alarms independently of the deep model
  score.

Therefore, results produced by this adapter can be attributed to the deep
backbone and its learned score calibration. They may be much lower than the
OFP ML baselines, but the comparison is clean.

## Current Script

The implementation is:

```text
OFP/deep_learning/official/legacy_protocol/run_deep_models_module_level.py
```

The result metadata records:

```text
adapter_type = pure_ofp_decision_adapter
```

Example smoke-test command:

```powershell
python -B OFP\deep_learning\official\legacy_protocol\run_deep_models_module_level.py `
  --models itransformer `
  --folds 1 `
  --epochs 1 `
  --seq_len 64 `
  --module_batch_size 2 `
  --score_batch_size 64 `
  --max_train_files 40 `
  --max_val_files 20 `
  --max_test_files 20 `
  --threshold_metric f1 `
  --out_root output/ofp_legacy_protocol/deep_pure_adapter_smoke `
  --device cuda
```

## Reporting Rule

Use this table label for clean deep baselines:

```text
iTransformer + pure OFP decision adapter
```

Do not report earlier rule-prior or legacy-timestamp results as pure deep-model
performance.
