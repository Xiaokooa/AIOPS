# Neural Rule-Prior OFP Results

This note records a historical experiment. The runnable scripts have since been
merged under `OFP_DL_official/legacy_protocol/` and the current code uses the
first-event 120h target with post-first-anomaly rows excluded.

## Setting

- Evaluator: OFP module-level first-warning evaluator.
- Data split: `dataset/train_test_set_index(in).csv`, folds 1--3.
- Input features: OFP model2 default 22 engineered features.
- Base neural head: MLP, 2 hidden layers, hidden size 128, LayerNorm.
- Historical training label: row-level `anomaly` label, matching the old OFP
  model2 setting. Current code treats this option as a compatibility alias for
  the first-event 120h target.
- Alarm prior: OFP RuleModel-style prior, fused by `rule_or`.
- Timestamp behavior: `--legacy_timestamp_float32`, matching the original
  OFP `FeatureData.py` float32 export behavior.
- Threshold selection: validation-fold F1, searched in `[0.5, 0.99]`.

This is a rule-prior neural alarm model rather than a pure sequence-only deep
model. The result shows that deep models can reach the OFP ML baseline scale
only after explicitly injecting the near-fault rule/threshold prior that
dominates the OFP scoring behavior.

## Command

```powershell
python -B OFP_DL_official\legacy_protocol\run_neural_model2_full.py `
  --folds 1 2 3 `
  --label_mode anomaly `
  --model_type mlp `
  --epochs 1 `
  --batch_size 8192 `
  --lr 0.001 `
  --hidden_dim 128 `
  --layers 2 `
  --dropout 0.05 `
  --norm_type layer `
  --clip_value 20 `
  --threshold_min 0.5 `
  --threshold_max 0.99 `
  --threshold_grid_size 50 `
  --threshold_metric f1 `
  --prior_mode rule_or `
  --neural_prior_cap 0.49 `
  --legacy_timestamp_float32 `
  --out_root output/ofp_legacy_protocol/neural_model2_rule_prior_legacyts_full_f1 `
  --device cuda
```

## Three-Fold Results

| Model | F1 | Precision | Recall | Hit | PredPos | Final | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 | 0.6445 | 1.0000 | 0.4755 | 650 | 650 | 1.4917 | 0.8391 |
| Fold 2 | 0.6425 | 1.0000 | 0.4733 | 647 | 647 | 1.4898 | 0.8385 |
| Fold 3 | 0.6644 | 0.9985 | 0.4978 | 681 | 682 | 1.5186 | 0.8457 |
| Mean | 0.6505 | 0.9995 | 0.4822 | 659.3 | 659.7 | 1.5000 | 0.8411 |

## Baseline Comparison

| Model | Mean F1 | Mean Precision | Mean Recall | Mean Hit | Mean Final |
|---|---:|---:|---:|---:|---:|
| RF(0.3) | 0.5736 | 1.0000 | 0.4022 | 550.0 | 1.4656 |
| RF(0.5) | 0.5729 | 1.0000 | 0.4015 | 549.0 | 1.4182 |
| XGBoost | 0.5545 | 0.9969 | 0.3842 | 525.3 | 1.4557 |
| Rule | 0.5686 | 0.9994 | 0.3974 | 543.3 | 1.3918 |
| RF(0.3)+XGBoost | 0.5741 | 0.9970 | 0.4032 | 551.3 | 1.5268 |
| Neural Rule-Prior | 0.6505 | 0.9995 | 0.4822 | 659.3 | 1.5000 |

## Interpretation

The gain is driven by explicitly preserving the OFP near-fault rule prior and
the legacy float32 timestamp behavior. Without this prior, full-data neural
models trained on the same OFP features produced F1 around 0.12--0.15 on fold 1,
even with short pre-event labels, balanced sampling, soft-bin features, or RF
distillation.
