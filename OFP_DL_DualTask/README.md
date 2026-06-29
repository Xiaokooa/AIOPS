# OFP DL OFP-Compatible Adapters

This folder implements OFP-compatible deep-learning adapters for the optical-module task.

The default protocol is now `dual_v2`: keep current/anomaly supervision as an auxiliary head, but use only the future-ahead head for OFP first-warning alerts.

- **dual_v2**: trains current + ahead heads, evaluates OFP alerts with `ahead_score` only.
- **forecast_only**: drops modules that are already faulty at the first timestamp and keeps only pre-first-fault windows for faulty modules.
- **hybrid_tree**: trains the deep encoder, extracts deep embeddings/scores, concatenates OFP model2 engineered features, and trains a tree classifier for OFP alerts.
- **dual_legacy**: legacy diagnostic mode that ORs current and ahead heads, closer to the first DualTask version.

By default, modules that are already faulty at their first timestamp are excluded from train/validation/test frames. Use `--include_start_fault_modules` only when you need the old all-positive OFP-compatible setting.

The implementation reuses the existing deep backbones under `model/Optical_prediction_model/deep_learning`:

- `itransformer`
- `patchtst`
- `moderntcn`
- `fits`

## Quick Smoke Runs

```powershell
python -m OFP_DL_DualTask.run --protocol dual_v2 --model fits --profile tiny --epochs 1 --max_train_modules 30 --max_val_modules 12 --max_test_modules 12 --force_rebuild
python -m OFP_DL_DualTask.run --protocol forecast_only --model fits --profile tiny --epochs 1 --max_train_modules 40 --max_val_modules 20 --max_test_modules 20 --force_rebuild
python -m OFP_DL_DualTask.run --protocol hybrid_tree --model fits --profile tiny --epochs 1 --max_train_modules 40 --max_val_modules 20 --max_test_modules 20 --hybrid_tree_model rf
python -m OFP_DL_DualTask.run --protocol dual_v2 --model fits --profile tiny --epochs 1 --sensor_columns temperature,current,currentTXPower,currentRXPower,currentMultiRXPower1 --force_rebuild
```

This is a mechanism smoke test, not a formal experiment.

## Full Run Sketches

```powershell
python -m OFP_DL_DualTask.run --protocol dual_v2 --model patchtst --profile server --obs_minutes 1440 --ahead_hours 120 --threshold_metric ofp_f1_score --threshold_grid fine --export_predictions
python -m OFP_DL_DualTask.run --protocol forecast_only --model patchtst --profile server --obs_minutes 1440 --ahead_hours 120 --no_prefix_windows --threshold_metric ofp_f1_score --threshold_grid fine --export_predictions
python -m OFP_DL_DualTask.run --protocol hybrid_tree --model patchtst --profile server --obs_minutes 1440 --ahead_hours 120 --threshold_metric ofp_f1_score --threshold_grid fine --export_predictions
python -m OFP_DL_DualTask.run --protocol hybrid_tree --model patchtst --profile server --obs_minutes 1440 --ahead_hours 120 --eval_step_minutes 60 --sensor_columns temperature,current,currentTXPower,currentRXPower,currentMultiRXPower1 --hybrid_tree_model xgboost --threshold_metric ofp_f1_score --threshold_grid fine --export_predictions
```

Outputs are written under:

```text
output/OFP_DL_DualTask/
```

## Important Protocol Notes

- Splits are module-level, stratified by `dataset/train_test_set_index(in).csv`.
- Start-fault modules are excluded by default after splitting, so train/predict/evaluate all operate on the same non-start-fault module subset.
- The evaluation does **not** credit alerts at or after the first failure timestamp.
- `--sensor_columns` restricts only the raw channels sent into the deep encoder. Model2 engineered features used by `hybrid_tree` are still kept intact.
- Prefix-padded windows are supported, so the current head can emit alerts from the first timestamp of a module, matching the row-wise nature of OFP machine-learning baselines.
- Thresholds are selected on validation modules only. The test split is used only for final reporting.
- Server defaults use module-level sampled validation during training (`monitor_val_max_windows`) and a larger validation subset for final threshold selection (`final_val_max_windows`) to avoid scanning millions of validation windows every epoch. Set `--final_val_max_windows 0` for full validation threshold selection.

## Model2 Rule Visualization

```powershell
python -m OFP_DL_DualTask.visualize_model2_rules --file 000000000.csv --top_k_rules 12 --formats png,pdf,svg
```

This exports rule hit CSVs and figures under:

```text
output/OFP_DL_DualTask/rule_visualization/
```
