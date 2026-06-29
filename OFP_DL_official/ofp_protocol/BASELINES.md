# OFP Baseline Map

This folder is the new protocol layer.  The original `OFP/` directory should
remain read-only.

## Original Sources

- `OFP/model1`
  - XGBoost point-level model.
  - Uses `anomaly_ahead_120_hours` in `train.py`.
  - Existing prediction script outputs one CSV per module with `timestamp,predict`.

- `OFP/model2`
  - Teacher baseline framework with feature extraction, cross-folder training,
    prediction, evaluation, and result merging.
  - Includes RuleModel, TreeModel/RF-style models, MUTANT, and merge utilities.
  - `Tools/EvaluateResult.py` defines the module-level metrics used by the task.

## Unified Protocol

Every baseline or deep model should export:

```text
<prediction_dir>/<module>.csv
timestamp,predict
...
```

Optional columns such as `score` or `proba` are allowed, but the official
evaluation reads only `timestamp` and `predict` unless configured otherwise.
For faulted modules, predictions at or after the first anomaly timestamp are
ignored; runners may also write `valid_for_eval` to make this explicit.

Use:

```powershell
python -B OFP_DL_official\ofp_protocol\evaluator.py `
  --prediction_dir <prediction_dir> `
  --label_dir D:\AIOps\dataset\training `
  --out_dir <out_eval_dir>
```

Main paper metrics should follow the OFP order:

```text
F1, Precision, Recall, Accuracy, Avg lead hour, Min lead hour, Final score
```
