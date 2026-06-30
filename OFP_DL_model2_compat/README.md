# OFP DL Model2-Compatible Deep Row Suite

This directory is a compatibility/diagnostic experiment for explaining why the
legacy ML model2 protocol can score much higher than strict deep first-warning
forecasting.

## What This Implements

The five deep encoders are reused from `OFP_DL_official`:

- PatchTST
- iTransformer
- FTEformer
- ModernTCN
- FITS

The task is deliberately changed to match the ML model2 mechanism:

- input: ML model2 engineered features on each row, with a short causal window;
- target: configurable row label:
  - `module_fault` default: every row of a faulty module is positive, matching the aggressive module-risk mechanism that can exploit legacy timestamps;
  - `anomaly`: current anomaly row label, closer to the original `TrainLabel=NAnomaly` source setting;
  - `ahead120`: strict first-event 120h row label for rows before the first anomaly;
- prediction: row-level binary alarm with validation-set threshold selection;
- timestamp: legacy feature timestamp path, so near-simultaneous alarms can be
  evaluated as slightly earlier than the true first anomaly;
- rule OR: deep score alarms are OR-merged with model2-style rules. The default
  `model2_simple` includes `Temp == -255` extra rows plus simple current/temperature
  and lane-power abnormal-state rules available in this compat feature table.

This is not the strict 120h first-warning setup used in `OFP_DL_official`.
Use it as an ablation/diagnostic path when comparing against ML model2.

## Why Short `seq_len`

The lead-time histogram shows that most faulty modules have no long pre-event
history. A long forecasting input such as 288 or 576 rows is therefore weakly
matched to this compatibility task. The default here is `SEQ_LEN=64`, because
the goal is row-level abnormal/near-abnormal state detection rather than long
lead-time temporal forecasting.

## Single-Model Commands

```bash
GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_patchtst.sh
GPU_ID=1 bash OFP_DL_model2_compat/scripts/run_itransformer.sh
GPU_ID=2 bash OFP_DL_model2_compat/scripts/run_fteformer.sh
GPU_ID=3 bash OFP_DL_model2_compat/scripts/run_moderntcn.sh
GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_fits.sh
```

The single-model scripts default to:

```bash
TARGET_MODE=module_fault
RULE_MODE=model2_simple
THRESHOLD_SEARCH=1
THRESHOLD_METRIC=f1_score
```

For ablations:

```bash
TARGET_MODE=anomaly GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_moderntcn.sh
TARGET_MODE=ahead120 RULE_MODE=temp GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_moderntcn.sh
THRESHOLD_SEARCH=0 THRESHOLD=0.3 GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_moderntcn.sh
```

## Full Suite

```bash
GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_model2_compat_suite.sh
```

To evaluate the stricter "at least 1 hour before the first anomaly" hit rule:

```bash
MIN_HIT_LEAD_HOURS=1 GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_model2_compat_suite.sh
```

## Paper-Oriented Three-Protocol Suite

For paper tables, prefer the Python suite below because it keeps the three
protocols in separate output folders and uses full data by default:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal
```

The default models are exactly the four requested backbones:

```text
fits itransformer moderntcn patchtst
```

Methods:

- `pure_deep`: deep backbone only, no rule OR and no tabular ML. Default target
  is `ahead120`, so this is the strict first-warning diagnostic adapter.
- `ofp_compat_deep`: direct deep row classifier with `module_fault`,
  `model2_plus`, and `model2_simple` rule OR. This is the main
  model2-compatible deep adapter.
- `hybrid_fusion`: trains the deep backbone, extracts final-layer embeddings and
  deep scores, concatenates them with model2/model2_plus features, then trains
  RF/XGB/LightGBM/CatBoost tabular models. Default ML models are
  `rf,xgb,lgbm,catboost`; missing optional dependencies are skipped unless
  `--require_all_ml` is set.

The paper suite enables DRAM-inspired training defaults:

- `--sample_selection hybrid`: half high-signal top-k rows and half random rows,
  following the "high-quality sample selection" idea.
- `--temporal_positive_weight 2`: upweights positive rows closer to the first
  anomaly within `--temporal_weight_horizon_hours 120`.
- `--adaptive_negative_weight 1`: after a warmup epoch, scores selected negative
  rows and upweights harder negatives for the remaining epochs.
- `model2_plus` includes multi-scale rule-like storm/count/rate features over
  short and long windows.

Before a long run, print the exact commands:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --dry_run
```

Quick smoke test:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py --profile smoke --methods all --device cpu
```

Useful formal variants:

```bash
# Isolate hybrid fusion without explicit rule OR.
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods hybrid_fusion --hybrid_rule_mode none

# Evaluate a stricter hit definition requiring at least one hour lead time.
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --min_hit_lead_hours 1

# Lead-time sensitivity table for operational constraints.
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --min_hit_lead_hours 2
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --min_hit_lead_hours 5

# Fail instead of skipping if LightGBM/CatBoost are missing.
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods hybrid_fusion --require_all_ml
```

Outputs:

- `<out_root>/suite_manifest.json`: method definitions, commands, status, and logs.
- `<out_root>/pure_deep/fold_metrics.csv`: strict pure deep OFP adapter.
- `<out_root>/ofp_compat_deep/fold_metrics.csv`: model2-compatible direct deep adapter.
- `<out_root>/hybrid_fusion/fold_metrics.csv`: deep embedding + model2 features + RF/XGB.

See `EXPERIMENT_COMMANDS.md` for copy-paste commands covering single-method
runs, ablations, lead-time sensitivity, and server pull/setup steps.

## Deep Embedding + Model2 Tabular Fusion

This path first trains a deep model in the model2-compatible row task, extracts
the embedding entering the final linear layer, concatenates it with model2
engineered features, performs feature selection, and trains OFP-style tabular
models (`rf`, `xgb`, `xgbrf`, `lgbm`, `catboost`; optional packages are skipped
unless `--require_all_ml` is set).

Default: four deep embeddings (`patchtst`, `itransformer`, `fteformer`,
`moderntcn`) + model2 features + deep score, followed by ExtraTrees top-k
selection.

```bash
GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_tabular_fusion.sh
```

Useful variants:

```bash
# Embedding-only auxiliary features after model2 feature concatenation.
DEEP_FEATURE_PARTS=embedding GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_tabular_fusion.sh

# ML model2 baseline under the same trained deep run, using model2 features only.
ML_FEATURE_SET=model2 ML_MODELS="rf xgb lgbm catboost" GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_tabular_fusion.sh

# Same tabular fusion experiment under the 1-hour-minimum hit rule.
MIN_HIT_LEAD_HOURS=1 GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_tabular_fusion.sh
```

Main outputs:

- `fold_metrics.csv`: one row per deep model / ML model / fold / prediction mode.
- `model_metrics_mean_std.csv`: mean/std grouped by deep model and mode.
- `<deep_model>/fold_<n>/tabular/selected_features.csv`: selected features after fusion.
- `<deep_model>/fold_<n>/predictions/...`: prediction files for evaluation upload or inspection.

## XGB + Deep Parallel Fusion

This approximates the OFP two-model style by training XGB and the deep model in
parallel, thresholding both on the validation split, and OR-merging their
prediction results.

```bash
GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_xgb_deep_parallel.sh
```

For the 1-hour hit rule:

```bash
MIN_HIT_LEAD_HOURS=1 GPU_ID=0 bash OFP_DL_model2_compat/scripts/run_xgb_deep_parallel.sh
```

## Four-GPU Launcher

```bash
GPU_IDS="0 1 2 3" bash OFP_DL_model2_compat/scripts/run_model2_compat_4gpu.sh
```

## Smoke Test

```bash
GPU_ID=0 \
FOLDS=1 \
MODELS=fits \
EPOCHS=1 \
MAX_TRAIN_FILES=50 \
MAX_TEST_FILES=20 \
bash OFP_DL_model2_compat/scripts/run_model2_compat_suite.sh
```

The smoke test only checks that the path runs end to end. It is not performance
evidence.
