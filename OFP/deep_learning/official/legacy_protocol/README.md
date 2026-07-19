# OFP Legacy-Compatible Protocol

This directory keeps the older reproduction scripts that used to live outside
`OFP_DL_official`. They now import the in-repo protocol packages and follow the
same first-event 120h target used by the official runners.

The legacy-compatible line intentionally differs from the stricter formal
protocol in `OFP/deep_learning/official/ofp_formal_protocol/` only in model family and
report aggregation:

- `model1` teacher checkpoints are kept for reproduction only.
- `model2` uses the existing model2-style Rule/RF/XGBoost prediction artifacts.
- RF/XGBoost/Rule fusions use timestamp-level OR over prediction files.
- Table aggregation follows the README style: fold-level decisions are pooled
  across all three folds before precision/recall/F1/final-score are computed.
- Training/evaluation code in this directory now excludes rows at or after the
  first anomaly; post-fault predictions are written with `valid_for_eval=0`.

## Baseline Reproduction

The README XGBoost row is reproduced by the teacher-provided model1 checkpoint,
not by retraining `model1` from scratch with the current external wrapper.

```powershell
python -B OFP\deep_learning\official\legacy_protocol\evaluate_teacher_model1.py `
  --model teacher_model1_original OFP\model1\xgboost_model_optical_original_features_ahead_120.json
```

This produces `teacher_model1_original`, which matches the `OFP/readme.md`
XGBoost row exactly:

| Model | Final | F1 | Precision | Recall | Hit | PredPos | AvgLead(h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher_model1_original | 2.057151 | 0.255344 | 0.678947 | 0.157240 | 645 | 950 | 68.445736 |

Reproduce the already-generated model2/readme-compatible artifacts and fusions:

```powershell
python -B OFP\deep_learning\official\legacy_protocol\reproduce_readme_baselines.py
```

Outputs are written to `output/ofp_legacy_protocol/readme_repro/`.

The main README-scale OR fusions use the teacher XGBoost checkpoint:

```powershell
python -B OFP\deep_learning\official\legacy_protocol\run_or_fusion.py `
  --model_name rf0.5_plus_teacher_model1_original `
  --source 'output/ofp_protocol/model2_suite/rf_thresh0.5/fold_{fold}/predictions' `
  --source 'output/ofp_legacy_protocol/readme_repro/teacher_model1_original/fold_{fold}/predictions'

python -B OFP\deep_learning\official\legacy_protocol\run_or_fusion.py `
  --model_name rf0.5_plus_teacher_model1_original_plus_rule `
  --source 'output/ofp_protocol/model2_suite/rf_thresh0.5/fold_{fold}/predictions' `
  --source 'output/ofp_legacy_protocol/readme_repro/teacher_model1_original/fold_{fold}/predictions' `
  --source 'output/ofp_protocol/model2_suite/rule/fold_{fold}/predictions'
```

Summarize any completed model root that contains
`<model>/fold_<k>/evaluation/module_decisions.csv`:

```powershell
python -B OFP\deep_learning\official\legacy_protocol\summarize_existing.py `
  --root output/ofp_legacy_protocol/readme_repro
```

Current pooled reproduction summary:

| Model | Final | F1 | Precision | Recall | Hit | PredPos | AvgLead(h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule | 1.391854 | 0.568638 | 0.999387 | 0.397367 | 1630 | 1631 | 0.007600 |
| RF(0.5) | 1.417727 | 0.572969 | 1.000000 | 0.401511 | 1647 | 1647 | 0.027802 |
| XGBoost teacher checkpoint | 2.057151 | 0.255344 | 0.678947 | 0.157240 | 645 | 950 | 68.445736 |
| RF(0.5)+XGBoost | 2.455536 | 0.632216 | 0.869769 | 0.496587 | 2037 | 2342 | 21.677884 |
| RF(0.5)+XGBoost+Rule | 2.455363 | 0.632118 | 0.869398 | 0.496587 | 2037 | 2343 | 21.677884 |
| RF(0.3) | 1.465147 | 0.573713 | 1.000000 | 0.402243 | 1650 | 1650 | 0.074383 |
| RF(0.3)+XGBoost | 2.455536 | 0.632216 | 0.869769 | 0.496587 | 2037 | 2342 | 21.677905 |

## Deep Model Interface

Deep models should write exactly the same prediction interface:

```text
<root>/<model>/fold_<k>/predictions/<module>.csv
timestamp,predict
...
```

An optional `score` column is allowed and ignored by the official evaluator.
Existing `OFP\deep_learning\official\ofp_protocol\run_deep_models.py` already writes this layout, so a full
legacy-compatible window-level run can use:

```powershell
python -B OFP\deep_learning\official\ofp_protocol\run_deep_models.py `
  --out_root output/ofp_legacy_protocol/deep_legacy `
  --models itransformer patchtst moderntcn fits fteformer `
  --folds 1 2 3 `
  --device cuda

python -B OFP\deep_learning\official\legacy_protocol\summarize_existing.py `
  --root output/ofp_legacy_protocol/deep_legacy
```

Smoke or partial roots must be summarized with matching folds, for example
`--folds 1`; they are interface checks only and should not be reported as formal
paper results.

## Module-Level Deep Training

`OFP\deep_learning\official\legacy_protocol\run_deep_models_module_level.py` is the evaluator-aligned
deep runner. It changes the learning problem from point/window classification to
module-bag training:

- one training item is one module;
- each module is represented by sampled timestamp windows;
- faulty modules are trained to have at least one high-scoring pre-anomaly
  window within the 120-hour warning region;
- healthy modules are trained so their maximum window score stays low;
- the validation threshold is selected by OFP `final_score`, not window F1;
- prediction still writes `timestamp,predict,score` because the official
  evaluator requires timestamp-level CSV files.

Small smoke run:

```powershell
python -B OFP\deep_learning\official\legacy_protocol\run_deep_models_module_level.py `
  --out_root output/ofp_legacy_protocol/deep_module_level_smoke `
  --models fteformer `
  --folds 1 `
  --epochs 1 `
  --max_train_files 20 `
  --max_val_files 20 `
  --max_test_files 20 `
  --device cuda `
  --module_batch_size 4 `
  --score_batch_size 32 `
  --windows_per_module 16 `
  --positive_windows_per_faulty 8
```

Resource-conscious full run template for a 3060 GPU:

```powershell
python -B OFP\deep_learning\official\legacy_protocol\run_deep_models_module_level.py `
  --out_root output/ofp_legacy_protocol/deep_module_level_3060_e3 `
  --models itransformer patchtst moderntcn fits fteformer `
  --folds 1 2 3 `
  --epochs 3 `
  --module_batch_size 8 `
  --score_batch_size 64 `
  --windows_per_module 32 `
  --positive_windows_per_faulty 16 `
  --device cuda
```
