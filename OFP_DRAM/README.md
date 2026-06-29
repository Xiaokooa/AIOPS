# OFP_DRAM

DRAM-paper-inspired optical module failure prediction experiments.

This folder translates the HPCA 2026 DRAM failure paper into an OFP setting:

- CE before UCE becomes optical telemetry before `anomaly=1`.
- DQ/beat micro-patterns become lane-level RX/TX imbalance patterns.
- Static architecture/vendor features become available OFP static groups such as `folder_index`.
- Temporal weighting becomes lead-aware positive weighting.
- Adaptive negative reweighting becomes hard-negative two-stage training.

## What Is Implemented

- Multi-scale optical features over 15 min, 1 h, 6 h, and 24 h histories.
- Sensor statistics for temperature, current, total TX/RX power, and lane TX/RX power.
- Lane-level micro features: RX/TX lane spread, coefficient of variation, strongest/weakest lane ratios, per-lane TX-RX gaps, and total-vs-lane residuals.
- Lead-aware positive sample weights for the 120 h ahead task.
- Two-stage training:
  - stage 1 trains on lead-weighted positives plus half of negatives;
  - stage 2 scores held-out negatives and gives harder negatives larger weights.
- Models:
  - `random_forest`
  - `xgboost`
  - `lightgbm`
  - `catboost` when the optional package is installed.

## Smoke Run

From `D:\AIOps`:

```powershell
python -B -m OFP_DRAM.run_experiment --smoke --models random_forest,lightgbm
```

The smoke run writes:

```text
output/OFP_DRAM/smoke/results/summary.json
output/OFP_DRAM/smoke/results/model_metrics.csv
output/OFP_DRAM/smoke/models/*.joblib
```

## Full Run

```powershell
python -B -m OFP_DRAM.run_experiment `
  --output_dir output/OFP_DRAM/full `
  --models random_forest,xgboost,lightgbm,catboost `
  --eval_step_minutes 60 `
  --progress_every 200 `
  --n_jobs 4
```

If CatBoost is not installed, the run records `missing_dependency` for CatBoost and continues with the other models.

For the first non-smoke run, start with one model:

```powershell
python -B -m OFP_DRAM.run_experiment `
  --output_dir output/OFP_DRAM/lgbm_first `
  --models lightgbm `
  --eval_step_minutes 60 `
  --progress_every 100 `
  --n_jobs 4
```

Avoid `--eval_step_minutes 5` on the full dataset unless you intentionally want dense timestamp-level scoring. It can create very large validation/test feature caches and spend a long time in feature construction before training starts.

## Official Test Prediction Files

After training a model, write OFP-style `timestamp,predict` files:

```powershell
python -B -m OFP_DRAM.run_experiment `
  --output_dir output/OFP_DRAM/full `
  --models lightgbm `
  --predict_official_test `
  --official_model lightgbm
```

Predictions are written under:

```text
output/OFP_DRAM/full/predictions/lightgbm/
```

## Main Comparison

Use the same feature frame and compare:

1. XGBoost/RF baselines from existing OFP style.
2. New LightGBM.
3. New CatBoost.
4. One-stage vs two-stage training with `--one_stage`.
5. Feature ablations by changing `--history_minutes` or editing `features.py` groups.
