# OFP_DRAM Experiment Plan

## Goal

Convert the DRAM paper's failure prediction recipe into an OFP experiment for optical modules.
The target is first-warning optical module prediction: alert before the first `anomaly=1`, with the 120 h horizon as the main setting.

## Hypotheses

H1. Lane-level optical micro-features improve OFP prediction more than plain per-sensor statistics.

H2. Lead-aware positive weighting improves the OFP final score because earlier useful alerts receive more emphasis than very late pre-failure windows.

H3. Two-stage hard-negative reweighting reduces false positives from healthy but noisy optical modules.

H4. CatBoost and LightGBM are competitive with or better than existing RF/XGBoost-style OFP baselines when the feature set is hardware-aware.

## DRAM-to-OFP Mapping

| DRAM paper concept | OFP equivalent |
| --- | --- |
| CE history | optical telemetry before anomaly |
| UCE | first `anomaly=1` |
| vendor/architecture | `folder_index` and any future module static metadata |
| DQ count | per-lane RX/TX power imbalance |
| beat count | short-range burst/jump behavior in optical telemetry |
| MTBE/frequency | slope, deltas, diff statistics, window volatility |
| temporal positive weighting | lead-aware positive weighting |
| adaptive negative reweighting | hard-negative two-stage training |
| event-driven prediction | first warning over timestamp-level predictions |

## Minimal Viable Experiment

1. Build multi-scale features for every train/val/test split.
2. Train `random_forest`, `xgboost`, `lightgbm`, and `catboost` when installed.
3. Select thresholds on validation by OFP-style module final score.
4. Report window metrics and module-level first-warning metrics on the held-out test split.
5. Save feature importance for tree models.

## Ablations

Run each model with:

- `--one_stage`: disables hard-negative stage.
- `--history_minutes 60,360,1440`: removes 15 min fast precursor features.
- `--history_minutes 15,60`: keeps only short windows for a smoke-scale temporal ablation.
- Feature-group ablations can be done by temporarily removing lane features in `features.py`.

## Evidence Criteria

The DRAM-inspired design is supported if:

- LightGBM/CatBoost/XGBoost improve module-level final score or F1 over RF.
- Two-stage training reduces `false_positive` or improves precision without unacceptable recall loss.
- Feature importance highlights lane spread, lane CV, TX-RX gap, or total-vs-lane residual features.

The design is weakened if:

- Gains only appear in window metrics but not first-warning module metrics.
- Threshold selection produces very late alerts with poor lead hours.
- Feature importance is dominated by static folder features or coverage artifacts.

## Caveats

- Many positive modules have the first anomaly at the first row. These modules cannot provide true pre-failure evidence, so the implementation samples only windows before the first anomaly.
- `folder_index` is the only available static grouping in the public local dataset. Better metadata such as vendor, rate, distance class, or deployment region would make the static branch closer to the DRAM paper.
- CatBoost is optional in this environment. The code supports it, but the Python package must be installed before it can run.

