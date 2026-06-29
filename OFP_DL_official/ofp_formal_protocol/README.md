# Formal OFP Protocol

This directory contains new code for the paper protocol.  It does not modify
`OFP/`, which remains the teacher-provided reference code.

## Frozen Protocol

1. The event time of a faulty module is the first row where `anomaly > 0`.
2. The primary training label is first-event `1h` ahead with a rolling 24h
   lookback in the deep runners:
   `0 < first_anomaly_ts - timestamp <= 1h`.
3. Rows with `timestamp >= first_anomaly_ts` are outside the prediction task
   and must not participate in training, normalization, threshold selection, or
   evaluation decisions.
4. The outer test split is module-level and follows
   `dataset/train_test_set_index(in).csv`.
5. For each outer fold, validation modules are sampled only from the remaining
   training pool using stratified sampling.
6. Normalization, weak-label statistics, model fitting, and threshold selection
   must use only train/validation modules.
7. Final test reporting uses `timestamp,predict` files and
   `OFP_DL_official/ofp_protocol/evaluator.py`.
8. Thresholds are selected on validation modules by OFP `final_score`, not by
   window-level F1.
9. The current OFP-compatible official path intentionally uses legacy
   `float32 -> int64` feature timestamps to match the OFP/model2 evaluator
   behavior.  Strict timestamp runs are not part of this default setting.

## Commands

Build the fixed manifest:

```powershell
python -B OFP_DL_official\ofp_formal_protocol\build_splits.py
```

Quick smoke run for one deep model:

```powershell
python -B OFP_DL_official\ofp_formal_protocol\run_deep_models_formal.py --models patchtst --folds 1 --epochs 1 --max_train_files 20 --max_val_files 20 --max_test_files 20 --device cpu
```

Quick smoke run for one tree baseline:

```powershell
python -B OFP_DL_official\ofp_formal_protocol\run_tree_baselines_formal.py --models rf --folds 1 --rf_estimators 10 --max_train_files 50 --max_val_files 50 --max_test_files 50 --n_jobs 1
```

Formal deep-model run:

```powershell
python -B OFP_DL_official\ofp_formal_protocol\run_deep_models_formal.py --models itransformer patchtst moderntcn fits fteformer --folds 1 2 3
```

Formal tree-baseline run:

```powershell
python -B OFP_DL_official\ofp_formal_protocol\run_tree_baselines_formal.py --models rf xgboost --folds 1 2 3
```

Outputs are written under `output/ofp_formal_protocol/`.
