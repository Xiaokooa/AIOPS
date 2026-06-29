# OFP Official Deep-Learning Benchmarks

This directory is reserved for OFP-aligned experiments only.

Default protocol in this directory now follows the OFP README split:

- module-level three-fold split from `dataset/train_test_set_index(in).csv`;
- for each fold, `folder_index == fold` is test and all remaining modules are train;
- deep models use a rolling 24h lookback window (`seq_len=288` for the 5-minute
  data) and learn a 1h-ahead first-warning alarm: at timestamp `t`, the positive
  label is `0 < first_anomaly_ts - t <= 1h`;
- the default deep output is `[alarm, h1]`; the alarm head is used for
  first-warning threshold selection and the `h1` head is used for auxiliary BCE;
- rows at or after the first anomaly are excluded from training, train-only
  normalization, validation threshold selection, and evaluation decisions;
- full timestamp prediction files are still written for every evaluated test
  module, with post-first-anomaly rows marked as `valid_for_eval=0` when labels
  are available;
- the OFP-index deep runner now holds out a train-only validation subset by
  default (`index_val_fraction=0.1`) to select threshold/EMA/K by module-level
  `f1_score`;
- deep models use module-balanced training by default: each module contributes a
  capped number of pre-fault positive, pre-window negative, and normal windows,
  and left-censored faulty modules do not contribute to ahead-warning training;
- the first-warning event loss rewards first alarms inside the 1h primary
  window by default (`event_hit_mode=primary_window`) and naturally penalizes
  alarms fired too early;
- ML baselines keep OFP model2 feature/model settings but use the same
  first-event 1-hour training target as the deep models;
- ML and DL default to `legacy_float32` timestamp mode for OFP README compatibility;
- no smoke subsets or `max_*` caps in paper-grade runs;
- train-only normalization and weak-label references;
- saved config, counts, threshold, checkpoint, predictions, and evaluator output.

Development smoke tests and incomplete variants live under `dev/` or outside
this directory. Legacy reproduction scripts retained for reference live under
`legacy_protocol/`.

## Official Code Entrypoints

- `PatchTST/run.py`
- `iTransformer/run.py`
- `FTEformer/run.py`
- `ModernTCN/run.py`
- `FITS/run.py`
- `RF/run.py`
- `XGBoost/run.py`
- `run_official_benchmark.py`

Default output root:

`OFP_DL_official_results/lookback24h_ahead1h_legacy_benchmark`

## Server Bash Entrypoints

Use these scripts from the repository root after unzipping on Linux:

- `bash OFP_DL_official/scripts/run_first_warning_suite.sh`
- `bash OFP_DL_official/scripts/run_rf.sh`
- `bash OFP_DL_official/scripts/run_xgboost.sh`
- `bash OFP_DL_official/scripts/run_patchtst.sh`
- `bash OFP_DL_official/scripts/run_itransformer.sh`
- `bash OFP_DL_official/scripts/run_fteformer.sh`
- `bash OFP_DL_official/scripts/run_moderntcn.sh`
- `bash OFP_DL_official/scripts/run_fits.sh`

GPU scripts default to `GPU_ID=0`. To run on GPU 4:

`GPU_ID=4 bash OFP_DL_official/scripts/run_itransformer.sh`

For a 4-GPU server with device ids `0,1,2,3`, first check CUDA visibility:

```bash
python -B OFP_DL_official/scripts/check_cuda.py
CUDA_VISIBLE_DEVICES=2 python -B OFP_DL_official/scripts/check_cuda.py
```

Run the deep suite on one selected GPU:

```bash
GPU_ID=0 bash OFP_DL_official/scripts/run_first_warning_suite.sh
```

Run model/fold jobs in parallel over four GPUs:

```bash
GPU_IDS="0 1 2 3" bash OFP_DL_official/scripts/run_first_warning_4gpu.sh
```

Each deep script prints a compact run banner, one training summary line per
epoch, the final OFP evaluation metrics, and a clean fold/mean metric table.

For deep models, the bash scripts default to
`TRAIN_SAMPLING=module_balanced`, `SELECTION_METRIC=f1_score`,
`INDEX_VAL_FRACTION=0.1`, `AMP=0`, `POS_WEIGHT_CAP=20`, and the first-warning
event loss:

`Loss = event_loss_weight * first_warning_event_loss + aux_bce_weight * h1_BCE`

`AMP=0` is the stable default because several OFP deep models can produce
non-finite first-batch gradients under fp16 autocast. `POS_WEIGHT_CAP=20`
prevents the rare 1h-ahead positive rows from dominating the row-level BCE gradient.
Both can still be overridden from the shell for controlled ablations.

Module-balanced defaults:

```bash
MODULE_POSITIVE_WINDOWS_PER_MODULE=32
MODULE_NEGATIVE_WINDOWS_PER_FAULTY_MODULE=8
MODULE_NORMAL_WINDOWS_PER_MODULE=8
EVENT_HIT_MODE=primary_window
SEQ_LEN=288
```

Use `TRAIN_SAMPLING=full` when a run must consume every training row and be
directly comparable with the full-row RF/XGBoost protocol:

`GPU_ID=0 TRAIN_SAMPLING=full bash OFP_DL_official/scripts/run_moderntcn.sh`

## Paper-Readiness Gate

A full-row fair-comparison run is paper-eligible only if:

1. all requested folds finish without `max_*` caps;
2. `train_stride == 1` and `eval_stride == 1`;
3. `evaluated_module_cnt == test_modules`;
4. `train_pos_rows > 0` and `train_neg_rows > 0`;
5. ML baselines use OFP model2 settings (`RF=100 trees`, `XGBoost=10 trees`, fixed thresholds);
6. deep-model thresholds and alarm strategy are fixed or validation-selected
   according to the runner and recorded;
7. predictions are generated for every legacy OFP timestamp in every test module,
   while evaluation ignores rows at or after the first anomaly.
8. row-level deep models use `TRAIN_SAMPLING=full` / `--train_sampling full`
   when claimed as full-row evidence.

The sampled DL protocols (`TRAIN_SAMPLING=module_balanced` or
`TRAIN_SAMPLING=pos_all_neg_ratio`) are valid only when reported explicitly as
sampled protocols. `module_balanced` is the default F1-oriented deep protocol;
`pos_all_neg_ratio` keeps all positive rows and a fixed stratified negative
ratio for backward-compatible ablations. Neither is the same evidence level as
a full-row RF/XGBoost comparison.
