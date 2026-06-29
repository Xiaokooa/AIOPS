# OFP Semi-supervised MIL Pipeline

This path implements a separate experimental pipeline for the OFP first-warning
task:

1. pretrain a temporal encoder on normal modules only with masked reconstruction;
2. fine-tune the same encoder with first-warning event-level MIL supervision;
3. select the warning threshold on validation modules with the first-warning
   evaluator and write official `timestamp,predict,score,valid_for_eval` CSVs.

The supervised fine-tuning target is trace-level/MIL-oriented: a faulty module
is encouraged to fire its first alarm on a valid pre-first-anomaly window, while
normal modules are encouraged to keep all sampled window scores low.
Point-level first-event `[16h, 24h, 72h, 120h]` labels are still used as
auxiliary window targets.

Linux entrypoint:

```bash
bash OFP_DL_semisup_mil/scripts/run_semisup_first_warning.sh
```

Single-model strong run:

```bash
GPU_ID=0 bash OFP_DL_semisup_mil/scripts/run_semisup_temporal_mil.sh
```

The strong single-model wrapper selects the validation alarm strategy by
`SELECTION_METRIC=f1_score` by default, because the OFP `final_score` can favor
high accuracy and long lead time even when event-level F1 is low. Use
`SELECTION_METRIC=final_score` to reproduce the final-score-selected policy.

Short alias:

```bash
GPU_ID=0 bash OFP_DL_semisup_mil/scripts/run_semisup_mil.sh
```

Parallel fold launcher for a 4-GPU server:

```bash
GPU_IDS="0 1 2 3" bash OFP_DL_semisup_mil/scripts/run_semisup_first_warning_4gpu.sh
```

## Smoke Run

```powershell
python -B OFP_DL_semisup_mil\run_semisup_mil.py `
  --fold 1 `
  --device cpu `
  --pretrain_epochs 1 `
  --finetune_epochs 1 `
  --max_train_files 40 `
  --max_pretrain_files 20 `
  --max_val_files 20 `
  --max_test_files 20 `
  --seq_len 64 `
  --pretrain_batch_size 16 `
  --module_batch_size 4 `
  --windows_per_module 8 `
  --positive_windows_per_faulty 4
```

Outputs are written under `OFP_DL_semisup_mil_results/fold_<k>/`.

## Full Run Template

```powershell
python -B OFP_DL_semisup_mil\run_semisup_mil.py `
  --fold 1 `
  --device cuda `
  --pretrain_epochs 3 `
  --finetune_epochs 5 `
  --seq_len 288 `
  --windows_per_module 32 `
  --positive_windows_per_faulty 16
```

Run folds 1/2/3 separately and aggregate `evaluation/evaluate_result.csv` files
if using this as a paper baseline.
