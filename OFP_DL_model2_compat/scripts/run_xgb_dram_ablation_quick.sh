#!/usr/bin/env bash
set -euo pipefail

FOLD="${FOLD:-1}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/xgb_dram_ablation_quick_fold${FOLD}}"

python -u -B OFP_DL_model2_compat/run_xgb_dram_ablation.py \
  --folds "${FOLD}" \
  --out_root "${OUT_ROOT}" \
  --variants all \
  --seq_len 32 \
  --batch_size 192 \
  --sampling_mode module_balanced \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --max_train_files 2500 \
  --max_test_files 1200 \
  --selector extra_trees \
  --select_k 96 \
  --ml_n_estimators 300 \
  --xgb_tree_method hist \
  --n_jobs 1 \
  "$@"
