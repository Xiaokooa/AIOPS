#!/usr/bin/env bash
set -euo pipefail

FOLDS="${FOLDS:-1 2 3}"
VARIANTS="${VARIANTS:-full_dram_xgb}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/xgb_dram_lead_sweep}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-0}"
MAX_TEST_FILES="${MAX_TEST_FILES:-0}"
LEAD_TIME_GRID="${LEAD_TIME_GRID:-1m,5m,15m,30m,1h,2h,5h,12h,24h}"
N_JOBS="${N_JOBS:-4}"

# shellcheck disable=SC2086
python -u -B OFP_DL_model2_compat/run_xgb_dram_ablation.py \
  --folds ${FOLDS} \
  --out_root "${OUT_ROOT}" \
  --variants ${VARIANTS} \
  --seq_len 32 \
  --batch_size 192 \
  --sampling_mode module_balanced \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --max_train_files "${MAX_TRAIN_FILES}" \
  --max_test_files "${MAX_TEST_FILES}" \
  --max_cached_files 512 \
  --selector extra_trees \
  --select_k 96 \
  --ml_n_estimators 300 \
  --xgb_tree_method hist \
  --n_jobs "${N_JOBS}" \
  --lead_time_grid "${LEAD_TIME_GRID}" \
  "$@"
