#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
FOLD="${FOLD:-1}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/patchtst_stat_aligned_xgb_precision_quick_fold${FOLD}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -u -B OFP_DL_model2_compat/run_patchtst_stat_aligned_xgb.py \
  --folds "${FOLD}" \
  --device cuda \
  --out_root "${OUT_ROOT}" \
  --seq_len 32 \
  --epochs 3 \
  --batch_size 192 \
  --feature_mode model2_plus \
  --stat_feature_mode all_engineered \
  --fusion_mode gated_attn \
  --sampling_mode module_balanced \
  --rule_mode none \
  --sample_selection hybrid \
  --temporal_positive_weight 1.0 \
  --adaptive_negative_weight 0.0 \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --max_train_files 2500 \
  --max_test_files 1200 \
  --threshold_grid 0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99 \
  --xgb_balance_mode none \
  --no_xgb_sample_weight \
  --ml_n_estimators 300 \
  --xgb_tree_method hist \
  --n_jobs 1 \
  "$@"
