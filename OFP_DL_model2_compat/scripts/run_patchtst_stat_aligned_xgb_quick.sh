#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
FOLD="${FOLD:-1}"
TEMPORAL_ENCODER="${TEMPORAL_ENCODER:-patchtst}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/${TEMPORAL_ENCODER}_stat_aligned_xgb_quick_fold${FOLD}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -u -B OFP_DL_model2_compat/run_patchtst_stat_aligned_xgb.py \
  --temporal_encoder "${TEMPORAL_ENCODER}" \
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
  --sample_selection hybrid \
  --temporal_positive_weight 2.0 \
  --adaptive_negative_weight 1.0 \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --max_train_files 2500 \
  --max_test_files 1200 \
  --ml_n_estimators 300 \
  --xgb_tree_method hist \
  --n_jobs 1 \
  "$@"
