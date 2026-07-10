#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
FOLDS="${FOLDS:-1}"
TOP_K="${TOP_K:-64}"
EPOCHS="${EPOCHS:-3}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
ML_N_ESTIMATORS="${ML_N_ESTIMATORS:-300}"
SELECTOR_ESTIMATORS="${SELECTOR_ESTIMATORS:-300}"
SELECTOR_MAX_ROWS="${SELECTOR_MAX_ROWS:-300000}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/htsf_latent_probe_top${TOP_K}_fold1}"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u -B OFP_DL_model2_compat/run_patchtst_stat_aligned_xgb.py \
  --experiment_id "htsf_latent_probe_top${TOP_K}" \
  --method_label HTSF-XGB \
  --temporal_encoder patchtst \
  --decision_layer xgb \
  --folds ${FOLDS} \
  --out_root "${OUT_ROOT}" \
  --seq_len 32 \
  --epochs "${EPOCHS}" \
  --batch_size 192 \
  --target_mode ahead120 \
  --feature_mode ofp \
  --stat_feature_mode ofp_expert_stat \
  --stat_feature_groups statistical,expert \
  --temporal_summary_mode none \
  --sampling_mode module_balanced \
  --sample_selection hybrid \
  --sample_topk_fraction 0.5 \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --temporal_positive_weight 0.0 \
  --adaptive_negative_weight 0.0 \
  --rule_mode none \
  --fusion_mode gated_attn \
  --latent_dim 128 \
  --latent_probe_topk "${TOP_K}" \
  --latent_probe_estimators "${SELECTOR_ESTIMATORS}" \
  --latent_probe_max_rows "${SELECTOR_MAX_ROWS}" \
  --xgb_balance_mode none \
  --no_xgb_sample_weight \
  --ml_n_estimators "${ML_N_ESTIMATORS}" \
  --xgb_tree_method hist \
  --max_train_files "${MAX_TRAIN_FILES}" \
  --max_test_files "${MAX_TEST_FILES}" \
  --max_cached_files 512 \
  --log_batches 200 \
  --n_jobs 4 \
  --device cuda \
  "$@"
