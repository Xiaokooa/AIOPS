#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
FOLDS="${FOLDS:-1 2 3}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/tsf_table5_ablation}"
TSF_VARIANTS="${TSF_VARIANTS:-all}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-0}"
MAX_TEST_FILES="${MAX_TEST_FILES:-0}"
N_JOBS="${N_JOBS:-4}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ "${TSF_VARIANTS}" == "all" ]]; then
  VARIANT_LIST="full no_stat_branch no_temporal_branch no_cross_attention no_gated_fusion"
else
  VARIANT_LIST="${TSF_VARIANTS}"
fi

for variant in ${VARIANT_LIST}; do
  case "${variant}" in
    full)
      TSF_ABLATION="full"
      FUSION_MODE="gated_attn"
      ;;
    no_stat_branch)
      TSF_ABLATION="no_stat_branch"
      FUSION_MODE="gated_attn"
      ;;
    no_temporal_branch)
      TSF_ABLATION="no_temporal_branch"
      FUSION_MODE="gated_attn"
      ;;
    no_cross_attention)
      TSF_ABLATION="no_cross_attention"
      FUSION_MODE="gated_attn"
      ;;
    no_gated_fusion)
      TSF_ABLATION="full"
      FUSION_MODE="attn_mean"
      ;;
    *)
      echo "Unknown TSF variant: ${variant}" >&2
      exit 2
      ;;
  esac

  echo "========================================================================"
  echo "[table5] variant=${variant} folds=${FOLDS} out_root=${OUT_ROOT}"
  echo "========================================================================"

  # shellcheck disable=SC2086
  python -u -B OFP_DL_model2_compat/run_patchtst_stat_aligned_xgb.py \
    --folds ${FOLDS} \
    --device cuda \
    --out_root "${OUT_ROOT}" \
    --seq_len 32 \
    --epochs 3 \
    --batch_size 192 \
    --feature_mode model2_plus \
    --stat_feature_mode all_engineered \
    --fusion_mode "${FUSION_MODE}" \
    --tsf_ablation "${TSF_ABLATION}" \
    --sampling_mode module_balanced \
    --rule_mode none \
    --sample_selection hybrid \
    --temporal_positive_weight 1.0 \
    --adaptive_negative_weight 0.0 \
    --positive_windows_per_module 32 \
    --negative_windows_per_faulty_module 8 \
    --normal_windows_per_module 8 \
    --max_train_files "${MAX_TRAIN_FILES}" \
    --max_test_files "${MAX_TEST_FILES}" \
    --max_cached_files 512 \
    --threshold_grid 0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99 \
    --xgb_balance_mode none \
    --no_xgb_sample_weight \
    --ml_n_estimators 300 \
    --xgb_tree_method hist \
    --n_jobs "${N_JOBS}" \
    "$@"
done
