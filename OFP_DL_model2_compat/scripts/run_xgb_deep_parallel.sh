#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export RUN_ID
export ML_MODELS="${ML_MODELS:-xgb}"
export ML_FEATURE_SET="${ML_FEATURE_SET:-model2}"
export SELECTOR="${SELECTOR:-extra_trees}"
export SELECT_K="${SELECT_K:-64}"
export ENABLE_PARALLEL_FUSION="${ENABLE_PARALLEL_FUSION:-1}"
export PARALLEL_ML_MODEL="${PARALLEL_ML_MODEL:-xgb}"
export PARALLEL_FEATURE_SET="${PARALLEL_FEATURE_SET:-model2}"
export PARALLEL_SELECTOR="${PARALLEL_SELECTOR:-extra_trees}"
export PARALLEL_SELECT_K="${PARALLEL_SELECT_K:-64}"
export WRITE_DEEP_PREDICTIONS="${WRITE_DEEP_PREDICTIONS:-1}"
export OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/xgb_deep_parallel_${RUN_ID}}"

bash "$SCRIPT_DIR/run_tabular_fusion.sh"
