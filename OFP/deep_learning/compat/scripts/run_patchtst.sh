#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID
export MODELS="${MODELS:-patchtst}"
export OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/model2_feature_patchtst_${RUN_ID}}"
export EPOCHS="${EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-192}"
export SEQ_LEN="${SEQ_LEN:-32}"
export NEGATIVE_RATIO="${NEGATIVE_RATIO:-10}"
export POS_WEIGHT_CAP="${POS_WEIGHT_CAP:-20}"
export TARGET_MODE="${TARGET_MODE:-module_fault}"
export RULE_MODE="${RULE_MODE:-model2_simple}"
export FEATURE_MODE="${FEATURE_MODE:-model2_plus}"
export SAMPLING_MODE="${SAMPLING_MODE:-module_balanced}"
export POSITIVE_WINDOWS_PER_MODULE="${POSITIVE_WINDOWS_PER_MODULE:-32}"
export NEGATIVE_WINDOWS_PER_FAULTY_MODULE="${NEGATIVE_WINDOWS_PER_FAULTY_MODULE:-8}"
export NORMAL_WINDOWS_PER_MODULE="${NORMAL_WINDOWS_PER_MODULE:-8}"
export THRESHOLD_SEARCH="${THRESHOLD_SEARCH:-1}"
export THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1_score}"
export LR="${LR:-0.0003}"
export AMP="${AMP:-0}"

bash "$SCRIPT_DIR/run_model2_compat_suite.sh"
