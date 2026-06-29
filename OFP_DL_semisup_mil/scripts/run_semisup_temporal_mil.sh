#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID
export OUT_ROOT="${OUT_ROOT:-OFP_DL_semisup_mil_results/first_warning_temporal_mil_${RUN_ID}}"

# Lightweight preview defaults for a 4090-class GPU.
# Override any value from the shell when running formal experiments.
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
export SEQ_LEN="${SEQ_LEN:-96}"
export D_MODEL="${D_MODEL:-64}"
export LAYERS="${LAYERS:-2}"
export N_HEADS="${N_HEADS:-4}"
export PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
export MODULE_BATCH_SIZE="${MODULE_BATCH_SIZE:-8}"
export SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-512}"
export WINDOWS_PER_MODULE="${WINDOWS_PER_MODULE:-12}"
export POSITIVE_WINDOWS_PER_FAULTY="${POSITIVE_WINDOWS_PER_FAULTY:-8}"
export AUX_BCE_WEIGHT="${AUX_BCE_WEIGHT:-0.3}"
export EVENT_LOSS_WEIGHT="${EVENT_LOSS_WEIGHT:-1.0}"
export SELECTION_METRIC="${SELECTION_METRIC:-f1_score}"
export LR="${LR:-0.0003}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export MAX_CACHED_FILES="${MAX_CACHED_FILES:-128}"
export MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
export MAX_PRETRAIN_FILES="${MAX_PRETRAIN_FILES:-2500}"
export MAX_VAL_FILES="${MAX_VAL_FILES:-600}"
export MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
export THRESHOLD_GRID_SIZE="${THRESHOLD_GRID_SIZE:-25}"

bash "$SCRIPT_DIR/run_semisup_first_warning.sh"
