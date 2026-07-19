#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
MODELS="${MODELS:-patchtst itransformer fteformer moderntcn fits}"
FOLDS="${FOLDS:-1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-192}"
SEQ_LEN="${SEQ_LEN:-32}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-10}"
POS_WEIGHT_CAP="${POS_WEIGHT_CAP:-20}"
THRESHOLD="${THRESHOLD:-0.3}"
TARGET_MODE="${TARGET_MODE:-module_fault}"
FEATURE_MODE="${FEATURE_MODE:-model2_plus}"
SAMPLING_MODE="${SAMPLING_MODE:-module_balanced}"
POSITIVE_WINDOWS_PER_MODULE="${POSITIVE_WINDOWS_PER_MODULE:-32}"
NEGATIVE_WINDOWS_PER_FAULTY_MODULE="${NEGATIVE_WINDOWS_PER_FAULTY_MODULE:-8}"
NORMAL_WINDOWS_PER_MODULE="${NORMAL_WINDOWS_PER_MODULE:-8}"
RULE_MODE="${RULE_MODE:-model2_simple}"
MIN_HIT_LEAD_HOURS="${MIN_HIT_LEAD_HOURS:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
THRESHOLD_SEARCH="${THRESHOLD_SEARCH:-1}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1_score}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1}"
MAX_CACHED_FILES="${MAX_CACHED_FILES:-128}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AMP="${AMP:-0}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/model2_feature_deep_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"
LOG_PATH="$LOG_DIR/run_${RUN_ID}.log"

mkdir -p "$LOG_DIR"

read -r -a MODEL_ARGS <<< "$MODELS"
read -r -a FOLD_ARGS <<< "$FOLDS"

AMP_ARGS=()
if [[ "$AMP" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi
SEARCH_ARGS=()
if [[ "$THRESHOLD_SEARCH" != "1" ]]; then
  SEARCH_ARGS+=(--no_threshold_search)
fi

{
  echo "============================================================"
  echo "OFP model2-compatible deep row suite"
  echo "============================================================"
  echo "run_id:       $RUN_ID"
  echo "models:       $MODELS"
  echo "folds:        $FOLDS"
  echo "device:       cuda:$GPU_ID"
  echo "epochs:       $EPOCHS"
  echo "seq_len:      $SEQ_LEN"
  echo "batch_size:   $BATCH_SIZE"
  echo "amp:          $AMP"
  echo "features:     ML model2 engineered features"
  echo "target:       $TARGET_MODE"
  echo "feature_mode: $FEATURE_MODE"
  echo "sampling:     $SAMPLING_MODE"
  echo "module caps:  pos=$POSITIVE_WINDOWS_PER_MODULE faulty_neg=$NEGATIVE_WINDOWS_PER_FAULTY_MODULE normal=$NORMAL_WINDOWS_PER_MODULE"
  echo "quick caps:   train=$MAX_TRAIN_FILES test=$MAX_TEST_FILES"
  echo "rule_mode:    $RULE_MODE"
  echo "min_hit_h:    $MIN_HIT_LEAD_HOURS"
  echo "prediction:   legacy feature timestamps + deep/rule OR"
  echo "threshold:    val search=$THRESHOLD_SEARCH metric=$THRESHOLD_METRIC fallback=$THRESHOLD"
  echo "thresholds:   $THRESHOLD_GRID"
  echo "out_root:     $OUT_ROOT"
  echo "log:          $LOG_PATH"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -B OFP/deep_learning/compat/run_model2_compat_deep.py \
    --models "${MODEL_ARGS[@]}" \
    --folds "${FOLD_ARGS[@]}" \
    --data_dir dataset/training \
    --index_path "dataset/train_test_set_index(in).csv" \
    --out_root "$OUT_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --device cuda \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --grad_clip "$GRAD_CLIP" \
    --negative_ratio "$NEGATIVE_RATIO" \
    --pos_weight_cap "$POS_WEIGHT_CAP" \
    --fixed_threshold "$THRESHOLD" \
    --target_mode "$TARGET_MODE" \
    --feature_mode "$FEATURE_MODE" \
    --sampling_mode "$SAMPLING_MODE" \
    --positive_windows_per_module "$POSITIVE_WINDOWS_PER_MODULE" \
    --negative_windows_per_faulty_module "$NEGATIVE_WINDOWS_PER_FAULTY_MODULE" \
    --normal_windows_per_module "$NORMAL_WINDOWS_PER_MODULE" \
    --rule_mode "$RULE_MODE" \
    --min_hit_lead_hours "$MIN_HIT_LEAD_HOURS" \
    --val_fraction "$VAL_FRACTION" \
    --threshold_grid "$THRESHOLD_GRID" \
    --threshold_metric "$THRESHOLD_METRIC" \
    --max_cached_files "$MAX_CACHED_FILES" \
    --max_train_files "$MAX_TRAIN_FILES" \
    --max_test_files "$MAX_TEST_FILES" \
    --num_workers "$NUM_WORKERS" \
    --log_batches 0 \
    "${SEARCH_ARGS[@]}" \
    "${AMP_ARGS[@]}"

  python -u -B OFP/deep_learning/official/scripts/print_result_summary.py --root "$OUT_ROOT"
} 2>&1 | tee "$LOG_PATH"
