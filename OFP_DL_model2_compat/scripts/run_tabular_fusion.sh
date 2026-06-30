#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
MODELS="${MODELS:-patchtst itransformer fteformer moderntcn}"
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
ML_MODELS="${ML_MODELS:-rf xgb lgbm catboost}"
ML_FEATURE_SET="${ML_FEATURE_SET:-fusion}"
DEEP_FEATURE_PARTS="${DEEP_FEATURE_PARTS:-embedding,score}"
SELECTOR="${SELECTOR:-extra_trees}"
SELECT_K="${SELECT_K:-96}"
ML_N_ESTIMATORS="${ML_N_ESTIMATORS:-300}"
N_JOBS="${N_JOBS:-4}"
ENABLE_PARALLEL_FUSION="${ENABLE_PARALLEL_FUSION:-0}"
PARALLEL_ML_MODEL="${PARALLEL_ML_MODEL:-xgb}"
PARALLEL_FEATURE_SET="${PARALLEL_FEATURE_SET:-model2}"
PARALLEL_SELECTOR="${PARALLEL_SELECTOR:-extra_trees}"
PARALLEL_SELECT_K="${PARALLEL_SELECT_K:-64}"
WRITE_DEEP_PREDICTIONS="${WRITE_DEEP_PREDICTIONS:-0}"
MAX_CACHED_FILES="${MAX_CACHED_FILES:-128}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AMP="${AMP:-0}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/tabular_fusion_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"
LOG_PATH="$LOG_DIR/run_${RUN_ID}.log"

mkdir -p "$LOG_DIR"

read -r -a MODEL_ARGS <<< "$MODELS"
read -r -a FOLD_ARGS <<< "$FOLDS"
read -r -a ML_MODEL_ARGS <<< "$ML_MODELS"

AMP_ARGS=()
if [[ "$AMP" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi
SEARCH_ARGS=()
if [[ "$THRESHOLD_SEARCH" != "1" ]]; then
  SEARCH_ARGS+=(--no_threshold_search)
fi
PARALLEL_ARGS=()
if [[ "$ENABLE_PARALLEL_FUSION" == "1" ]]; then
  PARALLEL_ARGS+=(--enable_parallel_fusion)
fi
DEEP_ARGS=()
if [[ "$WRITE_DEEP_PREDICTIONS" == "1" ]]; then
  DEEP_ARGS+=(--write_deep_predictions)
fi

{
  echo "============================================================"
  echo "OFP model2-compatible tabular fusion suite"
  echo "============================================================"
  echo "run_id:       $RUN_ID"
  echo "deep models:  $MODELS"
  echo "ml models:    $ML_MODELS"
  echo "folds:        $FOLDS"
  echo "device:       cuda:$GPU_ID"
  echo "epochs:       $EPOCHS"
  echo "seq_len:      $SEQ_LEN"
  echo "feature_set:  $ML_FEATURE_SET"
  echo "deep parts:   $DEEP_FEATURE_PARTS"
  echo "selector:     $SELECTOR top_k=$SELECT_K"
  echo "parallel:     $ENABLE_PARALLEL_FUSION model=$PARALLEL_ML_MODEL feature_set=$PARALLEL_FEATURE_SET"
  echo "target:       $TARGET_MODE"
  echo "rule_mode:    $RULE_MODE"
  echo "min_hit_h:    $MIN_HIT_LEAD_HOURS"
  echo "threshold:    val search=$THRESHOLD_SEARCH metric=$THRESHOLD_METRIC fallback=$THRESHOLD"
  echo "quick caps:   train=$MAX_TRAIN_FILES test=$MAX_TEST_FILES"
  echo "out_root:     $OUT_ROOT"
  echo "log:          $LOG_PATH"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -B OFP_DL_model2_compat/run_model2_compat_tabular_fusion.py \
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
    --ml_models "${ML_MODEL_ARGS[@]}" \
    --ml_feature_set "$ML_FEATURE_SET" \
    --deep_feature_parts "$DEEP_FEATURE_PARTS" \
    --selector "$SELECTOR" \
    --select_k "$SELECT_K" \
    --ml_n_estimators "$ML_N_ESTIMATORS" \
    --n_jobs "$N_JOBS" \
    --parallel_ml_model "$PARALLEL_ML_MODEL" \
    --parallel_feature_set "$PARALLEL_FEATURE_SET" \
    --parallel_selector "$PARALLEL_SELECTOR" \
    --parallel_select_k "$PARALLEL_SELECT_K" \
    --max_cached_files "$MAX_CACHED_FILES" \
    --max_train_files "$MAX_TRAIN_FILES" \
    --max_test_files "$MAX_TEST_FILES" \
    --num_workers "$NUM_WORKERS" \
    --log_batches 0 \
    "${SEARCH_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${DEEP_ARGS[@]}" \
    "${AMP_ARGS[@]}"

  python -u -B OFP_DL_official/scripts/print_result_summary.py --root "$OUT_ROOT"
} 2>&1 | tee "$LOG_PATH"
