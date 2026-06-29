#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
FOLDS="${FOLDS:-1}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
SEQ_LEN="${SEQ_LEN:-96}"
D_MODEL="${D_MODEL:-64}"
LAYERS="${LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
MODULE_BATCH_SIZE="${MODULE_BATCH_SIZE:-8}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-512}"
WINDOWS_PER_MODULE="${WINDOWS_PER_MODULE:-12}"
POSITIVE_WINDOWS_PER_FAULTY="${POSITIVE_WINDOWS_PER_FAULTY:-8}"
AUX_BCE_WEIGHT="${AUX_BCE_WEIGHT:-0.3}"
EVENT_LOSS_WEIGHT="${EVENT_LOSS_WEIGHT:-1.0}"
SELECTION_METRIC="${SELECTION_METRIC:-final_score}"
ALARM_SMOOTHING="${ALARM_SMOOTHING:-ema}"
ALARM_SMOOTH_WINDOWS="${ALARM_SMOOTH_WINDOWS:-1 3 6 12}"
ALARM_CONSECUTIVE_KS="${ALARM_CONSECUTIVE_KS:-1 2 3}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MAX_CACHED_FILES="${MAX_CACHED_FILES:-128}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
MAX_PRETRAIN_FILES="${MAX_PRETRAIN_FILES:-2500}"
MAX_VAL_FILES="${MAX_VAL_FILES:-600}"
MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
THRESHOLD_GRID_SIZE="${THRESHOLD_GRID_SIZE:-25}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_semisup_mil_results/first_warning_semisup_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"
LOG_PATH="$LOG_DIR/run_${RUN_ID}.log"

mkdir -p "$LOG_DIR"
read -r -a FOLD_ARGS <<< "$FOLDS"
read -r -a SMOOTH_ARGS <<< "$ALARM_SMOOTH_WINDOWS"
read -r -a K_ARGS <<< "$ALARM_CONSECUTIVE_KS"

{
  echo "============================================================"
  echo "OFP semi-supervised first-warning MIL"
  echo "============================================================"
  echo "run_id:          $RUN_ID"
  echo "folds:           $FOLDS"
  echo "device:          cuda:$GPU_ID"
  echo "pretrain_epochs: $PRETRAIN_EPOCHS"
  echo "finetune_epochs: $FINETUNE_EPOCHS"
  echo "seq_len:         $SEQ_LEN"
  echo "module_batch:    $MODULE_BATCH_SIZE"
  echo "loss:            event=${EVENT_LOSS_WEIGHT}, aux_bce=${AUX_BCE_WEIGHT}"
  echo "select metric:   $SELECTION_METRIC"
  echo "alarm search:    smoothing=${ALARM_SMOOTHING}, windows=${ALARM_SMOOTH_WINDOWS}, K=${ALARM_CONSECUTIVE_KS}"
  echo "quick caps:      train=${MAX_TRAIN_FILES}, pretrain=${MAX_PRETRAIN_FILES}, val=${MAX_VAL_FILES}, test=${MAX_TEST_FILES}"
  echo "out_root:        $OUT_ROOT"
  echo "log:             $LOG_PATH"
  echo "============================================================"

  for fold in "${FOLD_ARGS[@]}"; do
    echo
    echo "---------------- fold ${fold} ----------------"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -B OFP_DL_semisup_mil/run_semisup_mil.py \
      --fold "$fold" \
      --data_dir dataset/training \
      --index_path "dataset/train_test_set_index(in).csv" \
      --out_root "$OUT_ROOT" \
      --val_fraction "$VAL_FRACTION" \
      --max_train_files "$MAX_TRAIN_FILES" \
      --max_pretrain_files "$MAX_PRETRAIN_FILES" \
      --max_val_files "$MAX_VAL_FILES" \
      --max_test_files "$MAX_TEST_FILES" \
      --seq_len "$SEQ_LEN" \
      --d_model "$D_MODEL" \
      --layers "$LAYERS" \
      --n_heads "$N_HEADS" \
      --pretrain_epochs "$PRETRAIN_EPOCHS" \
      --finetune_epochs "$FINETUNE_EPOCHS" \
      --pretrain_batch_size "$PRETRAIN_BATCH_SIZE" \
      --module_batch_size "$MODULE_BATCH_SIZE" \
      --score_batch_size "$SCORE_BATCH_SIZE" \
      --windows_per_module "$WINDOWS_PER_MODULE" \
      --positive_windows_per_faulty "$POSITIVE_WINDOWS_PER_FAULTY" \
      --aux_bce_weight "$AUX_BCE_WEIGHT" \
      --event_loss_weight "$EVENT_LOSS_WEIGHT" \
      --selection_metric "$SELECTION_METRIC" \
      --alarm_smoothing "$ALARM_SMOOTHING" \
      --alarm_search_smooth_windows "${SMOOTH_ARGS[@]}" \
      --alarm_search_consecutive_ks "${K_ARGS[@]}" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --threshold_grid_size "$THRESHOLD_GRID_SIZE" \
      --timestamp_mode legacy_float32 \
      --device cuda \
      --max_cached_files "$MAX_CACHED_FILES"
  done

  python -u -B OFP_DL_official/scripts/print_result_summary.py --root "$OUT_ROOT"
} 2>&1 | tee "$LOG_PATH"
