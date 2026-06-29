#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
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
SELECTION_METRIC="${SELECTION_METRIC:-f1_score}"
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
OUT_ROOT="${OUT_ROOT:-OFP_DL_semisup_mil_results/first_warning_semisup_4gpu_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"
read -r -a GPU_ARGS <<< "$GPU_IDS"
read -r -a FOLD_ARGS <<< "$FOLDS"
read -r -a SMOOTH_ARGS <<< "$ALARM_SMOOTH_WINDOWS"
read -r -a K_ARGS <<< "$ALARM_CONSECUTIVE_KS"

echo "============================================================"
echo "OFP semi-supervised first-warning MIL: 4-GPU launcher"
echo "============================================================"
echo "run_id:   $RUN_ID"
echo "gpus:     $GPU_IDS"
echo "folds:    $FOLDS"
echo "quick:    pretrain=${PRETRAIN_EPOCHS}, finetune=${FINETUNE_EPOCHS}, seq_len=${SEQ_LEN}"
echo "caps:     train=${MAX_TRAIN_FILES}, pretrain=${MAX_PRETRAIN_FILES}, val=${MAX_VAL_FILES}, test=${MAX_TEST_FILES}"
echo "select:   $SELECTION_METRIC"
echo "out_root: $OUT_ROOT"
echo "============================================================"

PIDS=()

wait_batch() {
  local status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  PIDS=()
  if [[ "$status" != "0" ]]; then
    echo "[launcher] at least one job failed; see $LOG_DIR"
    exit "$status"
  fi
}

job_idx=0
for fold in "${FOLD_ARGS[@]}"; do
  gpu="${GPU_ARGS[$((job_idx % ${#GPU_ARGS[@]}))]}"
  job_root="$OUT_ROOT/jobs/fold${fold}"
  job_log="$LOG_DIR/semisup_fold${fold}_gpu${gpu}.log"
  mkdir -p "$job_root"
  echo "[launcher] start fold=${fold} gpu=${gpu} log=${job_log}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python -u -B OFP_DL_semisup_mil/run_semisup_mil.py \
      --fold "$fold" \
      --data_dir dataset/training \
      --index_path "dataset/train_test_set_index(in).csv" \
      --out_root "$job_root" \
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
  ) > "$job_log" 2>&1 &
  PIDS+=("$!")
  job_idx=$((job_idx + 1))
  if [[ "${#PIDS[@]}" -ge "${#GPU_ARGS[@]}" ]]; then
    wait_batch
  fi
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_batch
fi

python -u -B OFP_DL_official/scripts/print_result_summary.py --root "$OUT_ROOT"
echo "[launcher] done. logs=$LOG_DIR"
