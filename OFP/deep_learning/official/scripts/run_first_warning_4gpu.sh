#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
MODELS="${MODELS:-patchtst itransformer fteformer moderntcn fits}"
FOLDS="${FOLDS:-1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-96}"
SEQ_LEN="${SEQ_LEN:-288}"
INPUT_MODE="${INPUT_MODE:-model2_features}"
TRAIN_SAMPLING="${TRAIN_SAMPLING:-module_balanced}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-10}"
POS_WEIGHT_CAP="${POS_WEIGHT_CAP:-20}"
AUX_BCE_WEIGHT="${AUX_BCE_WEIGHT:-0.3}"
EVENT_LOSS_WEIGHT="${EVENT_LOSS_WEIGHT:-1.0}"
EVENT_HIT_MODE="${EVENT_HIT_MODE:-primary_window}"
EVENT_MODULES_PER_EPOCH="${EVENT_MODULES_PER_EPOCH:-96}"
EVENT_MAX_WINDOWS_PER_MODULE="${EVENT_MAX_WINDOWS_PER_MODULE:-96}"
EVENT_MODULE_BATCH_SIZE="${EVENT_MODULE_BATCH_SIZE:-4}"
EVENT_POSITIVE_FRACTION="${EVENT_POSITIVE_FRACTION:-0.5}"
MODULE_POSITIVE_WINDOWS_PER_MODULE="${MODULE_POSITIVE_WINDOWS_PER_MODULE:-32}"
MODULE_NEGATIVE_WINDOWS_PER_FAULTY_MODULE="${MODULE_NEGATIVE_WINDOWS_PER_FAULTY_MODULE:-8}"
MODULE_NORMAL_WINDOWS_PER_MODULE="${MODULE_NORMAL_WINDOWS_PER_MODULE:-8}"
INDEX_VAL_FRACTION="${INDEX_VAL_FRACTION:-0.1}"
SELECTION_METRIC="${SELECTION_METRIC:-f1_score}"
ALARM_SMOOTHING="${ALARM_SMOOTHING:-ema}"
ALARM_SMOOTH_WINDOW="${ALARM_SMOOTH_WINDOW:-1}"
ALARM_CONSECUTIVE_K="${ALARM_CONSECUTIVE_K:-1}"
ALARM_SMOOTH_WINDOWS="${ALARM_SMOOTH_WINDOWS:-1 3 6 12 24}"
ALARM_CONSECUTIVE_KS="${ALARM_CONSECUTIVE_KS:-1 2 3}"
THRESHOLD="${THRESHOLD:-0.5}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1}"
MAX_CACHED_FILES="${MAX_CACHED_FILES:-128}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
MAX_VAL_FILES="${MAX_VAL_FILES:-600}"
MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AMP="${AMP:-0}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_official_results/lookback24h_ahead1h_legacy_4gpu_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"
read -r -a GPU_ARGS <<< "$GPU_IDS"
read -r -a MODEL_ARGS <<< "$MODELS"
read -r -a FOLD_ARGS <<< "$FOLDS"
read -r -a SMOOTH_ARGS <<< "$ALARM_SMOOTH_WINDOWS"
read -r -a K_ARGS <<< "$ALARM_CONSECUTIVE_KS"

AMP_ARGS=()
if [[ "$AMP" == "1" ]]; then
  AMP_ARGS+=(--amp)
elif [[ "$AMP" == "0" ]]; then
  AMP_ARGS+=(--no_amp)
fi

echo "============================================================"
echo "OFP first-warning deep suite: 4-GPU launcher"
echo "============================================================"
echo "run_id:     $RUN_ID"
echo "gpus:       $GPU_IDS"
echo "models:     $MODELS"
echo "folds:      $FOLDS"
echo "epochs:     $EPOCHS"
echo "input_mode: $INPUT_MODE"
echo "task:       24h lookback -> 1h-ahead first warning, legacy timestamp"
echo "sampling:   $TRAIN_SAMPLING"
echo "quick caps: train=${MAX_TRAIN_FILES}, val=${MAX_VAL_FILES}, test=${MAX_TEST_FILES}"
echo "amp:        $AMP"
echo "pos_w_cap:  $POS_WEIGHT_CAP"
echo "val_select: ${INDEX_VAL_FRACTION}/${SELECTION_METRIC}"
echo "out_root:   $OUT_ROOT"
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
for model in "${MODEL_ARGS[@]}"; do
  for fold in "${FOLD_ARGS[@]}"; do
    gpu="${GPU_ARGS[$((job_idx % ${#GPU_ARGS[@]}))]}"
    job_root="$OUT_ROOT/jobs/${model}_fold${fold}"
    job_log="$LOG_DIR/${model}_fold${fold}_gpu${gpu}.log"
    job_cache="$job_root/module_cache"
    mkdir -p "$job_root" "$job_cache"
    echo "[launcher] start model=${model} fold=${fold} gpu=${gpu} log=${job_log}"
    (
      CUDA_VISIBLE_DEVICES="$gpu" python -u -B OFP/deep_learning/official/run_official_benchmark.py \
        --models "$model" \
        --folds "$fold" \
        --data_dir dataset/training \
        --index_path "dataset/train_test_set_index(in).csv" \
        --out_root "$job_root" \
        --timestamp_mode legacy_float32 \
        --input_mode "$INPUT_MODE" \
        --train_sampling "$TRAIN_SAMPLING" \
        --negative_ratio "$NEGATIVE_RATIO" \
        --pos_weight_cap "$POS_WEIGHT_CAP" \
        --aux_bce_weight "$AUX_BCE_WEIGHT" \
        --event_loss_weight "$EVENT_LOSS_WEIGHT" \
        --event_hit_mode "$EVENT_HIT_MODE" \
        --event_modules_per_epoch "$EVENT_MODULES_PER_EPOCH" \
        --event_max_windows_per_module "$EVENT_MAX_WINDOWS_PER_MODULE" \
        --event_module_batch_size "$EVENT_MODULE_BATCH_SIZE" \
        --event_positive_fraction "$EVENT_POSITIVE_FRACTION" \
        --module_positive_windows_per_module "$MODULE_POSITIVE_WINDOWS_PER_MODULE" \
        --module_negative_windows_per_faulty_module "$MODULE_NEGATIVE_WINDOWS_PER_FAULTY_MODULE" \
        --module_normal_windows_per_module "$MODULE_NORMAL_WINDOWS_PER_MODULE" \
        --index_val_fraction "$INDEX_VAL_FRACTION" \
        --selection_metric "$SELECTION_METRIC" \
        --alarm_smoothing "$ALARM_SMOOTHING" \
        --alarm_smooth_window "$ALARM_SMOOTH_WINDOW" \
        --alarm_consecutive_k "$ALARM_CONSECUTIVE_K" \
        --alarm_search_smooth_windows "${SMOOTH_ARGS[@]}" \
        --alarm_search_consecutive_ks "${K_ARGS[@]}" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --seq_len "$SEQ_LEN" \
        --device cuda \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --grad_clip "$GRAD_CLIP" \
        --fixed_threshold "$THRESHOLD" \
        --max_cached_files "$MAX_CACHED_FILES" \
        --max_train_files "$MAX_TRAIN_FILES" \
        --max_val_files "$MAX_VAL_FILES" \
        --max_test_files "$MAX_TEST_FILES" \
        --num_workers "$NUM_WORKERS" \
        --module_cache_dir "$job_cache" \
        --log_batches 0 \
        --no_tqdm \
        "${AMP_ARGS[@]}"
    ) > "$job_log" 2>&1 &
    PIDS+=("$!")
    job_idx=$((job_idx + 1))
    if [[ "${#PIDS[@]}" -ge "${#GPU_ARGS[@]}" ]]; then
      wait_batch
    fi
  done
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_batch
fi

python -u -B OFP/deep_learning/official/scripts/print_result_summary.py --root "$OUT_ROOT"
echo "[launcher] done. logs=$LOG_DIR"
