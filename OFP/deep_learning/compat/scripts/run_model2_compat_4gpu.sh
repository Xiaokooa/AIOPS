#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
MODELS="${MODELS:-patchtst itransformer fteformer moderntcn fits}"
FOLDS="${FOLDS:-1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-192}"
SEQ_LEN="${SEQ_LEN:-32}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-10}"
POS_WEIGHT_CAP="${POS_WEIGHT_CAP:-20}"
THRESHOLD="${THRESHOLD:-0.3}"
TARGET_MODE="${TARGET_MODE:-module_fault}"
RULE_MODE="${RULE_MODE:-model2_simple}"
MIN_HIT_LEAD_HOURS="${MIN_HIT_LEAD_HOURS:-0}"
FEATURE_MODE="${FEATURE_MODE:-model2_plus}"
SAMPLING_MODE="${SAMPLING_MODE:-module_balanced}"
POSITIVE_WINDOWS_PER_MODULE="${POSITIVE_WINDOWS_PER_MODULE:-32}"
NEGATIVE_WINDOWS_PER_FAULTY_MODULE="${NEGATIVE_WINDOWS_PER_FAULTY_MODULE:-8}"
NORMAL_WINDOWS_PER_MODULE="${NORMAL_WINDOWS_PER_MODULE:-8}"
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
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/model2_feature_deep_4gpu_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"
read -r -a GPU_ARGS <<< "$GPU_IDS"
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

echo "============================================================"
echo "OFP model2-compatible deep row suite: 4-GPU launcher"
echo "============================================================"
echo "run_id:     $RUN_ID"
echo "gpus:       $GPU_IDS"
echo "models:     $MODELS"
echo "folds:      $FOLDS"
echo "epochs:     $EPOCHS"
echo "seq_len:    $SEQ_LEN"
echo "target:     $TARGET_MODE"
echo "rule_mode:  $RULE_MODE"
echo "min_hit_h:  $MIN_HIT_LEAD_HOURS"
echo "features:   $FEATURE_MODE"
echo "sampling:   $SAMPLING_MODE"
echo "quick caps: train=$MAX_TRAIN_FILES test=$MAX_TEST_FILES"
echo "threshold:  val search=$THRESHOLD_SEARCH metric=$THRESHOLD_METRIC fallback=$THRESHOLD"
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
    mkdir -p "$job_root"
    echo "[launcher] start model=${model} fold=${fold} gpu=${gpu} log=${job_log}"
    (
      CUDA_VISIBLE_DEVICES="$gpu" python -u -B OFP/deep_learning/compat/run_model2_compat_deep.py \
        --models "$model" \
        --folds "$fold" \
        --data_dir dataset/training \
        --index_path "dataset/train_test_set_index(in).csv" \
        --out_root "$job_root" \
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
        --rule_mode "$RULE_MODE" \
        --min_hit_lead_hours "$MIN_HIT_LEAD_HOURS" \
        --feature_mode "$FEATURE_MODE" \
        --sampling_mode "$SAMPLING_MODE" \
        --positive_windows_per_module "$POSITIVE_WINDOWS_PER_MODULE" \
        --negative_windows_per_faulty_module "$NEGATIVE_WINDOWS_PER_FAULTY_MODULE" \
        --normal_windows_per_module "$NORMAL_WINDOWS_PER_MODULE" \
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
