#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
FOLDS="${FOLDS:-1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WIN_SIZE="${WIN_SIZE:-288}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-4}"
ENCODER="${ENCODER:-lstm}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
PROJ_DIM="${PROJ_DIM:-64}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NOISE_LEVEL="${NOISE_LEVEL:-0.05}"
BIAS_SCALE="${BIAS_SCALE:-0.5}"
BIAS_PERCENT="${BIAS_PERCENT:-0.5}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.1}"
SCORE_STRIDE="${SCORE_STRIDE:-12}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1024}"
CENTROID_WINDOWS_PER_FILE="${CENTROID_WINDOWS_PER_FILE:-2}"
MAX_CENTROID_WINDOWS="${MAX_CENTROID_WINDOWS:-10000}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1_score}"
SMOOTHING="${SMOOTHING:-ema}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-1}"
CONSECUTIVE_K="${CONSECUTIVE_K:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
TRAIN_SOURCE="${TRAIN_SOURCE:-normal_modules}"
MODEL_NAME="${MODEL_NAME:-carots_ofp}"
TASK_NAME="${TASK_NAME:-24h lookback score -> first-warning evaluator, legacy timestamp}"
OUT_ROOT="${OUT_ROOT:-OFP_external_adapter_results/lookback24h_ahead1h_carots_ofp_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"
LOG_PATH="$LOG_DIR/run_${RUN_ID}.log"

mkdir -p "$LOG_DIR"
read -r -a FOLD_ARGS <<< "$FOLDS"

TRAIN_ARGS=()
if [[ "$TRAIN_SOURCE" == "all_modules" ]]; then
  TRAIN_ARGS+=(--train_on_all_modules)
else
  TRAIN_ARGS+=(--train_on_normal_modules_only)
fi

{
  echo "============================================================"
  echo "OFP CAROTS train-score-evaluate"
  echo "============================================================"
  echo "run_id:       $RUN_ID"
  echo "folds:        $FOLDS"
  echo "device:       cuda:$GPU_ID"
  echo "model_name:   $MODEL_NAME"
  echo "task:         $TASK_NAME"
  echo "encoder:      $ENCODER"
  echo "epochs:       $EPOCHS"
  echo "batch_size:   $BATCH_SIZE"
  echo "win_size:     $WIN_SIZE"
  echo "samples/file: $SAMPLES_PER_FILE"
  echo "train_source: $TRAIN_SOURCE"
  echo "loss:         sim_threshold=$SIM_THRESHOLD temp=$TEMPERATURE"
  echo "augment:      noise=$NOISE_LEVEL bias=$BIAS_SCALE percent=$BIAS_PERCENT"
  echo "score:        stride=$SCORE_STRIDE centroid_per_file=$CENTROID_WINDOWS_PER_FILE"
  echo "quick caps:   train=$MAX_TRAIN_FILES test=$MAX_TEST_FILES"
  echo "alarm:        smoothing=$SMOOTHING window=$SMOOTH_WINDOW K=$CONSECUTIVE_K"
  echo "threshold:    metric=$THRESHOLD_METRIC grid=$THRESHOLD_GRID"
  echo "out_root:     $OUT_ROOT"
  echo "log:          $LOG_PATH"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -B OFP_external_adapters/run_carots_ofp_train_eval.py \
    --model_name "$MODEL_NAME" \
    --data_dir dataset/training \
    --index_path "dataset/train_test_set_index(in).csv" \
    --out_root "$OUT_ROOT" \
    --folds "${FOLD_ARGS[@]}" \
    --device cuda \
    --encoder "$ENCODER" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --win_size "$WIN_SIZE" \
    --samples_per_file "$SAMPLES_PER_FILE" \
    --hidden_dim "$HIDDEN_DIM" \
    --proj_dim "$PROJ_DIM" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --noise_level "$NOISE_LEVEL" \
    --bias_scale "$BIAS_SCALE" \
    --bias_percent "$BIAS_PERCENT" \
    --sim_threshold "$SIM_THRESHOLD" \
    --temperature "$TEMPERATURE" \
    --score_stride "$SCORE_STRIDE" \
    --score_batch_size "$SCORE_BATCH_SIZE" \
    --centroid_windows_per_file "$CENTROID_WINDOWS_PER_FILE" \
    --max_centroid_windows "$MAX_CENTROID_WINDOWS" \
    --threshold_grid "$THRESHOLD_GRID" \
    --threshold_metric "$THRESHOLD_METRIC" \
    --val_fraction "$VAL_FRACTION" \
    --smoothing "$SMOOTHING" \
    --smooth_window "$SMOOTH_WINDOW" \
    --consecutive_k "$CONSECUTIVE_K" \
    --num_workers "$NUM_WORKERS" \
    --max_train_files "$MAX_TRAIN_FILES" \
    --max_test_files "$MAX_TEST_FILES" \
    "${TRAIN_ARGS[@]}"

  python -u -B OFP_DL_official/scripts/print_result_summary.py --root "$OUT_ROOT"
} 2>&1 | tee "$LOG_PATH"
