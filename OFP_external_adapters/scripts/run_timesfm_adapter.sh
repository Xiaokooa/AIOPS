#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FOLDS="${FOLDS:-1}"
GPU_ID="${GPU_ID:-0}"
MODEL_NAME="${MODEL_NAME:-timesfm}"
MODEL_ID="${MODEL_ID:-google/timesfm-2.5-200m-pytorch}"
CACHE_DIR="${CACHE_DIR:-model/timesfm_hf_cache}"
TASK_NAME="${TASK_NAME:-24h lookback -> 1h-ahead first warning on 5min grid, legacy timestamp}"
CONTEXT="${CONTEXT:-288}"
MIN_CONTEXT="${MIN_CONTEXT:-24}"
FORECAST_MIN_CONTEXT="${FORECAST_MIN_CONTEXT:-288}"
HORIZON="${HORIZON:-12}"
RESAMPLE_SECONDS="${RESAMPLE_SECONDS:-300}"
STRIDE="${STRIDE:-1}"
MAX_WINDOWS_PER_FILE="${MAX_WINDOWS_PER_FILE:-16}"
POSITION_SAMPLE="${POSITION_SAMPLE:-even}"
PER_CORE_BATCH_SIZE="${PER_CORE_BATCH_SIZE:-128}"
MOCK_FORECAST="${MOCK_FORECAST:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
RISK_MODE="${RISK_MODE:-hybrid_sum}"
NORMAL_SIGMA="${NORMAL_SIGMA:-2.0}"
DRIFT_SIGMA="${DRIFT_SIGMA:-1.0}"
RECENT_TAIL="${RECENT_TAIL:-24}"
FORECAST_WEIGHT="${FORECAST_WEIGHT:-1.0}"
CURRENT_WEIGHT="${CURRENT_WEIGHT:-1.0}"
RECENT_WEIGHT="${RECENT_WEIGHT:-0.7}"
DRIFT_WEIGHT="${DRIFT_WEIGHT:-0.3}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.000001,0.00001,0.0001,0.001,0.003,0.005,0.01,0.03,0.05,0.10,0.20,0.30,0.50,0.75,1.0,1.5,2.0,3.0,5.0,8.0,10.0}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1_score}"
SMOOTHING="${SMOOTHING:-ema}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-1}"
CONSECUTIVE_K="${CONSECUTIVE_K:-1}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-1000}"
MAX_TEST_FILES="${MAX_TEST_FILES:-300}"
OUT_ROOT="${OUT_ROOT:-OFP_external_adapter_results/lookback24h_ahead1h_timesfm_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"
LOG_PATH="$LOG_DIR/run_${RUN_ID}.log"

mkdir -p "$LOG_DIR"
read -r -a FOLD_ARGS <<< "$FOLDS"

MOCK_ARGS=()
if [[ "$MOCK_FORECAST" == "1" ]]; then
  MOCK_ARGS+=(--mock_forecast)
fi

{
  echo "============================================================"
  echo "OFP TimesFM forecast-risk adapter"
  echo "============================================================"
  echo "run_id:      $RUN_ID"
  echo "folds:       $FOLDS"
  echo "device:      cuda:$GPU_ID"
  echo "model_id:    $MODEL_ID"
  echo "task:        $TASK_NAME"
  echo "cache_dir:   $CACHE_DIR"
  echo "mock:        $MOCK_FORECAST"
  echo "risk_mode:   $RISK_MODE"
  echo "context:     $CONTEXT"
  echo "min_context: $MIN_CONTEXT"
  echo "forecast_min_context: $FORECAST_MIN_CONTEXT"
  echo "horizon:     $HORIZON"
  echo "resample_s:  $RESAMPLE_SECONDS"
  echo "stride:      $STRIDE"
  echo "max_windows: $MAX_WINDOWS_PER_FILE"
  echo "sample_mode: $POSITION_SAMPLE"
  echo "quick caps:  train=$MAX_TRAIN_FILES test=$MAX_TEST_FILES"
  echo "risk parts:  sigma=$NORMAL_SIGMA drift_sigma=$DRIFT_SIGMA recent_tail=$RECENT_TAIL"
  echo "weights:     forecast=$FORECAST_WEIGHT current=$CURRENT_WEIGHT recent=$RECENT_WEIGHT drift=$DRIFT_WEIGHT"
  echo "alarm:       smoothing=$SMOOTHING window=$SMOOTH_WINDOW K=$CONSECUTIVE_K"
  echo "threshold:   metric=$THRESHOLD_METRIC grid=$THRESHOLD_GRID"
  echo "out_root:    $OUT_ROOT"
  echo "log:         $LOG_PATH"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -B OFP_external_adapters/run_timesfm_ofp_adapter.py \
    --model_name "$MODEL_NAME" \
    --model_id "$MODEL_ID" \
    --cache_dir "$CACHE_DIR" \
    --data_dir dataset/training \
    --index_path "dataset/train_test_set_index(in).csv" \
    --out_root "$OUT_ROOT" \
    --folds "${FOLD_ARGS[@]}" \
    --context "$CONTEXT" \
    --min_context "$MIN_CONTEXT" \
    --forecast_min_context "$FORECAST_MIN_CONTEXT" \
    --horizon "$HORIZON" \
    --resample_seconds "$RESAMPLE_SECONDS" \
    --stride "$STRIDE" \
    --max_windows_per_file "$MAX_WINDOWS_PER_FILE" \
    --position_sample "$POSITION_SAMPLE" \
    --per_core_batch_size "$PER_CORE_BATCH_SIZE" \
    --risk_mode "$RISK_MODE" \
    --normal_sigma "$NORMAL_SIGMA" \
    --drift_sigma "$DRIFT_SIGMA" \
    --recent_tail "$RECENT_TAIL" \
    --forecast_weight "$FORECAST_WEIGHT" \
    --current_weight "$CURRENT_WEIGHT" \
    --recent_weight "$RECENT_WEIGHT" \
    --drift_weight "$DRIFT_WEIGHT" \
    --threshold_grid "$THRESHOLD_GRID" \
    --threshold_metric "$THRESHOLD_METRIC" \
    --val_fraction "$VAL_FRACTION" \
    --smoothing "$SMOOTHING" \
    --smooth_window "$SMOOTH_WINDOW" \
    --consecutive_k "$CONSECUTIVE_K" \
    --max_train_files "$MAX_TRAIN_FILES" \
    --max_test_files "$MAX_TEST_FILES" \
    "${MOCK_ARGS[@]}"

  python -u -B OFP_DL_official/scripts/print_result_summary.py --root "$OUT_ROOT"
} 2>&1 | tee "$LOG_PATH"
