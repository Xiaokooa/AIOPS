#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FOLDS="${FOLDS:-1}"
SCORE_DIR="${CAROTS_SCORE_DIR:-}"
SCORE_MODE="${SCORE_MODE:-external}"
MODEL_NAME="${MODEL_NAME:-carots}"
TASK_NAME="${TASK_NAME:-24h lookback score -> first-warning evaluator, legacy timestamp}"
WIN_SIZE="${WIN_SIZE:-288}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50}"
THRESHOLD_METRIC="${THRESHOLD_METRIC:-f1_score}"
SMOOTHING="${SMOOTHING:-ema}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-1}"
CONSECUTIVE_K="${CONSECUTIVE_K:-1}"
MAX_TRAIN_FILES="${MAX_TRAIN_FILES:-2500}"
MAX_TEST_FILES="${MAX_TEST_FILES:-1200}"
OUT_ROOT="${OUT_ROOT:-OFP_external_adapter_results/lookback24h_ahead1h_carots_${RUN_ID}}"
LOG_DIR="$OUT_ROOT/logs"
LOG_PATH="$LOG_DIR/run_${RUN_ID}.log"

mkdir -p "$LOG_DIR"
read -r -a FOLD_ARGS <<< "$FOLDS"

SCORE_ARGS=()
if [[ -n "$SCORE_DIR" ]]; then
  SCORE_ARGS+=(--score_dir "$SCORE_DIR")
else
  SCORE_ARGS+=(--score_mode "$SCORE_MODE")
fi

{
  echo "============================================================"
  echo "OFP CAROTS score adapter"
  echo "============================================================"
  echo "run_id:      $RUN_ID"
  echo "folds:       $FOLDS"
  echo "score_dir:   ${SCORE_DIR:-<none>}"
  echo "score_mode:  $SCORE_MODE"
  echo "model_name:  $MODEL_NAME"
  echo "task:        $TASK_NAME"
  echo "win_size:    $WIN_SIZE"
  echo "quick caps:  train=$MAX_TRAIN_FILES test=$MAX_TEST_FILES"
  echo "alarm:       smoothing=$SMOOTHING window=$SMOOTH_WINDOW K=$CONSECUTIVE_K"
  echo "threshold:   metric=$THRESHOLD_METRIC grid=$THRESHOLD_GRID"
  echo "out_root:    $OUT_ROOT"
  echo "log:         $LOG_PATH"
  echo "============================================================"

  python -u -B OFP_external_adapters/run_carots_ofp_adapter.py \
    "${SCORE_ARGS[@]}" \
    --model_name "$MODEL_NAME" \
    --data_dir dataset/training \
    --index_path "dataset/train_test_set_index(in).csv" \
    --out_root "$OUT_ROOT" \
    --folds "${FOLD_ARGS[@]}" \
    --win_size "$WIN_SIZE" \
    --threshold_grid "$THRESHOLD_GRID" \
    --threshold_metric "$THRESHOLD_METRIC" \
    --val_fraction "$VAL_FRACTION" \
    --smoothing "$SMOOTHING" \
    --smooth_window "$SMOOTH_WINDOW" \
    --consecutive_k "$CONSECUTIVE_K" \
    --max_train_files "$MAX_TRAIN_FILES" \
    --max_test_files "$MAX_TEST_FILES"

  python -u -B OFP_DL_official/scripts/print_result_summary.py --root "$OUT_ROOT"
} 2>&1 | tee "$LOG_PATH"
