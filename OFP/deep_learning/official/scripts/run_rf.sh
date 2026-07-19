#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

FOLDS="${FOLDS:-1}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_official_results/lookback24h_ahead1h_legacy_rf}"
N_JOBS="${N_JOBS:-16}"
RF_ESTIMATORS="${RF_ESTIMATORS:-30}"
THRESHOLD="${THRESHOLD:-0.5}"

mkdir -p "$OUT_ROOT/logs"
LOG_PATH="$OUT_ROOT/logs/rf_$(date +%Y%m%d_%H%M%S).log"

echo "[run-rf] model=rf"
echo "[run-rf] folds=$FOLDS n_jobs=$N_JOBS rf_estimators=$RF_ESTIMATORS threshold=$THRESHOLD"
echo "[run-rf] protocol=OFP index, 1h-ahead label, legacy_float32 timestamp, OFP model2 features"
echo "[run-rf] out_root=$OUT_ROOT"
echo "[run-rf] log=$LOG_PATH"

python -u -B OFP/deep_learning/official/run_official_benchmark.py \
  --models rf \
  --folds $FOLDS \
  --data_dir dataset/training \
  --index_path "dataset/train_test_set_index(in).csv" \
  --out_root "$OUT_ROOT" \
  --timestamp_mode legacy_float32 \
  --n_jobs "$N_JOBS" \
  --rf_estimators "$RF_ESTIMATORS" \
  --rf_threshold "$THRESHOLD" \
  --log_batches 0 \
  --no_tqdm \
  2>&1 | tee "$LOG_PATH"
