#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-quick}"
GPU_IDS="${GPU_IDS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/htsf_two_stage_${PROFILE}}"
FOLDS="${FOLDS:-1}"
EPOCHS="${EPOCHS:-6}"
ONLY="${ONLY:-training_hard_negative}"

# Six epochs leave two branch-warm-up epochs, one initial fusion epoch,
# and three epochs after adaptive hard-negative weights are introduced.
# shellcheck disable=SC2086
python -u -B OFP_DL_model2_compat/run_htsf_experiments.py \
  --profile "${PROFILE}" \
  --suites training \
  --only "${ONLY}" \
  --folds ${FOLDS} \
  --epochs "${EPOCHS}" \
  --gpus "${GPU_IDS}" \
  --max_parallel "${MAX_PARALLEL}" \
  --out_root "${OUT_ROOT}" \
  --resume \
  "$@"
