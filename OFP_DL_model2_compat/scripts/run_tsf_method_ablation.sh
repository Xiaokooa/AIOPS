#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TSF_VARIANTS="${TSF_VARIANTS:-all}"
export OUT_ROOT="${OUT_ROOT:-OFP_DL_model2_compat_results/tsf_method_ablation}"

bash "${SCRIPT_DIR}/run_tsf_table5_ablation.sh" "$@"
