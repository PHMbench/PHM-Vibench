#!/usr/bin/env bash
set -euo pipefail

# Paper02 ablation launcher for the current PHM-Vibench repo.
# This wrapper delegates to run_ablation_study.py and keeps GPU binding within
# the local two-card policy: CUDA_VISIBLE_DEVICES must be 0 or 1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/paper/UXFD_paper/1D-2D_fusion_explainable/experiments/ablation_study}"
GPU_ID="${GPU_ID:-0}"

case "${GPU_ID}" in
  0|1) ;;
  *)
    echo "GPU_ID must be 0 or 1 under the local 2x4090 execution policy; got ${GPU_ID}" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python "${SCRIPT_DIR}/run_ablation_study.py" \
  --output_dir "${OUTPUT_DIR}" \
  --gpu_id "${GPU_ID}" \
  --configs 1D_only 2D_only No_Statistical
