#!/bin/bash

# Experiment 1 (GFS, optional): HSE Few-Shot Baseline

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [GFS] 开始实验1: HSE Few-Shot 基线"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456 789 999)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/GFS_config/experiment_1_gfs_hse.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_gfs/experiment_1_gfs_hse"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for seed in "${SEEDS[@]}"; do
  seed_dir="${RESULTS_DIR}/seed_${seed}"
  mkdir -p "${seed_dir}"
  log_file="${seed_dir}/experiment_1_gfs_seed_${seed}.log"
  config_backup="${seed_dir}/experiment_1_gfs_seed_${seed}.yaml"
  cp "${CONFIG_PATH}" "${config_backup}"

  python main.py \
    --config_path "${CONFIG_PATH}" \
    --pipeline "Pipeline_01_default" \
    --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
    --override "environment.seed=${seed}" \
    --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_1_gfs_hse/seed_${seed}" \
    --override "trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_1_gfs_hse/seed_${seed}" \
    2>&1 | tee "${log_file}"
done

echo "🎯 [GFS] 实验1 HSE Few-Shot 基线完成"

