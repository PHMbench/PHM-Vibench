#!/bin/bash

# Experiment 0 (GFS, optional): Backbone+Head Few-Shot Baseline

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [GFS] 开始实验0: Backbone+Head Few-Shot 基线"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

declare -a DATASETS=(
  "1:CWRU"
  "13:Ottawa"
  "6:THU"
  "12:JNU"
  "19:HUST"
)

SEEDS=(42)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/GFS_config/experiment_0_gfs_baseline.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_gfs/experiment_0_gfs_baseline"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for dataset_info in "${DATASETS[@]}"; do
  IFS=':' read -r dataset_id dataset_name <<< "${dataset_info}"
  echo "📊 [GFS] 数据集 ${dataset_id}: ${dataset_name}"

  for seed in "${SEEDS[@]}"; do
    seed_dir="${RESULTS_DIR}/dataset_${dataset_id}/seed_${seed}"
    mkdir -p "${seed_dir}"
    log_file="${seed_dir}/experiment_0_gfs_dataset_${dataset_id}_seed_${seed}.log"
    config_backup="${seed_dir}/experiment_0_gfs_dataset_${dataset_id}_seed_${seed}.yaml"
    cp "${CONFIG_PATH}" "${config_backup}"

    python main.py \
      --config_path "${CONFIG_PATH}" \
      --pipeline "Pipeline_01_default" \
      --override "task.target_system_id=[${dataset_id}]" \
      --override "environment.seed=${seed}" \
      --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_0_gfs_baseline/dataset_${dataset_id}" \
      --override "trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_0_gfs_baseline/dataset_${dataset_id}" \
      2>&1 | tee "${log_file}"
  done
done

echo "🎯 [GFS] 实验0 Few-Shot 基线完成"

