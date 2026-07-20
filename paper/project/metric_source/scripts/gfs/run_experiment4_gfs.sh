#!/bin/bash

# =============================================================================
# Experiment 4 (GFS): Component Ablation under GFS/Few-shot Training
# Config: configs/GFS_config/experiment_4_gfs_ablation.yaml
# Pipeline: Pipeline_02_pretrain_fewshot
# =============================================================================

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [GFS] 开始实验4: 组件消融 (Few-shot/GFS)"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456 789 999)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/GFS_config/experiment_4_gfs_ablation.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_gfs/experiment_4_ablation_gfs"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for seed in "${SEEDS[@]}"; do
  echo "🔄 [GFS] 运行种子 ${seed} ..."
  seed_result_dir="${RESULTS_DIR}/seed_${seed}"
  mkdir -p "${seed_result_dir}"
  log_file="${seed_result_dir}/experiment_4_gfs_seed_${seed}.log"
  config_backup="${seed_result_dir}/experiment_4_gfs_seed_${seed}.yaml"
  cp "${CONFIG_PATH}" "${config_backup}"

  python main.py \
    --config_path "${CONFIG_PATH}" \
    --pipeline "Pipeline_02_pretrain_fewshot" \
    --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
    --override "environment.seed=${seed}" \
    --override "trainer.deterministic=true" \
    --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_4_ablation_gfs/seed_${seed}" \
    --override "trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_4_ablation_gfs/seed_${seed}" \
    2>&1 | tee "${log_file}"

  echo "✅ [GFS] 种子 ${seed} 实验完成"
  sleep 3
done

echo "🎯 [GFS] 实验4完成，结果目录: ${RESULTS_DIR}"

