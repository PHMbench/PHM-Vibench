#!/bin/bash

# CDDG Experiment 4: Component Ablation (Cross-Domain Classification)

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [CDDG] 开始实验4: 组件消融 (CDDG)"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456 789 999)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/CDDG_config/experiment_4_cddg_ablation.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_cddg/experiment_4_cddg_ablation"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for seed in "${SEEDS[@]}"; do
  seed_result_dir="${RESULTS_DIR}/seed_${seed}"
  mkdir -p "${seed_result_dir}"
  log_file="${seed_result_dir}/experiment_4_cddg_seed_${seed}.log"

  python main.py \
    --config_path "${CONFIG_PATH}" \
    --pipeline "Pipeline_02_pretrain_fewshot" \
    --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
    --override "environment.seed=${seed}" \
    --override "trainer.deterministic=true" \
    --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_4_cddg_ablation/seed_${seed}" \
    --override "trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_4_cddg_ablation/seed_${seed}" \
    2>&1 | tee "${log_file}"
done

echo "🎯 [CDDG] 实验4 完成"

