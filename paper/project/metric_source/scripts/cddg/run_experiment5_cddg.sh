#!/bin/bash

# CDDG Experiment 5 (optional): Few-Shot Gradient Sweep as CDDG Control

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [CDDG] 开始实验5: Few-shot 梯度扫描 (CDDG 对照)"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SHOTS_CONFIG=(1 3 5 10 15 20)
SEEDS=(42 123 456)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/experiment_5_unified.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_cddg/experiment_5_fewshot_sweep_cddg"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for shots in "${SHOTS_CONFIG[@]}"; do
  shots_result_dir="${RESULTS_DIR}/shots_${shots}"
  mkdir -p "${shots_result_dir}"
  for seed in "${SEEDS[@]}"; do
    seed_result_dir="${shots_result_dir}/seed_${seed}"
    mkdir -p "${seed_result_dir}"
    log_file="${seed_result_dir}/experiment_5_shots_${shots}_seed_${seed}.log"

    python main.py \
      --config_path "${CONFIG_PATH}" \
      --pipeline "Pipeline_02_pretrain_fewshot" \
      --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
      --override "task.few_shot.shots=[${shots}]" \
      --override "task.few_shot.samples_per_class=[${shots}]" \
      --override "environment.seed=${seed}" \
      --override "trainer.deterministic=true" \
      --override "environment.output_dir=results/experiment_5_fewshot_sweep/shots_${shots}/seed_${seed}" \
      --override "trainer.save_dir=results/experiment_5_fewshot_sweep/shots_${shots}/seed_${seed}" \
      2>&1 | tee "${log_file}"
  done
done

echo "🎯 [CDDG] 实验5 CDDG 对照完成"

