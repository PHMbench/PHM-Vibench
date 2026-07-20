#!/bin/bash

# Experiment 2 (GFS, optional): HSE Contrastive Pretraining + GFS Finetuning

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [GFS] 开始实验2: HSE 预训练 + GFS 微调"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456 789 999)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/GFS_config/experiment_2_gfs_hse_pretrain.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_gfs/experiment_2_gfs_hse_pretrain"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for seed in "${SEEDS[@]}"; do
  echo "🔄 [GFS] 种子 ${seed} 两阶段训练..."
  seed_result_dir="${RESULTS_DIR}/seed_${seed}"
  twostage_dir="${seed_result_dir}/two_stage"
  mkdir -p "${seed_result_dir}" "${twostage_dir}"

  log_file="${twostage_dir}/experiment_2_gfs_seed_${seed}_twostage.log"
  config_backup="${seed_result_dir}/experiment_2_gfs_seed_${seed}.yaml"
  cp "${CONFIG_PATH}" "${config_backup}"

  python main.py \
    --config_path "${CONFIG_PATH}" \
    --pipeline "Pipeline_02_pretrain_fewshot" \
    --override "environment.seed=${seed}" \
    --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
    --override "trainer.num_workers=0" \
    --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_2_gfs_hse_pretrain/seed_${seed}/two_stage" \
    --override "stages[0].trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_2_gfs_hse_pretrain/seed_${seed}/two_stage/pretrain" \
    --override "stages[1].trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_2_gfs_hse_pretrain/seed_${seed}/two_stage/finetune" \
    --override "stages[0].trainer.max_epochs=1" \
    --override "stages[1].trainer.max_epochs=1" \
    2>&1 | tee "${log_file}"

  echo "✅ [GFS] 种子 ${seed} 实验2 完成"
  sleep 5
done

echo "🎯 [GFS] 实验2 HSE 预训练 + GFS 微调 完成"

