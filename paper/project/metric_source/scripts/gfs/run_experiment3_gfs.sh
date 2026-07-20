#!/bin/bash

# =============================================================================
# Experiment 3 (GFS): HSE-Prompt Unified Method (Pretrain + GFS Finetuning)
# 使用 GFS 专用配置: configs/GFS_config/experiment_3_gfs_hse_prompt.yaml
# Pipeline: Pipeline_02_pretrain_fewshot
# =============================================================================

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [GFS] 开始实验3: HSE-Prompt 多阶段 (预训练 + GFS 微调)"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
echo "📊 目标数据集: ${TARGET_SYSTEM_ID} (CWRU, Ottawa, THU, JNU, HUST)"

SEEDS=(42 123 456 789 999)
echo "🎲 随机种子: ${SEEDS[*]}"

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/GFS_config/experiment_3_gfs_hse_prompt.yaml"
echo "📄 使用 GFS 配置: ${CONFIG_PATH}"

RESULTS_DIR="${SCRIPT_DIR}/../results_gfs/experiment_3_hse_prompt_gfs"
mkdir -p "${RESULTS_DIR}"

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "🔄 [GFS] 运行种子 ${seed} 的两阶段训练..."

  seed_result_dir="${RESULTS_DIR}/seed_${seed}"
  twostage_dir="${seed_result_dir}/two_stage"
  mkdir -p "${seed_result_dir}" "${twostage_dir}"

  log_file="${twostage_dir}/experiment_3_gfs_seed_${seed}_twostage.log"
  config_backup="${seed_result_dir}/experiment_3_gfs_seed_${seed}.yaml"
  cp "${PROJECT_ROOT}/${CONFIG_PATH}" "${config_backup}"

  cd "${PROJECT_ROOT}"

  python main.py \
    --config_path "${CONFIG_PATH}" \
    --pipeline "Pipeline_02_pretrain_fewshot" \
    --override "environment.seed=${seed}" \
    --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
    --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_3_hse_prompt_gfs/seed_${seed}/two_stage" \
    --override "trainer.num_workers=0" \
    --override "stages[0].trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_3_hse_prompt_gfs/seed_${seed}/two_stage/pretrain" \
    --override "stages[1].trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_3_hse_prompt_gfs/seed_${seed}/two_stage/finetune" \
    --override "stages[0].trainer.max_epochs=1" \
    --override "stages[1].trainer.max_epochs=1" \
    --override "stages[0].environment.project=experiment_3_gfs_seed_${seed}_pretrain" \
    --override "stages[1].environment.project=experiment_3_gfs_seed_${seed}_finetune" \
    2>&1 | tee "${log_file}"

  echo "✅ [GFS] 种子 ${seed} 两阶段训练完成"
  echo "⏳ 休息 5 秒..."
  sleep 5
done

echo ""
echo "🎯 [GFS] 实验3完成: HSE-Prompt 预训练 + GFS 微调"
echo "📁 结果保存在: ${RESULTS_DIR}"

