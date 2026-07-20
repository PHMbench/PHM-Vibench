#!/bin/bash

# CDDG Experiment 3: HSE-Prompt Pretraining + CDDG Finetuning

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [CDDG] 开始实验3: HSE-Prompt 预训练 + CDDG 微调"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456 789 999)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/CDDG_config/experiment_3_cddg_hse_prompt.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_cddg/experiment_3_cddg_hse_prompt"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for seed in "${SEEDS[@]}"; do
  seed_result_dir="${RESULTS_DIR}/seed_${seed}"
  twostage_dir="${seed_result_dir}/two_stage"
  mkdir -p "${seed_result_dir}" "${twostage_dir}"
  log_file="${twostage_dir}/experiment_3_seed_${seed}_twostage.log"

  python main.py \
    --config_path "${CONFIG_PATH}" \
    --pipeline "Pipeline_02_pretrain_fewshot" \
    --override "environment.seed=${seed}" \
    --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
    --override "environment.deterministic=true" \
    --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_3_hse_prompt_pretrain/seed_${seed}/two_stage" \
    --override "trainer.num_workers=0" \
    --override "stages[0].trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_3_hse_prompt_pretrain/seed_${seed}/two_stage/pretrain" \
    --override "stages[1].trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_3_hse_prompt_pretrain/seed_${seed}/two_stage/finetune" \
    --override "stages[0].trainer.max_epochs=1" \
    --override "stages[1].trainer.max_epochs=1" \
    --override "stages[0].environment.project=experiment_3_seed_${seed}_pretrain" \
    --override "stages[1].environment.project=experiment_3_seed_${seed}_finetune" \
    2>&1 | tee "${log_file}"
done

echo "🎯 [CDDG] 实验3 完成"

