#!/bin/bash

# CDDG Experiment 7: Noise Robustness

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [CDDG] 开始实验7: 噪声鲁棒性 (CDDG)"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/CDDG_config/experiment_7_cddg_noise.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_cddg/experiment_7_cddg_noise"
mkdir -p "${RESULTS_DIR}"

declare -a SNR_CONFIGS=(
  "20:20dB - 清晰信号"
  "10:10dB - 轻微噪声"
  "5:5dB - 中等噪声"
  "0:0dB - 高噪声"
)

cd "${PROJECT_ROOT}"

for snr_info in "${SNR_CONFIGS[@]}"; do
  IFS=':' read -r snr_value snr_desc <<< "${snr_info}"
  snr_result_dir="${RESULTS_DIR}/SNR_${snr_value}dB"
  mkdir -p "${snr_result_dir}"
  for seed in "${SEEDS[@]}"; do
    seed_result_dir="${snr_result_dir}/seed_${seed}"
    mkdir -p "${seed_result_dir}"
    log_file="${seed_result_dir}/experiment_7_SNR_${snr_value}dB_seed_${seed}.log"

    python main.py \
      --config_path "${CONFIG_PATH}" \
      --pipeline "Pipeline_02_pretrain_fewshot" \
      --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
      --override "data.noise_snr=${snr_value}" \
      --override "environment.seed=${seed}" \
      --override "trainer.deterministic=true" \
      --override "environment.output_dir=results/experiment_7_noise_robustness/SNR_${snr_value}dB/seed_${seed}" \
      --override "trainer.save_dir=results/experiment_7_noise_robustness/SNR_${snr_value}dB/seed_${seed}" \
      2>&1 | tee "${log_file}"
  done
done

echo "🎯 [CDDG] 实验7 完成"

