#!/bin/bash

# =============================================================================
# Experiment 7 (GFS): Noise Robustness under GFS/Few-shot Training
# Config: configs/GFS_config/experiment_7_gfs_noise.yaml
# Pipeline: Pipeline_02_pretrain_fewshot
# =============================================================================

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [GFS] 开始实验7: 噪声鲁棒性 (Few-shot/GFS)"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/GFS_config/experiment_7_gfs_noise.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_gfs/experiment_7_noise_gfs"
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
  echo "🔄 [GFS] 测试 SNR=${snr_value}dB (${snr_desc})"

  snr_result_dir="${RESULTS_DIR}/SNR_${snr_value}dB"
  mkdir -p "${snr_result_dir}"

  for seed in "${SEEDS[@]}"; do
    echo "🎲 [GFS] 种子 ${seed} ..."
    seed_result_dir="${snr_result_dir}/seed_${seed}"
    mkdir -p "${seed_result_dir}"
    log_file="${seed_result_dir}/experiment_7_SNR_${snr_value}dB_gfs_seed_${seed}.log"
    config_backup="${seed_result_dir}/experiment_7_SNR_${snr_value}dB_gfs_seed_${seed}.yaml"
    cp "${CONFIG_PATH}" "${config_backup}"

    python main.py \
      --config_path "${CONFIG_PATH}" \
      --pipeline "Pipeline_02_pretrain_fewshot" \
      --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
      --override "data.noise_snr=${snr_value}" \
      --override "environment.seed=${seed}" \
      --override "trainer.deterministic=true" \
      --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_7_noise_robustness_gfs/SNR_${snr_value}dB/seed_${seed}" \
      --override "trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_7_noise_robustness_gfs/SNR_${snr_value}dB/seed_${seed}" \
      2>&1 | tee "${log_file}"

    echo "✅ [GFS] SNR ${snr_value}dB, 种子 ${seed} 完成"
    sleep 3
  done
done

echo "🎯 [GFS] 实验7 噪声鲁棒性 (GFS) 完成"

