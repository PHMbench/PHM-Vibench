#!/bin/bash

# =============================================================================
# Experiment 6 (GFS): Backbone Universality under GFS/Few-shot Training
# Config: configs/GFS_config/experiment_6_gfs_backbone.yaml
# Pipeline: Pipeline_02_pretrain_fewshot
# =============================================================================

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [GFS] 开始实验6: Backbone 普适性 (Few-shot/GFS)"
echo "📂 项目根目录: ${PROJECT_ROOT}"
echo "📅 开始时间: $(date)"
echo ""

TARGET_SYSTEM_ID="[1,13,6,12,19]"
SEEDS=(42 123 456)

CONFIG_PATH="paper/2025-10_foundation_model_0_metric/configs/GFS_config/experiment_6_gfs_backbone.yaml"
RESULTS_DIR="${SCRIPT_DIR}/../results_gfs/experiment_6_backbone_gfs"
mkdir -p "${RESULTS_DIR}"

declare -a BACKBONES=(
  "B_04_Dlinear:Dlinear - 线性预测模型"
  "B_06_TimesNet:TimesNet - 时序分析网络"
  "B_08_PatchTST:PatchTST - 补丁时间序列Transformer"
  "B_09_FNO:FNO - 傅里叶神经算子"
)

cd "${PROJECT_ROOT}"

for backbone_info in "${BACKBONES[@]}"; do
  IFS=':' read -r backbone_name backbone_desc <<< "${backbone_info}"
  echo "🔄 [GFS] 测试 Backbone: ${backbone_name} (${backbone_desc})"

  backbone_result_dir="${RESULTS_DIR}/backbone_${backbone_name}"
  mkdir -p "${backbone_result_dir}"

  for seed in "${SEEDS[@]}"; do
    echo "🎲 [GFS] 种子 ${seed} ..."
    seed_result_dir="${backbone_result_dir}/seed_${seed}"
    mkdir -p "${seed_result_dir}"
    log_file="${seed_result_dir}/experiment_6_${backbone_name}_gfs_seed_${seed}.log"
    config_backup="${seed_result_dir}/experiment_6_${backbone_name}_gfs_seed_${seed}.yaml"
    cp "${CONFIG_PATH}" "${config_backup}"

    python main.py \
      --config_path "${CONFIG_PATH}" \
      --pipeline "Pipeline_02_pretrain_fewshot" \
      --override "task.target_system_id=${TARGET_SYSTEM_ID}" \
      --override "model.backbone=${backbone_name}" \
      --override "environment.seed=${seed}" \
      --override "trainer.deterministic=true" \
      --override "environment.output_dir=paper/2025-10_foundation_model_0_metric/results/experiment_6_backbone_universality_gfs/backbone_${backbone_name}/seed_${seed}" \
      --override "trainer.save_dir=paper/2025-10_foundation_model_0_metric/results/experiment_6_backbone_universality_gfs/backbone_${backbone_name}/seed_${seed}" \
      2>&1 | tee "${log_file}"

    echo "✅ [GFS] Backbone ${backbone_name}, 种子 ${seed} 完成"
    sleep 3
  done
done

echo "🎯 [GFS] 实验6 Backbone 普适性 (GFS) 完成"

