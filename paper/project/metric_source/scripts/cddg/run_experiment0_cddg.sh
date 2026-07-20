#!/bin/bash

# CDDG Experiment 0: Backbone+Head Independent Baseline (wrapper moved from top-level)

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 [CDDG] 开始实验0: Backbone+Head 独立基线"
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

RESULTS_DIR="${SCRIPT_DIR}/../results_cddg/experiment_0_backbone_head"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_ROOT}"

for dataset_info in "${DATASETS[@]}"; do
  IFS=':' read -r dataset_id dataset_name <<< "$dataset_info"
  echo "📊 [CDDG] 数据集 ${dataset_id}: ${dataset_name}"

  case "${dataset_id}" in
    1) INPUT_DIM=2 ;;
    13) INPUT_DIM=2 ;;
    6) INPUT_DIM=1 ;;
    12) INPUT_DIM=1 ;;
    19) INPUT_DIM=3 ;;
    *) INPUT_DIM=2 ;;
  esac

  for seed in "${SEEDS[@]}"; do
    dataset_result_dir="${RESULTS_DIR}/dataset_${dataset_id}"
    mkdir -p "${dataset_result_dir}"
    log_file="${dataset_result_dir}/experiment_0_dataset_${dataset_id}_seed_${seed}.log"

    python main.py \
      --config_path "paper/2025-10_foundation_model_0_metric/configs/CDDG_config/experiment_0_cddg_baseline.yaml" \
      --pipeline "Pipeline_01_default" \
      --override "task.target_system_id=[${dataset_id}]" \
      --override "model.input_dim=${INPUT_DIM}" \
      --override "environment.seed=${seed}" \
      --override "trainer.deterministic=true" \
      --override "environment.output_dir=results/experiment_0_backbone_head/dataset_${dataset_id}" \
      --override "trainer.save_dir=results/experiment_0_backbone_head/dataset_${dataset_id}" \
      2>&1 | tee "${log_file}"
  done
done

echo "🎯 [CDDG] 实验0 完成"

