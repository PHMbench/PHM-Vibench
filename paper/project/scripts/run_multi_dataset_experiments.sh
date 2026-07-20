#!/bin/bash

# 1D-2D Fusion 多数据集验证实验脚本
# 运行在多个数据集上的验证实验

echo "=========================================================="
echo "1D-2D Fusion 多数据集验证实验"
echo "=========================================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 创建结果目录
RESULTS_DIR="Paper/1D-2D_fusion_explainable/results/multi_dataset"
mkdir -p $RESULTS_DIR

# 实验列表
declare -a experiments=(
    "CWRU"
    "XJTU"
    "THU_006"
)

# GPU分配策略
declare -a gpu_assignments=(
    "0"  # CWRU
    "1"  # XJTU
    "2"  # THU_006
)

# 日志文件
LOG_FILE="$RESULTS_DIR/experiments_$(date +%Y%m%d_%H%M%S).log"
echo "实验日志: $LOG_FILE"

# 运行实验
for i in "${!experiments[@]}"; do
    exp_name="${experiments[$i]}"
    gpu="${gpu_assignments[$i]}"
    config_file="Paper/1D-2D_fusion_explainable/configs/config_${exp_name}.yaml"

    echo "----------------------------------------------------------"
    echo "开始实验: $exp_name (GPU: $gpu)"
    echo "配置文件: $config_file"
    echo "----------------------------------------------------------"

    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        echo "错误: 配置文件不存在 $config_file"
        continue
    fi

    # 记录实验开始时间
    start_time=$(date +%s)

    # 运行实验
    {
        echo "[$(date)] 开始 $exp_name 实验"
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate LQ_signal
        export CUDA_VISIBLE_DEVICES=$gpu

        python main_com.py --config_dir "$config_file" 2>&1

        echo "[$(date)] $exp_name 实验完成"
    } | tee -a "$LOG_FILE"

    # 记录实验结束时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "[$(date)] $exp_name 实验耗时: $((duration / 60)) 分 $((duration % 60)) 秒" | tee -a "$LOG_FILE"

    # 移动结果文件
    if [ -d "save/task_THU_018_basic" ]; then
        mv "save/task_THU_018_basic" "$RESULTS_DIR/run_${exp_name}"
        echo "[$(date)] 结果已移动到: $RESULTS_DIR/run_${exp_name}"
    fi

    echo "----------------------------------------------------------"
    echo "$exp_name 实验完成"
    echo "----------------------------------------------------------"
    echo ""
done

echo "=========================================================="
echo "所有多数据集验证实验完成！"
echo "结果目录: $RESULTS_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================================="

# 生成实验摘要
echo "生成实验摘要..."
python Paper/1D-2D_fusion_explainable/scripts/collect_multi_dataset_results.py \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/multi_dataset_summary.json"

echo "实验完成！"