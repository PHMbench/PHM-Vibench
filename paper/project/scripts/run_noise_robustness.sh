#!/bin/bash

# 1D-2D Fusion 噪声鲁棒性测试脚本
# 测试不同信噪比下的模型性能

echo "=========================================================="
echo "1D-2D Fusion 噪声鲁棒性测试"
echo "=========================================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 创建结果目录
RESULTS_DIR="Paper/1D-2D_fusion_explainable/results/noise_robustness"
mkdir -p $RESULTS_DIR

# SNR水平列表
declare -a snr_levels=(
    "snr0"
    "snr5"
    "snr10"
    "snr20"
)

# GPU分配策略
declare -a gpu_assignments=(
    "0"  # snr0
    "1"  # snr5
    "2"  # snr10
    "3"  # snr20
)

# 日志文件
LOG_FILE="$RESULTS_DIR/noise_robustness_$(date +%Y%m%d_%H%M%S).log"
echo "实验日志: $LOG_FILE"

# 运行噪声鲁棒性实验
for i in "${!snr_levels[@]}"; do
    snr_name="${snr_levels[$i]}"
    gpu="${gpu_assignments[$i]}"
    config_file="Paper/1D-2D_fusion_explainable/configs/noise/config_${snr_name}.yaml"

    echo "----------------------------------------------------------"
    echo "开始噪声实验: $snr_name (GPU: $gpu)"
    echo "配置文件: $config_file"
    echo "----------------------------------------------------------"

    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        echo "错误: 配置文件不存在 $config_file"
        continue
    fi

    # 提取SNR值用于命名
    snr_value=$(echo "$snr_name" | sed 's/snr//')

    # 记录实验开始时间
    start_time=$(date +%s)

    # 运行实验
    {
        echo "[$(date)] 开始 SNR=${snr_value}dB 噪声实验"
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate LQ_signal
        export CUDA_VISIBLE_DEVICES=$gpu

        python main_com.py --config_dir "$config_file" 2>&1

        echo "[$(date)] SNR=${snr_value}dB 噪声实验完成"
    } | tee -a "$LOG_FILE"

    # 记录实验结束时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "[$(date)] SNR=${snr_value}dB 实验耗时: $((duration / 60)) 分 $((duration % 60)) 秒" | tee -a "$LOG_FILE"

    # 移动结果文件
    if [ -d "save/task_THU_018_basic" ]; then
        mv "save/task_THU_018_basic" "$RESULTS_DIR/run_${snr_name}"
        echo "[$(date)] 结果已移动到: $RESULTS_DIR/run_${snr_name}"
    fi

    echo "----------------------------------------------------------"
    echo "SNR=${snr_value}dB 噪声实验完成"
    echo "----------------------------------------------------------"
    echo ""
done

echo "=========================================================="
echo "所有噪声鲁棒性测试完成！"
echo "结果目录: $RESULTS_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================================="

# 生成噪声鲁棒性摘要
echo "生成噪声鲁棒性摘要..."
python Paper/1D-2D_fusion_explainable/scripts/collect_noise_results.py \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/noise_robustness_summary.json"

echo "噪声鲁棒性测试完成！"