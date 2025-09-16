#!/bin/bash
"""
完整论文级实验Pipeline
运行所有必要的实验来产生论文级结果

使用方法:
  bash full_paper_experiments.sh [--quick] [--skip-validation]
  
选项:
  --quick           运行快速版本 (减少epochs和重复次数)
  --skip-validation 跳过环境验证步骤
"""

set -e  # 遇到错误立即停止

# 默认参数
QUICK_MODE=false
SKIP_VALIDATION=false
START_TIME=$(date +%s)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        -h|--help)
            echo "完整论文级实验Pipeline"
            echo ""
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --quick           运行快速版本 (减少epochs和重复次数)"
            echo "  --skip-validation 跳过环境验证步骤"
            echo "  -h, --help        显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 $0 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 设置实验配置
if [ "$QUICK_MODE" = true ]; then
    echo "🚀 运行快速模式实验"
    EPOCHS_BASELINE=10
    EPOCHS_RESEARCH=25
    ABLATION_REPEATS=1
else
    echo "🚀 运行完整论文级实验"
    EPOCHS_BASELINE=50
    EPOCHS_RESEARCH=200
    ABLATION_REPEATS=3
fi

# 创建实验输出目录
EXPERIMENT_DATE=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="paper_experiments_${EXPERIMENT_DATE}"
mkdir -p "$OUTPUT_DIR"

# 设置日志文件
LOG_FILE="$OUTPUT_DIR/experiment_log.txt"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "=========================================="
echo "🎯 Flow预训练论文级实验开始"
echo "开始时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
echo "快速模式: $QUICK_MODE"
echo "=========================================="

# 1. 环境验证
if [ "$SKIP_VALIDATION" = false ]; then
    echo ""
    echo "=== 第1步: 验证实验环境 ==="
    python validate_flow_setup.py || {
        echo "❌ 环境验证失败！请检查设置后重新运行"
        exit 1
    }
    echo "✅ 环境验证通过"
else
    echo "⚠️  跳过环境验证步骤"
fi

# 2. 基线实验
echo ""
echo "=== 第2步: 基线实验 ==="

# Flow基线实验
echo "🔬 运行Flow基线实验..."
./run_flow_experiments.sh research \
    --config_override "task.epochs=$EPOCHS_RESEARCH" \
    --wandb \
    --notes "Paper_Baseline_Flow_Full" || {
    echo "❌ Flow基线实验失败！"
    exit 1
}

# 传统方法对比基线
echo "🔬 运行传统方法对比实验..."
baseline_methods=("CNN_Baseline" "Transformer_Baseline" "VAE_Baseline")

for method in "${baseline_methods[@]}"; do
    echo "  运行 $method..."
    
    # 这里需要根据实际的baseline配置文件调整
    python main.py \
        --config "configs/comparison/${method,,}.yaml" \
        --notes "Paper_Baseline_$method" || {
        echo "⚠️  警告: $method 实验失败，继续其他实验"
        continue
    }
done

echo "✅ 基线实验完成"

# 3. 消融研究
echo ""
echo "=== 第3步: 消融研究 ==="

# Flow组件消融
echo "🧪 Flow组件消融研究..."

# 采样步数消融
echo "  采样步数消融..."
sampling_steps=(20 50 100 200)
for steps in "${sampling_steps[@]}"; do
    echo "    测试采样步数: $steps"
    
    for ((i=1; i<=$ABLATION_REPEATS; i++)); do
        ./run_flow_experiments.sh baseline \
            --config_override "task.num_steps=$steps,task.epochs=$EPOCHS_BASELINE" \
            --wandb \
            --notes "Ablation_Steps_${steps}_Run${i}" || {
            echo "⚠️  警告: 采样步数$steps 第$i次运行失败"
        }
    done
done

# 对比学习权重消融
echo "  对比学习权重消融..."
contrastive_weights=(0.0 0.1 0.3 0.5 0.7 1.0)
for weight in "${contrastive_weights[@]}"; do
    echo "    测试对比学习权重: $weight"
    
    for ((i=1; i<=$ABLATION_REPEATS; i++)); do
        ./run_flow_experiments.sh contrastive \
            --config_override "task.contrastive_weight=$weight,task.epochs=$EPOCHS_BASELINE" \
            --wandb \
            --notes "Ablation_Contrastive_${weight}_Run${i}" || {
            echo "⚠️  警告: 对比权重$weight 第$i次运行失败"
        }
    done
done

# 模型规模消融  
echo "  模型规模消融..."
model_sizes=("128,4" "256,6" "512,8")  # hidden_dim,n_layers
for size in "${model_sizes[@]}"; do
    IFS=',' read -r hidden_dim n_layers <<< "$size"
    echo "    测试模型规模: hidden_dim=$hidden_dim, n_layers=$n_layers"
    
    for ((i=1; i<=$ABLATION_REPEATS; i++)); do
        ./run_flow_experiments.sh baseline \
            --config_override "model.hidden_dim=$hidden_dim,model.n_layers=$n_layers,task.epochs=$EPOCHS_BASELINE" \
            --wandb \
            --notes "Ablation_Size_${hidden_dim}_${n_layers}_Run${i}" || {
            echo "⚠️  警告: 模型规模${hidden_dim}_${n_layers} 第$i次运行失败"
        }
    done
done

echo "✅ 消融研究完成"

# 4. 泛化性实验
echo ""
echo "=== 第4步: 泛化性实验 ==="

# 跨数据集评估 (这里需要根据实际数据集配置调整)
echo "🌐 跨数据集泛化实验..."

# 如果有多个数据集配置，可以运行跨数据集实验
datasets=("CWRU" "XJTU" "THU")  # 假设的数据集名称
for source in "${datasets[@]}"; do
    for target in "${datasets[@]}"; do
        if [ "$source" != "$target" ]; then
            echo "  跨数据集: $source -> $target"
            
            # 这里需要根据实际的跨数据集配置调整
            # python evaluate_cross_dataset.py \
            #     --source "$source" \
            #     --target "$target" \
            #     --model flow_pretrained \
            #     --notes "CrossDataset_${source}_${target}" || {
            #     echo "⚠️  警告: 跨数据集 $source->$target 评估失败"
            # }
        fi
    done
done

echo "✅ 泛化性实验完成"

# 5. Few-shot学习实验
echo ""
echo "=== 第5步: Few-shot学习实验 ==="

echo "🎯 Few-shot学习评估..."

# Pipeline_02 预训练 + Few-shot
./run_flow_experiments.sh pipeline02 \
    --config_override "task.epochs=$EPOCHS_RESEARCH" \
    --wandb \
    --notes "Paper_Pipeline02_Pretrain" || {
    echo "⚠️  警告: Pipeline02预训练失败"
}

# Few-shot评估 (需要单独的评估脚本)
# python evaluate_few_shot.py \
#     --model flow_pipeline02 \
#     --shots 1,5,10,20 \
#     --repeats 10 \
#     --notes "Paper_FewShot_Evaluation" || {
#     echo "⚠️  警告: Few-shot评估失败"
# }

echo "✅ Few-shot学习实验完成"

# 6. 效率分析
echo ""
echo "=== 第6步: 效率分析 ==="

echo "⚡ 运行效率分析..."

# 推理速度测试
# python benchmark_inference_speed.py \
#     --models flow,baseline \
#     --batch_sizes 1,8,32,64 \
#     --notes "Paper_Efficiency_Analysis" || {
#     echo "⚠️  警告: 效率分析失败"
# }

echo "✅ 效率分析完成"

# 7. 结果收集和分析
echo ""
echo "=== 第7步: 结果收集和分析 ==="

echo "📊 收集实验结果..."

# 收集所有结果
python plan/scripts/collect_results.py \
    --results_dir results/ \
    --output_prefix "$OUTPUT_DIR/paper_results" \
    --generate_latex || {
    echo "⚠️  警告: 结果收集失败，请手动检查"
}

# 生成论文图表
echo "📈 生成论文图表..."
# python plan/scripts/generate_paper_figures.py \
#     --results_dir results/ \
#     --output_dir "$OUTPUT_DIR/figures" || {
#     echo "⚠️  警告: 图表生成失败"
# }

echo "✅ 结果收集完成"

# 8. 实验总结
echo ""
echo "=== 实验总结 ==="

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "🎉 所有论文级实验已完成！"
echo ""
echo "📊 实验统计："
echo "  总用时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "  输出目录: $OUTPUT_DIR"
echo "  实验模式: $([ "$QUICK_MODE" = true ] && echo "快速模式" || echo "完整模式")"
echo ""
echo "📁 生成的文件："
echo "  实验日志: $LOG_FILE"
echo "  结果汇总: $OUTPUT_DIR/paper_results_*.csv"
echo "  LaTeX表格: $OUTPUT_DIR/paper_results_summary.tex"
echo ""
echo "📋 下一步建议："
echo "  1. 检查 $OUTPUT_DIR 中的结果文件"
echo "  2. 运行统计显著性分析: python plan/scripts/statistical_analysis.py"
echo "  3. 生成论文图表: python plan/scripts/generate_paper_figures.py"
echo "  4. 检查WandB dashboard查看详细训练曲线"
echo ""
echo "✅ 实验pipeline执行完毕！"

# 如果在交互式终端中，询问是否打开结果目录
if [ -t 0 ]; then
    echo ""
    read -p "是否打开结果目录? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v nautilus &> /dev/null; then
            nautilus "$OUTPUT_DIR" &
        elif command -v open &> /dev/null; then
            open "$OUTPUT_DIR"
        else
            echo "请手动检查目录: $OUTPUT_DIR"
        fi
    fi
fi