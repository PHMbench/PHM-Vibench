#!/bin/bash
# Flow预训练实验运行脚本
# 用于快速启动不同类型的Flow实验

set -e  # 出错时退出

echo "🚀 Flow预训练实验管理脚本"
echo "=================================="

# 定义配置路径
CONFIG_DIR="configs/demo/Pretraining/Flow"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 实验类型定义
declare -A EXPERIMENTS
EXPERIMENTS[quick]="flow_quick_validation.yaml"
EXPERIMENTS[baseline]="flow_baseline_experiment.yaml" 
EXPERIMENTS[contrastive]="flow_contrastive_experiment.yaml"
EXPERIMENTS[pipeline02]="flow_pipeline02_pretrain.yaml"
EXPERIMENTS[research]="flow_research_experiment.yaml"

# 帮助函数
show_usage() {
    echo "用法: $0 <实验类型> [选项]"
    echo ""
    echo "实验类型:"
    echo "  quick       - 快速验证 (5轮次, ~5分钟)"
    echo "  baseline    - 基线实验 (50轮次, ~1小时)"
    echo "  contrastive - Flow+对比学习 (60轮次, ~1.5小时)"
    echo "  pipeline02  - Pipeline_02预训练 (100轮次, ~2.5小时)"  
    echo "  research    - 研究级实验 (200轮次, ~5小时)"
    echo ""
    echo "选项:"
    echo "  --dry-run   - 显示将要运行的命令，不实际执行"
    echo "  --gpu N     - 指定GPU编号 (默认: 0)"
    echo "  --notes 'X' - 添加实验备注"
    echo "  --wandb     - 启用WandB跟踪"
    echo ""
    echo "示例:"
    echo "  $0 quick                          # 快速验证"
    echo "  $0 baseline --gpu 1              # 在GPU 1上运行基线"
    echo "  $0 contrastive --wandb --notes '对比学习测试'"
    echo ""
}

# 检查参数
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# 解析参数
EXPERIMENT_TYPE="$1"
shift

DRY_RUN=false
GPU_ID=0
NOTES=""
ENABLE_WANDB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --notes)
            NOTES="$2"
            shift 2
            ;;
        --wandb)
            ENABLE_WANDB=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ 未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 验证实验类型
if [[ ! -v EXPERIMENTS[$EXPERIMENT_TYPE] ]]; then
    echo "❌ 无效的实验类型: $EXPERIMENT_TYPE"
    echo "可用类型: ${!EXPERIMENTS[@]}"
    exit 1
fi

CONFIG_FILE="${EXPERIMENTS[$EXPERIMENT_TYPE]}"
CONFIG_PATH="$CONFIG_DIR/$CONFIG_FILE"

# 检查配置文件
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "❌ 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 构建命令
COMMAND="python main.py --config_path $CONFIG_PATH"

if [[ -n "$NOTES" ]]; then
    COMMAND="$COMMAND --notes '$NOTES'"
fi

# 显示实验信息
echo "📋 实验配置:"
echo "   类型: $EXPERIMENT_TYPE"
echo "   配置: $CONFIG_FILE"
echo "   GPU: $GPU_ID"
echo "   备注: ${NOTES:-'无'}"
echo "   WandB: $([ "$ENABLE_WANDB" = true ] && echo '启用' || echo '禁用')"
echo ""

# 显示预期时间
case $EXPERIMENT_TYPE in
    quick)
        echo "⏱️  预期时间: ~5分钟"
        ;;
    baseline)
        echo "⏱️  预期时间: ~1小时"
        ;;
    contrastive)
        echo "⏱️  预期时间: ~1.5小时"
        ;;
    pipeline02)
        echo "⏱️  预期时间: ~2.5小时"
        ;;
    research)
        echo "⏱️  预期时间: ~5小时"
        ;;
esac
echo ""

# 启用WandB的话需要修改配置
if [[ "$ENABLE_WANDB" = true ]]; then
    echo "🔄 启用WandB跟踪..."
    # 创建临时配置文件
    TEMP_CONFIG=$(mktemp --suffix=.yaml)
    cp "$CONFIG_PATH" "$TEMP_CONFIG"
    
    # 修改WandB设置
    sed -i 's/WANDB_MODE: "disabled"/WANDB_MODE: "online"/' "$TEMP_CONFIG"
    CONFIG_PATH="$TEMP_CONFIG"
    COMMAND="python main.py --config_path $CONFIG_PATH"
    if [[ -n "$NOTES" ]]; then
        COMMAND="$COMMAND --notes '$NOTES'"
    fi
fi

echo "🚀 执行命令: $COMMAND"
echo ""

# 执行或预演
if [[ "$DRY_RUN" = true ]]; then
    echo "🔍 试运行模式 - 将要执行的命令:"
    echo "$COMMAND"
    echo ""
    echo "环境变量:"
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID"
    
    if [[ "$ENABLE_WANDB" = true && -f "$TEMP_CONFIG" ]]; then
        echo ""
        echo "临时配置文件: $TEMP_CONFIG"
        echo "WandB设置已启用"
    fi
else
    echo "🚀 开始实验..."
    echo "=================================="
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 运行实验
    if eval "$COMMAND"; then
        # 计算运行时间
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        seconds=$((duration % 60))
        
        echo ""
        echo "🎉 实验完成!"
        echo "⏱️  运行时间: ${hours}小时${minutes}分钟${seconds}秒"
        
        # 显示结果路径
        case $EXPERIMENT_TYPE in
            quick)
                echo "📊 结果路径: results/flow_quick_validation/"
                ;;
            baseline)
                echo "📊 结果路径: results/flow_baseline/"
                ;;
            contrastive)
                echo "📊 结果路径: results/flow_contrastive/"
                ;;
            pipeline02)
                echo "📊 结果路径: results/flow_pipeline02_pretrain/"
                ;;
            research)
                echo "📊 结果路径: results/flow_research/"
                ;;
        esac
        
        echo "✨ Flow预训练实验成功完成!"
        
    else
        echo "❌ 实验执行失败"
        exit 1
    fi
    
    # 清理临时文件
    if [[ "$ENABLE_WANDB" = true && -f "$TEMP_CONFIG" ]]; then
        rm -f "$TEMP_CONFIG"
    fi
fi

echo "=================================="
echo "🎯 Flow预训练实验脚本完成"