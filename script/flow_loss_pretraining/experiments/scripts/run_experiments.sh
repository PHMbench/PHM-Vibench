#!/bin/bash

# ==============================================================================
# PHM-Vibench Flow预训练完整实验脚本
# 版本: v2.0
# 用途: 自动化运行Flow预训练的完整实验流程
# ==============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# 参数解析
QUICK_MODE=false
BASELINE_ONLY=false
ABLATION_ONLY=false
SKIP_VALIDATION=false
USE_WANDB=false
FULL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --baseline)
            BASELINE_ONLY=true
            shift
            ;;
        --ablation)
            ABLATION_ONLY=true
            shift
            ;;
        --full)
            FULL_MODE=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --quick           快速验证模式 (30分钟)"
            echo "  --baseline        仅运行基线实验 (6小时)"
            echo "  --ablation        仅运行消融研究 (12小时)"
            echo "  --full            完整研究模式 (7天)"
            echo "  --skip-validation 跳过环境验证"
            echo "  --wandb          启用W&B日志"
            echo "  -h, --help       显示此帮助"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 脚本配置
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../" && pwd )"
CONFIG_DIR="$SCRIPT_DIR/../configs"
RESULTS_BASE_DIR="$PROJECT_ROOT/save/flow_experiments_$(date +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"

log_info "Flow预训练实验启动"
log_info "项目根目录: $PROJECT_ROOT"
log_info "配置目录: $CONFIG_DIR"
log_info "结果保存目录: $RESULTS_BASE_DIR"

# ==============================================================================
# 环境验证
# ==============================================================================

validate_environment() {
    log_info "开始环境验证..."

    # 检查Python环境
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi

    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "未检测到NVIDIA GPU"
    else
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log_info "检测到 $GPU_COUNT 个GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read -r name memory; do
            log_info "GPU: $name, 显存: ${memory}MB"
        done
    fi

    # 检查Flow模型
    if ! python -c "from src.model_factory.ISFM.M_04_ISFM_Flow import Model" 2>/dev/null; then
        log_error "Flow模型导入失败"
        exit 1
    fi

    # 检查数据
    if [ ! -f "data/metadata_6_11.xlsx" ]; then
        log_warning "数据文件不存在: data/metadata_6_11.xlsx"
        log_info "实验将使用模拟数据"
    fi

    # 检查配置文件
    for config in quick_validation.yaml baseline.yaml full_research.yaml; do
        if [ ! -f "$CONFIG_DIR/$config" ]; then
            log_error "配置文件不存在: $CONFIG_DIR/$config"
            exit 1
        fi
    done

    log_success "环境验证完成"
}

# ==============================================================================
# 实验执行函数
# ==============================================================================

run_experiment() {
    local config_name=$1
    local experiment_name=$2
    local max_epochs=${3:-100}

    log_info "开始实验: $experiment_name"
    log_info "配置文件: $config_name"
    log_info "最大轮次: $max_epochs"

    local config_file="$CONFIG_DIR/$config_name"
    local result_dir="$RESULTS_BASE_DIR/$experiment_name"

    # 创建结果目录
    mkdir -p "$result_dir"

    # 构建命令
    local cmd="python main.py --config $config_file"

    # 添加额外参数
    if [ "$USE_WANDB" = true ]; then
        cmd="$cmd --wandb --wandb_project flow_pretraining --wandb_run_name $experiment_name"
    fi

    # 设置结果保存路径
    export RESULTS_DIR="$result_dir"

    log_info "执行命令: $cmd"

    # 执行实验
    if eval "$cmd"; then
        log_success "实验完成: $experiment_name"

        # 保存配置文件副本
        cp "$config_file" "$result_dir/config_used.yaml"

        # 生成实验摘要
        cat > "$result_dir/experiment_summary.txt" << EOF
实验名称: $experiment_name
配置文件: $config_name
开始时间: $(date)
状态: 成功
EOF

    else
        log_error "实验失败: $experiment_name"
        return 1
    fi
}

# ==============================================================================
# 快速验证模式
# ==============================================================================

run_quick_validation() {
    log_info "🚀 启动快速验证模式"
    log_info "预计时间: 30分钟"

    run_experiment "quick_validation.yaml" "quick_validation" 10

    log_success "✅ 快速验证完成"
}

# ==============================================================================
# 基线实验
# ==============================================================================

run_baseline_experiments() {
    log_info "⚖️ 启动基线实验"
    log_info "预计时间: 6小时"

    # Flow基线
    run_experiment "baseline.yaml" "flow_baseline" 200

    # 传统CNN基线 (如果配置存在)
    if [ -f "$CONFIG_DIR/cnn_baseline.yaml" ]; then
        run_experiment "cnn_baseline.yaml" "cnn_baseline" 200
    fi

    # Transformer基线 (如果配置存在)
    if [ -f "$CONFIG_DIR/transformer_baseline.yaml" ]; then
        run_experiment "transformer_baseline.yaml" "transformer_baseline" 200
    fi

    log_success "✅ 基线实验完成"
}

# ==============================================================================
# 消融研究
# ==============================================================================

run_ablation_studies() {
    log_info "🔬 启动消融研究"
    log_info "预计时间: 12小时"

    # 不同Flow步数
    for steps in 20 50 100 200; do
        local config_file="$CONFIG_DIR/ablation_steps_$steps.yaml"
        if [ -f "$config_file" ]; then
            run_experiment "ablation_steps_$steps.yaml" "ablation_steps_$steps" 100
        fi
    done

    # 不同损失权重
    for weight in 0.05 0.1 0.2 0.5; do
        local config_file="$CONFIG_DIR/ablation_weight_$weight.yaml"
        if [ -f "$config_file" ]; then
            run_experiment "ablation_weight_$weight.yaml" "ablation_weight_$weight" 100
        fi
    done

    log_success "✅ 消融研究完成"
}

# ==============================================================================
# 完整研究模式
# ==============================================================================

run_full_research() {
    log_info "🎯 启动完整研究模式"
    log_info "预计时间: 7天"

    # 多数据集预训练
    run_experiment "full_research.yaml" "multi_dataset_pretrain" 1000

    # Few-shot评估
    if [ -f "$CONFIG_DIR/few_shot_evaluation.yaml" ]; then
        run_experiment "few_shot_evaluation.yaml" "few_shot_evaluation" 50
    fi

    # 跨域泛化
    if [ -f "$CONFIG_DIR/cross_domain.yaml" ]; then
        run_experiment "cross_domain.yaml" "cross_domain_generalization" 200
    fi

    log_success "✅ 完整研究完成"
}

# ==============================================================================
# 结果收集和分析
# ==============================================================================

collect_and_analyze_results() {
    log_info "📊 开始结果收集和分析"

    local collect_script="$SCRIPT_DIR/collect_results.py"
    local analysis_script="$SCRIPT_DIR/statistical_analysis.py"

    if [ -f "$collect_script" ]; then
        log_info "收集实验结果..."
        python "$collect_script" \
            --results_dir "$RESULTS_BASE_DIR" \
            --generate_latex \
            --output_prefix "flow_experiments"

        if [ $? -eq 0 ]; then
            log_success "结果收集完成"
        else
            log_error "结果收集失败"
        fi
    fi

    # 统计分析
    local results_csv="$RESULTS_BASE_DIR/experiment_results.csv"
    if [ -f "$analysis_script" ] && [ -f "$results_csv" ]; then
        log_info "进行统计分析..."
        python "$analysis_script" \
            --results_file "$results_csv" \
            --confidence_level 0.95

        if [ $? -eq 0 ]; then
            log_success "统计分析完成"
        else
            log_error "统计分析失败"
        fi
    fi
}

# ==============================================================================
# 生成实验报告
# ==============================================================================

generate_experiment_report() {
    log_info "📋 生成实验报告"

    local report_file="$RESULTS_BASE_DIR/EXPERIMENT_REPORT.md"

    cat > "$report_file" << EOF
# Flow预训练实验报告

## 实验概况

- **实验开始时间**: $(date)
- **实验模式**: $1
- **结果目录**: $RESULTS_BASE_DIR
- **使用W&B**: $USE_WANDB

## 实验配置

EOF

    # 添加已执行的实验列表
    if [ -d "$RESULTS_BASE_DIR" ]; then
        echo "## 已完成实验" >> "$report_file"
        echo "" >> "$report_file"
        for exp_dir in "$RESULTS_BASE_DIR"/*/; do
            if [ -d "$exp_dir" ]; then
                local exp_name=$(basename "$exp_dir")
                echo "- $exp_name" >> "$report_file"
            fi
        done
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

## 下一步操作

1. 查看详细结果：\`ls -la $RESULTS_BASE_DIR\`
2. 分析实验数据：\`python experiments/scripts/statistical_analysis.py --results_file experiment_results.csv\`
3. 生成论文图表：查看 \`paper/figures/\` 目录
4. 撰写实验部分：参考 \`paper/latex_template.tex\`

## 联系信息

如有问题，请查看故障排除指南或提交Issue。
EOF

    log_success "实验报告已生成: $report_file"
}

# ==============================================================================
# 主执行逻辑
# ==============================================================================

main() {
    log_info "Flow预训练实验开始执行"

    # 环境验证
    if [ "$SKIP_VALIDATION" != true ]; then
        validate_environment
    fi

    # 根据模式执行对应实验
    if [ "$QUICK_MODE" = true ]; then
        run_quick_validation
        generate_experiment_report "quick_validation"

    elif [ "$BASELINE_ONLY" = true ]; then
        run_baseline_experiments
        collect_and_analyze_results
        generate_experiment_report "baseline_experiments"

    elif [ "$ABLATION_ONLY" = true ]; then
        run_ablation_studies
        collect_and_analyze_results
        generate_experiment_report "ablation_studies"

    elif [ "$FULL_MODE" = true ]; then
        run_full_research
        collect_and_analyze_results
        generate_experiment_report "full_research"

    else
        # 默认：完整流程
        log_info "🎯 启动完整实验流程"

        # 1. 快速验证
        run_quick_validation

        # 2. 基线实验
        run_baseline_experiments

        # 3. 核心研究
        run_full_research

        # 4. 消融研究
        run_ablation_studies

        # 5. 结果分析
        collect_and_analyze_results

        generate_experiment_report "complete_pipeline"
    fi

    log_success "🎉 所有实验执行完成！"
    log_info "结果保存在: $RESULTS_BASE_DIR"
    log_info "查看实验报告: $RESULTS_BASE_DIR/EXPERIMENT_REPORT.md"
}

# 执行主函数
main "$@"