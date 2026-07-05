#!/bin/bash
# Pipeline 资源管理修复状态检查脚本
# 用于验证 GOAL-FFU-P1-007 的修复进度

echo "=== Pipeline 资源管理修复状态检查 ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_file_pattern() {
    local file=$1
    local pattern=$2
    local description=$3

    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $description"
        return 0
    else
        echo -e "${RED}❌${NC} $description"
        return 1
    fi
}

# 1. 检查 Pipeline_02 资源清理
echo "### Pipeline_02_pretrain_fewshot.py ###"
check_file_pattern "src/Pipeline_02_pretrain_fewshot.py" "data.close()\|data_factory.data.close()" \
    "包含 data.close() 调用"
check_file_pattern "src/Pipeline_02_pretrain_fewshot.py" "finally" \
    "包含 finally 块（异常安全）"
echo ""

# 2. 检查 Pipeline_03 资源清理
echo "### Pipeline_03_multitask_pretrain_finetune.py ###"
check_file_pattern "src/Pipeline_03_multitask_pretrain_finetune.py" "data.close()\|data_factory.data.close()" \
    "包含 data.close() 调用"
check_file_pattern "src/Pipeline_03_multitask_pretrain_finetune.py" "finally" \
    "包含 finally 块（异常安全）"
echo ""

# 3. 检查 Pipeline_04 资源清理
echo "### Pipeline_04_unified_metric.py ###"
check_file_pattern "src/Pipeline_04_unified_metric.py" "data.close()\|data_factory.data.close()" \
    "包含 data.close() 调用"
check_file_pattern "src/Pipeline_04_unified_metric.py" "finally" \
    "包含 finally 块（异常安全）"
echo ""

# 4. 检查 TwoStageOrchestrator 清理
echo "### TwoStageOrchestrator ###"
check_file_pattern "src/utils/training/two_stage_orchestrator.py" "run_pretrain.*finally\|finally:.*data" \
    "run_pretrain 包含 finally 清理"
check_file_pattern "src/utils/training/two_stage_orchestrator.py" "run_adapt.*finally\|finally:.*data" \
    "run_adapt 包含 finally 清理"
echo ""

# 5. 检查配置验证
echo "### 配置验证 ###"
check_file_pattern "src/Pipeline_02_pretrain_fewshot.py" "stages.*验证\|stages.*validation\|missing.*required" \
    "Pipeline_02 包含 stages 验证逻辑"
check_file_pattern "src/Pipeline_03_multitask_pretrain_finetune.py" "stages.*验证\|stages.*validation\|missing.*required" \
    "Pipeline_03 包含 stages 验证逻辑"
echo ""

# 6. 检查边界情况处理
echo "### 边界情况处理 ###"
check_file_pattern "src/data_factory/dataset_task/Default_dataset.py" "len(processed_data).*== 0\|空数据集\|empty dataset" \
    "包含空数据集检测"
check_file_pattern "src/data_factory/dataset_task/Default_dataset.py" "不支持的归一化\|Unknown normalization\|supported normalization" \
    "包含归一化参数验证"
check_file_pattern "src/data_factory/dataset_task/Default_dataset.py" "denominator.*== 0\|std_vals.*== 0\|常数信号" \
    "包含常数信号处理"
echo ""

# 7. 检查测试文件
echo "### 测试覆盖 ###"
if [ -f "test/test_pipeline_resource_management.py" ]; then
    echo -e "${GREEN}✅${NC} 资源管理测试文件存在"
else
    echo -e "${RED}❌${NC} 资源管理测试文件缺失"
fi
echo ""

# 总结
echo "=== 修复状态总结 ==="
echo "详细内容见: .specify/goals/v2/GOAL-FFU-P1-007-pipeline-resource-management.md"
echo "审查报告: .env/素材/5-4/review/6-24/pipeline/orchestration_bugs.md"
