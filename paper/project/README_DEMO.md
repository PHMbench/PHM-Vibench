# LLM Explainable Fault Diagnosis Toolkit Demo

## 项目概述

这是一个基于模板的LLM故障诊断解释生成系统演示，实现了**阶段1：不依赖外部API的模板LLM stub demo**。

## 核心功能

### 1. 中间表示系统
- **LLMIntermediateRepresentation**: 结构化的故障诊断中间表示
- **FaultInfo**: 故障信息（类型、置信度、概率分布等）
- **SignalAnalysis**: 信号分析结果（统计特征、频域分析、关键发现）
- **TechnicalExplanation**: 技术解释（重要特征、频率成分、处理阶段）
- **DeviceContext**: 设备上下文信息

### 2. 模板化LLM系统
- **LocalTemplateLLM**: 本地模板LLM，无需外部API
- 支持7种解释类型：
  - 一般故障解释
  - 原因分析
  - 维修指导
  - 严重程度评估
  - 技术细节分析
  - 预防策略
  - 监测建议

### 3. 适配器系统
- **ExplanationToIRAdapter**: 将各种解释格式转换为中间表示
- **MockDataAdapter**: 创建演示用的模拟数据

## 快速开始

### 运行完整流程演示
```bash
cd Paper/LLM_Explainable_FD_Toolkit
python experiments/scripts/run_minimal_llm_demo_standalone.py --mode pipeline
```

### 运行单个案例演示
```bash
# 案例1：轴承内圈故障
python experiments/scripts/run_minimal_llm_demo_standalone.py --mode single --case 0

# 案例2：设备不对中故障
python experiments/scripts/run_minimal_llm_demo_standalone.py --mode single --case 1
```

## 系统架构

```
LLM工具包结构
├── code/llm_explainable_toolkit/
│   ├── core/
│   │   ├── intermediate_representation.py  # 中间表示定义
│   │   └── adapters.py                     # 适配器实现
│   └── llm_integration/
│       └── local_template_llm.py           # 本地模板LLM
└── experiments/scripts/
    └── run_minimal_llm_demo_standalone.py  # 演示脚本
```

## 核心组件说明

### 中间表示 (Intermediate Representation)
提供标准化的数据结构来连接技术分析和自然语言生成：

```python
@dataclass
class LLMIntermediateRepresentation:
    explanation_id: str
    timestamp: str
    fault_info: FaultInfo
    signal_analysis: SignalAnalysis
    technical_explanation: TechnicalExplanation
    device_context: DeviceContext
    user_query: Optional[str]
    explanation_style: str
    # ... 更多字段
```

### 本地模板LLM (LocalTemplateLLM)
基于模板和规则的文本生成系统：

```python
llm = LocalTemplateLLM(style="standard")
context = {"intermediate_representation": ir}
response = llm.generate("请解释这个故障的原因", context)
```

### 支持的故障类型
- **内圈故障**: 滚动轴承内圈表面损伤
- **外圈故障**: 滚动轴承外圈表面损伤
- **不对中**: 设备轴线偏移
- **齿轮故障**: 齿面磨损或断裂
- **不平衡**: 转子质量分布不均

## 演示输出示例

### 故障诊断结果
```
# 故障诊断结果

**检测到故障类型：** 内圈故障
**诊断置信度：** 87.0%
**设备类型：** 滚动轴承

## 主要发现
检测到 内圈故障 特征频率成分；振动RMS值显著增高；频域分析显示明显谐波

## 故障描述
滚动轴承内圈表面出现疲劳、剥落或裂纹等损伤

**特征频率：** 157.5 Hz
```

### 维修指导
```
# 内圈故障 维修指导

**紧急程度：** 紧急（置信度：87.0%）

## 维修步骤
1. 停止设备运行，确保安全
2. 拆卸轴承检查内圈损伤情况
3. 评估损伤程度决定更换或修复
4. 检查相关部件（轴、密封等）状态
5. 安装新轴承并正确调整间隙
6. 更换润滑剂并进行试运行测试

**建议执行时间：** 24小时内
```

## 技术特点

### 1. 无外部依赖
- 完全本地运行，无需网络连接
- 不依赖外部LLM API
- 所有功能模块自包含

### 2. 结构化数据流
```
信号数据 → 模型预测 → 中间表示 → 模板LLM → 自然语言解释
```

### 3. 智能意图识别
自动识别用户查询类型并生成相应的解释：
- 原因分析关键词："原因", "为什么", "why", "cause"
- 维修指导关键词："维修", "维护", "修复", "repair", "fix"
- 严重程度关键词："严重", "风险", "危险", "severity", "risk"

### 4. 知识库支持
内置故障知识库，包含：
- 故障机理描述
- 常见原因分析
- 标准维修步骤
- 预防措施建议

## 阶段1完成状态

✅ **已完成**：
- [x] 中间表示结构定义
- [x] 适配脚本实现
- [x] 本地模板LLM stub
- [x] 最小demo脚本
- [x] 数据流验证测试
- [x] 完整演示流程

🎯 **核心产出**：
1. **对话演示系统**: 支持多轮问答的自然语言解释
2. **提示模板系统**: 7种类型的专业化解释模板
3. **解释生成样例**: 完整的故障诊断解释案例
4. **数据处理流程**: 从原始数据到自然语言解释的完整链路

## 下一步计划（阶段2）

- 接入真实Explainable_FD_Toolkit输出
- 多样化模板和对话风格
- 与主仓库模型的真实数据集成
- 交互式对话界面

## 文件说明

### 核心模块
- `intermediate_representation.py`: 中间表示数据结构定义
- `adapters.py`: 数据格式转换适配器
- `local_template_llm.py`: 本地模板LLM实现

### 演示脚本
- `run_minimal_llm_demo_standalone.py`: 独立运行版本（推荐使用）
- `run_minimal_llm_demo.py`: 完整版本（依赖外部模块）

## 使用说明

### 环境要求
- Python 3.7+
- numpy

### 运行命令
```bash
# 确保在正确目录
cd /home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/LLM_Explainable_FD_Toolkit

# 运行完整流程演示
python experiments/scripts/run_minimal_llm_demo_standalone.py --mode pipeline

# 运行单个案例
python experiments/scripts/run_minimal_llm_demo_standalone.py --mode single --case 0
```

### 自定义使用
```python
from llm_explainable_toolkit.core.intermediate_representation import create_mock_ir
from llm_explainable_toolkit.llm_integration.local_template_llm import LocalTemplateLLM

# 创建中间表示
ir = create_mock_ir(fault_type="内圈故障", confidence=0.85)

# 创建LLM实例
llm = LocalTemplateLLM(style="standard")

# 生成解释
context = {"intermediate_representation": ir}
response = llm.generate("请解释这个故障的原因", context)
print(response)
```

---

**项目状态**: ✅ 阶段1完成
**更新时间**: 2024-11-26
**版本**: 1.0.0