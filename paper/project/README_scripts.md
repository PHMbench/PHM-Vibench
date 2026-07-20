# Fuzzy-XFD: Fuzzy Logic Explainable Fault Diagnosis - Scripts Guide

## 项目概述

基于模糊逻辑和一阶谓词逻辑的可解释故障诊断方法，结合模糊推理和神经网络，提供透明化的决策过程和人类可理解的诊断规则。

## 测试脚本

### test_unified_fuzzylogic_simple_init.py

**用途**：验证模糊逻辑模型在统一基线框架下的初始化和前向传播

**功能验证**：
- ✅ 统一配置兼容性
- ✅ 模糊隶属度函数
- ✅ 一阶谓词逻辑推理
- ✅ 模糊规则库
- ✅ 可解释性分析

**运行方式**：
```bash
cd Paper/Paper_fuzzy_XFD/scripts
python test_unified_fuzzylogic_simple_init.py
```

**预期输出**：
```
[FuzzyLogic_simple Unified Check] forward ok, output shape = [2, 5]
[FuzzyLogic_simple Unified Check] output range = [min, max]
[Debug] Reduced features shape: [2, 32]
[Debug] Membership values shape: [2, 32, 3]
[Debug] Fuzzy output shape: [2, 5]
```

**技术细节**：

1. **输入格式**：
   - 张量形状：`(batch_size, in_dim=4096, in_channels=3)`
   - 数据类型：`torch.float32`
   - 设备：CUDA/CPU自动检测

2. **处理流程**：
   ```
   Signal Processing → Feature Reduction → Fuzzy Membership → Rule Application → Defuzzification → Classification
   ```

3. **模型架构**：
   - **信号处理层**：4层TSPN信号处理（I, WF, I）
   - **特征降维**：`out_channels * 64 → 32` 维度
   - **模糊隶属度函数**：32个特征 × 3个隶属函数（低、中、高）
   - **模糊规则库**：10条模糊推理规则
   - **解模糊化**：从模糊输出到清晰分类

## 核心创新点

1. **模糊逻辑集成**：将人类专家知识编码为模糊规则
2. **一阶谓词逻辑**：支持复杂的逻辑推理表达式
3. **可解释推理**：每个决策都有明确的逻辑解释
4. **知识驱动**：结合数据驱动和知识驱动方法

## 模糊逻辑系统

**隶属度函数**：
```python
# 高斯隶属度函数参数
num_fuzzy_features: 32
num_membership_functions: 3  # Low, Medium, High
centers: 随机初始化，可学习
widths: 正值初始化，可学习
```

**模糊规则库**：
- Rule 1: IF features[0] is High AND features[1] is Low THEN fault_type[0]
- Rule 2: IF features[2] is Medium AND features[3] is High THEN fault_type[1]
- ... 共10条规则

**规则权重**：
- 每条规则都有可学习的权重
- 支持规则重要性自动调整

## 一阶谓词逻辑

**谓词定义**：
```python
# 基本谓词
has_fault(x): 表示x存在故障
severity(x, s): 故障严重程度s ∈ {low, medium, high}
location(x, l): 故障位置l ∈ {bearing, gear, motor}
frequency(x, f): 振动频率特征f ∈ {low, medium, high}
```

**逻辑规则**：
```
∀x (has_fault(x) ∧ severity(x, high) → immediate_action(x))
∃x (has_fault(x) ∧ location(x, bearing) ∧ frequency(x, high))
∀x (normal(x) → ¬has_fault(x))
```

## 可解释性功能

1. **隶属度可视化**：
   ```python
   # 获取隶属度值
   membership_values = model.compute_membership(features)
   # 可视化每个特征的模糊隶属度
   ```

2. **规则激活分析**：
   ```python
   # 规则激活强度
   rule_activations = model.apply_rules(membership_values)
   # 识别最强激活的规则
   ```

3. **决策路径追踪**：
   - 记录输入特征到输出的完整推理路径
   - 显示每条规则的贡献度
   - 生成自然语言解释

## 模糊推理过程

**Fuzzification（模糊化）**：
```python
# 计算每个特征对每个隶属函数的隶属度
membership = torch.exp(-((x - centers) ** 2) / (2 * widths ** 2))
```

**Rule Application（规则应用）**：
```python
# 聚合隶属度值
aggregated_membership = torch.mean(membership_values, dim=2)
# 加权规则激活
rule_activation = torch.sum(aggregated_membership * rule_weights, dim=1)
```

**Defuzzification（解模糊化）**：
```python
# 加权平均得到清晰输出
final_output = torch.sum(rule_outputs * rule_activation, dim=1)
```

## 实验配置

**模型参数**：
```yaml
in_dim: 4096
in_channels: 3
out_channels: 3
num_classes: 5
scale: 3
skip_connection: True
```

**模糊逻辑参数**：
```yaml
num_fuzzy_features: 32
num_membership_functions: 3
num_fuzzy_rules: 10
fuzzy_inference_method: "mamdani"
defuzzification_method: "centroid"
```

## 性能指标

**模糊系统性能**：
- ✅ 隶属度函数有效性
- ✅ 规则库完整性
- ✅ 推理逻辑一致性

**可解释性指标**：
- 规则激活强度分布
- 隶属度函数形状
- 决策透明度

## 可视化分析

1. **隶属度函数图**：
   - 32个特征的隶属度函数曲线
   - 低、中、高三个级别的分布
   - 学习前后的变化对比

2. **规则激活热图**：
   - 不同输入下的规则激活模式
   - 规则与故障类型的关联性
   - 规则重要性排名

3. **推理流程图**：
   - 从输入到输出的完整路径
   - 关键决策节点
   - 置信度传播

## 依赖项

- torch >= 1.9.0
- numpy
- matplotlib（可选，用于可视化）
- scikit-fuzzy（可选，高级模糊功能）
- 统一基线框架：`model/FuzzyLogic_simple.py`

## 故障排除

**常见问题**：
1. **隶属度函数重叠**：调整centers和widths初始化
2. **规则冲突**：检查规则逻辑一致性
3. **训练不稳定**：使用较小的学习率

**调试建议**：
- 可视化隶属度函数
- 分析规则激活模式
- 验证推理逻辑正确性

## 扩展应用

1. **自适应模糊系统**：动态调整隶属度函数
2. **层次化规则**：多层模糊推理结构
3. **神经-模糊混合**：结合神经网络学习能力
4. **专家系统集成**：导入领域专家知识

## 应用场景

1. **故障诊断**：透明化的故障识别和解释
2. **预测性维护**：基于模糊规则的故障预测
3. **质量控制**：模糊边界的产品分类
4. **决策支持**：人类可理解的决策辅助

## 模糊规则示例

**典型故障规则**：
```
IF 振动强度 is High AND 频率特征 is High THEN 内圈故障
IF 温度 is Medium AND 噪声 is Low THEN 正常状态
IF 振动强度 is Medium AND 位移 is High THEN 外圈故障
```

## 相关论文

- "Fuzzy Logic for Fault Diagnosis in Rotating Machinery"
- "Explainable AI: A Survey of Interpretable Methods"
- "Neuro-Fuzzy Systems: A Comprehensive Review"

## 更新日志

- 2025-11-27: 创建统一基线兼容版本
- 2025-11-27: 修复reshape vs view兼容性问题
- 2025-11-27: 增强模糊规则应用逻辑
- 2025-11-27: 添加一阶谓词逻辑支持