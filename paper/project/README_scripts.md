# MoE-Explainable Fault Diagnosis - Scripts Guide

## 项目概述

基于混合专家（Mixture of Experts, MoE）架构的可解释故障诊断方法，通过多个专门化专家网络和门控网络实现条件计算和模型可解释性。

## 测试脚本

### test_unified_moe_simple_init.py

**用途**：验证MoE混合专家模型在统一基线框架下的初始化和前向传播

**功能验证**：
- ✅ 统一配置兼容性
- ✅ 多专家网络架构
- ✅ 门控机制（Gating Network）
- ✅ 专家负载均衡
- ✅ 可解释性路由分析

**运行方式**：
```bash
cd Paper/MOE_explainable/scripts
python test_unified_moe_simple_init.py
```

**预期输出**：
```
[MoE_simple Unified Check] forward ok, output shape = [2, 10]
[MoE_simple Unified Check] output range = [min, max]
[Debug] Expert loads: [expert1: 1, expert2: 1]
```

**技术细节**：

1. **输入格式**：
   - 张量形状：`(batch_size, in_dim=4096, in_channels=3)`
   - 数据类型：`torch.float32`
   - 设备：CUDA/CPU自动检测

2. **处理流程**：
   ```
   Input Signal → Signal Processing → Gating Network → Expert Selection → Expert Processing → Aggregation → Classification
   ```

3. **模型架构**：
   - **信号处理层**：4层TSPN信号处理（I, WF, I）
   - **门控网络**：计算专家选择权重
   - **专家网络**：多个专门化前馈网络
   - **聚合机制**：加权组合专家输出
   - **分类层**：10类分类（演示用）

4. **输出**：
   - 形状：`(batch_size, num_classes=10)`
   - 内容：多类别分类的logits

## 核心创新点

1. **条件计算**：只激活相关的专家网络
2. **专门化学习**：每个专家学习特定模式
3. **可解释路由**：门控权重提供决策解释
4. **负载均衡**：自动平衡专家使用频率

## 专家架构

**专家网络配置**：
```python
num_experts: 4
expert_hidden_dim: 256
expert_layers: 2
```

**门控网络特性**：
- 输入：处理后的信号特征
- 输出：专家选择权重（softmax归一化）
- 负载均衡：自动平衡专家使用

**专家类型**：
- Expert 0-1: 线性专家（轻量级）
- Expert 2-3: MLP专家（复杂模式）

## 可解释性功能

1. **专家选择分析**：
   ```python
   # 门控权重显示专家选择偏好
   gating_weights = model.gating_network(x)
   selected_experts = torch.argmax(gating_weights, dim=-1)
   ```

2. **负载均衡监控**：
   ```python
   # 专家使用统计
   expert_loads = torch.bincount(selected_experts)
   ```

3. **决策路径追踪**：
   - 记录每批次的专家选择
   - 分析专家使用模式
   - 识别故障类型与专家关联

## 实验配置

**模型参数**：
```yaml
in_dim: 4096
in_channels: 3
out_channels: 3
num_classes: 10  # 演示用10类
scale: 3
skip_connection: True
```

**MoE特定参数**：
```yaml
num_experts: 4
expert_hidden_dim: 256
gating_temperature: 1.0
load_balance_weight: 0.01
```

## 性能指标

**模型性能**：
- ✅ 专家路由有效性
- ✅ 负载均衡特性
- ✅ 多专家协同工作

**可解释性指标**：
- 专家选择清晰度
- 专家专门化程度
- 决策路径可追踪性

## 依赖项

- torch >= 1.9.0
- numpy
- 统一基线框架：`model/MoE_simple.py`

## 故障排除

**常见问题**：
1. **专家负载不均**：调整负载均衡权重
2. **门控网络过拟合**：添加正则化
3. **专家冗余**：减少专家数量或增加专门化约束

**调试建议**：
- 监控专家选择分布
- 检查门控权重范围
- 验证专家输出一致性

## 扩展应用

1. **动态专家**：根据数据分布调整专家
2. **层次化MoE**：多层专家结构
3. **专家剪枝**：移除冗余专家
4. **专家蒸馏**：压缩专家知识

## 应用场景

1. **多故障模式**：不同故障类型使用专门专家
2. **运行条件**：不同工况使用专门专家
3. **设备类型**：不同设备使用专门专家
4. **多模态融合**：不同模态使用专门专家

## 可视化分析

1. **专家热图**：显示专家选择模式
2. **负载分布**：专家使用频率分析
3. **门控轨迹**：决策过程可视化
4. **专家贡献**：各专家输出贡献度

## 相关论文

- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- "Switch Transformers: Scaling to Trillion Parameter Models"
- "Conditional Computation in Neural Networks"

## 更新日志

- 2025-11-27: 创建统一基线兼容版本
- 2025-11-27: 添加负载均衡和专家选择分析
- 2025-11-27: 增强可解释性功能