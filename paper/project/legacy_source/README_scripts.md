# TII Operator Attention - Scripts Guide

## 项目概述

基于时域-频域算子注意力机制（Time-domain Operator Attention, TII）的可解释故障诊断方法，通过信号处理算子的注意力权重实现端到端的可解释性分析。

## 测试脚本

### test_unified_operator_attention_simple_init.py

**用途**：验证Operator Attention模型在统一基线框架下的初始化和前向传播

**功能验证**：
- ✅ 统一配置兼容性
- ✅ 算子注意力机制
- ✅ 时域-频域联合处理
- ✅ 可解释性权重生成
- ✅ 多头算子注意力

**运行方式**：
```bash
cd Paper/TII_operator_attention/scripts
python test_unified_operator_attention_simple_init.py
```

**预期输出**：
```
[OperatorAttention_simple Unified Check] forward ok, output shape = [2, 10]
[OperatorAttention_simple Unified Check] output range = [min, max]
[Debug] Operator attention weights shape: [batch_size, num_operators, num_heads]
```

**技术细节**：

1. **输入格式**：
   - 张量形状：`(batch_size, in_dim=4096, in_channels=3)`
   - 数据类型：`torch.float32`
   - 设备：CUDA/CPU自动检测

2. **处理流程**：
   ```
   Input Signal → Operator Extraction → Attention Weights → Weighted Combination → Classification → Explainability
   ```

3. **模型架构**：
   - **信号处理层**：4层TSPN信号处理（I, WF, I）
   - **算子提取器**：提取多种信号处理算子特征
   - **注意力机制**：计算算子重要性权重
   - **多头注意力**：多个注意力头捕获不同模式
   - **可解释性模块**：生成算子重要性解释

4. **输出**：
   - 分类输出：`(batch_size, num_classes=10)`
   - 注意力权重：`(batch_size, num_operators, num_heads)`

## 核心创新点

1. **算子级注意力**：直接在信号处理算子上学习注意力
2. **端到端可解释**：注意力权重提供天然解释
3. **时域-频域融合**：联合处理不同域的信号特征
4. **自适应算子选择**：根据输入动态选择重要算子

## 算子库

**支持的算子类型**：
- **时域算子**：
  - 移动平均（Moving Average）
  - 微分算子（Derivative）
  - 积分算子（Integral）
  - 统计算子（Statistical）

- **频域算子**：
  - FFT变换（Fast Fourier Transform）
  - 小波变换（Wavelet Transform）
  - 滤波器组（Filter Banks）
  - 希尔伯特变换（Hilbert Transform）

- **混合算子**：
  - 时频联合算子
  - 自适应滤波算子
  - 非线性变换算子

## 注意力机制

**多头注意力配置**：
```python
num_heads: 8
attention_dim: 64
num_operators: 12
dropout: 0.1
```

**注意力计算**：
```python
# Query, Key, Value生成
Q = self.query_proj(operator_features)
K = self.key_proj(operator_features)
V = self.value_proj(operator_features)

# 注意力权重计算
attention_weights = torch.softmax(Q @ K.T / sqrt(d_k), dim=-1)
weighted_output = attention_weights @ V
```

## 可解释性分析

1. **算子重要性排名**：
   ```python
   # 获取算子重要性
   importance_scores = torch.mean(attention_weights, dim=(0, 2))
   ranked_operators = torch.argsort(importance_scores, descending=True)
   ```

2. **故障-算子关联**：
   - 分析不同故障类型的算子偏好
   - 识别特定故障的关键算子
   - 生成算子重要性热图

3. **时域-频域贡献**：
   - 时域算子 vs 频域算子贡献比
   - 不同故障类型的域偏好分析
   - 自适应域选择策略

## 实验配置

**模型参数**：
```yaml
in_dim: 4096
in_channels: 3
out_channels: 3
num_classes: 10
scale: 3
skip_connection: True
```

**注意力参数**：
```yaml
num_heads: 8
attention_dim: 64
num_operators: 12
temperature: 1.0
dropout: 0.1
```

## 性能指标

**模型性能**：
- ✅ 注意力权重收敛性
- ✅ 算子选择有效性
- ✅ 多头注意力协同

**可解释性指标**：
- 算子重要性清晰度
- 注意力权重分布
- 故障-算子关联强度

## 可视化功能

1. **注意力权重热图**：
   - 显示各算子的注意力分布
   - 多头注意力模式分析
   - 时序变化趋势

2. **算子贡献分析**：
   - 算子重要性柱状图
   - 故障类型vs算子矩阵
   - 域贡献饼图

3. **决策路径可视化**：
   - 算子激活序列
   - 注意力流动轨迹
   - 特征传播路径

## 依赖项

- torch >= 1.9.0
- numpy
- matplotlib（可选，用于可视化）
- seaborn（可选，用于高级可视化）
- 统一基线框架：`model/OperatorAttention_simple.py`

## 故障排除

**常见问题**：
1. **注意力权重退化**：调整温度参数
2. **算子冗余**：减少算子数量或添加正则化
3. **多头注意力发散**：增加dropout或减小学习率

**调试建议**：
- 可视化注意力权重分布
- 检查算子特征相关性
- 验证多头注意力一致性

## 扩展应用

1. **动态算子库**：根据数据自动扩展算子
2. **层次化注意力**：多层注意力结构
3. **跨模态注意力**：融合不同传感器数据
4. **在线学习**：实时更新注意力权重

## 应用场景

1. **故障诊断**：识别关键故障特征算子
2. **预测性维护**：通过算子变化预测故障
3. **质量控制**：产品缺陷的关键算子识别
4. **设备监控**：实时算子重要性分析

## 算子解释报告

**自动生成报告内容**：
- Top-5重要算子排名
- 算子贡献百分比
- 时域-频域贡献比
- 故障特异性分析
- 建议改进措施

## 相关论文

- "Attention Is All You Need"
- "Explainable AI: A Review of the Main Debates and Technical Challenges"
- "Interpretable Deep Learning for Fault Diagnosis"

## 更新日志

- 2025-11-27: 创建统一基线兼容版本
- 2025-11-27: 添加多头算子注意力机制
- 2025-11-27: 增强可解释性可视化功能