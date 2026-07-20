# 1D-2D Fusion Explainable Fault Diagnosis - Scripts Guide

## 项目概述

1D-2D融合可解释故障诊断方法，结合一维时序信号和二维频谱图的多模态学习，提升故障诊断性能和可解释性。

## 测试脚本

### test_unified_fusion1d2d_simple_init.py

**用途**：验证1D-2D融合模型在统一基线框架下的初始化和前向传播

**功能验证**：
- ✅ 统一配置兼容性
- ✅ 1D信号处理分支
- ✅ 2D频谱图分支
- ✅ 多模态特征融合
- ✅ 统计特征提取

**运行方式**：
```bash
cd Paper/1D-2D_fusion_explainable/scripts
python test_unified_fusion1d2d_simple_init.py
```

**预期输出**：
```
[Fusion1D2D_simple Unified Check] forward ok, output shape = [batch_size, 5]
[Fusion1D2D_simple Unified Check] output range = [min, max]
```

**技术细节**：

1. **输入格式**：
   - 张量形状：`(batch_size, in_dim=4096, in_channels=3)`
   - 数据类型：`torch.float32`
   - 设备：CUDA/CPU自动检测

2. **处理流程**：
   ```
   Input Signal → Signal Processing → 1D Branch + 2D Branch + Statistical Features → Fusion → Classification
   ```

3. **模型架构**：
   - **信号处理层**：4层TSPN信号处理（I, WF, I）
   - **1D分支**：1D CNN + 自适应池化
   - **2D分支**：STFT频谱图转换 + 2D CNN
   - **统计特征**：均值、方差、最大值、最小值、RMS
   - **融合层**：特征拼接 + 全连接分类

4. **输出**：
   - 形状：`(batch_size, num_classes=5)`
   - 内容：5类故障的logits

## 核心创新点

1. **多模态融合**：同时利用时域和频域信息
2. **可解释性**：保持1D和2D路径的可解释性
3. **端到端训练**：联合优化所有分支
4. **统一框架**：与TSPN基线框架完全兼容

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

**信号处理配置**：
```yaml
layer1: ["I", "WF", "I"]
layer2: ["I", "WF", "I"]
layer3: ["I", "WF", "I"]
layer4: ["I", "WF", "I"]
```

## 故障类别

| 类别 | 编码 | 描述 |
|------|------|------|
| 正常状态 | 0 | Normal |
| 内圈故障 | 1 | Inner Race Fault |
| 外圈故障 | 2 | Outer Race Fault |
| 滚动体故障 | 3 | Ball Fault |
| 保持架故障 | 4 | Cage Fault |

## 性能指标

测试脚本验证：
- ✅ 模型初始化成功
- ✅ 前向传播无错误
- ✅ 输出维度正确
- ✅ 数值范围合理

## 依赖项

- torch >= 1.9.0
- numpy
- 统一基线框架：`model/Fusion1D2D_simple.py`

## 故障排除

**常见问题**：
1. **维度不匹配**：检查`in_channels`和`scale`参数
2. **内存不足**：减少`batch_size`或`in_dim`
3. **设备错误**：确保CUDA环境正确配置

**调试建议**：
- 使用小批量测试：`batch_size=2`
- 检查中间特征维度
- 验证梯度计算正常

## 扩展应用

1. **自定义信号处理**：修改`layer1-4`配置
2. **不同的2D表示**：替换STFT为小波变换
3. **注意力机制**：在融合前添加注意力权重
4. **多尺度特征**：添加不同尺度的卷积核

## 相关论文

- Original 1D-2D Fusion Method Paper
- TSPN: Transparent Signal Processing Networks
- Multi-modal Learning for Fault Diagnosis

## 更新日志

- 2025-11-27: 创建统一基线兼容版本
- 2025-11-27: 添加测试脚本验证