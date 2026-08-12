# Explainable FD Toolkit - Scripts Guide

## 项目概述

统一的可解释性故障诊断工具包，集成GradCAM可视化、模型插件系统、1D-2D融合和神经符号集成，提供全方位的可解释性分析工具。

## 测试脚本

### test_unified_modelplugin_tspn_resnet.py

**用途**：验证ModelPlugin系统的核心功能和可解释性接口

**功能验证**：
- ✅ TSPN和ResNet模型动态加载
- ✅ GradCAM可解释性可视化生成
- ✅ 1D-2D多模态融合支持
- ✅ 神经符号集成功能
- ✅ 统一的可解释性接口

**运行方式**：
```bash
cd Paper/Explainable_FD_Toolkit/scripts
python test_unified_modelplugin_tspn_resnet.py
```

**预期输出**：
```
[Testing TSPN ModelPlugin]
  - Input shape: [1, 4096, 3]
  - Model output shape: [1, 5]
  - GradCAM shape: [1, 4096]
  - ✅ TSPN ModelPlugin test completed

[Testing 1D-2D Fusion Explainability]
  - Multi-modal features: 1D + 2D + statistical
  - ✅ 1D-2D Fusion explainability test completed
```

**技术细节**：

1. **输入格式**：
   - 张量形状：`(batch_size, in_dim=4096, in_channels=3)`
   - 数据类型：`torch.float32`
   - 设备：CUDA/CPU自动检测

2. **ModelPlugin架构**：
   ```
   Model → Feature Extraction → GradCAM → Feature Importance → Natural Language Explanation
   ```

3. **可解释性方法**：
   - **GradCAM**：梯度加权的类激活映射
   - **特征重要性**：各特征的贡献度分析
   - **多模态可视化**：1D/2D特征可视化
   - **符号推理**：逻辑解释生成

## ModelPlugin系统

**核心功能**：
```python
class ModelPlugin:
    def get_feature_maps(self, x, layer_name=None)    # 获取特征图
    def generate_gradcam(self, x, target_class=None)   # 生成GradCAM
    def explain_prediction(self, x, method="gradcam")  # 解释预测
```

**支持的模型**：
- TSPN（透明信号处理网络）
- ResNet（深度残差网络）
- Fusion1D-2D（多模态融合）
- 任何继承nn.Module的模型

**插件特性**：
- 动态模型加载
- 自动特征提取
- 多种可解释性方法
- 统一的接口设计

## GradCAM可视化

**实现原理**：
```python
# 1. 前向传播获取特征图
feature_maps = self.get_feature_maps(x)

# 2. 计算类别得分梯度
class_score.backward()

# 3. 全局平均池化梯度
pooled_gradients = torch.mean(gradients, dim=1, keepdim=True)

# 4. 加权特征图并ReLU激活
cam = torch.sum(feature_maps * pooled_gradients, dim=2)
cam = F.relu(cam)
```

**可视化输出**：
- 热力图显示重要区域
- 归一化到[0,1]范围
- 支持不同输入维度

## 1D-2D融合可解释性

**融合架构**：
- **1D分支**：时序信号卷积特征
- **2D分支**：频谱图卷积特征
- **统计特征**：均值、方差、RMS等
- **融合层**：特征拼接与分类

**可解释性分析**：
- 各模态贡献度分析
- 特征重要性排名
- 决策路径可视化

## 神经符号集成

**符号映射**：
```python
# 神经网络输出 → 符号表示
symbolic_facts = [
    "has_fault(outer_race_fault)",
    "severity(outer_race_fault, high)",
    "location(outer_race_fault, bearing)"
]
```

**逻辑推理**：
- 基于规则的推理引擎
- 一阶谓词逻辑支持
- 可解释的推理链

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

**可解释性参数**：
```yaml
gradcam_method: "guided"
feature_importance_method: "permutation"
visualization_size: [224, 224]
symbolic_threshold: 0.5
```

## 性能指标

**ModelPlugin性能**：
- ✅ 动态模型加载成功
- ✅ GradCAM生成正确
- ✅ 特征提取有效
- ✅ 多方法兼容

**可解释性指标**：
- 解释覆盖率
- 特征重要性清晰度
- 可视化质量
- 推理逻辑一致性

## 可视化功能

1. **GradCAM热力图**：
   - 输入信号的重要区域
   - 多层特征融合
   - 时间维度可视化

2. **特征重要性图**：
   - 特征贡献排名
   - 类别特异性分析
   - 时间序列重要性

3. **融合可视化**：
   - 1D/2D模态贡献
   - 特征融合权重
   - 多模态决策边界

## 依赖项

- torch >= 1.9.0
- torchvision（用于ResNet）
- numpy
- matplotlib（可视化）
- opencv-python（图像处理）
- scikit-learn（特征重要性）
- 统一基线框架：`model/Fusion1D2D_simple.py`

## 故障排除

**常见问题**：
1. **GradCAM为空**：检查模型梯度计算
2. **特征维度错误**：验证输入形状
3. **可视化失败**：确保matplotlib后端正确

**调试建议**：
- 使用简单模型验证流程
- 检查中间张量形状
- 验证梯度计算路径

## 扩展应用

1. **新可解释性方法**：添加SHAP、LIME等
2. **多模型集成**：同时分析多个模型
3. **实时可解释性**：在线诊断解释
4. **交互式可视化**：用户界面集成

## 应用场景

1. **故障诊断**：提供诊断依据和解释
2. **模型验证**：理解模型决策过程
3. **用户培训**：帮助用户理解AI决策
4. **合规要求**：满足可解释性法规

## 解释报告生成

**自动报告内容**：
- 主要发现和建议
- 特征重要性分析
- GradCAM可视化结果
- 逻辑推理链条
- 置信度评估

## 工具包API

**核心接口**：
```python
# 初始化插件
plugin = ModelPlugin("TSPN", model, args)

# 生成解释
explanation = plugin.explain_prediction(x, method="gradcam")

# 获取GradCAM
gradcam = plugin.generate_gradcam(x, target_class=2)

# 特征重要性
importance = plugin.explain_prediction(x, method="feature_importance")
```

## 相关论文

- "Grad-CAM: Why Did You Say That?"
- "Explainable AI for Fault Diagnosis: A Review"
- "Visual Explanations from Deep Networks via Gradient-based Localization"

## 更新日志

- 2025-11-27: 创建ModelPlugin系统
- 2025-11-27: 集成GradCAM可视化
- 2025-11-27: 添加1D-2D融合支持
- 2025-11-27: 扩展神经符号集成功能