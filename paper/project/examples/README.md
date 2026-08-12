# Explainable_FD_Toolkit 使用示例

本目录包含了Explainable_FD_Toolkit的各种使用示例，展示了如何为不同类型的故障诊断模型生成解释。

## 示例列表

1. **[TSPN信号路径解释](01_tspn_signal_path.ipynb)**
   - 展示透明信号处理网络的信号路径追踪
   - 可视化信号在各层的变换过程
   - 分析算子重要性

2. **[NNSPN神经信号处理网络解释](02_nnspn_explanation.ipynb)**
   - 神经信号处理网络的梯度解释
   - 特征重要性分析
   - 层级贡献度分析

3. **[TKAN时间Kolmogorov-Arnold网络解释](03_tkan_temporal_explanation.ipynb)**
   - 时间模式解释
   - Kolmogorov-Arnold分解分析
   - 时序归因分析

4. **[批量解释与比较](04_batch_comparison.ipynb)**
   - 多模型解释方法比较
   - 批量处理优化
   - 解释质量评估

5. **[LLM增强解释](05_llm_enhanced_explanation.ipynb)**
   - 自然语言解释生成
   - 对话式解释系统
   - 个性化解释定制

6. **[实际工业应用案例](06_industrial_case_study.ipynb)**
   - 真实振动信号分析
   - 轴承故障诊断解释
   - 工程师友好的解释展示

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install torch torchvision
pip install captum matplotlib seaborn
pip install jupyter notebook
```

### 运行示例

```bash
# 启动Jupyter Notebook
jupyter notebook

# 或者直接运行Python脚本
python examples/01_tspn_signal_path.py
```

### 数据准备

部分示例需要预训练模型和测试数据：

```bash
# 下载示例数据（如果有）
wget https://example.com/data/demo_signals.pkl

# 或者使用内置的模拟数据生成器
python examples/utils/data_generator.py --type vibration --length 1000
```

## 示例特色

### 🎯 目标导向
- 每个示例解决特定的解释需求
- 从基础概念到高级应用循序渐进
- 包含完整的代码和详细注释

### 🔧 实用性强
- 基于真实工业场景设计
- 提供可直接运行的代码
- 包含最佳实践和常见陷阱

### 📊 可视化丰富
- 多种解释可视化方法
- 交互式图表展示
- 专业的工程图表

### 🌐 多模型支持
- TSPN、NNSPN、TKAN等模型适配
- 不同解释方法的对比
- 模型特定功能的展示

## 自定义示例

你可以基于这些示例创建自己的解释脚本：

```python
# 导入核心模块
from toolkit_integration.explainability import UnifiedExplainer
from toolkit_integration.TSPN_explainable import TSPN_Explainable

# 加载你的模型和数据
model = load_your_model("path/to/model.pth")
data = load_your_data("path/to/data.pkl")

# 创建解释器
explainer = UnifiedExplainer(model, method='auto')

# 生成解释
explanation = explainer.explain(data)

# 可视化结果
explanation.visualize()
```

## 贡献

欢迎贡献新的示例！请遵循以下规范：

1. **命名**: 使用数字前缀和描述性名称，如 `07_custom_model.ipynb`
2. **文档**: 包含完整的问题描述、代码注释和结果分析
3. **依赖**: 明确列出所需的依赖包
4. **数据**: 如果需要特殊数据，请提供数据生成脚本

## 技术支持

如果遇到问题，请：

1. 查看[API参考文档](../doc/api_reference.md)
2. 检查[常见问题解答](../doc/faq.md)
3. 在GitHub Issues中提问
4. 联系开发团队

## 更新日志

- **v1.0.0** (2024-11): 初始版本，包含6个核心示例
- **v1.1.0** (计划): 添加更多工业案例和高级功能示例

---

开始探索可解释性故障诊断的强大功能吧！