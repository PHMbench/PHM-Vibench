# Explainable_FD_Toolkit 使用指南

## 环境配置

### 基础依赖
```bash
# 核心依赖
pip install torch torchvision
pip install numpy pandas scipy
pip install matplotlib seaborn
pip install scikit-learn

# 可解释性相关
pip install shap lime
pip install captum
pip install explainable-ai

# 可选：LLM支持
pip install transformers
pip install openai
```

### 环境验证
```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 检查基础环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 导入工具包模块
try:
    from toolkit_integration.TSPN_explainable import TSPN_Explainable
    print("✅ TSPN可解释模块加载成功")
except ImportError as e:
    print(f"❌ TSPN模块加载失败: {e}")
```

## 快速开始

### 1. 基础使用示例

```python
from toolkit_integration.TSPN_explainable import TSPN_Explainable
from toolkit_integration.llm_explainable_base import LLM_Explainer
import torch

# 初始化TSPN可解释模型
config = {
    'model_type': 'TSPN',
    'signal_processing_layers': ['FFT', 'HT', 'WF', 'I'],
    'feature_extractor': 'statistical',
    'num_classes': 10
}

explainer = TSPN_Explainable(config)

# 生成示例数据
batch_size = 32
signal_length = 4096
dummy_signals = torch.randn(batch_size, signal_length)
fault_labels = torch.randint(0, 10, (batch_size,))

# 获取模型预测和解释
with torch.no_grad():
    predictions = explainer.predict(dummy_signals)
    explanations = explainer.explain(dummy_signals, fault_labels)

print(f"预测结果形状: {predictions.shape}")
print(f"解释信息: {explanations.keys()}")
```

### 2. 可视化解释结果

```python
import matplotlib.pyplot as plt

# 可视化信号处理路径
signal_idx = 0  # 选择第0个信号样本
explainer.visualize_processing_path(
    dummy_signals[signal_idx],
    save_path="figures/processing_path.png"
)

# 可视化特征重要性
explainer.visualize_feature_importance(
    explanations['feature_importance'][signal_idx],
    save_path="figures/feature_importance.png"
)

# 生成综合解释报告
explainer.generate_explanation_report(
    signal_data=dummy_signals[signal_idx],
    prediction=predictions[signal_idx],
    explanation=explanations,
    save_path="doc/explanation_report.pdf"
)
```

### 3. LLM增强解释

```python
# 初始化LLM解释器（需要API密钥）
llm_explainer = LLM_Explainer(
    model_name="gpt-3.5-turbo",
    api_key="your_api_key_here"
)

# 生成自然语言解释
fault_type = "内圈故障"
confidence = 0.92
key_features = explanations['feature_importance'][signal_idx].topk(5)

natural_explanation = llm_explainer.generate_explanation(
    fault_type=fault_type,
    confidence=confidence,
    key_features=key_features,
    context="工业电机故障诊断"
)

print("LLM生成的解释:")
print(natural_explanation)
```

## 高级功能

### 1. 批量解释分析

```python
def batch_explain_analysis(signals, labels, explainer):
    """批量分析多个样本的解释模式"""

    all_explanations = []

    for i in range(len(signals)):
        # 获取单个样本的解释
        explanation = explainer.explain(
            signals[i:i+1],
            labels[i:i+1]
        )
        all_explanations.append(explanation)

    # 分析解释模式
    feature_patterns = analyze_feature_patterns(all_explanations)
    fault_explanations = group_by_fault_type(all_explanations, labels)

    return {
        'feature_patterns': feature_patterns,
        'fault_explanations': fault_explanations
    }

def analyze_feature_patterns(explanations):
    """分析特征重要性模式"""
    import numpy as np

    # 提取所有样本的特征重要性
    importances = [exp['feature_importance'].squeeze() for exp in explanations]
    importances = np.array(importances)

    # 计算统计信息
    mean_importance = np.mean(importances, axis=0)
    std_importance = np.std(importances, axis=0)

    return {
        'mean_importance': mean_importance,
        'std_importance': std_importance,
        'consistent_features': np.where(std_importance < 0.1)[0]
    }
```

### 2. 实时解释系统

```python
class RealTimeExplainer:
    """实时故障诊断解释系统"""

    def __init__(self, model_path, config):
        self.explainer = TSPN_Explainable(config)
        self.explainer.load_model(model_path)
        self.llm_explainer = LLM_Explainer() if config.get('use_llm', False) else None

    def process_signal(self, signal_data):
        """处理实时信号数据"""
        # 预测故障类型
        with torch.no_grad():
            prediction = self.explainer.predict(signal_data.unsqueeze(0))
            fault_type = prediction.argmax().item()
            confidence = torch.softmax(prediction, dim=1).max().item()

        # 生成解释
        explanation = self.explainer.explain(signal_data.unsqueeze(0))

        # 如果启用LLM，生成自然语言解释
        if self.llm_explainer:
            natural_explanation = self.llm_explainer.generate_explanation(
                fault_type=self._get_fault_name(fault_type),
                confidence=confidence,
                key_features=explanation['feature_importance'].squeeze()
            )
        else:
            natural_explanation = None

        return {
            'fault_type': fault_type,
            'confidence': confidence,
            'technical_explanation': explanation,
            'natural_explanation': natural_explanation,
            'timestamp': datetime.now()
        }

    def _get_fault_name(self, fault_idx):
        """根据索引获取故障名称"""
        fault_names = ['正常', '内圈故障', '外圈故障', '滚动体故障', '保持架故障',
                      '不平衡', '不对中', '摩擦', '松动', '其他故障']
        return fault_names[fault_idx] if fault_idx < len(fault_names) else '未知故障'
```

## 配置文件模板

### TSPN配置文件 (configs/tspn_config.yaml)
```yaml
model:
  type: "TSPN"
  input_dim: 4096
  output_dim: 10
  layers:
    - {type: "I", out_channels: 16}
    - {type: "FFT", out_channels: 32}
    - {type: "HT", out_channels: 32}
    - {type: "WF", out_channels: 64}

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: "adam"

explanation:
  methods: ["gradient", "integrated_gradients", "shap"]
  visualizations: ["feature_importance", "processing_path", "attention_weights"]

llm:
  enabled: false
  model: "gpt-3.5-turbo"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.7
```

## 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖正确安装，检查Python路径
2. **CUDA错误**: 检查PyTorch的CUDA版本匹配
3. **内存不足**: 减小batch_size或使用模型并行
4. **LLM API错误**: 检查API密钥和网络连接

### 性能优化

1. **模型量化**: 使用半精度浮点数减少内存使用
2. **批处理**: 增加batch_size提高GPU利用率
3. **缓存**: 缓存常用的解释结果
4. **异步处理**: 对于实时应用使用异步处理

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型输出形状
explainer = TSPN_Explainable(config)
dummy_input = torch.randn(1, 4096)

with torch.no_grad():
    output = explainer.model(dummy_input)
    print(f"模型输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

# 检查解释组件
explanation = explainer.explain(dummy_input, torch.tensor([0]))
for key, value in explanation.items():
    if torch.is_tensor(value):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")
```

## 扩展开发

### 添加新的解释方法

```python
class CustomExplainer(TSPN_Explainable):
    """自定义解释器"""

    def explain_with_custom_method(self, signal_data, target_class):
        """实现自定义解释方法"""

        # 1. 计算自定义解释指标
        custom_attribution = self._compute_custom_attribution(
            signal_data, target_class
        )

        # 2. 生成自定义可视化
        visualization = self._generate_custom_visualization(
            custom_attribution
        )

        return {
            'custom_attribution': custom_attribution,
            'visualization': visualization,
            'method_name': 'custom_method'
        }

    def _compute_custom_attribution(self, signal_data, target_class):
        """实现自定义归因计算"""
        # 这里实现你的自定义算法
        pass

    def _generate_custom_visualization(self, attribution):
        """生成自定义可视化"""
        # 这里实现你的可视化方法
        pass
```

这个使用指南提供了从基础安装到高级应用的完整指导，帮助用户快速上手并深入使用Explainable_FD_Toolkit的各项功能。