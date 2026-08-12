# Explainable_FD_Toolkit API 参考文档

## 概述

Explainable_FD_Toolkit 提供统一的可解释性API，支持多种故障诊断模型的解释生成。本文档详细介绍了工具集的所有核心接口、类和方法。

## 目录

- [核心架构](#核心架构)
- [统一解释器](#统一解释器)
- [解释对象](#解释对象)
- [基础解释器](#基础解释器)
- [模型适配器](#模型适配器)
- [LLM增强接口](#llm增强接口)
- [配置规范](#配置规范)
- [异常处理](#异常处理)

## 核心架构

### 模块组织

```
toolkit_integration/
├── explainability/
│   ├── core/
│   │   ├── unified_explainer.py     # 统一解释器
│   │   ├── base_explainer.py        # 基础解释器
│   │   └── explanation.py           # 解释对象
│   ├── methods/
│   │   ├── intrinsic/               # 本征解释方法
│   │   └── posthoc/                 # 事后解释方法
│   ├── llm/                         # LLM增强接口
│   └── conversation/                # 对话式解释
├── TSPN_explainable.py             # TSPN模型适配器
└── explainability_demo.py          # 使用示例
```

### 导入方式

```python
# 核心模块导入
from toolkit_integration.explainability import UnifiedExplainer
from toolkit_integration.explainability.core import Explanation
from toolkit_integration.explainability.core.base_explainer import BaseExplainer

# 模型适配器导入
from toolkit_integration.TSPN_explainable import TSPN_Explainable

# 快速函数导入
from toolkit_integration.explainability.core.unified_explainer import explain_model
```

## 统一解释器 (UnifiedExplainer)

### 类定义

```python
class UnifiedExplainer:
    def __init__(self,
                 model: torch.nn.Module,
                 config: Optional[Dict[str, Any]] = None,
                 method: str = 'auto')
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `torch.nn.Module` | ✓ | - | 要解释的PyTorch模型 |
| `config` | `Optional[Dict[str, Any]]` | ✗ | `{}` | 配置字典，包含解释方法的参数 |
| `method` | `str` | ✗ | `'auto'` | 解释方法，见下方支持的-methods |

### 支持的解释方法

| 方法名 | 类型 | 说明 | 适用模型 |
|--------|------|------|----------|
| `'auto'` | 自动 | 自动选择最佳解释方法 | 所有模型 |
| `'signal_path'` | 本征 | 信号路径追踪 | TSPN、支持信号路径的模型 |
| `'integrated_gradients'` | 事后 | 积分梯度法 | 所有模型 |
| `'deeplift'` | 事后 | DeepLift算法 | 所有模型 |
| `'saliency'` | 事后 | 梯度显著性分析 | 所有模型 |
| `'captum'` | 事后 | Captum通用接口 | 所有模型 |

### 核心方法

#### explain()

```python
def explain(self,
            input_data: torch.Tensor,
            target_class: Optional[int] = None,
            method: Optional[str] = None,
            **kwargs) -> Explanation
```

**参数说明:**
- `input_data`: 输入张量，形状 `[batch_size, sequence_length, channels]`
- `target_class`: 目标类别ID（可选）
- `method`: 临时覆盖默认解释方法
- `**kwargs`: 传递给特定解释器的额外参数

**返回值:**
- `Explanation`: 标准化解释对象

**使用示例:**
```python
# 基础使用
explainer = UnifiedExplainer(model, method='signal_path')
explanation = explainer.explain(signal_data, target_class=2)

# 临时切换方法
explanation = explainer.explain(signal_data, method='integrated_gradients')
```

#### explain_batch()

```python
def explain_batch(self,
                  input_data: torch.Tensor,
                  target_classes: Optional[List[int]] = None,
                  **kwargs) -> List[Explanation]
```

**参数说明:**
- `input_data`: 批量输入张量
- `target_classes`: 每个样本的目标类别列表
- `**kwargs`: 额外参数

**返回值:**
- `List[Explanation]`: 解释对象列表

**使用示例:**
```python
explanations = explainer.explain_batch(
    batch_signals,
    target_classes=[0, 1, 2, 1],
    method='signal_path'
)
```

#### compare_methods()

```python
def compare_methods(self,
                    input_data: torch.Tensor,
                    target_class: Optional[int] = None,
                    methods: Optional[List[str]] = None,
                    **kwargs) -> Dict[str, Explanation]
```

**参数说明:**
- `input_data`: 输入张量
- `target_class`: 目标类别
- `methods`: 要比较的方法列表，None表示使用默认方法

**返回值:**
- `Dict[str, Explanation]`: 方法名到解释对象的映射

**使用示例:**
```python
comparisons = explainer.compare_methods(
    signal_data,
    target_class=1,
    methods=['signal_path', 'integrated_gradients', 'deeplift']
)

for method, explanation in comparisons.items():
    print(f"Method: {method}")
    metrics = explanation.get_metrics()
    print(f"Fidelity: {metrics.get('fidelity', 'N/A')}")
```

### 工具方法

#### get_available_methods()

```python
def get_available_methods() -> Dict[str, str]
```

**返回值:**
- `Dict[str, str]`: 方法名到描述的映射

#### get_model_explainability_info()

```python
def get_model_explainability_info() -> Dict[str, Any]
```

**返回值:**
- 包含模型可解释性信息的字典:
  - `model_type`: 模型类型
  - `supported_methods`: 支持的方法列表
  - `explainability_features`: 特殊功能列表

#### 工厂方法

```python
@staticmethod
def create_explainer(model: torch.nn.Module,
                    method: str = 'auto',
                    **config_kwargs) -> 'UnifiedExplainer'
```

**使用示例:**
```python
explainer = UnifiedExplainer.create_explainer(
    model,
    method='signal_path',
    include_frequency_analysis=True,
    baseline='zero'
)
```

## 解释对象 (Explanation)

### 类定义

```python
class Explanation:
    def __init__(self,
                 data: Dict[str, Any],
                 meta: Optional[Dict[str, Any]] = None)
```

### 数据访问方法

#### get_data() / get_meta()

```python
def get_data(self, key: str, default: Any = None) -> Any
def get_meta(self, key: str, default: Any = None) -> Any
```

**使用示例:**
```python
# 获取归因值
attribution = explanation.get_data('attributions')

# 获取元数据
method_name = explanation.get_meta('method')
model_name = explanation.get_meta('model_name')
```

#### get_attribution()

```python
def get_attribution() -> Optional[np.ndarray]
```

**返回值:**
- 主要归因值的numpy数组，如果不可用则返回None

**说明:**
自动按优先级查找归因值：`attributions` → `importance_scores` → `saliency` → `path`

#### get_method_name() / get_model_name()

```python
def get_method_name() -> str
def get_model_name() -> str
```

### 可视化方法

#### visualize()

```python
def visualize(self, mode: str = 'auto') -> matplotlib.figure.Figure
```

**可视化模式:**
- `'auto'`: 自动选择最佳可视化方式
- `'attribution'`: 归因值可视化
- `'path'`: 信号路径可视化
- `'importance'`: 重要性分数可视化

**使用示例:**
```python
import matplotlib.pyplot as plt

# 自动选择可视化
fig = explanation.visualize(mode='auto')
plt.title('故障诊断解释')
plt.show()

# 保存可视化
explanation.save_visualization('explanation.png', mode='path')
```

#### save_visualization()

```python
def save_visualization(self,
                      filepath: Union[str, Path],
                      mode: str = 'auto') -> None
```

### 序列化方法

#### to_dict() / to_json()

```python
def to_dict() -> Dict[str, Any]
def to_json(self, filepath: Union[str, Path]) -> None
```

**使用示例:**
```python
# 转换为字典
data = explanation.to_dict()

# 保存为JSON
explanation.to_json('explanation_result.json')
```

### 指标计算

#### get_metrics()

```python
def get_metrics() -> Dict[str, float]
```

**返回指标:**
- `attribution_mean`: 归因值均值
- `attribution_std`: 归因值标准差
- `attribution_max`: 最大绝对归因值
- `attribution_sparsity`: 归因值稀疏度（小于0.01的比例）

## 基础解释器 (BaseExplainer)

### 抽象基类

```python
from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, input_data: torch.Tensor, target_class: Optional[int] = None, **kwargs) -> Explanation:
        """生成解释的抽象方法"""
        pass
```

### 自定义解释器示例

```python
class CustomExplainer(BaseExplainer):
    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config)
        # 初始化自定义解释器

    def explain(self, input_data: torch.Tensor, target_class: Optional[int] = None, **kwargs) -> Explanation:
        # 实现自定义解释逻辑
        explanation_data = {
            'attributions': custom_attributions,
            'original_signal': input_data
        }

        explanation_meta = {
            'method': 'custom_method',
            'model_name': type(self.model).__name__,
            'input_shape': list(input_data.shape)
        }

        return Explanation(explanation_data, explanation_meta)
```

## 模型适配器

### TSPN模型适配器

#### TSPN_Explainable类

```python
class TSPN_Explainable:
    def __init__(self, config_path: str)
    def load_model(self, model_path: str) -> None
    def load_data(self, data_path: str) -> None
    def diagnose_and_explain(self, signal_data: torch.Tensor, fault_type: str) -> Tuple[Any, List[Explanation]]
    def visualize_explanations(self, explanations: List[Explanation], save_path: str) -> None
```

**使用示例:**
```python
# 初始化TSPN解释器
tspn_explainer = TSPN_Explainable(config_path="configs/tspn_config.yaml")
tspn_explainer.load_model("models/tspn_best.pth")
tspn_explainer.load_data("data/test_signals.pkl")

# 生成诊断和解释
diagnosis, explanations = tspn_explainer.diagnose_and_explain(signal_data, "inner_race")

# 可视化结果
tspn_explainer.visualize_explanations(explanations, save_path="figures/tspn_explanations/")
```

### 其他模型适配器

#### NNSPN适配器

```python
class NNSPN_Explainable:
    def __init__(self, config_path: str)
    def explain_signal_transformations(self, input_data: torch.Tensor) -> Explanation
    def analyze_neural_contributions(self, input_data: torch.Tensor) -> Explanation
```

#### TKAN适配器

```python
class TKAN_Explainable:
    def __init__(self, config_path: str)
    def explain_kolmogorov_arnold_decomposition(self, input_data: torch.Tensor) -> Explanation
    def analyze_temporal_patterns(self, input_data: torch.Tensor) -> Explanation
```

## LLM增强接口

### LLM增强解释器

```python
class LLMEnhancedExplainer:
    def __init__(self,
                 base_explainer: UnifiedExplainer,
                 llm_config: Dict[str, Any])

    def generate_natural_explanation(self,
                                   explanation: Explanation,
                                   target_audience: str = 'engineer') -> str

    def conversational_explain(self,
                              query: str,
                              context: Optional[Dict[str, Any]] = None) -> str
```

**配置参数:**
```python
llm_config = {
    "model": "gpt-4",           # LLM模型名称
    "language": "zh",           # 语言设置
    "api_key": "your_api_key",  # API密钥
    "max_tokens": 1000,         # 最大token数
    "temperature": 0.7          # 生成温度
}
```

**使用示例:**
```python
# 初始化LLM增强解释器
llm_explainer = LLMEnhancedExplainer(explainer, llm_config)

# 生成技术解释
technical_expl = explainer.explain(signal_data)
natural_expl = llm_explainer.generate_natural_explanation(
    technical_expl,
    target_audience="engineer"
)

# 对话式解释
answer = llm_explainer.conversational_explain(
    "为什么模型认为这个信号存在内圈故障？"
)
```

### 提示管理器

```python
from toolkit_integration.explainability.llm.prompt_manager import PromptManager

prompt_manager = PromptManager()

# 获取预设提示模板
prompt = prompt_manager.get_prompt(
    task="fault_explanation",
    audience="engineer",
    method="signal_path"
)

# 自定义提示
custom_prompt = prompt_manager.build_prompt(
    context="旋转机械故障诊断",
    explanation_data=explanation_data,
    user_question="请解释故障原因"
)
```

## 配置规范

### 完整配置示例

```yaml
# config/explainer_config.yaml
explainer:
  method: "auto"                    # 解释方法
  baseline: "zero"                  # 基线设置
  n_steps: 20                       # 积分步数
  internal_batch_size: 4            # 批处理大小

  # 可视化配置
  visualization:
    save_path: "figures/"
    dpi: 300
    format: "png"
    figsize: [12, 8]
    colormap: "viridis"

  # 指标计算配置
  metrics:
    compute_faithfulness: true      # 计算忠实性
    compute_stability: true         # 计算稳定性
    compute_complexity: true        # 计算复杂度
    compute_completeness: true      # 计算完整性

  # 信号路径特定配置
  signal_path:
    include_frequency_analysis: true
    include_energy_analysis: true
    include_statistics: true

  # LLM增强配置
  llm:
    enabled: false
    model: "gpt-4"
    language: "zh"
    max_tokens: 1000
    temperature: 0.7

# 模型特定配置
models:
  TSPN:
    default_method: "signal_path"
    signal_processing_layers: 4
    feature_types: ["Mean", "Std", "Max", "Min", "RMS"]

  NNSPN:
    default_method: "integrated_gradients"
    hidden_layers: [128, 64, 32]

  TKAN:
    default_method: "saliency"
    temporal_window: 100
```

### 配置加载

```python
import yaml
from pathlib import Path

# 加载配置文件
config_path = Path("configs/explainer_config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 使用配置创建解释器
explainer = UnifiedExplainer(model, config=config['explainer'])
```

## 异常处理

### 常见异常类型

```python
# 解释方法不支持
class UnsupportedMethodError(Exception):
    pass

# 模型不兼容
class ModelCompatibilityError(Exception):
    pass

# 数据格式错误
class DataFormatError(Exception):
    pass

# 配置错误
class ConfigurationError(Exception):
    pass
```

### 异常处理示例

```python
try:
    explainer = UnifiedExplainer(model, method='signal_path')
    explanation = explainer.explain(signal_data)
except UnsupportedMethodError as e:
    print(f"不支持的解释方法: {e}")
    # 降级到默认方法
    explainer = UnifiedExplainer(model, method='auto')
    explanation = explainer.explain(signal_data)
except DataFormatError as e:
    print(f"输入数据格式错误: {e}")
    # 重新格式化数据
    signal_data = preprocess_signal(signal_data)
    explanation = explainer.explain(signal_data)
except Exception as e:
    print(f"解释生成失败: {e}")
    # 记录错误并继续
    logger.error(f"Explanation failed: {e}")
```

## 快速函数参考

### explain_model()

```python
def explain_model(model: torch.nn.Module,
                  input_data: torch.Tensor,
                  method: str = 'auto',
                  target_class: Optional[int] = None,
                  **kwargs) -> Explanation
```

**参数说明:**
- `model`: 要解释的模型
- `input_data`: 输入张量
- `method`: 解释方法
- `target_class`: 目标类别
- `**kwargs`: 额外配置参数

**使用示例:**
```python
# 一行代码生成解释
explanation = explain_model(
    model=your_model,
    input_data=signal_tensor,
    method='signal_path',
    target_class=2
)
```

## 最佳实践

### 1. 方法选择指南

- **TSPN模型**: 优先使用 `signal_path` 方法获得最详细的解释
- **深度神经网络**: 使用 `integrated_gradients` 或 `deeplift`
- **实时应用**: 使用 `saliency` 方法获得最快的解释速度
- **不确定时**: 使用 `auto` 让系统自动选择

### 2. 批量处理优化

```python
# 启用批量解释以提高效率
explainer = UnifiedExplainer(model, config={'batch_size': 8})

# 使用批量方法而非循环
explanations = explainer.explain_batch(batch_data)  # 推荐
# 而不是:
# explanations = [explainer.explain(x) for x in batch_data]  # 不推荐
```

### 3. 内存管理

```python
# 大数据集处理时注意内存使用
import torch

# 使用小批量处理
batch_size = 16
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    explanations = explainer.explain_batch(batch)

    # 及时清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 4. 错误恢复

```python
# 实现优雅的错误恢复
def robust_explain(model, data, methods=None):
    if methods is None:
        methods = ['signal_path', 'integrated_gradients', 'saliency']

    for method in methods:
        try:
            return explainer.explain(data, method=method)
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            continue

    raise RuntimeError("All explanation methods failed")
```

---

## 更新日志

### v1.0.0 (2024-11)
- 初始API设计
- 统一解释器接口
- 基础解释方法实现
- LLM增强接口设计

### 贡献指南

如需为API贡献代码或提出改进建议，请遵循以下规范：

1. **代码风格**: 遵循PEP 8规范
2. **文档**: 所有公共接口必须有完整的docstring
3. **测试**: 新功能需要包含相应的单元测试
4. **向后兼容**: API更改需要保持向后兼容性

### 联系方式

- API维护: [开发团队邮箱]
- 问题报告: [GitHub Issues链接]
- 技术讨论: [讨论区链接]