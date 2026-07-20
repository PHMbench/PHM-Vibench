# Explainable FD Toolkit 标准接口说明

> **阶段2完成报告**：本文档详细说明了Explainable FD Toolkit的标准接口规范，支持多模型多方法接入。

## 一、核心接口概览

Explainable FD Toolkit提供了一套标准化的接口，确保不同解释方法和模型之间的互操作性。核心接口包括：

1. **SignalData** - 统一信号数据容器
2. **ExplainabilityMethod** - 可解释性方法接口协议
3. **ModelPlugin** - 模型插件接口协议
4. **Explanation** - 统一解释结果数据结构

## 二、接口详细说明

### 2.1 SignalData - 统一信号容器

**用途**：标准化信号数据的存储和访问，支持原始信号、处理特征和元数据。

**核心属性**：
```python
class SignalData:
    def __init__(self,
                 raw_signal: Union[np.ndarray, torch.Tensor],  # 原始信号 [T] 或 [C, T]
                 sampling_rate: int,                           # 采样率
                 metadata: Optional[Dict[str, Any]] = None,    # 元数据
                 processed_features: Optional[...],            # 处理特征
                 time_stamps: Optional[...],                   # 时间戳
                 channel_names: Optional[List[str]] = None,    # 通道名称
                 label: Optional[Union[int, str]] = None):     # 故障标签
```

**关键方法**：
- `get_shape()` - 获取信号形状
- `get_num_channels()` - 获取通道数
- `get_duration()` - 获取信号时长
- `get_channel_data(channel_idx)` - 获取指定通道数据
- `get_time_window(start, end)` - 提取时间窗口
- `save(filepath)` / `load(filepath)` - 序列化操作

**使用示例**：
```python
from toolkit_integration.explainability.core import SignalData

# 创建信号数据
signal_data = SignalData(
    raw_signal=raw_signal_array,
    sampling_rate=1024,
    channel_names=['acc_x', 'acc_y', 'acc_z'],
    label='bearing_fault',
    metadata={'sensor_type': 'accelerometer', 'location': 'bearing_1'}
)

# 访问信号信息
duration = signal_data.get_duration()  # 获取时长
channel_data = signal_data.get_channel_data(0)  # 获取第一个通道
```

### 2.2 ExplainabilityMethod - 可解释性方法接口

**用途**：定义所有解释方法必须实现的标准接口，确保方法间的可互换性。

**核心接口**：
```python
@runtime_checkable
class ExplainabilityMethod(Protocol):
    def explain(self, signal: SignalData, prediction: Any, **kwargs) -> Explanation:
        """生成解释结果"""
        ...

    def visualize(self, explanation: Explanation, mode: str = 'auto', **kwargs) -> Figure:
        """可视化解释结果"""
        ...

    def evaluate(self, explanations: Sequence[Explanation], **kwargs) -> Dict[str, float]:
        """评估解释质量"""
        ...

    def get_method_name() -> str:
        """获取方法名称"""
        ...

    def get_method_type() -> str:
        """获取方法类型（'intrinsic', 'posthoc', 'hybrid'）"""
        ...
```

**实现要求**：
- 继承自 `BaseExplainerAdapter` 获得基础功能
- 实现所有必需方法
- 提供标准化的配置接口

**示例实现**：
```python
from toolkit_integration.explainability.core import BaseExplainerAdapter, ExplainabilityMethod

class MyExplainer(BaseExplainerAdapter, ExplainabilityMethod):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._method_name = "MyMethod"
        self._method_type = "posthoc"

    def explain(self, signal: SignalData, prediction: Any, **kwargs) -> Explanation:
        # 实现解释逻辑
        return Explanation(data, meta)
```

### 2.3 ModelPlugin - 模型插件接口

**用途**：标准化模型与工具集的交互接口，支持任意模型的接入。

**核心接口**：
```python
@runtime_checkable
class ModelPlugin(Protocol):
    def fit(self, data: Sequence[SignalData], labels: Sequence[Any], **kwargs) -> None:
        """训练模型"""
        ...

    def predict(self, signal: SignalData, **kwargs) -> Any:
        """单样本预测"""
        ...

    def predict_batch(self, signals: Sequence[SignalData], **kwargs) -> List[Any]:
        """批量预测"""
        ...

    def get_explanation(self, signal: SignalData, method: ExplainabilityMethod, **kwargs) -> Explanation:
        """生成解释"""
        ...

    def get_intermediate_features(self, signal: SignalData, **kwargs) -> Dict[str, np.ndarray]:
        """提取中间特征"""
        ...
```

**实现要求**：
- 继承自 `BaseModelAdapter` 获得基础功能
- 实现 `get_intermediate_features` 方法支持本征解释
- 提供模型元数据接口

### 2.4 Explanation - 统一解释结果

**用途**：标准化解释结果的存储、可视化和评估。

**核心结构**：
```python
class Explanation:
    def __init__(self, data: Dict[str, Any], meta: Optional[Dict[str, Any]] = None):
        # data: 解释数据（归因、路径、重要性等）
        # meta: 元数据（方法信息、模型信息等）

    def get_attribution() -> Optional[np.ndarray]:
        """获取主要归因值"""
        ...

    def visualize(mode: str = 'auto') -> Figure:
        """生成可视化"""
        ...

    def get_metrics() -> Dict[str, float]:
        """计算基础指标"""
        ...
```

**标准数据格式**：
- `attributions` - 特征归因值
- `path` - 信号转换路径
- `importance_scores` - 重要性分数
- `original_signal` - 原始信号
- `method_specific` - 方法特定数据

## 三、已实现的方法

### 3.1 本征方法 (Intrinsic Methods)

#### PathAnalysisExplainer - 路径分析解释器
- **功能**：跟踪信号在模型中的转换路径
- **适用模型**：TSPN、NNSPN等透明信号处理网络
- **特点**：支持频率分析、能量分析、统计分析
- **配置选项**：
  ```python
  config = {
      'include_frequency_analysis': True,
      'include_energy_analysis': True,
      'max_path_depth': 10,
      'importance_threshold': 0.1
  }
  ```

#### OperatorWeightExplainer - 算子权重解释器
- **功能**：分析模型算子权重和参数
- **适用模型**：任何权重可见的神经网络
- **特点**：支持权重分析、激活模式、算子重要性排序
- **配置选项**：
  ```python
  config = {
      'weight_analysis_method': 'magnitude',  # 'magnitude', 'variance', 'spectral'
      'include_activation_patterns': True,
      'top_k_operators': 10
  }
  ```

### 3.2 事后方法 (Post-hoc Methods)

#### GradCAMExplainer - 梯度加权类激活映射
- **功能**：生成基于梯度的热力图
- **适用模型**：CNN、RNN等神经网络
- **特点**：支持多目标层、梯度平滑、多种插值方法
- **配置选项**：
  ```python
  config = {
      'target_layers': [],  # 自动检测
      'use_abs_gradients': True,
      'attribution_smoothing': True,
      'interpolation_method': 'linear'
  }
  ```

#### SHAPExplainer - SHAP值解释器
- **功能**：计算基于博弈论的特征归因
- **适用模型**：任何可微模型
- **特点**：支持梯度SHAP、核SHAP、分段计算
- **配置选项**：
  ```python
  config = {
      'explanation_method': 'gradient',  # 'gradient', 'kernel', 'deep'
      'use_segments': True,
      'n_segments': 50,
      'background_samples': 10
  }
  ```

## 四、配置管理系统

### 4.1 统一配置接口

```python
from toolkit_integration.explainability.config import get_method_config, create_method

# 获取方法配置
config = get_method_config('PathAnalysis')
config['max_path_depth'] = 15  # 修改配置

# 创建方法实例
explainer = create_method('PathAnalysis', config)
```

### 4.2 配置文件支持

支持YAML和JSON格式的配置文件：

```python
from toolkit_integration.explainability.config import config_manager

# 加载配置
config_manager.load_config_from_file('PathAnalysis', 'path_analysis_config.yaml')

# 保存配置
config_manager.save_config_to_file('PathAnalysis', 'my_config.yaml')
```

### 4.3 实验配置

支持多方法实验配置：

```python
# 创建实验配置
experiment_config = config_manager.create_experiment_config(
    method_names=['PathAnalysis', 'GradCAM', 'SHAP'],
    experiment_name='comparison_study',
    sampling_rate=1024.0
)

# 保存实验配置
config_manager.save_experiment_config(experiment_config, 'experiment.yaml')
```

## 五、集成规范

### 5.1 新模型接入

1. **创建模型适配器**：
```python
from toolkit_integration.explainability.core import BaseModelAdapter

class MyModelAdapter(BaseModelAdapter, ModelPlugin):
    def __init__(self, model, config=None):
        super().__init__(model, config)

    def get_intermediate_features(self, signal, layer_names=None):
        # 实现特征提取
        return features
```

2. **注册方法**：
```python
adapter = MyModelAdapter(my_model)
explanation = adapter.get_explanation(signal, method)
```

### 5.2 新方法接入

1. **继承基础适配器**：
```python
class MyMethod(BaseExplainerAdapter, ExplainabilityMethod):
    def explain(self, signal, prediction, **kwargs):
        # 实现解释逻辑
        return Explanation(data, meta)
```

2. **注册到配置系统**：
```python
# 在 method_configs.py 中添加
DEFAULT_CONFIGS['MyMethod'] = {...}
METHOD_CLASSES['MyMethod'] = 'MyMethodExplainer'
```

## 六、使用示例

### 6.1 基本使用

```python
from toolkit_integration.explainability.core import SignalData
from toolkit_integration.explainability.methods import PathAnalysisExplainer
from toolkit_integration.explainability.config import create_method

# 准备数据
signal_data = SignalData(raw_signal, sampling_rate=1024, label='fault')

# 创建解释器
explainer = create_method('PathAnalysis')

# 生成解释
explanation = explainer.explain(signal_data, prediction)

# 可视化
fig = explainer.visualize(explanation, mode='auto')
```

### 6.2 批量解释

```python
# 创建统一接口
from toolkit_integration.explainability.core import UnifiedExplainer

unified = UnifiedExplainer()

# 添加多个方法
unified.add_method('PathAnalysis', config1)
unified.add_method('GradCAM', config2)

# 批量解释
explanations = unified.explain_batch(signals, model, methods=['PathAnalysis', 'GradCAM'])

# 评估比较
metrics = unified.evaluate_explanations(explanations)
```

## 七、阶段2完成总结

### 7.1 已完成的任务

✅ **接口规范化**：
- 实现了SignalData统一信号容器
- 定义了ExplainabilityMethod和ModelPlugin协议
- 创建了BaseExplainerAdapter和BaseModelAdapter基类

✅ **方法实现**：
- 本征方法：PathAnalysisExplainer（路径分析）、OperatorWeightExplainer（算子权重）
- 事后方法：GradCAMExplainer、SHAPExplainer

✅ **配置系统**：
- 统一的配置管理接口
- YAML/JSON配置文件支持
- 实验配置模板

✅ **文档更新**：
- 完整的接口说明文档
- 使用示例和集成规范

### 7.2 技术特点

- **标准化**：所有方法遵循统一接口
- **可扩展**：支持新方法和新模型的接入
- **配置化**：灵活的配置管理系统
- **可视化**：多种可视化模式
- **评估**：标准化的解释质量评估

### 7.3 下一阶段计划

阶段3将重点实现：
- 多模型插件实现（TSPN、NNSPN-MoE等）
- 实验结果组织和标准化
- 批量运行脚本

这套标准接口为Explainable FD Toolkit奠定了坚实的基础，使其能够作为"可解释性OS"支持各种故障诊断模型和方法的统一接入和评估。