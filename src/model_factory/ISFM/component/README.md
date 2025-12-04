# ISFM工业信号基础模型组件技术文档

## 目录

1. [架构概览](#1-架构概览)
2. [组件技术规格](#2-组件技术规格)
3. [ISFM集成指南](#3-isfm集成指南)
4. [使用示例](#4-使用示例)
5. [高级主题](#5-高级主题)
6. [最佳实践和故障排除](#6-最佳实践和故障排除)

---

## 1. 架构概览

### 1.1 ISFM三层架构体系

ISFM (Industrial Signal Foundation Model) 采用标准化的三层架构模式：

```
输入信号 (B, L, C)
    ↓
嵌入层 (E_XX_HSE/E_XX_Patch)
    ↓ 补丁提取 + 时间编码
主干网络 (B_XX_PatchTST/B_XX_Dlinear)
    ↓ 特征提取/序列建模
任务头 (H_XX_Linear_cla/H_XX_multiple_task)
    ↓ 任务特定处理
输出 (logits/features/representations)
```

#### 架构层次说明

| 层次 | 功能 | 组件类型 | 示例 |
|------|------|----------|------|
| **嵌入层** | 信号预处理和特征嵌入 | Embedding | E_01_HSE, E_02_HSE_v2, E_03_Patch |
| **主干网络** | 特征提取和序列建模 | Backbone | B_01_basic_transformer, B_04_Dlinear, B_08_PatchTST |
| **任务头** | 任务特定的输出处理 | TaskHead | H_01_Linear_cla, H_02_distance_cla, H_09_multiple_task |

### 1.2 组件注册系统

ISFM采用字典注册机制，支持动态组件加载：

```python
# 嵌入层组件字典
Embedding_dict = {
    'E_01_HSE': E_01_HSE,           # HSE信号嵌入
    'E_02_HSE_v2': E_02_HSE_v2,   # HSE增强版嵌入
    'E_03_Patch': E_03_Patch,      # 补丁嵌入
}

# 主干网络组件字典
Backbone_dict = {
    'B_01_basic_transformer': B_01_basic_transformer,
    'B_04_Dlinear': B_04_Dlinear,
    'B_08_PatchTST': B_08_PatchTST,
    'B_09_FNO': B_09_FNO,
}

# 任务头组件字典
TaskHead_dict = {
    'H_01_Linear_cla': H_01_Linear_cla,
    'H_02_distance_cla': H_02_distance_cla,
    'H_09_multiple_task': H_09_multiple_task,
}
```

### 1.3 工厂模式集成

系统采用工厂模式实现基于配置的动态加载：

```python
def resolve_model_module(args_model: Any) -> str:
    """解析模型模块路径"""
    return f"src.model_factory.{args_model.type}.{args_model.name}"

# 使用示例：自动解析为 "src.model_factory.ISFM.M_01_ISFM"
module_path = resolve_model_module(args_model)
```

### 1.4 数据流和处理管道

#### 标准前向传播流程
```python
def forward(self, x, file_id=False, task_id=False, return_feature=False):
    """
    ISFM标准前向传播流程
    """
    # 阶段1：信号嵌入
    x = self._embed(x, file_id)

    # 阶段2：主干网络处理
    x = self._encode(x)

    # 阶段3：任务头处理
    x = self._head(x, file_id, task_id)

    return x
```

#### 元数据驱动处理
```python
# 基于数据集特性的自适应处理
fs = self.metadata[file_id]['Sample_rate']        # 采样率
system_id = str(self.metadata[file_id]['Dataset_id'])  # 系统ID

# 自动适应不同数据集的特征
x = self.adapt_to_dataset(x, fs, system_id)
```

---

## 2. 组件技术规格

### 2.1 ContrastiveSSL - 自监督对比学习

#### 组件概述
ContrastiveSSL实现了工业信号的自监督对比学习，支持多种时间序列增强策略和对比损失函数。

#### 核心功能
- **时间序列增强**：噪声添加、抖动、缩放、掩码等
- **对比学习**：InfoNCE损失实现
- **投影头**：特征投影和表示学习
- **多视图支持**：双视图对比学习架构

#### 技术架构
```python
class ContrastiveSSL(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()
        self.augmentation = TimeSeriesAugmentation()      # 数据增强
        self.encoder = ContrastiveEncoder()               # 特征编码器
        self.projection_head = ProjectionHead()           # 投影头
        self.contrastive_loss = InfoNCELoss()            # 对比损失

    def forward(self, x, return_features=False):
        # 数据增强
        x1, x2 = self.augmentation(x)

        # 特征编码
        z1, z2 = self.encoder(x1), self.encoder(x2)

        # 特征投影
        p1, p2 = self.projection_head(z1), self.projection_head(z2)

        return p1, p2, z1, z2
```

#### 配置参数
```yaml
model:
  type: "ContrastiveSSL"

  # 增强参数
  augmentation_strength: 0.1      # 增强强度
  noise_std: 0.05                 # 噪声标准差
  jitter_ratio: 0.1               # 抖动比例

  # 对比学习参数
  temperature: 0.07               # 温度参数
  projection_dim: 128             # 投影维度

  # 网络参数
  encoder_depth: 6                # 编码器深度
  num_heads: 8                    # 注意力头数
```

#### 工业应用场景
- **故障诊断**：无标签故障特征学习
- **异常检测**：正常模式学习
- **预训练**：为下游任务提供初始化

---

### 2.2 MaskedAutoencoder - 掩码自编码器

#### 组件概述
MaskedAutoencoder实现工业信号的掩码重建，通过补丁级别的掩码和重建任务学习信号表示。

#### 核心功能
- **补丁嵌入**：将信号分割为补丁并嵌入
- **随机掩码**：75%高比例掩码策略
- **编码器-解码器架构**：深度编码器 + 轻量级解码器
- **位置编码**：时间和位置信息编码

#### 技术架构
```python
class MaskedAutoencoder(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()
        self.patch_embedding = PatchEmbedding()           # 补丁嵌入
        self.position_encoding = PositionalEncoding()    # 位置编码
        self.encoder = MaskedAutoencoderEncoder()        # 掩码编码器
        self.decoder = MaskedAutoencoderDecoder()        # 解码器
        self.mask_ratio = args_model.mask_ratio          # 掩码比例

    def forward(self, x):
        # 补丁嵌入和位置编码
        patches = self.patch_embedding(x)
        patches = self.position_encoding(patches)

        # 随机掩码
        visible_patches, mask = self.random_mask(patches)

        # 编码器处理
        encoded = self.encoder(visible_patches)

        # 解码器重建
        reconstructed = self.decoder(encoded, mask)

        return reconstructed, mask
```

#### 配置参数
```yaml
model:
  type: "MaskedAutoencoder"

  # 掩码参数
  mask_ratio: 0.75                 # 掩码比例 (推荐0.75)
  patch_size: 64                   # 补丁大小

  # 网络参数
  encoder_depth: 12                # 编码器深度
  decoder_depth: 4                 # 解码器深度
  embed_dim: 768                   # 嵌入维度

  # 重建参数
  reconstruction_loss: "MSE"       # 重建损失函数
```

#### 工业应用场景
- **信号去噪**：学习清洁信号表示
- **缺失数据补全**：填补传感器数据缺失
- **特征学习**：无监督表示学习

---

### 2.3 MultiModalFM - 多模态基础模型

#### 组件概述
MultiModalFM支持多种传感器数据的融合处理，包括振动、声学、热成像等工业传感器数据。

#### 核心功能
- **多模态编码**：不同模态数据的独立编码
- **跨模态注意力**：模态间的信息交互
- **融合策略**：注意力、拼接、加法等融合方式
- **自适应权重**：学习模态重要性权重

#### 技术架构
```python
class MultiModalFM(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()
        self.modality_encoders = nn.ModuleDict({      # 模态编码器
            'vibration': ModalityEncoder('vibration'),
            'acoustic': ModalityEncoder('acoustic'),
            'thermal': ModalityEncoder('thermal'),
        })

        self.cross_modal_attention = CrossModalAttention()  # 跨模态注意力
        self.fusion_strategy = args_model.fusion_strategy  # 融合策略
        self.fusion_layer = FusionLayer()              # 融合层

    def forward(self, modality_inputs):
        # 模态编码
        modality_features = {}
        for modality, data in modality_inputs.items():
            if data is not None:
                modality_features[modality] = self.modality_encoders[modality](data)

        # 跨模态交互
        enhanced_features = self.cross_modal_attention(modality_features)

        # 多模态融合
        fused_features = self.fusion_layer(enhanced_features, self.fusion_strategy)

        return fused_features, modality_features
```

#### 配置参数
```yaml
model:
  type: "MultiModalFM"

  # 模态配置
  modalities: ["vibration", "acoustic", "thermal"]  # 支持的模态
  fusion_strategy: "attention"                      # 融合策略: attention/concat/add

  # 网络参数
  modality_dim: 512                 # 单模态特征维度
  fusion_dim: 1024                  # 融合后特征维度
  num_heads: 8                      # 注意力头数

  # 注意力参数
  attention_dropout: 0.1            # 注意力dropout
```

#### 工业应用场景
- **设备健康监测**：多传感器数据融合分析
- **故障诊断**：综合多种传感器的故障特征
- **预测性维护**：基于多模态数据的剩余寿命预测

---

### 2.4 SignalLanguageFM - 信号文本对齐

#### 组件概述
SignalLanguageFM实现工业信号与自然语言描述的对齐，支持零样本学习和信号语义理解。

#### 核心功能
- **信号编码器**：工业信号的语义编码
- **文本编码器**：自然语言描述的编码
- **对比对齐**：信号-文本对比学习
- **零样本推理**：基于文本描述的信号分类

#### 技术架构
```python
class SignalLanguageFM(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()
        self.signal_encoder = SignalEncoder()          # 信号编码器
        self.text_encoder = TextEncoder()              # 文本编码器
        self.contrastive_loss = SignalTextContrastiveLoss()  # 对比损失
        self.temperature = args_model.temperature        # 温度参数

    def forward(self, signals, texts=None):
        # 信号编码
        signal_features = self.signal_encoder(signals)

        if texts is not None:
            # 文本编码
            text_features = self.text_encoder(texts)

            # 对比学习
            loss = self.contrastive_loss(signal_features, text_features, self.temperature)

            return signal_features, text_features, loss

        return signal_features

    def zero_shot_classify(self, signals, class_descriptions):
        """零样本分类"""
        signal_features = self.forward(signals)
        text_features = self.text_encoder(class_descriptions)

        # 计算相似度
        similarities = F.cosine_similarity(signal_features, text_features)
        return similarities
```

#### 配置参数
```yaml
model:
  type: "SignalLanguageFM"

  # 编码器参数
  signal_encoder_depth: 6          # 信号编码器深度
  text_encoder_depth: 6            # 文本编码器深度
  feature_dim: 512                 # 特征维度

  # 对比学习参数
  temperature: 0.07                # 温度参数
  max_text_length: 77              # 最大文本长度

  # 预训练参数
  pretraining_corpus: "industrial_descriptions"  # 预训练语料
```

#### 工业应用场景
- **智能故障诊断**：基于故障描述的零样本诊断
- **语义检索**：基于文本描述检索相关信号
- **人机交互**：自然语言描述的信号分析

---

### 2.5 TemporalDynamicsSSL - 时间动态自监督学习

#### 组件概述
TemporalDynamicsSSL专注于工业信号的时间动态建模，通过多种时间预测任务学习信号的时序特征。

#### 核心功能
- **时间增强**：时序特定的数据增强
- **预测任务**：下一步预测、排列检测、掩码重建
- **动态建模**：时间依赖关系建模
- **多任务学习**：多种时间任务的联合学习

#### 技术架构
```python
class TemporalDynamicsSSL(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()
        self.temporal_encoder = TemporalEncoder()      # 时间编码器
        self.temporal_augmentation = TemporalAugmentation()  # 时间增强
        self.prediction_heads = nn.ModuleDict({         # 预测头
            'next_step': NextStepPredictionHead(),
            'permutation': PermutationDetectionHead(),
            'masked': MaskedReconstructionHead(),
        })

    def forward(self, x, tasks=['next_step']):
        # 时间增强
        augmented_x = self.temporal_augmentation(x)

        # 时间编码
        temporal_features = self.temporal_encoder(augmented_x)

        outputs = {}
        for task in tasks:
            if task in self.prediction_heads:
                outputs[task] = self.prediction_heads[task](temporal_features, x)

        return outputs
```

#### 配置参数
```yaml
model:
  type: "TemporalDynamicsSSL"

  # 时间任务
  temporal_tasks: ["next_step", "permutation", "masked"]  # 时间任务列表
  prediction_horizon: 1           # 预测视界

  # 增强参数
  time_warping: 0.1               # 时间扭曲强度
  magnitude_warping: 0.1          # 幅度扭曲强度

  # 网络参数
  encoder_depth: 8                # 编码器深度
  hidden_dim: 512                 # 隐藏层维度

  # 多任务参数
  task_weights:                   # 任务权重
    next_step: 1.0
    permutation: 0.5
    masked: 0.5
```

#### 工业应用场景
- **时序预测**：设备状态的时间演化预测
- **异常检测**：基于时间动态的异常识别
- **趋势分析**：设备性能趋势分析

---

## 3. ISFM集成指南

### 3.1 组件注册流程

#### 步骤1：创建组件类
```python
# 在 src/model_factory/ISFM/component/ 目录下创建组件
class CustomSSLComponent(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()
        # 组件初始化
        self.args_model = args_model
        self.metadata = metadata

    def forward(self, x):
        # 前向传播逻辑
        return processed_x
```

#### 步骤2：注册到组件字典
```python
# 在相应的组件模块中注册
# src/model_factory/ISFM/embedding/__init__.py
from .E_04_CustomSSL import E_04_CustomSSL

__all__ = ["E_01_HSE", "E_02_HSE_v2", "E_03_Patch", "E_04_CustomSSL"]
```

#### 步骤3：更新主模型注册
```python
# 在 src/model_factory/ISFM/M_01_ISFM.py 中注册
Embedding_dict = {
    'E_01_HSE': E_01_HSE,
    'E_02_HSE_v2': E_02_HSE_v2,
    'E_03_Patch': E_03_Patch,
    'E_04_CustomSSL': E_04_CustomSSL,  # 新注册的组件
}
```

### 3.2 配置驱动集成

#### 基础配置模板
```yaml
# configs/ISFM/custom_config.yaml
model:
  name: "M_01_ISFM"              # 主模型名称
  type: "ISFM"                   # 模型工厂类型

  # 组件选择
  embedding: "E_04_CustomSSL"    # 自定义嵌入组件
  backbone: "B_08_PatchTST"      # 主干网络
  task_head: "H_01_Linear_cla"   # 任务头

  # 网络参数
  d_model: 512                   # 模型维度
  num_heads: 8                   # 注意力头数
  num_layers: 6                  # 层数
  dropout: 0.1                   # Dropout率

  # 组件特定参数
  custom_param1: 0.1
  custom_param2: "enabled"
```

#### 数据集配置
```yaml
data:
  train_datasets: ["CWRU", "XJTU", "THU"]
  test_datasets: ["Ottawa", "JNU"]

  # 数据预处理
  sample_rate: 12000             # 采样率
  window_length: 4096            # 窗口长度
  normalize: true                # 标准化

  # 元数据
  metadata_file: "metadata.xlsx"
```

### 3.3 工厂模式使用

#### 动态模型加载
```python
from src.model_factory import model_factory
from src.configs.utils import create_namespace

# 加载配置
args_model = create_namespace(config.model)
metadata = load_metadata(config.data.metadata_file)

# 工厂自动加载和实例化
model = model_factory(args_model, metadata)

# 自动解析为: src.model_factory.ISFM.M_01_ISFM
print(f"Loaded model: {type(model).__name__}")
```

#### 组件参数验证
```python
def validate_component_config(config):
    """验证组件配置的有效性"""
    valid_embeddings = list(Embedding_dict.keys())
    valid_backbones = list(Backbone_dict.keys())
    valid_task_heads = list(TaskHead_dict.keys())

    if config.embedding not in valid_embeddings:
        raise ValueError(f"Invalid embedding: {config.embedding}")
    if config.backbone not in valid_backbones:
        raise ValueError(f"Invalid backbone: {config.backbone}")
    if config.task_head not in valid_task_heads:
        raise ValueError(f"Invalid task_head: {config.task_head}")

    print("✅ Component configuration validated")
```

### 3.4 参数传递和继承

#### 参数层次结构
```python
class ComponentBase(nn.Module):
    def __init__(self, args_model, metadata):
        super().__init__()

        # 模型级参数（所有组件共享）
        self.d_model = getattr(args_model, 'd_model', 512)
        self.num_heads = getattr(args_model, 'num_heads', 8)
        self.dropout = getattr(args_model, 'dropout', 0.1)

        # 组件级参数（特定组件）
        self.component_specific_param = getattr(args_model, 'component_specific_param', 'default')

        # 元数据驱动参数
        self.metadata = metadata
        self.num_classes = len(metadata['class_mapping'])
```

#### 元数据集成
```python
def get_dataset_metadata(file_id, metadata):
    """获取数据集特定元数据"""
    try:
        dataset_info = metadata[file_id]
        return {
            'sample_rate': dataset_info['Sample_rate'],
            'system_id': dataset_info['Dataset_id'],
            'fault_type': dataset_info['Fault_type'],
            'load_condition': dataset_info['Load_condition'],
        }
    except KeyError:
        return {
            'sample_rate': 12000,
            'system_id': 'unknown',
            'fault_type': 'unknown',
            'load_condition': 'unknown',
        }
```

### 3.5 元数据驱动适配

#### 自适应采样率处理
```python
def adapt_to_sampling_rate(self, x, sample_rate):
    """根据采样率自适应处理"""
    if sample_rate == 12000:
        return self.process_12khz(x)
    elif sample_rate == 48000:
        return self.process_48khz(x)
    else:
        # 自适应重采样
        return self.resample_and_process(x, sample_rate)
```

#### 系统ID感知处理
```python
def system_aware_processing(self, x, system_id):
    """系统感知的处理策略"""
    if system_id == '1':  # CWRU
        return self.process_cwru_style(x)
    elif system_id == '2':  # XJTU
        return self.process_xjtu_style(x)
    else:
        return self.process_generic(x)
```

---

## 4. 使用示例

### 4.1 快速开始

#### 基础使用示例
```python
from src.model_factory import model_factory
from src.configs.utils import create_namespace
from src.data_factory import load_metadata

# 1. 准备配置
config = {
    'model': {
        'name': 'M_01_ISFM',
        'type': 'ISFM',
        'embedding': 'E_01_HSE',
        'backbone': 'B_08_PatchTST',
        'task_head': 'H_01_Linear_cla',
        'd_model': 512,
        'num_heads': 8,
    }
}

# 2. 创建模型
args_model = create_namespace(config['model'])
metadata = load_metadata('metadata.xlsx')
model = model_factory(args_model, metadata)

# 3. 前向传播
import torch
x = torch.randn(32, 2, 1024)  # (batch_size, channels, length)
file_id = 'CWRU_001'
output = model(x, file_id=file_id)

print(f"Output shape: {output.shape}")
print(f"Model loaded successfully: {type(model).__name__}")
```

#### 组件组合示例
```python
# 尝试不同的组件组合
configurations = [
    {
        'embedding': 'E_01_HSE',
        'backbone': 'B_04_Dlinear',
        'task_head': 'H_01_Linear_cla',
        'description': 'HSE + Dlinear + Linear'
    },
    {
        'embedding': 'E_03_Patch',
        'backbone': 'B_08_PatchTST',
        'task_head': 'H_09_multiple_task',
        'description': 'Patch + PatchTST + Multi-task'
    },
    {
        'embedding': 'E_02_HSE_v2',
        'backbone': 'B_09_FNO',
        'task_head': 'H_02_distance_cla',
        'description': 'HSEv2 + FNO + Metric Learning'
    }
]

for config in configurations:
    args_model = create_namespace(config)
    model = model_factory(args_model, metadata)
    print(f"✅ {config['description']}: {model.__class__.__name__}")
```

### 4.2 ISFM模型集成

#### 主模型定制
```python
class CustomISFM(nn.Module):
    """自定义ISFM模型"""

    def __init__(self, args_model, metadata):
        super().__init__()

        # 组件字典
        Embedding_dict = {...}
        Backbone_dict = {...}
        TaskHead_dict = {...}

        # 动态加载组件
        self.embedding = Embedding_dict[args_model.embedding](args_model, metadata)
        self.backbone = Backbone_dict[args_model.backbone](args_model, metadata)
        self.task_head = TaskHead_dict[args_model.task_head](args_model, metadata)

    def forward(self, x, file_id=False, task_id=False, return_feature=False):
        # 三阶段处理流程
        x = self._embed(x, file_id)      # 嵌入
        x = self._encode(x)              # 编码
        x = self._head(x, file_id, task_id)  # 任务头

        return x if not return_feature else (x, x)
```

#### 组件接口实现
```python
class CustomEmbedding(nn.Module):
    """自定义嵌入组件示例"""

    def __init__(self, args_model, metadata):
        super().__init__()
        self.args_model = args_model
        self.metadata = metadata

        # 核心组件
        self.hse_embedding = HSEEmbedding(args_model)
        self.patch_embedding = PatchEmbedding(args_model)

    def forward(self, x, file_id=None):
        # 获取数据集信息
        if file_id:
            dataset_info = self.metadata[file_id]
            sample_rate = dataset_info['Sample_rate']
        else:
            sample_rate = 12000

        # 自适应处理
        x = self.adapt_to_sample_rate(x, sample_rate)

        # HSE特征提取
        hse_features = self.hse_embedding(x)

        return hse_features
```

### 4.3 配置模板

#### 完整配置示例
```yaml
# configs/ISFM/experiment_template.yaml
experiment:
  name: "ISFM故障诊断实验"
  description: "使用ISFM进行轴承故障诊断"

# 模型配置
model:
  name: "M_01_ISFM"
  type: "ISFM"

  # 组件选择
  embedding: "E_01_HSE"
  backbone: "B_08_PatchTST"
  task_head: "H_01_Linear_cla"

  # 网络架构
  d_model: 512
  num_heads: 8
  num_layers: 6
  dropout: 0.1

  # HSE特定参数
  patch_size: 64
  overlap_ratio: 0.5

# 数据配置
data:
  train_datasets: ["CWRU", "XJTU", "THU"]
  test_datasets: ["Ottawa", "JNU"]

  # 信号处理
  sample_rate: 12000
  window_length: 4096
  overlap: 0.5
  normalize: true

  # 元数据
  metadata_file: "data/metadata/combined_metadata.xlsx"

# 训练配置
training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-4
  weight_decay: 1e-4

  # 优化器
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"

  # 损失函数
  loss: "CrossEntropyLoss"
  label_smoothing: 0.1
```

#### 多任务配置
```yaml
# 多任务学习配置
model:
  name: "M_03_ISFM"
  type: "ISFM"
  embedding: "E_02_HSE_v2"
  backbone: "B_08_PatchTST"
  task_head: "H_09_multiple_task"

  # 多任务权重
  task_weights:
    classification: 1.0
    reconstruction: 0.5
    contrastive: 0.1

# 任务配置
tasks:
  classification:
    num_classes: 10
    loss: "CrossEntropyLoss"

  reconstruction:
    loss: "MSELoss"
    weight: 0.5

  contrastive:
    loss: "InfoNCE"
    temperature: 0.07
    weight: 0.1
```

### 4.4 自定义组件开发

#### 完整开发流程
```python
# 步骤1: 创建组件文件
# src/model_factory/ISFM/component/CustomTimeFreq.py

import torch
import torch.nn as nn
from typing import Dict, Any

class CustomTimeFreqEmbedding(nn.Module):
    """自定义时频嵌入组件"""

    def __init__(self, args_model, metadata):
        super().__init__()
        self.args_model = args_model
        self.metadata = metadata

        # 可配置参数
        self.n_fft = getattr(args_model, 'n_fft', 512)
        self.hop_length = getattr(args_model, 'hop_length', 256)
        self.n_mels = getattr(args_model, 'n_mels', 128)

        # 时频变换
        self.stft = nn.Sequential(
            STFTLayer(self.n_fft, self.hop_length),
            MelScale(self.n_mels),
            nn.BatchNorm2d(1),
        )

        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.n_mels * 64 + 1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, args_model.d_model),
        )

    def forward(self, x, file_id=None):
        # 时域特征
        time_features = self.extract_time_features(x)

        # 频域特征
        freq_features = self.extract_freq_features(x)

        # 特征融合
        combined_features = torch.cat([time_features, freq_features], dim=-1)
        embedded = self.fusion_layer(combined_features)

        return embedded

    def extract_time_features(self, x):
        """提取时域特征"""
        # 实现时域特征提取逻辑
        return x

    def extract_freq_features(self, x):
        """提取频域特征"""
        # 实现频域特征提取逻辑
        return self.stft(x)
```

#### 组件注册
```python
# 步骤2: 注册到组件模块
# src/model_factory/ISFM/embedding/__init__.py

from .E_05_CustomTimeFreq import CustomTimeFreqEmbedding

__all__ = ["E_01_HSE", "E_02_HSE_v2", "E_03_Patch", "E_05_CustomTimeFreq"]

# 步骤3: 添加到主模型字典
# src/model_factory/ISFM/M_01_ISFM.py

Embedding_dict = {
    'E_01_HSE': E_01_HSE,
    'E_02_HSE_v2': E_02_HSE_v2,
    'E_03_Patch': E_03_Patch,
    'E_05_CustomTimeFreq': CustomTimeFreqEmbedding,  # 新组件
}
```

#### 组件测试
```python
# 组件单元测试
def test_custom_component():
    """测试自定义组件"""
    import torch

    # 创建测试配置
    class MockArgs:
        d_model = 512
        n_fft = 512
        hop_length = 256
        n_mels = 128

    # 创建模拟元数据
    metadata = {'test': {'Sample_rate': 12000}}

    # 初始化组件
    args_model = MockArgs()
    component = CustomTimeFreqEmbedding(args_model, metadata)

    # 测试前向传播
    x = torch.randn(8, 2, 1024)  # (batch, channels, length)
    output = component(x, file_id='test')

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Custom component test passed!")

if __name__ == '__main__':
    test_custom_component()
```

---

## 5. 高级主题

### 5.1 多模态集成

#### 模态数据对齐
```python
class MultiModalAligner(nn.Module):
    """多模态数据对齐器"""

    def __init__(self, modality_dims, align_dim=512):
        super().__init__()
        self.aligners = nn.ModuleDict({
            modality: nn.Linear(dim, align_dim)
            for modality, dim in modality_dims.items()
        })

    def forward(self, modality_inputs):
        aligned_features = {}
        for modality, data in modality_inputs.items():
            if data is not None:
                aligned_features[modality] = self.aligners[modality](data)
        return aligned_features
```

#### 跨模态注意力
```python
class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, modality_features):
        # 将所有模态特征堆叠
        features = []
        modality_names = []

        for modality, feature in modality_features.items():
            features.append(feature)
            modality_names.append(modality)

        stacked_features = torch.stack(features, dim=1)

        # 计算跨模态注意力
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )

        return attended_features, attention_weights, modality_names
```

### 5.2 自监督学习管道

#### 预训练任务设计
```python
class SSLTaskScheduler:
    """自监督学习任务调度器"""

    def __init__(self, task_configs):
        self.task_configs = task_configs
        self.current_epoch = 0

    def get_active_tasks(self, epoch):
        """根据epoch返回活跃任务"""
        active_tasks = []

        for task, config in self.task_configs.items():
            start_epoch = config.get('start_epoch', 0)
            end_epoch = config.get('end_epoch', float('inf'))

            if start_epoch <= epoch <= end_epoch:
                active_tasks.append(task)

        return active_tasks

    def get_task_weights(self, epoch):
        """获取任务权重"""
        weights = {}
        active_tasks = self.get_active_tasks(epoch)

        for task in active_tasks:
            config = self.task_configs[task]
            weight = config.get('weight', 1.0)

            # 权重调度
            if 'schedule' in config:
                weight = self.apply_schedule(weight, config['schedule'], epoch)

            weights[task] = weight

        return weights
```

#### 代理任务组合
```python
class MultiTaskSSL(nn.Module):
    """多任务自监督学习"""

    def __init__(self, ssl_tasks, task_weights):
        super().__init__()
        self.ssl_tasks = nn.ModuleDict(ssl_tasks)
        self.task_weights = task_weights

    def forward(self, x, active_tasks=None):
        if active_tasks is None:
            active_tasks = list(self.ssl_tasks.keys())

        losses = {}
        features = {}

        for task_name in active_tasks:
            if task_name in self.ssl_tasks:
                task = self.ssl_tasks[task_name]
                task_output = task(x)

                if 'loss' in task_output:
                    losses[task_name] = task_output['loss']
                if 'features' in task_output:
                    features[task_name] = task_output['features']

        # 计算加权总损失
        total_loss = 0
        for task_name, loss in losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss

        return {
            'total_loss': total_loss,
            'task_losses': losses,
            'features': features
        }
```

### 5.3 Prompt系统集成

#### Prompt引导的嵌入
```python
class PromptGuidedEmbedding(nn.Module):
    """Prompt引导的信号嵌入"""

    def __init__(self, base_embedding, prompt_dim=128):
        super().__init__()
        self.base_embedding = base_embedding
        self.prompt_dim = prompt_dim

        # Prompt embedding
        self.prompt_embedding = nn.Parameter(
            torch.randn(prompt_dim) * 0.02
        )

        # Prompt融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(base_embedding.output_dim + prompt_dim,
                     base_embedding.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x, file_id=None):
        # 基础嵌入
        base_features = self.base_embedding(x, file_id)

        # Prompt扩展
        batch_size = base_features.size(0)
        prompt_features = self.prompt_embedding.unsqueeze(0).expand(batch_size, -1)

        # 特征融合
        fused_features = torch.cat([base_features, prompt_features], dim=-1)
        enhanced_features = self.fusion_layer(fused_features)

        return enhanced_features
```

#### 任务特定Prompt
```python
class TaskSpecificPrompts(nn.Module):
    """任务特定Prompt"""

    def __init__(self, num_tasks, prompt_dim=128):
        super().__init__()
        self.task_prompts = nn.Parameter(
            torch.randn(num_tasks, prompt_dim) * 0.02
        )

    def get_prompt(self, task_id):
        """获取任务特定Prompt"""
        if isinstance(task_id, str):
            # 将任务名称映射到索引
            task_map = {'classification': 0, 'regression': 1, 'detection': 2}
            task_idx = task_map.get(task_id, 0)
        else:
            task_idx = task_id

        return self.task_prompts[task_idx]

    def forward(self, features, task_id):
        prompt = self.get_prompt(task_id)
        prompt_features = prompt.unsqueeze(0).expand(features.size(0), -1)

        # Prompt融合
        enhanced_features = features + prompt_features
        return enhanced_features
```

### 5.4 跨域泛化

#### 域自适应层
```python
class DomainAdaptiveLayer(nn.Module):
    """域自适应层"""

    def __init__(self, input_dim, num_domains):
        super().__init__()
        self.domain_specific_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim)
            for _ in range(num_domains)
        ])
        self.shared_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x, domain_id=None):
        if domain_id is not None:
            # 域特定处理
            domain_output = self.domain_specific_layers[domain_id](x)
            shared_output = self.shared_layer(x)
            return domain_output + shared_output
        else:
            # 共享处理
            return self.shared_layer(x)
```

#### 跨域对比学习
```python
class CrossDomainContrastiveLoss(nn.Module):
    """跨域对比损失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, domain_ids):
        """
        Args:
            features: [N, D] 特征向量
            labels: [N] 类别标签
            domain_ids: [N] 域ID
        """
        # 特征归一化
        features = F.normalize(features, dim=-1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix / self.temperature

        # 构建正负样本掩码
        class_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        domain_mask = domain_ids.unsqueeze(0) != domain_ids.unsqueeze(1)

        # 正样本：同类别，不同域
        positive_mask = class_mask & domain_mask

        # 负样本：不同类别或同域
        negative_mask = ~positive_mask

        # 计算对比损失
        losses = []
        for i in range(len(features)):
            pos_sims = similarity_matrix[i][positive_mask[i]]
            neg_sims = similarity_matrix[i][negative_mask[i]]

            if len(pos_sims) > 0 and len(neg_sims) > 0:
                numerator = torch.logsumexp(pos_sims, dim=0)
                denominator = torch.logsumexp(torch.cat([pos_sims, neg_sims]), dim=0)
                loss = -(numerator - denominator)
                losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0)
```

---

## 6. 最佳实践和故障排除

### 6.1 性能优化建议

#### 内存优化
```python
class MemoryEfficientProcessing:
    """内存优化处理策略"""

    @staticmethod
    def gradient_checkpointing(model, x):
        """梯度检查点"""
        from torch.utils.checkpoint import checkpoint

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # 对大模块使用梯度检查点
        x = checkpoint(create_custom_forward(model.embedding), x)
        x = checkpoint(create_custom_forward(model.backbone), x)
        x = checkpoint(create_custom_forward(model.task_head), x)

        return x

    @staticmethod
    def mixed_precision_training(model, optimizer):
        """混合精度训练"""
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()

        def train_step(batch):
            optimizer.zero_grad()

            with autocast():
                output = model(batch)
                loss = compute_loss(output, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            return loss.item()

        return train_step
```

#### 计算优化
```python
class ComputationalOptimization:
    """计算优化策略"""

    @staticmethod
    def tensor_parallel_processing(model, x):
        """张量并行处理"""
        device_count = torch.cuda.device_count()

        if device_count > 1 and x.size(0) >= device_count:
            # 分割批次到多个GPU
            batch_splits = torch.chunk(x, device_count)

            outputs = []
            for i, batch_chunk in enumerate(batch_splits):
                device = torch.device(f'cuda:{i}')
                chunk_output = model(batch_chunk.to(device))
                outputs.append(chunk_output.cpu())

            return torch.cat(outputs, dim=0)
        else:
            return model(x)
```

### 6.2 常见配置问题

#### 组件兼容性检查
```python
def check_component_compatibility(config):
    """检查组件兼容性"""

    # 嵌入层与主干网络的兼容性
    embedding_backbone_compatibility = {
        'E_01_HSE': ['B_01_basic_transformer', 'B_08_PatchTST', 'B_09_FNO'],
        'E_02_HSE_v2': ['B_01_basic_transformer', 'B_08_PatchTST'],
        'E_03_Patch': ['B_04_Dlinear', 'B_08_PatchTST'],
    }

    embedding = config.get('embedding', 'E_01_HSE')
    backbone = config.get('backbone', 'B_08_PatchTST')

    compatible_backbones = embedding_backbone_compatibility.get(embedding, [])

    if backbone not in compatible_backbones:
        print(f"⚠️  Warning: {embedding} may not be compatible with {backbone}")
        print(f"   Compatible backbones for {embedding}: {compatible_backbones}")

    # 任务头兼容性
    task_head_compatibility = {
        'H_01_Linear_cla': ['classification'],
        'H_02_distance_cla': ['metric_learning', 'classification'],
        'H_09_multiple_task': ['multi_task', 'classification'],
    }

    task_head = config.get('task_head', 'H_01_Linear_cla')
    expected_tasks = task_head_compatibility.get(task_head, ['general'])

    print(f"✅ Component compatibility checked")
    print(f"   Expected tasks for {task_head}: {expected_tasks}")
```

#### 参数范围验证
```python
def validate_parameter_ranges(config):
    """验证参数范围"""

    # 模型维度检查
    d_model = config.get('d_model', 512)
    if d_model % 8 != 0:
        raise ValueError(f"d_model must be divisible by 8, got {d_model}")

    if d_model < 128 or d_model > 2048:
        print(f"⚠️  Warning: d_model={d_model} is outside recommended range [128, 2048]")

    # 注意力头数检查
    num_heads = config.get('num_heads', 8)
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

    # Dropout检查
    dropout = config.get('dropout', 0.1)
    if not 0 <= dropout <= 1:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")

    if dropout > 0.5:
        print(f"⚠️  Warning: dropout={dropout} is high, may affect training stability")

    print("✅ Parameter validation passed")
```

### 6.3 调试和验证工具

#### 模型检查工具
```python
class ModelChecker:
    """模型检查工具"""

    @staticmethod
    def check_model_architecture(model, input_shape):
        """检查模型架构"""
        model.eval()

        try:
            with torch.no_grad():
                x = torch.randn(input_shape)
                output = model(x)

            print(f"✅ Forward pass successful")
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {output.shape}")

            # 检查模型参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")

            return True

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False

    @staticmethod
    def check_gradient_flow(model, input_shape):
        """检查梯度流"""
        model.train()

        x = torch.randn(input_shape, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        print("Gradient flow check:")
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"   {name}: {grad_norm:.6f}")

                if grad_norm < 1e-8:
                    print(f"     ⚠️  Very small gradient")
                elif grad_norm > 10:
                    print(f"     ⚠️  Large gradient - possible instability")

    @staticmethod
    def memory_usage_check(model, input_shape, batch_sizes=[1, 4, 16, 32]):
        """检查内存使用"""
        print("Memory usage check:")

        for batch_size in batch_sizes:
            try:
                x = torch.randn(batch_size, *input_shape[1:])

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    model = model.cuda()
                    x = x.cuda()

                    output = model(x)

                    memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    print(f"   Batch size {batch_size}: {memory_used:.1f} MB")

                    model = model.cpu()
                    torch.cuda.empty_cache()
                else:
                    output = model(x)
                    print(f"   Batch size {batch_size}: OK")

            except RuntimeError as e:
                print(f"   Batch size {batch_size}: OOM - {e}")
                break
```

#### 数据流可视化
```python
class DataFlowVisualizer:
    """数据流可视化工具"""

    @staticmethod
    def visualize_feature_shapes(model, input_shape):
        """可视化特征形状变化"""
        model.eval()

        def hook_fn(module, input, output):
            layer_name = module.__class__.__name__
            if isinstance(output, torch.Tensor):
                shape_str = f"{layer_name}: {input[0].shape} -> {output.shape}"
            else:
                shape_str = f"{layer_name}: {input[0].shape} -> {type(output)}"
            print(shape_str)

        # 注册hook
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        # 前向传播
        x = torch.randn(input_shape)
        with torch.no_grad():
            model(x)

        # 移除hook
        for hook in hooks:
            hook.remove()
```

### 6.4 故障排除指南

#### 常见错误及解决方案

| 错误类型 | 常见原因 | 解决方案 |
|----------|----------|----------|
| `RuntimeError: CUDA out of memory` | 批次太大或模型太大 | 减小批次大小、使用梯度检查点、混合精度训练 |
| `ValueError: Expected input to have 3 dimensions` | 输入维度错误 | 检查输入格式，调整为 (batch, channels, length) |
| `ImportError: No module named 'xxx'` | 组件未正确注册 | 检查 `__init__.py` 文件和导入路径 |
| `KeyError: 'Sample_rate'` | 元数据缺失 | 检查元数据文件格式和内容 |
| `TypeError: forward() got an unexpected keyword argument` | 方法签名不匹配 | 检查模型 `forward` 方法的参数定义 |

#### 性能基准测试
```python
def benchmark_model(model, input_shape, num_runs=100):
    """模型性能基准测试"""
    model.eval()

    # 预热
    for _ in range(10):
        with torch.no_grad():
            x = torch.randn(input_shape)
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 计时
    import time
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            x = torch.randn(input_shape)
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    throughput = input_shape[0] / avg_time

    print(f"Performance benchmark:")
    print(f"   Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"   Throughput: {throughput:.1f} samples/second")

    return avg_time, throughput
```

---

## 总结

本技术文档详细介绍了ISFM工业信号基础模型的五大核心组件及其与整体架构的集成方法。通过标准化的三层架构设计、灵活的组件注册系统和工厂模式，ISFM为工业信号处理提供了强大而可扩展的基础。

### 关键优势
- **模块化设计**：组件可独立开发和替换
- **配置驱动**：通过YAML配置灵活组合组件
- **元数据集成**：自动适应不同数据集特性
- **多模态支持**：支持多种工业传感器数据
- **自监督学习**：丰富的预训练策略

### 最佳实践
1. **组件选择**：根据具体任务选择合适的组件组合
2. **参数调优**：注意模型维度的兼容性和参数范围
3. **性能优化**：使用混合精度训练和梯度检查点
4. **调试验证**：利用内置工具检查模型架构和数据流

本文档为ISFM组件的开发、使用和维护提供了全面的技术指导，助力工业智能应用的研究和实践。