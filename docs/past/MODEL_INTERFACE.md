# PHM基础模型接口规范

> 📐 **规范指南** - PHM基础模型开发者必读，定义标准接口和最佳实践

## 🎯 文档目的

本文档为PHM基础模型开发者提供：
- ✅ **标准接口规范** - 必须遵循的API定义
- 🏗️ **实现指导** - 具体开发步骤和注意事项  
- 🔧 **调试技巧** - 常见问题和解决方案
- 📊 **最佳实践** - 性能优化和代码规范

## 📋 核心接口规范

### 基础模型接口

**所有模型必须遵循以下接口规范：**

```python
import torch.nn as nn

class Model(nn.Module):
    """
    PHM基础模型标准接口
    
    注意：类名必须是 'Model'，这是框架约定
    """
    
    def __init__(self, args_m, metadata=None):
        """
        初始化模型
        
        Args:
            args_m (Namespace): 模型配置参数
                - 包含所有配置文件中 model 节的参数
                - 可通过 args_m.parameter_name 访问
                
            metadata (MetadataAccessor): 数据集元信息访问器
                - 用于获取数据集相关信息（可选）
                - 访问方式：metadata[file_id]['field_name']
        
        必须设置的属性：
            - 模型的所有网络层和参数
            - 从args_m中提取必要的配置参数
        """
        super(Model, self).__init__()
        
        # ✅ 正确示例：提取配置参数
        self.input_dim = getattr(args_m, 'input_dim', 1)
        self.num_classes = getattr(args_m, 'num_classes', 10)
        self.dropout = getattr(args_m, 'dropout', 0.1)
        
        # ✅ 保存引用用于后续使用
        self.metadata = metadata
        self.args_m = args_m
    
    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        """
        前向传播 - 核心接口
        
        Args:
            x (torch.Tensor): 输入张量
                - 标准形状: (batch_size, sequence_length, channels)
                - 也支持: (batch_size, channels, sequence_length)
                
            file_id (str/int, optional): 样本文件ID
                - 用于从metadata获取样本特定信息
                - 例如：采样率、数据集ID等
                
            task_id (str, optional): 任务类型标识
                - 'classification': 分类任务
                - 'prediction': 预测任务
                - 'regression': 回归任务
                
            return_feature (bool): 是否返回特征而非最终输出
                - True: 返回中间特征表示
                - False: 返回任务相关的最终输出
        
        Returns:
            torch.Tensor: 模型输出
                - 分类任务: (batch_size, num_classes)
                - 预测任务: (batch_size, pred_length, channels)
                - 特征输出: (batch_size, feature_dim)
        """
        # 你的前向传播实现
        pass
```

### 传统模型实现示例

```python
# src/model_factory/CNN/MyResNet.py
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args_m, metadata=None):
        super(Model, self).__init__()
        
        # 从配置提取参数
        self.input_channels = getattr(args_m, 'input_channels', 1)
        self.num_classes = getattr(args_m, 'num_classes', 10)
        self.dropout_rate = getattr(args_m, 'dropout', 0.1)
        
        # 构建网络层
        self.conv1 = nn.Conv1d(self.input_channels, 64, kernel_size=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5) 
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(256, self.num_classes)
        
    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        # 输入形状处理: (B, L, C) -> (B, C, L)
        if x.dim() == 3 and x.shape[-1] < x.shape[1]:
            x = x.transpose(1, 2)
        
        # 特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))
        
        # 全局池化
        features = self.pool(x).squeeze(-1)  # (B, 256)
        
        # 根据需求返回特征或分类结果
        if return_feature:
            return features
        
        x = self.dropout(features)
        output = self.classifier(x)
        
        return output
```

## 🏗️ ISFM基础模型规范

### ISFM模型架构

ISFM (Industrial Signal Foundation Model) 采用模块化设计：

```
Input Signal → Embedding → Backbone → Task Head → Output
     ↓           ↓          ↓          ↓         ↓
   (B,L,C)   Hierarchical  Transformer  Linear   Task-specific
             Signal Embed    / CNN       Head     Output
```

### ISFM版本说明

| 版本 | 文件名 | 特点 | 推荐用途 | 状态 |
|-----|--------|------|----------|------|
| **M_01** | `M_01_ISFM.py` | 基础版本，最小功能集 | ✅ **新手推荐** | 稳定 |
| **M_02** | `M_02_ISFM.py` | 增强版，支持多通道和系统感知 | 🚀 **生产推荐** | 稳定 |
| **M_03** | `M_03_ISFM.py` | 实验版本，功能不完整 | ❌ **不推荐使用** | 有Bug |

### ISFM标准接口

```python
class Model(nn.Module):
    """ISFM基础模型标准实现"""
    
    def __init__(self, args_m, metadata):
        super(Model, self).__init__()
        self.metadata = metadata
        self.args_m = args_m
        
        # 构建三大组件
        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        self.task_head = TaskHead_dict[args_m.task_head](args_m)
    
    def _embed(self, x, file_id=None):
        """
        Step 1: 信号嵌入
        
        Args:
            x: 原始信号 (B, L, C)
            file_id: 用于获取采样率等信息
            
        Returns:
            embedded_x: 嵌入后的信号
            context_info: 上下文信息（可选）
        """
        # 获取信号相关元信息
        if file_id is not None and self.args_m.embedding in ('E_01_HSE', 'E_02_HSE_v2'):
            fs = self.metadata[file_id]['Sample_rate']
            system_id = self.metadata[file_id]['Dataset_id']
            return self.embedding(x, system_id, fs)
        else:
            return self.embedding(x)
    
    def _encode(self, x, context=None):
        """
        Step 2: 特征编码
        
        Args:
            x: 嵌入后的信号
            context: 上下文信息
            
        Returns:
            encoded_features: 编码后的特征
        """
        return self.backbone(x, context)
    
    def _head(self, x, file_id=None, task_id=None, return_feature=False):
        """
        Step 3: 任务输出
        
        Args:
            x: 编码后的特征
            file_id: 样本ID
            task_id: 任务类型
            return_feature: 是否返回特征
            
        Returns:
            task_output: 任务相关输出
        """
        system_id = None
        if file_id is not None:
            system_id = self.metadata[file_id]['Dataset_id']
            
        return self.task_head(
            x, 
            system_id=system_id, 
            task_id=task_id,
            return_feature=return_feature
        )
    
    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        """完整前向传播流程"""
        # 记录输入形状供后续使用
        self.shape = x.shape
        
        # 三步处理流程
        x, context = self._embed(x, file_id)      # Step 1
        x = self._encode(x, context)              # Step 2  
        x = self._head(x, file_id, task_id, return_feature)  # Step 3
        
        return x
```

## 🔧 模型组件开发

### 1. 嵌入层开发 (E_XX)

```python
# src/model_factory/ISFM/embedding/E_XX_YourEmbedding.py
import torch.nn as nn

class E_XX_YourEmbedding(nn.Module):
    """自定义信号嵌入层"""
    
    def __init__(self, configs):
        super().__init__()
        self.patch_size = configs.patch_size_L
        self.embed_dim = configs.output_dim
        
        # 你的嵌入网络
        self.projection = nn.Linear(self.patch_size, self.embed_dim)
        
    def forward(self, x, system_id=None, fs=None):
        """
        Args:
            x: 输入信号 (B, L, C)
            system_id: 系统ID（可选，用于系统感知）
            fs: 采样频率（可选，用于频率感知）
            
        Returns:
            embedded_signal: 嵌入后信号 (B, N_patches, embed_dim)
            context_info: 上下文信息
        """
        # 实现你的嵌入逻辑
        batch_size, seq_len, channels = x.shape
        
        # Patch分割示例
        n_patches = seq_len // self.patch_size
        x_patches = x[:, :n_patches * self.patch_size, :].reshape(
            batch_size, n_patches, self.patch_size, channels
        )
        
        # 投影到嵌入空间
        embedded = self.projection(x_patches.mean(dim=-1))
        
        return embedded, None
```

### 2. 骨干网络开发 (B_XX)

```python
# src/model_factory/ISFM/backbone/B_XX_YourBackbone.py
import torch.nn as nn

class B_XX_YourBackbone(nn.Module):
    """自定义骨干网络"""
    
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.num_layers = configs.num_layers
        
        # 构建你的网络架构
        self.layers = nn.ModuleList([
            YourTransformerLayer(self.d_model) 
            for _ in range(self.num_layers)
        ])
        
    def forward(self, x, context=None):
        """
        Args:
            x: 嵌入后信号 (B, N_patches, embed_dim)
            context: 上下文信息
            
        Returns:
            processed_features: 处理后特征 (B, N_patches, d_model)
        """
        for layer in self.layers:
            x = layer(x, context)
        return x
```

### 3. 任务头开发 (H_XX)

```python
# src/model_factory/ISFM/task_head/H_XX_YourHead.py
import torch.nn as nn

class H_XX_YourHead(nn.Module):
    """自定义任务头"""
    
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        
        # 根据任务类型构建不同输出层
        if hasattr(configs, 'num_classes'):
            self.classifier = nn.Linear(self.d_model, configs.num_classes)
        
    def forward(self, x, system_id=None, task_id=None, return_feature=False):
        """
        Args:
            x: 骨干网络输出 (B, N_patches, d_model)
            system_id: 系统ID（用于多系统任务）
            task_id: 任务类型
            return_feature: 是否返回特征
            
        Returns:
            task_output: 任务相关输出
        """
        # 全局特征聚合
        if x.dim() == 3:
            features = x.mean(dim=1)  # (B, d_model)
        else:
            features = x
            
        if return_feature:
            return features
            
        # 根据任务类型输出
        if task_id == 'classification':
            return self.classifier(features)
        elif task_id == 'prediction':
            # 实现预测任务逻辑
            return self.predictor(features)
        else:
            return features
```

## ⚙️ 配置参数规范

### 标准参数命名

```yaml
model:
  # === 基础参数 ===
  name: "ModelName"           # 模型名称（必需）
  type: "ModelType"           # 模型类型（必需）
  
  # === 网络结构参数 ===
  input_dim: 1                # 输入通道数
  d_model: 128               # 模型隐藏维度
  num_layers: 6              # 网络层数
  num_heads: 8               # 注意力头数
  d_ff: 256                  # 前馈网络维度
  
  # === 训练参数 ===
  dropout: 0.1               # Dropout概率
  activation: "relu"         # 激活函数
  
  # === 任务参数 ===
  num_classes: 10            # 分类类别数
  pred_length: 96            # 预测长度
  
  # === ISFM专用参数 ===
  embedding: "E_01_HSE"      # 嵌入层类型
  backbone: "B_08_PatchTST"  # 骨干网络类型  
  task_head: "H_01_Linear_cla" # 任务头类型
  
  # Patch相关参数
  patch_size_L: 16           # Patch长度
  patch_size_C: 1            # Patch通道
  num_patches: 64            # Patch数量
  output_dim: 128            # 输出维度
```

### 参数访问最佳实践

```python
def __init__(self, args_m, metadata=None):
    super().__init__()
    
    # ✅ 推荐：使用 getattr 提供默认值
    self.input_dim = getattr(args_m, 'input_dim', 1)
    self.dropout = getattr(args_m, 'dropout', 0.1)
    
    # ✅ 推荐：参数验证
    assert self.input_dim > 0, "input_dim must be positive"
    assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"
    
    # ❌ 避免：直接访问可能不存在的属性
    # self.some_param = args_m.some_param  # 可能报错
    
    # ✅ 推荐：处理复杂参数
    if hasattr(args_m, 'layer_sizes'):
        self.layer_sizes = args_m.layer_sizes
    else:
        self.layer_sizes = [128, 256, 128]  # 默认值
```

## 🚀 模型注册和使用

### 1. 注册新模型

```python
# 方法1：在 __init__.py 中注册
# src/model_factory/YourType/__init__.py
from .YourModel import Model as YourModel

# 方法2：使用注册装饰器（推荐）
from ...utils.registry import Registry
from ..model_factory import register_model

@register_model("YourType", "YourModel")
class Model(nn.Module):
    # 你的实现
```

### 2. 配置文件使用

```yaml
# configs/your_experiment.yaml
model:
  name: "YourModel"
  type: "YourType"
  
  # 你的自定义参数
  custom_param1: 128
  custom_param2: true
  custom_layers: [64, 128, 256]
```

### 3. 编程方式使用

```python
from src.model_factory.model_factory import model_factory
from argparse import Namespace

# 创建配置
args_model = Namespace(
    name="YourModel",
    type="YourType", 
    input_dim=1,
    num_classes=4
)

# 实例化模型
model = model_factory(args_model, metadata)
```

## 🧪 测试和调试

### 单元测试模板

```python
# 在你的模型文件末尾添加
if __name__ == '__main__':
    import torch
    from argparse import Namespace
    
    def test_model():
        """测试模型基本功能"""
        # 创建配置
        args_m = Namespace(
            input_dim=1,
            num_classes=4,
            d_model=64,
            dropout=0.1
        )
        
        # 创建模型
        model = Model(args_m)
        model.eval()
        
        # 测试不同输入形状
        batch_sizes = [1, 4, 16]
        seq_lengths = [512, 1024, 2048]
        channels = [1, 3]
        
        for B in batch_sizes:
            for L in seq_lengths:
                for C in channels:
                    x = torch.randn(B, L, C)
                    
                    # 测试前向传播
                    with torch.no_grad():
                        output = model(x)
                        print(f"Input: {x.shape} -> Output: {output.shape}")
                    
                    # 验证输出形状
                    assert output.shape[0] == B, "Batch size mismatch"
                    assert output.shape[1] == args_m.num_classes, "Class number mismatch"
        
        print("✅ 所有测试通过!")
    
    # 运行测试
    test_model()
```

### 常见错误和调试

#### 1. 形状错误

```python
# ❌ 常见错误：未处理输入形状变化
def forward(self, x):
    return self.conv1d(x)  # 假设输入是 (B, C, L) 但实际是 (B, L, C)

# ✅ 正确处理：
def forward(self, x):
    # 检查并转换输入形状
    if x.dim() == 3 and x.shape[-1] < x.shape[1]:
        x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
    return self.conv1d(x)
```

#### 2. 参数访问错误

```python
# ❌ 可能出错：
self.param = args_m.param  # 如果 param 不存在会报错

# ✅ 安全访问：
self.param = getattr(args_m, 'param', default_value)
```

#### 3. ISFM组件错误

```python
# ❌ 常见错误：忘记处理可选参数
def forward(self, x, file_id=None, task_id=None):
    system_id = self.metadata[file_id]['Dataset_id']  # file_id 可能是 None

# ✅ 正确处理：
def forward(self, x, file_id=None, task_id=None):
    system_id = None
    if file_id is not None:
        system_id = self.metadata[file_id]['Dataset_id']
```

## 📊 性能优化建议

### 1. 内存优化

```python
# 使用梯度检查点减少内存
import torch.utils.checkpoint as checkpoint

def forward(self, x):
    # 对于大模型，使用检查点
    x = checkpoint.checkpoint(self.large_layer, x)
    return x

# 及时释放中间变量
def forward(self, x):
    intermediate = self.layer1(x)
    output = self.layer2(intermediate)
    del intermediate  # 显式释放内存
    return output
```

### 2. 计算优化

```python
# 使用 torch.jit.script 编译加速
@torch.jit.script
def efficient_computation(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)

# 预计算常用值
def __init__(self, args_m, metadata=None):
    super().__init__()
    # 预计算位置编码等
    self.register_buffer('pos_encoding', self._create_pos_encoding())
```

### 3. 并行化

```python
# 利用多GPU并行
def forward(self, x):
    if self.training and torch.cuda.device_count() > 1:
        # 模型并行或数据并行优化
        pass
    return x
```

## 🎯 最佳实践总结

### ✅ 推荐做法

1. **接口规范**：严格遵循 `forward(x, file_id, task_id, return_feature)` 接口
2. **参数处理**：使用 `getattr()` 安全访问配置参数
3. **形状处理**：始终验证和处理输入张量形状
4. **错误处理**：对None值和边界情况进行检查
5. **测试代码**：在 `if __name__ == '__main__':` 中添加单元测试
6. **文档注释**：为关键方法添加清晰的docstring

### ❌ 避免事项

1. **硬编码参数**：避免在代码中硬编码数值
2. **形状假设**：不要假设特定的输入张量形状
3. **直接属性访问**：避免 `args_m.param` 而不检查存在性
4. **内存泄漏**：及时释放大型中间变量
5. **接口不一致**：不要改变标准接口签名

---

🎉 **现在你已经掌握了PHM基础模型开发的全部要点！**

继续阅读：
- 📊 [QUICKSTART.md](QUICKSTART.md) - 快速上手指南
- 📈 [DATA_GUIDE.md](DATA_GUIDE.md) - 数据系统详解  
- 🎯 [TASK_GUIDE.md](TASK_GUIDE.md) - 任务类型说明