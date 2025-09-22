# ContrastiveIDTask API 参考

PHM-Vibench ContrastiveIDTask的完整API接口文档。

## 📋 快速索引

- [核心类](#核心类)
- [配置接口](#配置接口)
- [数据接口](#数据接口)
- [训练接口](#训练接口)
- [工具函数](#工具函数)

## 🏗️ 核心类

### ContrastiveIDTask

长信号对比学习任务的主要实现类。

```python
@register_task("contrastive_id", "pretrain")
class ContrastiveIDTask(BaseIDTask):
    """基于ID的对比学习预训练任务

    继承BaseIDTask的所有功能，专注于对比学习逻辑实现。
    """
```

#### 初始化

```python
def __init__(self, **kwargs):
    """初始化对比学习任务

    Args:
        temperature (float, optional): InfoNCE温度参数. 默认: 0.07
        **kwargs: 传递给BaseIDTask的其他参数

    Attributes:
        temperature (float): 对比学习温度参数
        criterion (nn.CrossEntropyLoss): 损失函数
    """
```

#### 核心方法

##### prepare_batch()

```python
def prepare_batch(self, batch_data: List[Tuple]) -> Dict[str, torch.Tensor]:
    """为对比学习准备批次数据

    为每个样本ID生成正样本对，构建用于InfoNCE损失计算的张量。

    Args:
        batch_data (List[Tuple]): 批次数据
            格式: [(sample_id, None, metadata), ...]

    Returns:
        Dict[str, torch.Tensor]: 包含以下键的字典
            - 'anchor': 锚点窗口 [batch_size, window_size, channels]
            - 'positive': 正样本窗口 [batch_size, window_size, channels]

    Raises:
        ValueError: 当批次为空或处理失败时
        RuntimeError: 当窗口生成失败时

    Example:
        ```python
        task = ContrastiveIDTask(temperature=0.07)
        batch = task.prepare_batch(dataloader_batch)
        print(f"锚点形状: {batch['anchor'].shape}")
        print(f"正样本形状: {batch['positive'].shape}")
        ```
    """
```

##### infonce_loss()

```python
def infonce_loss(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
    """计算InfoNCE对比损失

    实现标准的InfoNCE损失计算，包括特征归一化和温度缩放。

    Args:
        z_anchor (torch.Tensor): 锚点特征向量 [batch_size, d_model]
        z_positive (torch.Tensor): 正样本特征向量 [batch_size, d_model]

    Returns:
        torch.Tensor: InfoNCE损失值 (标量)

    Mathematical Formula:
        L = -Σ_i log(exp(s(z_i, z_i+) / τ) / Σ_j exp(s(z_i, z_j) / τ))

        其中:
        - s(·,·): 余弦相似度函数
        - τ: 温度参数 (self.temperature)

    Example:
        ```python
        # 假设模型输出特征维度为256
        z_anchor = torch.randn(32, 256)  # 批量大小32
        z_positive = torch.randn(32, 256)

        loss = task.infonce_loss(z_anchor, z_positive)
        print(f"InfoNCE损失: {loss.item():.4f}")
        ```
    """
```

##### compute_accuracy()

```python
def compute_accuracy(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
    """计算对比学习准确率

    计算正样本在相似度排序中位于第一位的比例。

    Args:
        z_anchor (torch.Tensor): 锚点特征向量 [batch_size, d_model]
        z_positive (torch.Tensor): 正样本特征向量 [batch_size, d_model]

    Returns:
        torch.Tensor: 准确率 (0-1之间的标量)

    Example:
        ```python
        accuracy = task.compute_accuracy(z_anchor, z_positive)
        print(f"对比学习准确率: {accuracy.item():.2%}")
        ```
    """
```

##### _shared_step()

```python
def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
    """训练/验证/测试的共享步骤

    Args:
        batch (Dict): 批次数据，包含'anchor'和'positive'
        stage (str): 阶段标识 ('train', 'val', 'test')

    Returns:
        torch.Tensor: 损失值
    """
```

## ⚙️ 配置接口

### load_config()

```python
from src.configs import load_config

# 基础用法
config = load_config('contrastive')

# 从文件加载
config = load_config('configs/id_contrastive/debug.yaml')

# 参数覆盖
config = load_config('contrastive', {
    'task.temperature': 0.05,
    'data.batch_size': 64,
    'model.d_model': 512
})
```

### 配置结构

#### 数据配置 (ConfigWrapper.data)

```python
class DataConfig:
    factory_name: str = "id"                    # 数据工厂名称
    dataset_name: str = "ID_dataset"            # 数据集类名
    data_dir: str = "data"                      # 数据根目录
    metadata_file: str = "metadata_6_1.xlsx"   # 元数据文件

    # 窗口化参数
    window_size: int = 1024                     # 窗口大小
    stride: int = 512                           # 窗口步长
    num_window: int = 2                         # 每ID窗口数量
    window_sampling_strategy: str = "random"    # 采样策略

    # 批处理参数
    batch_size: int = 32                        # 批量大小
    num_workers: int = 4                        # 数据加载进程

    # 预处理参数
    normalization: bool = True                  # Z-score标准化
    truncate_length: Optional[int] = None       # 截断长度
```

#### 任务配置 (ConfigWrapper.task)

```python
class TaskConfig:
    name: str = "contrastive_id"               # 任务名称
    type: str = "pretrain"                     # 任务类型

    # 对比学习参数
    temperature: float = 0.07                  # InfoNCE温度

    # 优化参数
    lr: float = 0.001                          # 学习率
    weight_decay: float = 1e-4                 # 权重衰减

    # 调度器参数
    scheduler: str = "cosine"                  # 学习率调度器
    warmup_steps: int = 1000                   # 预热步数
```

#### 模型配置 (ConfigWrapper.model)

```python
class ModelConfig:
    factory_name: str = "ISFM"                 # 模型工厂
    type: str = "ISFM"                         # 模型类型

    # 架构参数
    d_model: int = 256                         # 嵌入维度
    nhead: int = 8                             # 注意力头数
    nlayers: int = 6                           # 编码器层数

    # 输入参数
    input_dim: int = 1                         # 输入通道数
    seq_len: int = 1024                        # 序列长度
```

## 📊 数据接口

### ID_dataset

基础ID数据集类，由data_factory提供。

```python
from src.data_factory import create_dataset

# 创建数据集
dataset = create_dataset(
    factory_name="id",
    dataset_name="ID_dataset",
    data_dir="data",
    metadata_file="metadata_6_1.xlsx"
)

# 数据集接口
len(dataset)                    # 数据集大小
dataset[idx]                    # 获取样本: (sample_id, None, metadata)
dataset.get_sample_ids()        # 获取所有样本ID
```

### H5DataDict

延迟加载数据字典，提供内存高效的数据访问。

```python
# 通过BaseIDTask._get_data_for_id()访问
data = task._get_data_for_id(sample_id)
print(f"信号形状: {data.shape}")         # [seq_len, channels]
print(f"数据类型: {data.dtype}")         # torch.float32
```

## 🚀 训练接口

### PyTorch Lightning集成

ContrastiveIDTask继承了BaseIDTask的PyTorch Lightning接口。

```python
import pytorch_lightning as pl

# 创建任务实例
task = ContrastiveIDTask(
    temperature=0.07,
    lr=0.001,
    # ... 其他参数
)

# 创建训练器
trainer = pl.Trainer(
    max_epochs=100,
    devices=1,
    precision=16
)

# 开始训练
trainer.fit(task, train_dataloader, val_dataloader)
```

### 训练步骤方法

```python
# Lightning训练步骤
def training_step(self, batch, batch_idx) -> torch.Tensor:
    """训练步骤 - 自动调用"""
    return self._shared_step(batch, "train")

def validation_step(self, batch, batch_idx) -> torch.Tensor:
    """验证步骤 - 自动调用"""
    return self._shared_step(batch, "val")

def test_step(self, batch, batch_idx) -> torch.Tensor:
    """测试步骤 - 自动调用"""
    return self._shared_step(batch, "test")
```

### 优化器配置

```python
def configure_optimizers(self):
    """配置优化器和学习率调度器"""
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=self.lr,
        weight_decay=self.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )

    return [optimizer], [scheduler]
```

## 🛠️ 工具函数

### 特征提取

```python
def extract_features(model: ContrastiveIDTask, dataset_id: str) -> torch.Tensor:
    """提取预训练特征用于下游任务

    Args:
        model: 预训练的ContrastiveIDTask模型
        dataset_id: 数据集标识符

    Returns:
        torch.Tensor: 提取的特征 [num_samples, d_model]
    """
    model.eval()
    features = []

    with torch.no_grad():
        for batch in dataloader:
            batch_prepared = model.prepare_batch(batch)
            z_anchor = model.model(batch_prepared['anchor'])
            features.append(z_anchor)

    return torch.cat(features, dim=0)
```

### 相似度计算

```python
def cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """计算余弦相似度

    Args:
        z1, z2: 特征向量 [batch_size, d_model]

    Returns:
        torch.Tensor: 相似度矩阵 [batch_size, batch_size]
    """
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    return torch.mm(z1_norm, z2_norm.t())
```

### 可视化工具

```python
def visualize_embeddings(features: torch.Tensor, labels: List[int]):
    """使用t-SNE可视化特征嵌入

    Args:
        features: 特征矩阵 [num_samples, d_model]
        labels: 标签列表
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.numpy())

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('ContrastiveIDTask特征可视化')
    plt.show()
```

## 📁 文件结构

```
src/task_factory/task/pretrain/
└── ContrastiveIDTask.py              # 主实现文件

configs/id_contrastive/
├── debug.yaml                        # 调试配置
├── production.yaml                   # 生产配置
├── ablation.yaml                     # 消融实验配置
└── cross_dataset.yaml               # 跨数据集配置

test/unit/task_factory/
└── test_contrastive_id_task.py       # 单元测试

test/integration/
├── test_contrastive_full_training.py # 集成测试
└── test_contrastive_real_data.py     # 真实数据测试
```

## 🔗 相关链接

- **技术指南**: [technical_guide.md](technical_guide.md)
- **故障排除**: [troubleshooting.md](troubleshooting.md)
- **PHM-Vibench文档**: [../../../docs/](../../../docs/)
- **配置系统**: [../../../src/configs/CLAUDE.md](../../../src/configs/CLAUDE.md)

---

**API版本**: v1.0
**更新时间**: 2024年9月
**兼容性**: PHM-Vibench v5.0+