# 对比学习预训练使用指南

## 目录
- [快速开始](#快速开始)
- [配置参数说明](#配置参数说明)
- [使用场景](#使用场景)
- [API文档](#api文档)
- [常见问题解答](#常见问题解答)
- [性能调优建议](#性能调优建议)
- [集成示例](#集成示例)

## 快速开始

### 1. 基础使用

最简单的对比学习预训练实验:

```bash
# 使用默认配置运行预训练
python main.py --config configs/id_contrastive/pretrain.yaml
```

### 2. 调试模式

开发和调试时使用小规模配置:

```bash
# 调试模式（CPU，小数据，快速迭代）
python main.py --config configs/id_contrastive/debug.yaml
```

### 3. 生产环境

正式实验使用优化配置:

```bash
# 生产环境（GPU，大批量，高性能）
python main.py --config configs/id_contrastive/production.yaml
```

### 4. 参数覆盖

通过命令行覆盖配置参数:

```bash
# 覆盖温度参数和学习率
python main.py --config configs/id_contrastive/pretrain.yaml \
    --override task.temperature=0.05 task.lr=5e-4
```

## 配置参数说明

### 数据配置 (data)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `factory_name` | str | "id" | 数据工厂名称，固定为"id" |
| `dataset_name` | str | "ID_dataset" | 数据集类名 |
| `batch_size` | int | 32 | 批大小，影响内存使用和训练稳定性 |
| `num_workers` | int | 4 | 数据加载进程数 |
| `window_size` | int | 1024 | 窗口大小，决定输入序列长度 |
| `stride` | int | 512 | 窗口步长，影响窗口重叠度 |
| `num_window` | int | 2 | 每个ID生成的窗口数量 |
| `window_sampling_strategy` | str | "random" | 窗口采样策略: random/sequential/evenly_spaced |
| `normalization` | bool | true | 是否进行数据标准化 |
| `truncate_length` | int | 16384 | 信号截断长度 |

### 模型配置 (model)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | "M_01_ISFM" | 模型名称 |
| `backbone` | str | "B_08_PatchTST" | 主干网络类型 |
| `d_model` | int | 256 | 模型隐藏维度 |

### 任务配置 (task)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | str | "pretrain" | 任务类型，固定为"pretrain" |
| `name` | str | "contrastive_id" | 任务名称，固定为"contrastive_id" |
| `lr` | float | 1e-3 | 学习率 |
| `weight_decay` | float | 1e-4 | 权重衰减 |
| `temperature` | float | 0.07 | InfoNCE温度参数，影响对比学习的难度 |

### 训练配置 (trainer)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs` | int | 50 | 训练轮数 |
| `accelerator` | str | "gpu" | 加速器类型: gpu/cpu |
| `devices` | int | 1 | 设备数量 |
| `precision` | int | 16 | 精度: 16(混合精度)/32(单精度) |
| `gradient_clip_val` | float | 1.0 | 梯度裁剪阈值 |
| `check_val_every_n_epoch` | int | 5 | 验证频率 |
| `log_every_n_steps` | int | 50 | 日志记录频率 |

## 使用场景

### 场景1: 单数据集预训练

用于在单个数据集上进行对比学习预训练:

```yaml
# configs/my_experiment.yaml
data:
  factory_name: "id"
  dataset_name: "ID_dataset"  
  # ... 其他参数保持默认
```

```bash
python main.py --config configs/my_experiment.yaml
```

### 场景2: 跨数据集域泛化

在多个源数据集上预训练，然后在目标数据集上评估:

```bash
# 使用跨数据集配置
python main.py --config configs/id_contrastive/cross_dataset.yaml \
    --override data.source_datasets='["CWRU","XJTU"]' \
    --override data.target_datasets='["PU","MFPT"]'
```

### 场景3: 消融实验

系统性研究不同参数的影响:

```bash
# 温度参数消融实验
python scripts/ablation_studies.py \
    --base_config configs/id_contrastive/ablation.yaml \
    --ablation_param task.temperature \
    --ablation_values 0.01 0.05 0.07 0.1 0.5
```

### 场景4: 超参数搜索

使用grid search或random search优化超参数:

```bash
# 网格搜索示例
python scripts/hyperparameter_search.py \
    --config configs/id_contrastive/pretrain.yaml \
    --search_space configs/search_spaces/contrastive_search.json
```

## API文档

### ContrastiveIDTask类

```python
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask

# 创建任务实例
task = ContrastiveIDTask(
    temperature=0.07,    # InfoNCE温度参数
    lr=1e-3,            # 学习率
    weight_decay=1e-4,   # 权重衰减
    **other_params
)
```

#### 核心方法

**prepare_batch(batch_data)**
- 功能: 准备对比学习的批处理数据
- 输入: `List[Tuple[str, np.ndarray, Dict]]` - 原始批次数据
- 输出: `Dict[str, torch.Tensor]` - 包含anchor, positive, ids的字典
- 异常: 当窗口数量不足时抛出ValueError

**infonce_loss(anchor_features, positive_features)**
- 功能: 计算InfoNCE对比损失
- 输入: 锚点特征和正样本特征张量
- 输出: 标量损失值
- 数学公式: `loss = -log(exp(sim_pos/τ) / Σexp(sim_all/τ))`

**contrastive_accuracy(anchor_features, positive_features)**
- 功能: 计算对比学习Top-1准确率
- 输入: 锚点特征和正样本特征张量
- 输出: 准确率标量值

### 配置加载

```python
from src.configs import load_config

# 从预设配置加载
config = load_config('contrastive_id_pretrain')

# 从文件加载并覆盖参数
config = load_config(
    'configs/id_contrastive/pretrain.yaml',
    {'task.temperature': 0.05, 'trainer.epochs': 100}
)
```

## 常见问题解答

### Q1: 内存不足怎么办？

**A:** 尝试以下解决方案:
1. 减小`batch_size` (如从32改为16)
2. 减小`window_size` (如从1024改为512)  
3. 减小`d_model` (如从256改为128)
4. 使用`precision: 16`启用混合精度训练
5. 增加`gradient_accumulation_steps`模拟大批量

```yaml
# 内存优化配置示例
data:
  batch_size: 16          # 减小批量
  window_size: 512        # 减小窗口
model:
  d_model: 128           # 减小模型维度
trainer:
  precision: 16          # 混合精度
  accumulate_grad_batches: 2  # 梯度累积
```

### Q2: 训练不收敛怎么办？

**A:** 检查以下方面:
1. **学习率过大**: 尝试减小`lr` (如1e-4)
2. **温度参数不合适**: 尝试调整`temperature` (0.05-0.1)
3. **梯度爆炸**: 减小`gradient_clip_val` (如0.5)
4. **数据预处理问题**: 确保`normalization: true`

```yaml
# 收敛优化配置
task:
  lr: 5e-4               # 较小学习率
  temperature: 0.05      # 较低温度
trainer:
  gradient_clip_val: 0.5 # 严格梯度裁剪
```

### Q3: 如何加速训练？

**A:** 优化策略:
1. **增大批量**: 提高`batch_size`和`num_workers`
2. **混合精度**: 使用`precision: 16`
3. **减少验证频率**: 增大`check_val_every_n_epoch`
4. **并行训练**: 使用多GPU `devices: [0,1]`

```yaml
# 速度优化配置
data:
  batch_size: 64         # 大批量
  num_workers: 8         # 多进程
trainer:
  precision: 16          # 混合精度
  devices: [0, 1]        # 多GPU
  check_val_every_n_epoch: 10  # 减少验证
```

### Q4: 如何监控训练过程？

**A:** 监控工具:
1. **TensorBoard**: 自动生成训练曲线
2. **WandB**: 在线实验管理
3. **命令行输出**: 实时损失和准确率
4. **日志文件**: 详细训练记录

```bash
# 启动TensorBoard监控
tensorboard --logdir=save/contrastive_pretrain/lightning_logs

# 查看实时日志
tail -f save/contrastive_pretrain/log.txt
```

### Q5: 如何选择最佳超参数？

**A:** 推荐设置范围:
- `temperature`: 0.05-0.1 (较低值学习更精细特征)
- `lr`: 1e-4 to 1e-3 (根据批量大小调整)
- `window_size`: 512-2048 (根据信号特征调整)
- `batch_size`: 16-64 (根据GPU内存调整)

使用消融实验脚本进行系统性搜索:
```bash
python scripts/ablation_studies.py --config configs/id_contrastive/ablation.yaml
```

## 性能调优建议

### 内存优化

1. **批量大小调整**:
   ```yaml
   data:
     batch_size: 32        # 基线
     # batch_size: 16      # 内存不足时
     # batch_size: 64      # 内存充足时
   ```

2. **窗口大小优化**:
   ```yaml
   data:
     window_size: 1024     # 平衡选择
     # window_size: 512    # 快速实验
     # window_size: 2048   # 高质量特征
   ```

3. **混合精度训练**:
   ```yaml
   trainer:
     precision: 16         # 减少50%内存使用
   ```

### 速度优化

1. **数据加载优化**:
   ```yaml
   data:
     num_workers: 8        # 根据CPU核数调整
     pin_memory: true      # 加速GPU传输
     persistent_workers: true  # 保持worker进程
   ```

2. **模型编译**:
   ```yaml
   model:
     compile: true         # PyTorch 2.0编译优化
   ```

3. **验证频率调整**:
   ```yaml
   trainer:
     check_val_every_n_epoch: 10  # 减少验证开销
   ```

### 质量优化

1. **学习率调度**:
   ```yaml
   task:
     lr_scheduler: "cosine"    # 余弦退火
     warmup_steps: 1000       # 学习率预热
   ```

2. **数据增强**:
   ```yaml
   data:
     augmentation: true       # 启用数据增强
     noise_level: 0.01       # 添加噪声
   ```

## 集成示例

### 与Pipeline_02集成

对比学习预训练 + 下游任务微调:

```python
# scripts/pretrain_finetune_pipeline.py
from src.configs import load_config
from src.pipeline import Pipeline_02_pretrain_fewshot

# 1. 预训练配置
pretrain_config = load_config('configs/id_contrastive/pretrain.yaml')

# 2. 微调配置  
finetune_config = load_config('configs/demo/GFS/GFS_demo.yaml', {
    'model.pretrain_path': 'save/contrastive_pretrain/checkpoints/best.ckpt'
})

# 3. 运行Pipeline
pipeline = Pipeline_02_pretrain_fewshot(
    pretrain_config=pretrain_config,
    finetune_config=finetune_config
)
results = pipeline.run()
```

### 与Streamlit GUI集成

```python
# streamlit_contrastive.py
import streamlit as st
from src.configs import load_config

st.title("对比学习预训练实验")

# 参数配置界面
temperature = st.slider("温度参数", 0.01, 0.5, 0.07)
batch_size = st.selectbox("批量大小", [16, 32, 64])
epochs = st.number_input("训练轮数", 1, 100, 50)

# 构建配置
config = load_config('configs/id_contrastive/pretrain.yaml', {
    'task.temperature': temperature,
    'data.batch_size': batch_size,
    'trainer.epochs': epochs
})

# 启动训练按钮
if st.button("开始训练"):
    # 运行实验逻辑
    pass
```

### 批量实验脚本

```python
# scripts/batch_experiments.py
import itertools
from src.configs import load_config

# 定义实验矩阵
temperature_values = [0.01, 0.05, 0.07, 0.1]
batch_sizes = [16, 32, 64]
window_sizes = [512, 1024, 2048]

# 生成所有组合
experiments = itertools.product(temperature_values, batch_sizes, window_sizes)

for temp, batch, window in experiments:
    # 构建配置
    config = load_config('configs/id_contrastive/pretrain.yaml', {
        'task.temperature': temp,
        'data.batch_size': batch,
        'data.window_size': window,
        'environment.experiment_name': f'temp{temp}_batch{batch}_window{window}'
    })
    
    # 运行实验
    # main_function(config)
```

## 进阶主题

### 自定义损失函数

扩展ContrastiveIDTask以支持其他对比损失:

```python
class CustomContrastiveIDTask(ContrastiveIDTask):
    def __init__(self, loss_type="infonce", **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
    
    def compute_loss(self, anchor, positive):
        if self.loss_type == "infonce":
            return self.infonce_loss(anchor, positive)
        elif self.loss_type == "ntxent":
            return self.ntxent_loss(anchor, positive)
        # 添加更多损失函数
```

### 动态温度调整

实现温度参数的自适应调整:

```python
class AdaptiveTemperatureTask(ContrastiveIDTask):
    def on_epoch_end(self):
        # 根据验证性能调整温度
        if self.current_val_acc < self.best_val_acc:
            self.temperature *= 0.95  # 降低温度
        else:
            self.temperature = min(self.temperature * 1.05, 0.5)  # 提高温度
```

### 特征可视化

分析学到的表征:

```python
# scripts/feature_analysis.py
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_features(model, dataloader):
    features, labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            feat = model.encode(batch)
            features.append(feat.cpu())
            labels.append(batch['labels'].cpu())
    
    # t-SNE降维可视化
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title('对比学习特征分布')
    plt.show()
```

---

本指南涵盖了PHM-Vibench对比学习预训练的完整使用方法。如有问题，请参考项目文档或提交Issue。