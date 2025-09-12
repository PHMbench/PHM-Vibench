# ContrastiveID Pretraining Guide

PHM-Vibench对比学习预训练任务完整使用指南。基于长信号ID对比学习，适用于工业设备振动信号分析的自监督预训练。

## 目录
- [快速开始](#快速开始)
- [配置参数详解](#配置参数详解)
- [实验工作流](#实验工作流)
- [集成指南](#集成指南)
- [故障排除](#故障排除)
- [高级用法](#高级用法)
- [API参考](#api参考)
- [性能优化](#性能优化)

## 快速开始

### 基本概念

ContrastiveIDTask是基于InfoNCE损失的对比学习任务，通过多窗口采样机制从同一ID信号中生成正样本对，学习具有语义意义的信号表征。

核心思想：
- **正样本对**: 同一ID信号的不同窗口
- **负样本**: 批次中其他样本的窗口
- **学习目标**: 最大化正样本对相似度，最小化负样本相似度

### 1. 调试模式 - 快速验证

开发和调试时使用小规模配置，最小资源占用：

```bash
# CPU调试模式 - 单epoch快速验证
python main.py --config configs/id_contrastive/debug.yaml
```

**调试配置特点**:
- 使用CPU，单线程数据加载
- 小批量(4)和小窗口(256)
- 单epoch训练，详细日志
- 最小模型维度(64)

### 2. 生产环境 - 完整训练

正式实验使用GPU优化配置：

```bash
# GPU生产模式 - 完整训练
python main.py --config configs/id_contrastive/production.yaml
```

**生产配置特点**:
- GPU + 混合精度训练
- 大批量(64)和大窗口(2048)
- 100个epoch，余弦学习率调度
- 早停和检查点保存

### 3. 消融实验 - 参数研究

系统性研究不同参数的影响：

```bash
# 消融实验基础配置
python main.py --config configs/id_contrastive/ablation.yaml

# 温度参数消融
python main.py --config configs/id_contrastive/ablation.yaml \
    --override task.temperature=0.05

# 批量大小消融  
python main.py --config configs/id_contrastive/ablation.yaml \
    --override data.batch_size=64
```

### 4. 跨数据集泛化 - 域适应

评估模型的泛化能力：

```bash
# 跨数据集域泛化
python main.py --config configs/id_contrastive/cross_dataset.yaml \
    --override data.source_datasets='["CWRU","XJTU"]' \
    --override data.target_datasets='["PU","MFPT"]'
```

## 配置参数详解

### 数据配置 (data)

控制数据加载、预处理和窗口化的关键参数：

| 参数 | 类型 | 默认值 | 描述 | 调优建议 |
|------|------|--------|------|---------|
| `factory_name` | str | "id" | 数据工厂名称，使用ID数据架构 | 固定值，无需修改 |
| `dataset_name` | str | "ID_dataset" | 数据集类名，兼容所有ID格式数据 | 固定值，无需修改 |
| `batch_size` | int | 32 | 批大小，影响内存和训练稳定性 | 调试:4, 生产:64, 根据GPU内存调整 |
| `num_workers` | int | 4 | 数据加载进程数，影响I/O效率 | 调试:1, 生产:8, 不超过CPU核数 |
| `window_size` | int | 1024 | 窗口大小，决定输入序列长度 | 越大捕获更多信息，但增加内存消耗 |
| `stride` | int | 512 | 窗口步长，控制窗口重叠度 | 通常为window_size的1/2 |
| `num_window` | int | 2 | 每个ID生成的窗口数量 | 对比学习固定为2，生成正样本对 |
| `window_sampling_strategy` | str | "random" | 窗口采样策略 | random(推荐)/sequential/evenly_spaced |
| `normalization` | bool | true | 是否进行Z-score标准化 | 强烈推荐开启，提高训练稳定性 |
| `truncate_length` | int | 16384 | 原始信号截断长度 | 根据数据集特征调整 |

**跨数据集专用参数** (仅cross_dataset.yaml):
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `source_datasets` | list | ["CWRU", "XJTU"] | 源域数据集列表 |
| `target_datasets` | list | ["PU", "MFPT"] | 目标域数据集列表 |
| `dataset_balancing` | str | "weighted" | 数据集平衡策略 |

### 模型配置 (model)

定义模型架构和容量：

| 参数 | 类型 | 默认值 | 描述 | 可选值 |
|------|------|--------|------|-------|
| `name` | str | "M_01_ISFM" | ISFM模型变体 | M_01_ISFM/M_02_ISFM/M_03_ISFM |
| `backbone` | str | "B_08_PatchTST" | 主干网络架构 | B_08_PatchTST/B_04_Dlinear/B_06_TimesNet |
| `d_model` | int | 256 | 模型隐藏维度 | 调试:64, 标准:256, 大模型:512 |

**主干网络特点**:
- **B_08_PatchTST**: Patch-based Transformer，适合长序列
- **B_04_Dlinear**: 轻量级线性网络，计算效率高
- **B_06_TimesNet**: 时域特征提取网络，适合时序分析

### 任务配置 (task)

控制对比学习的核心参数：

| 参数 | 类型 | 默认值 | 描述 | 调优建议 |
|------|------|--------|------|---------|
| `type` | str | "pretrain" | 任务类型标识 | 固定值 |
| `name` | str | "contrastive_id" | 任务名称标识 | 固定值 |
| `lr` | float | 1e-3 | 学习率，控制梯度更新步长 | 小批量用1e-3，大批量用5e-4 |
| `weight_decay` | float | 1e-4 | L2正则化权重衰减 | 防止过拟合，通常1e-4~1e-5 |
| `temperature` | float | 0.07 | InfoNCE温度参数 | 关键参数！0.05-0.1，越小越严格 |

**温度参数详解**:
- `temperature < 0.05`: 非常严格的对比学习，可能难以收敛
- `temperature = 0.07`: 推荐值，平衡学习难度和收敛性
- `temperature > 0.1`: 较宽松的对比学习，可能学习到粗粒度特征

### 训练配置 (trainer)

控制训练过程和优化策略：

| 参数 | 类型 | 默认值 | 描述 | 场景建议 |
|------|------|--------|------|---------|
| `epochs` | int | 50 | 最大训练轮数 | 调试:1, 标准:50, 生产:100 |
| `accelerator` | str | "gpu" | 计算设备类型 | gpu(推荐)/cpu |
| `devices` | int/list | 1 | 使用的设备数量或列表 | 单卡:1, 多卡:[0,1] |
| `precision` | int | 16 | 数值精度 | 16(混合精度，省内存)/32(单精度，稳定) |
| `gradient_clip_val` | float | 1.0 | 梯度裁剪阈值 | 防止梯度爆炸，0.5-1.0 |
| `check_val_every_n_epoch` | int | 5 | 验证频率 | 调试:1, 生产:10 |
| `log_every_n_steps` | int | 50 | 日志记录步长 | 调试:1, 生产:100 |

**生产环境专用参数** (仅production.yaml):
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `enable_early_stopping` | bool | true | 启用早停机制 |
| `patience` | int | 15 | 早停耐心值 |
| `lr_scheduler` | str | "cosine" | 学习率调度策略 |
| `save_top_k` | int | 3 | 保存最佳k个检查点 |

### 环境配置 (environment)

控制实验环境和结果保存：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `save_dir` | str | "save/" | 结果保存根目录 |
| `experiment_name` | str | "contrastive_*" | 实验名称，影响保存路径 |

## 实验工作流

### 工作流1: 调试开发流程

新功能开发和问题调试的推荐流程：

```bash
# 1. 快速验证 - 使用最小配置
python main.py --config configs/id_contrastive/debug.yaml

# 2. 检查输出日志
tail -f debug/contrastive_debug/log.txt

# 3. 参数微调 - 覆盖特定参数
python main.py --config configs/id_contrastive/debug.yaml \
    --override task.temperature=0.05 data.window_size=512

# 4. 代码调试模式（如需要）
python -m pdb main.py --config configs/id_contrastive/debug.yaml
```

**调试检查清单**:
- [ ] 数据加载正常，无报错
- [ ] 损失函数下降趋势
- [ ] 准确率有意义提升
- [ ] 内存使用可控

### 工作流2: 消融实验流程

系统性参数研究的标准流程：

```bash
# 1. 基线实验 - 使用默认参数
python main.py --config configs/id_contrastive/ablation.yaml \
    --override environment.experiment_name=baseline_temp_0.07

# 2. 温度参数消融
for temp in 0.01 0.05 0.07 0.1 0.5; do
    python main.py --config configs/id_contrastive/ablation.yaml \
        --override task.temperature=$temp \
        --override environment.experiment_name=temp_${temp}
done

# 3. 批量大小消融
for batch in 16 32 64 128; do
    python main.py --config configs/id_contrastive/ablation.yaml \
        --override data.batch_size=$batch \
        --override environment.experiment_name=batch_${batch}
done

# 4. 窗口大小消融
for window in 512 1024 2048 4096; do
    python main.py --config configs/id_contrastive/ablation.yaml \
        --override data.window_size=$window \
        --override data.stride=$((window/2)) \
        --override environment.experiment_name=window_${window}
done
```

**消融实验分析**:
```python
# 分析脚本示例
import pandas as pd
import matplotlib.pyplot as plt

# 收集实验结果
results = []
for exp_name in experiment_names:
    metrics = load_experiment_metrics(f"save/ablation/{exp_name}")
    results.append({
        'experiment': exp_name,
        'final_loss': metrics['val_contrastive_loss'][-1],
        'final_acc': metrics['val_contrastive_acc'][-1],
        'parameter': parse_parameter_from_name(exp_name)
    })

# 可视化分析
df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(df['parameter'], df['final_loss'], 'bo-')
plt.title('Parameter vs Loss')
plt.subplot(1, 2, 2)
plt.plot(df['parameter'], df['final_acc'], 'ro-')
plt.title('Parameter vs Accuracy')
plt.show()
```

### 工作流3: 生产训练流程

正式实验的完整训练流程：

```bash
# 1. 预检查 - 确认环境和数据
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from src.configs import load_config; print('Config loading OK')"

# 2. 启动训练 - 后台运行
nohup python main.py --config configs/id_contrastive/production.yaml > train.log 2>&1 &

# 3. 监控训练 - 实时日志
tail -f train.log
# 或使用TensorBoard
tensorboard --logdir=save/production/contrastive_production/lightning_logs

# 4. 检查点管理
ls save/production/contrastive_production/checkpoints/
# best.ckpt - 验证性能最佳模型
# last.ckpt - 最终训练状态
# epoch=N.ckpt - 定期保存点

# 5. 结果评估
python scripts/evaluate_pretrain.py \
    --checkpoint save/production/contrastive_production/checkpoints/best.ckpt \
    --config configs/id_contrastive/production.yaml
```

### 工作流4: 跨数据集泛化流程

域适应实验的系统化流程：

```bash
# 1. 单域基线 - 在各数据集上独立训练
datasets=("CWRU" "XJTU" "PU" "MFPT")
for dataset in "${datasets[@]}"; do
    python main.py --config configs/id_contrastive/cross_dataset.yaml \
        --override data.source_datasets="[\"$dataset\"]" \
        --override data.target_datasets="[\"$dataset\"]" \
        --override environment.experiment_name=single_domain_$dataset
done

# 2. 跨域泛化 - 源域到目标域
python main.py --config configs/id_contrastive/cross_dataset.yaml \
    --override data.source_datasets='["CWRU","XJTU"]' \
    --override data.target_datasets='["PU","MFPT"]' \
    --override environment.experiment_name=cross_domain_bearing

# 3. 多域联合 - 使用所有源域
python main.py --config configs/id_contrastive/cross_dataset.yaml \
    --override data.source_datasets='["CWRU","XJTU","PU"]' \
    --override data.target_datasets='["MFPT"]' \
    --override environment.experiment_name=multi_source_to_mfpt

# 4. 域分析
python scripts/domain_analysis.py \
    --experiments single_domain_* cross_domain_* multi_source_*
```

**域泛化评估指标**:
- **源域性能**: 训练数据集上的表现
- **目标域性能**: 测试数据集上的零样本表现  
- **域差距**: 源域与目标域性能的差异
- **泛化比率**: 目标域性能/源域性能

## 集成指南

### 与PHM-Vibench工厂系统集成

ContrastiveIDTask完全集成到PHM-Vibench的工厂设计模式中：

#### 1. 任务注册机制

```python
# 自动注册到task_factory
@register_task("contrastive_id", "pretrain")
class ContrastiveIDTask(BaseIDTask):
    """对比学习预训练任务"""
    pass

# 使用任务工厂创建
from src.task_factory import TaskFactory
task_factory = TaskFactory()
task = task_factory.create_task(
    task_type="pretrain",
    task_name="contrastive_id",
    # ... 其他参数
)
```

#### 2. 数据工厂集成

与ID数据架构无缝集成：

```python
# 使用id_data_factory
data_factory = DataFactory()
dataloader = data_factory.create_data_loader(
    factory_name="id",
    dataset_name="ID_dataset",
    # ... 数据配置
)

# 支持所有ID格式数据集
supported_datasets = [
    "CWRU", "XJTU", "PU", "MFPT", "JNU", 
    "PHM2009", "FEMTO", "IMS", "PRONOSTIA"
]
```

#### 3. 模型工厂集成

支持所有ISFM模型架构：

```python
# 模型工厂创建
model_factory = ModelFactory()
network = model_factory.create_model(
    model_type="ISFM",
    model_name="M_01_ISFM",
    backbone="B_08_PatchTST",
    # ... 模型配置
)

# 支持的模型架构
isfm_models = ["M_01_ISFM", "M_02_ISFM", "M_03_ISFM"]
backbones = ["B_08_PatchTST", "B_04_Dlinear", "B_06_TimesNet", "B_09_FNO"]
```

### 与Pipeline_ID集成

专门用于ID数据处理的管道：

```python
# Pipeline_ID使用示例
from src.pipeline import Pipeline_ID

# 创建管道实例
pipeline = Pipeline_ID()

# 运行对比学习预训练
results = pipeline.run_pretraining(
    config_path="configs/id_contrastive/production.yaml",
    checkpoint_callback=True,
    early_stopping=True
)

# 管道支持的操作
pipeline_ops = [
    "data_loading",      # ID数据加载
    "preprocessing",     # 信号预处理
    "windowing",        # 多窗口采样
    "training",         # 对比学习训练
    "evaluation",       # 性能评估
    "visualization"     # 结果可视化
]
```

### 与下游任务集成

预训练模型用于下游任务微调：

```python
# 1. 加载预训练模型
pretrain_checkpoint = "save/production/checkpoints/best.ckpt"
pretrained_task = ContrastiveIDTask.load_from_checkpoint(pretrain_checkpoint)

# 2. 提取特征编码器
encoder = pretrained_task.network

# 3. 创建下游任务（如分类）
from src.task_factory.task.classification import ClassificationTask
downstream_task = ClassificationTask(
    network=encoder,
    num_classes=10,
    freeze_encoder=False,  # 或True进行冻结微调
    # ... 其他参数
)

# 4. 微调训练
trainer = pl.Trainer(max_epochs=20)
trainer.fit(downstream_task, downstream_dataloader)
```

### 配置系统集成

利用PHM-Vibench v5.0统一配置系统：

```python
from src.configs import load_config

# 1. 基础配置加载
config = load_config("configs/id_contrastive/production.yaml")

# 2. 预设配置使用
config = load_config("contrastive_id_preset", {
    "data.batch_size": 64,
    "task.temperature": 0.05
})

# 3. 动态配置构建
config = load_config("debug_preset").copy().update({
    "trainer.epochs": 10,
    "model.d_model": 512
})

# 4. 链式配置覆盖
config = (load_config("base_config")
          .update_from_dict(user_overrides)
          .update_from_file("experiment_specific.yaml"))
```

## 故障排除

### 常见错误及解决方案

#### 错误1: CUDA内存不足

**错误信息**: `RuntimeError: CUDA out of memory`

**解决方案**:
```yaml
# 方案1: 减少批量大小
data:
  batch_size: 16  # 从32减少到16

# 方案2: 减少模型维度  
model:
  d_model: 128    # 从256减少到128

# 方案3: 启用梯度累积
trainer:
  accumulate_grad_batches: 4  # 模拟大批量

# 方案4: 使用混合精度
trainer:
  precision: 16   # 减少50%内存使用
```

#### 错误2: 数据加载失败

**错误信息**: `FileNotFoundError: metadata file not found`

**解决方案**:
```bash
# 1. 检查数据目录结构
ls -la data/
# 应该包含: metadata_*.xlsx 文件

# 2. 检查metadata文件路径
python -c "
import os
data_dir = 'data'
metadata_file = 'metadata_6_1.xlsx' 
full_path = os.path.join(data_dir, metadata_file)
print(f'Checking: {full_path}')
print(f'Exists: {os.path.exists(full_path)}')
"

# 3. 验证数据集ID
python -c "
from src.data_factory.dataset.ID_dataset import ID_dataset
dataset = ID_dataset('data', 'metadata_6_1.xlsx')
print(f'Dataset length: {len(dataset)}')
print(f'Available IDs: {dataset.get_available_ids()[:5]}')  # 显示前5个ID
"
```

#### 错误3: 训练不收敛

**症状**: 损失不下降或准确率停滞

**诊断和解决**:
```python
# 诊断脚本
import torch
import matplotlib.pyplot as plt

# 1. 检查梯度
def check_gradients(model):
    total_norm = 0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm:.6f}, Params with grad: {param_count}")
    return total_norm

# 2. 检查权重更新
def check_weight_updates(model_before, model_after):
    updates = []
    for (name, p_before), (_, p_after) in zip(
        model_before.named_parameters(), 
        model_after.named_parameters()
    ):
        update = torch.norm(p_after - p_before).item()
        updates.append((name, update))
        print(f"{name}: {update:.8f}")
    return updates

# 解决方案
solutions = {
    "学习率过大": "减小lr到5e-4或1e-4",
    "温度过低": "增大temperature到0.1",
    "梯度消失": "检查网络深度，考虑残差连接",
    "梯度爆炸": "减小gradient_clip_val到0.5",
    "数据标准化问题": "确保normalization=true"
}
```

#### 错误4: 验证性能异常

**症状**: 验证准确率远低于随机水平

**解决步骤**:
```python
# 1. 检查正负样本分布
def analyze_batch_distribution(dataloader):
    for batch in dataloader:
        anchor = batch['anchor']
        positive = batch['positive'] 
        print(f"Batch size: {len(batch['ids'])}")
        print(f"Anchor shape: {anchor.shape}")
        print(f"Positive shape: {positive.shape}")
        
        # 检查是否为同一ID的不同窗口
        for i, sample_id in enumerate(batch['ids']):
            print(f"Sample {i}: ID={sample_id}")
        break

# 2. 检查InfoNCE实现
def validate_infonce_implementation():
    # 创建简单测试用例
    batch_size = 4
    d_model = 256
    
    anchor = torch.randn(batch_size, d_model)
    positive = torch.randn(batch_size, d_model)
    
    # 手动计算InfoNCE
    task = ContrastiveIDTask(...)
    loss = task.infonce_loss(anchor, positive)
    print(f"InfoNCE loss: {loss.item()}")
    
    # 检查相似度矩阵
    anchor_norm = F.normalize(anchor, dim=1)
    positive_norm = F.normalize(positive, dim=1)
    sim_matrix = torch.mm(anchor_norm, positive_norm.t()) / task.temperature
    print(f"Similarity matrix diagonal: {torch.diag(sim_matrix)}")
```

### 性能诊断工具

#### GPU使用监控

```bash
# 实时GPU监控
nvidia-smi -l 1

# GPU内存使用分析
python -c "
import torch
print(f'Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'Current reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
"
```

#### 训练速度分析

```python
# 性能分析脚本
import time
import torch.profiler as profiler

def profile_training_step(model, batch):
    with profiler.profile(
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(10):
            # 模拟训练步骤
            model.training_step(batch, step)
            prof.step()
    
    # 分析结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## 高级用法

### 自定义配置覆盖

#### 1. 环境变量配置

```bash
# 设置环境变量
export CONTRASTIVE_TEMPERATURE=0.05
export CONTRASTIVE_BATCH_SIZE=128
export CONTRASTIVE_EPOCHS=200

# 在配置中使用
python -c "
import os
from src.configs import load_config

config = load_config('configs/id_contrastive/production.yaml', {
    'task.temperature': float(os.getenv('CONTRASTIVE_TEMPERATURE', 0.07)),
    'data.batch_size': int(os.getenv('CONTRASTIVE_BATCH_SIZE', 32)),
    'trainer.epochs': int(os.getenv('CONTRASTIVE_EPOCHS', 50))
})
"
```

#### 2. 条件配置

```python
# 根据条件动态调整配置
import torch
from src.configs import load_config

def create_adaptive_config():
    # 检测硬件环境
    gpu_count = torch.cuda.device_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_count > 0 else 0
    
    # 基础配置
    config = load_config("configs/id_contrastive/production.yaml")
    
    # 根据GPU内存调整批量大小
    if gpu_memory > 24:  # > 24GB
        config.update({"data.batch_size": 128, "model.d_model": 512})
    elif gpu_memory > 12:  # 12-24GB
        config.update({"data.batch_size": 64, "model.d_model": 256})
    else:  # < 12GB
        config.update({"data.batch_size": 32, "model.d_model": 128})
    
    # 多GPU配置
    if gpu_count > 1:
        config.update({
            "trainer.devices": list(range(gpu_count)),
            "trainer.strategy": "ddp",
            "data.batch_size": config.data.batch_size * gpu_count
        })
    
    return config
```

### 批量实验管理

#### 实验矩阵生成

```python
# scripts/generate_experiment_matrix.py
import itertools
import json
from pathlib import Path

def generate_ablation_experiments():
    """生成完整消融实验矩阵"""
    
    # 实验变量定义
    variables = {
        "temperature": [0.01, 0.05, 0.07, 0.1, 0.5],
        "batch_size": [16, 32, 64, 128],
        "window_size": [512, 1024, 2048, 4096],
        "d_model": [128, 256, 512],
        "lr": [1e-4, 5e-4, 1e-3, 5e-3]
    }
    
    # 生成所有组合（警告：可能很多！）
    experiments = []
    for values in itertools.product(*variables.values()):
        experiment = dict(zip(variables.keys(), values))
        
        # 添加合理性检查
        if experiment["batch_size"] * experiment["window_size"] > 200000:
            continue  # 跳过内存需求过高的组合
            
        experiments.append(experiment)
    
    print(f"Generated {len(experiments)} experiments")
    
    # 保存实验矩阵
    output_file = "experiments/ablation_matrix.json"
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    return experiments

def run_experiment_batch(experiments_file, max_parallel=4):
    """并行运行实验批次"""
    import subprocess
    import concurrent.futures
    
    with open(experiments_file, 'r') as f:
        experiments = json.load(f)
    
    def run_single_experiment(exp_id, params):
        cmd = [
            "python", "main.py",
            "--config", "configs/id_contrastive/ablation.yaml"
        ]
        
        # 添加参数覆盖
        for key, value in params.items():
            cmd.extend(["--override", f"task.{key}={value}" if key in ["temperature", "lr"] 
                       else f"data.{key}={value}" if key in ["batch_size", "window_size"]
                       else f"model.{key}={value}"])
        
        # 添加实验名称
        exp_name = f"exp_{exp_id}_" + "_".join(f"{k}_{v}" for k, v in params.items())
        cmd.extend(["--override", f"environment.experiment_name={exp_name}"])
        
        print(f"Running experiment {exp_id}: {exp_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return exp_id, result.returncode, result.stdout, result.stderr
    
    # 并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(run_single_experiment, i, exp) 
            for i, exp in enumerate(experiments)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            exp_id, returncode, stdout, stderr = future.result()
            if returncode == 0:
                print(f"✓ Experiment {exp_id} completed successfully")
            else:
                print(f"✗ Experiment {exp_id} failed: {stderr}")
```

### 参数调优策略

#### 1. 贝叶斯优化

```python
# 使用Optuna进行超参数优化
import optuna
from src.configs import load_config

def objective(trial):
    # 定义搜索空间
    temperature = trial.suggest_float("temperature", 0.01, 0.5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    
    # 创建配置
    config = load_config("configs/id_contrastive/ablation.yaml", {
        "task.temperature": temperature,
        "data.batch_size": batch_size,
        "task.lr": lr,
        "model.d_model": d_model,
        "trainer.epochs": 10,  # 快速评估
        "environment.experiment_name": f"optuna_trial_{trial.number}"
    })
    
    # 运行实验（简化版）
    try:
        result = run_experiment(config)
        return result['val_contrastive_acc']  # 优化目标
    except Exception as e:
        return 0.0  # 失败案例返回最低分

# 运行优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
print("Best value:", study.best_value)
```

#### 2. 学习率调度

```python
# 自定义学习率调度器
class ContrastiveLRScheduler:
    def __init__(self, optimizer, warmup_steps=1000, max_lr=1e-3, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0
    
    def step(self, val_acc=None):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # 线性预热
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # 余弦退火
            progress = (self.step_count - self.warmup_steps) / (10000 - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + cos(pi * progress))
        
        # 应用学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# 集成到训练中
scheduler = ContrastiveLRScheduler(optimizer)
for epoch in range(epochs):
    for batch in dataloader:
        # 训练步骤
        loss = model.training_step(batch)
        optimizer.step()
        scheduler.step()
```

### 特征分析与可视化

#### 1. 特征质量评估

```python
# scripts/feature_analysis.py
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_feature_quality(model, dataloader):
    """评估学到的特征质量"""
    model.eval()
    
    features = []
    labels = []
    ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 提取特征
            anchor_feat = model.network(batch['anchor'])
            positive_feat = model.network(batch['positive'])
            
            features.extend([anchor_feat, positive_feat])
            labels.extend([batch['labels'], batch['labels']])  # 假设有标签
            ids.extend([batch['ids'], batch['ids']])
    
    features = torch.cat(features).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    # 1. 轮廓系数 - 衡量聚类质量
    silhouette = silhouette_score(features, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # 2. 同类样本相似度 vs 异类样本相似度
    intra_class_sim = []
    inter_class_sim = []
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            sim = np.dot(features[i], features[j]) / (
                np.linalg.norm(features[i]) * np.linalg.norm(features[j])
            )
            if labels[i] == labels[j]:
                intra_class_sim.append(sim)
            else:
                inter_class_sim.append(sim)
    
    print(f"Intra-class similarity: {np.mean(intra_class_sim):.4f} ± {np.std(intra_class_sim):.4f}")
    print(f"Inter-class similarity: {np.mean(inter_class_sim):.4f} ± {np.std(inter_class_sim):.4f}")
    
    return {
        'silhouette_score': silhouette,
        'intra_class_similarity': np.mean(intra_class_sim),
        'inter_class_similarity': np.mean(inter_class_sim),
        'features': features,
        'labels': labels
    }

def visualize_feature_space(features, labels, method='tsne', save_path=None):
    """可视化特征空间分布"""
    
    if method == 'tsne':
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
    elif method == 'pca':
        # PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1], 
            c=[color], 
            label=f'Class {label}',
            alpha=0.6,
            s=50
        )
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Feature Space Visualization ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_evolution(checkpoints, dataloader):
    """分析特征在训练过程中的演化"""
    
    evolution_data = []
    
    for epoch, checkpoint_path in enumerate(checkpoints):
        model = ContrastiveIDTask.load_from_checkpoint(checkpoint_path)
        metrics = evaluate_feature_quality(model, dataloader)
        
        evolution_data.append({
            'epoch': epoch,
            'silhouette_score': metrics['silhouette_score'],
            'intra_class_sim': metrics['intra_class_similarity'],
            'inter_class_sim': metrics['inter_class_similarity']
        })
    
    # 可视化演化过程
    df = pd.DataFrame(evolution_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(df['epoch'], df['silhouette_score'], 'b-o')
    axes[0].set_title('Silhouette Score Evolution')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Silhouette Score')
    
    axes[1].plot(df['epoch'], df['intra_class_sim'], 'g-o', label='Intra-class')
    axes[1].plot(df['epoch'], df['inter_class_sim'], 'r-o', label='Inter-class')
    axes[1].set_title('Similarity Evolution')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].legend()
    
    axes[2].plot(df['epoch'], df['intra_class_sim'] - df['inter_class_sim'], 'purple', marker='o')
    axes[2].set_title('Similarity Gap Evolution')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Intra - Inter Similarity')
    
    plt.tight_layout()
    plt.show()
    
    return evolution_data
```

## API参考

### ContrastiveIDTask类

```python
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask

# 创建任务实例
task = ContrastiveIDTask(
    network=network,
    args_data=data_config,
    args_model=model_config,
    args_task=task_config,
    args_trainer=trainer_config,
    args_environment=env_config,
    metadata=metadata
)
```

#### 核心方法

**`prepare_batch(batch_data: List[Tuple]) -> Dict[str, torch.Tensor]`**

准备对比学习的批处理数据，生成正样本对：

```python
# 输入: 原始批次数据
batch_data = [
    ("ID_001", signal_array_1, metadata_1),
    ("ID_002", signal_array_2, metadata_2),
    # ...
]

# 输出: 对比学习数据字典
batch = {
    'anchor': torch.tensor([...]),    # 锚点窗口 [B, L, 1]
    'positive': torch.tensor([...]),  # 正样本窗口 [B, L, 1] 
    'ids': ["ID_001", "ID_002", ...]  # 样本ID列表
}
```

**`infonce_loss(z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor`**

计算InfoNCE对比损失：

```python
# 数学公式
# InfoNCE = -log(exp(sim(anchor, positive)/τ) / Σexp(sim(anchor, all)/τ))

loss = task.infonce_loss(anchor_features, positive_features)
# 返回标量损失值
```

**`compute_accuracy(z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor`**

计算对比学习Top-1准确率：

```python
accuracy = task.compute_accuracy(anchor_features, positive_features)
# 返回0-1之间的准确率值
```

#### 继承的方法（来自BaseIDTask）

**`process_sample(data_array: np.ndarray, metadata: Dict) -> np.ndarray`**
- 信号预处理（标准化、截断等）

**`create_windows(data: np.ndarray, strategy: str, num_window: int) -> List[np.ndarray]`**
- 窗口化采样（支持random/sequential/evenly_spaced策略）

### 配置系统API

```python
from src.configs import load_config

# 1. 基础配置加载
config = load_config("configs/id_contrastive/production.yaml")

# 2. 预设配置 + 参数覆盖
config = load_config("contrastive_id_preset", {
    "task.temperature": 0.05,
    "trainer.epochs": 100
})

# 3. 链式配置构建
config = (load_config("base_config")
          .copy()
          .update({"data.batch_size": 64})
          .update_from_file("overrides.yaml"))

# 4. 条件配置
if torch.cuda.is_available():
    config.update({"trainer.accelerator": "gpu", "trainer.precision": 16})
else:
    config.update({"trainer.accelerator": "cpu", "trainer.precision": 32})
```

### 工厂系统API

```python
# 任务工厂
from src.task_factory import create_task
task = create_task("pretrain", "contrastive_id", 
                   network, data_config, model_config, 
                   task_config, trainer_config, env_config, metadata)

# 数据工厂  
from src.data_factory import create_data_loader
dataloader = create_data_loader("id", "ID_dataset", data_config, metadata)

# 模型工厂
from src.model_factory import create_model
network = create_model("ISFM", "M_01_ISFM", model_config)
```

## 性能优化

### 计算资源优化

#### GPU内存管理

```python
# 动态内存分配策略
def optimize_memory_config(available_memory_gb):
    """根据可用GPU内存动态调整配置"""
    
    if available_memory_gb >= 24:  # RTX 4090, A100等
        return {
            "data.batch_size": 128,
            "model.d_model": 512,
            "trainer.precision": 16,
            "trainer.accumulate_grad_batches": 1
        }
    elif available_memory_gb >= 12:  # RTX 3080Ti, RTX 4080等
        return {
            "data.batch_size": 64,
            "model.d_model": 256,
            "trainer.precision": 16,
            "trainer.accumulate_grad_batches": 2
        }
    elif available_memory_gb >= 8:  # RTX 3070, RTX 4060Ti等
        return {
            "data.batch_size": 32,
            "model.d_model": 256,
            "trainer.precision": 16,
            "trainer.accumulate_grad_batches": 4
        }
    else:  # < 8GB
        return {
            "data.batch_size": 16,
            "model.d_model": 128,
            "trainer.precision": 16,
            "trainer.accumulate_grad_batches": 8
        }

# 使用示例
import torch
gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
memory_config = optimize_memory_config(gpu_memory)
config.update(memory_config)
```

#### CPU优化

```python
# CPU资源配置优化
import os
import psutil

def optimize_cpu_config():
    """优化CPU相关配置"""
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    return {
        "data.num_workers": min(cpu_count, 8),  # 避免过多进程
        "data.pin_memory": True,                 # 加速GPU传输
        "data.persistent_workers": True,        # 保持worker进程
        "trainer.enable_checkpointing": memory_gb > 16  # 大内存时启用
    }
```

#### 多GPU扩展

```python
# 多GPU训练配置
def setup_multi_gpu_config():
    """配置多GPU训练"""
    gpu_count = torch.cuda.device_count()
    
    if gpu_count > 1:
        return {
            "trainer.devices": list(range(gpu_count)),
            "trainer.strategy": "ddp",           # 分布式数据并行
            "trainer.sync_batchnorm": True,      # 批归一化同步
            "data.batch_size": 32 * gpu_count,   # 按GPU数量缩放批量
            "task.lr": 1e-3 * gpu_count,        # 线性缩放学习率
        }
    else:
        return {"trainer.devices": 1}

# 应用多GPU配置
multi_gpu_config = setup_multi_gpu_config()
config.update(multi_gpu_config)
```

### 训练速度优化

#### 数据I/O优化

```python
# 数据加载性能调优
def optimize_dataloader_config(dataset_size, signal_length):
    """根据数据特征优化数据加载"""
    
    # 基于数据集大小调整缓存策略
    if dataset_size < 10000:  # 小数据集
        prefetch_factor = 4
        num_workers = 2
    elif dataset_size < 100000:  # 中等数据集  
        prefetch_factor = 8
        num_workers = 4
    else:  # 大数据集
        prefetch_factor = 16
        num_workers = 8
    
    # 基于信号长度调整预处理并行度
    if signal_length > 50000:  # 长信号
        persistent_workers = True
        pin_memory = True
    else:  # 短信号
        persistent_workers = False
        pin_memory = False
        
    return {
        "data.num_workers": num_workers,
        "data.prefetch_factor": prefetch_factor,
        "data.persistent_workers": persistent_workers,
        "data.pin_memory": pin_memory
    }
```

#### 模型编译优化

```python
# PyTorch 2.0+ 编译优化
def enable_torch_compile(model, mode="default"):
    """启用PyTorch编译优化"""
    
    # 编译模式选择
    compile_modes = {
        "default": {"mode": "default"},      # 平衡速度和内存
        "reduce-overhead": {"mode": "reduce-overhead"},  # 减少开销
        "max-autotune": {"mode": "max-autotune"}         # 最大优化
    }
    
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        compiled_model = torch.compile(model, **compile_modes[mode])
        return compiled_model
    else:
        return model

# 在训练中使用
task.network = enable_torch_compile(task.network, mode="reduce-overhead")
```

### 收敛性优化

#### 学习率调度策略

```python
# 自适应学习率调度
class ContrastiveOptimizer:
    def __init__(self, model_parameters, config):
        self.config = config
        
        # 优化器选择
        if config.task.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model_parameters,
                lr=config.task.lr,
                weight_decay=config.task.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif config.task.optimizer == "lion":  # 新兴优化器
            from lion_pytorch import Lion
            self.optimizer = Lion(
                model_parameters,
                lr=config.task.lr,
                weight_decay=config.task.weight_decay
            )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
    
    def _create_scheduler(self):
        if self.config.task.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.trainer.epochs,
                eta_min=self.config.task.lr * 0.01
            )
        elif self.config.task.lr_scheduler == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.task.lr,
                total_steps=self.config.trainer.epochs,
                pct_start=0.1,  # 10%时间预热
                anneal_strategy='cos'
            )
        else:
            return None
```

#### 损失函数改进

```python
# 改进的InfoNCE实现
class ImprovedInfoNCE(nn.Module):
    def __init__(self, temperature=0.07, use_cosine_sim=True, 
                 hard_negative_weight=0.0):
        super().__init__()
        self.temperature = temperature
        self.use_cosine_sim = use_cosine_sim
        self.hard_negative_weight = hard_negative_weight
    
    def forward(self, anchor, positive, negatives=None):
        batch_size = anchor.size(0)
        
        if self.use_cosine_sim:
            # 余弦相似度
            anchor = F.normalize(anchor, dim=1)
            positive = F.normalize(positive, dim=1)
            
            # 正样本相似度
            pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)
            
            # 负样本相似度（批次内其他样本）
            neg_sim = torch.mm(anchor, positive.t())
            # 移除对角线（正样本）
            neg_sim = neg_sim.masked_fill(
                torch.eye(batch_size, device=anchor.device).bool(), 
                float('-inf')
            )
            
            # 合并正负样本相似度
            all_sim = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
            
        # InfoNCE损失
        labels = torch.zeros(batch_size, device=anchor.device, dtype=torch.long)
        loss = F.cross_entropy(all_sim, labels)
        
        # 困难负样本加权（可选）
        if self.hard_negative_weight > 0 and negatives is not None:
            hard_neg_loss = self._compute_hard_negative_loss(anchor, negatives)
            loss = loss + self.hard_negative_weight * hard_neg_loss
            
        return loss
```

### 质量监控与分析

#### 实时质量监控

```python
# 训练质量监控回调
class ContrastiveQualityMonitor(pl.Callback):
    def __init__(self, monitor_frequency=100):
        super().__init__()
        self.monitor_frequency = monitor_frequency
        self.step_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        
        if self.step_count % self.monitor_frequency == 0:
            # 监控特征质量
            self._monitor_feature_quality(pl_module, batch)
            
            # 监控梯度健康度
            self._monitor_gradient_health(pl_module)
            
            # 监控相似度分布
            self._monitor_similarity_distribution(pl_module, batch)
    
    def _monitor_feature_quality(self, model, batch):
        """监控特征表示质量"""
        with torch.no_grad():
            anchor_feat = model.network(batch['anchor'])
            positive_feat = model.network(batch['positive'])
            
            # 特征范数
            anchor_norm = torch.norm(anchor_feat, dim=1).mean()
            positive_norm = torch.norm(positive_feat, dim=1).mean()
            
            model.log("monitor/anchor_feat_norm", anchor_norm)
            model.log("monitor/positive_feat_norm", positive_norm)
            
            # 特征相似度统计
            cosine_sim = F.cosine_similarity(anchor_feat, positive_feat)
            model.log("monitor/positive_pair_similarity", cosine_sim.mean())
            model.log("monitor/positive_pair_similarity_std", cosine_sim.std())
    
    def _monitor_gradient_health(self, model):
        """监控梯度健康度"""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            model.log("monitor/gradient_norm", total_norm)
```

---

## 总结

本文档提供了PHM-Vibench ContrastiveID预训练任务的完整使用指南，涵盖了：

**🚀 快速开始**
- 4种配置场景：调试、生产、消融、跨数据集
- 即开即用的命令行示例
- 核心概念清晰说明

**⚙️ 配置详解** 
- 完整参数表格，包含调优建议
- 不同场景的最佳实践配置
- 硬件资源适配策略

**🔄 实验工作流**
- 标准化的实验流程
- 消融实验自动化脚本
- 跨数据集泛化评估

**🔧 集成指南**
- 与PHM-Vibench工厂系统集成
- Pipeline_ID无缝对接
- 下游任务微调流程

**🔍 故障排除**
- 常见错误诊断与解决
- 性能监控工具
- 收敛性分析方法

**🎯 高级用法**
- 批量实验管理
- 超参数自动调优
- 特征质量分析

**📊 性能优化**
- 内存、计算、I/O全方位优化
- 多GPU扩展策略
- 实时质量监控

该框架为工业设备振动信号分析提供了强大的对比学习预训练能力，支持从研究探索到生产部署的全流程需求。通过本指南，研究人员和工程师可以高效地开展相关工作，获得高质量的信号表征用于下游任务。

**技术特色**:
- ✅ **即插即用**: 配置驱动，无需修改代码
- ✅ **高度可扩展**: 支持30+工业数据集
- ✅ **性能优化**: GPU内存自适应，多GPU并行
- ✅ **质量保证**: 实时监控，自动诊断
- ✅ **标准接口**: 兼容PHM-Vibench生态系统

如有疑问或需要技术支持，请参考项目文档或提交Issue。