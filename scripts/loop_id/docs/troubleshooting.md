# ContrastiveIDTask 故障排除指南

ContrastiveIDTask常见问题的诊断与解决方案。

## 🎯 快速诊断

### 环境检查脚本

运行此脚本快速检查环境状态：

```python
def diagnose_environment():
    """完整环境诊断"""
    print("🔍 ContrastiveIDTask环境诊断")
    print("="*50)

    # Python环境
    import sys
    print(f"Python版本: {sys.version}")

    # PyTorch检查
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}, {props.total_memory/1e9:.1f}GB")
    except ImportError:
        print("❌ PyTorch未安装")

    # 核心模块检查
    try:
        from src.configs import load_config
        print("✅ 配置系统可用")
    except ImportError as e:
        print(f"❌ 配置系统导入失败: {e}")

    try:
        from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
        print("✅ ContrastiveIDTask可用")
    except ImportError as e:
        print(f"❌ ContrastiveIDTask导入失败: {e}")

    # 配置文件检查
    from pathlib import Path
    config_files = [
        "configs/id_contrastive/debug.yaml",
        "configs/id_contrastive/production.yaml"
    ]

    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} 不存在")

if __name__ == "__main__":
    diagnose_environment()
```

## 🚨 常见错误解决

### 1. 内存相关错误

#### 错误：CUDA out of memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**原因**: GPU内存不足，通常由批量大小过大或窗口大小过大引起

**解决方案**:

```yaml
# 方案1: 减小批量大小
data:
  batch_size: 16  # 从32或64减少到16

# 方案2: 减小窗口大小
data:
  window_size: 512  # 从1024减少到512

# 方案3: 使用CPU训练
trainer:
  devices: "cpu"
  precision: 32
```

**动态调整脚本**:
```python
def adjust_batch_size_for_memory():
    """根据GPU内存动态调整批量大小"""
    import torch

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:
            return 16  # 小于8GB使用小批量
        elif gpu_memory < 16:
            return 32  # 8-16GB使用中等批量
        else:
            return 64  # 大于16GB使用大批量
    else:
        return 8  # CPU模式使用最小批量
```

#### 错误：系统内存不足

**症状**:
```
OSError: [Errno 12] Cannot allocate memory
```

**解决方案**:
```yaml
# 减少数据加载进程
data:
  num_workers: 1  # 从4减少到1

# 启用延迟加载
data:
  lazy_loading: true
```

### 2. 数据加载错误

#### 错误：文件未找到

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/metadata_6_1.xlsx'
```

**诊断**:
```bash
# 检查当前目录
pwd

# 检查数据文件
ls -la data/
ls -la data/metadata_6_1.xlsx

# 检查配置中的路径
python -c "
from src.configs import load_config
config = load_config('configs/id_contrastive/debug.yaml')
print(f'数据目录: {config.data.data_dir}')
print(f'元数据文件: {config.data.metadata_file}')
"
```

**解决方案**:
```yaml
# 修正配置文件中的路径
data:
  data_dir: "/absolute/path/to/data"  # 使用绝对路径
  metadata_file: "metadata_6_1.xlsx"
```

#### 错误：数据格式不匹配

**症状**:
```
ValueError: Expected tensor of shape [batch, seq_len, channels], got [batch, channels, seq_len]
```

**解决方案**:
```python
# 在配置中添加数据预处理
data:
  preprocessing:
    transpose_channels: true  # 转置通道维度
    normalize: true           # 标准化
```

### 3. 模型相关错误

#### 错误：模型维度不匹配

**症状**:
```
RuntimeError: size mismatch, m1: [32 x 256], m2: [512 x 256]
```

**解决方案**:
```yaml
# 确保模型输入维度与数据匹配
model:
  input_dim: 1      # 与数据通道数一致
  d_model: 256      # 与预期特征维度一致
```

#### 错误：模型未注册

**症状**:
```
KeyError: 'contrastive_id' not found in task registry
```

**解决方案**:
```python
# 确保导入了模型定义
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask

# 检查注册状态
from src.task_factory import TASK_REGISTRY
print("已注册任务:", list(TASK_REGISTRY.keys()))
```

### 4. 训练相关错误

#### 错误：损失为NaN

**症状**:
```
Training loss: nan
```

**原因与解决**:

1. **学习率过大**:
```yaml
task:
  lr: 0.0001  # 从0.001降低到0.0001
```

2. **温度参数过小**:
```yaml
task:
  temperature: 0.1  # 从0.01增加到0.1
```

3. **梯度爆炸**:
```yaml
trainer:
  gradient_clip_val: 1.0  # 添加梯度裁剪
```

#### 错误：训练不收敛

**症状**: 损失不下降，准确率始终很低

**诊断脚本**:
```python
def diagnose_training_issues(model, dataloader):
    """诊断训练问题"""
    model.eval()

    # 检查数据质量
    batch = next(iter(dataloader))
    prepared_batch = model.prepare_batch(batch)

    print(f"批量形状 - 锚点: {prepared_batch['anchor'].shape}")
    print(f"批量形状 - 正样本: {prepared_batch['positive'].shape}")

    # 检查特征分布
    with torch.no_grad():
        z_anchor = model.model(prepared_batch['anchor'])
        z_positive = model.model(prepared_batch['positive'])

        print(f"特征均值 - 锚点: {z_anchor.mean().item():.4f}")
        print(f"特征标准差 - 锚点: {z_anchor.std().item():.4f}")
        print(f"特征均值 - 正样本: {z_positive.mean().item():.4f}")
        print(f"特征标准差 - 正样本: {z_positive.std().item():.4f}")

        # 检查相似度分布
        sim_matrix = torch.mm(F.normalize(z_anchor, dim=1),
                              F.normalize(z_positive, dim=1).t())
        pos_sim = torch.diag(sim_matrix).mean()
        neg_sim = (sim_matrix.sum() - torch.diag(sim_matrix).sum()) / (sim_matrix.numel() - sim_matrix.size(0))

        print(f"正样本平均相似度: {pos_sim.item():.4f}")
        print(f"负样本平均相似度: {neg_sim.item():.4f}")
```

**解决方案**:

1. **调整温度参数**:
```yaml
task:
  temperature: 0.07  # 尝试0.05, 0.1, 0.2
```

2. **增加批量大小**:
```yaml
data:
  batch_size: 64  # 更多负样本
```

3. **检查数据预处理**:
```yaml
data:
  normalization: true  # 确保数据标准化
  window_sampling_strategy: "random"  # 确保随机性
```

### 5. 配置相关错误

#### 错误：配置参数缺失

**症状**:
```
ValueError: 缺少必需字段: data.data_dir
```

**解决方案**:
```yaml
# 确保配置包含所有必需字段
data:
  data_dir: "data"
  metadata_file: "metadata_6_1.xlsx"
  factory_name: "id"
  dataset_name: "ID_dataset"

model:
  type: "ISFM"
  factory_name: "ISFM"

task:
  name: "contrastive_id"
```

#### 错误：配置加载失败

**症状**:
```
FileNotFoundError: configs/id_contrastive/debug.yaml not found
```

**检查与修复**:
```bash
# 检查配置文件存在
find . -name "*.yaml" | grep contrastive

# 创建缺失的配置文件
mkdir -p configs/id_contrastive
cp configs/demo/debug.yaml configs/id_contrastive/debug.yaml
```

## 🔧 调试工具

### 1. 逐步调试脚本

```python
def debug_step_by_step():
    """逐步调试ContrastiveIDTask"""

    # 步骤1: 配置加载
    try:
        from src.configs import load_config
        config = load_config('configs/id_contrastive/debug.yaml')
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return

    # 步骤2: 数据集创建
    try:
        from src.data_factory import create_dataset
        dataset = create_dataset(**config.data.to_dict())
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return

    # 步骤3: 任务创建
    try:
        from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
        task = ContrastiveIDTask(**config.to_dict())
        print("✅ 任务创建成功")
    except Exception as e:
        print(f"❌ 任务创建失败: {e}")
        return

    # 步骤4: 批次处理测试
    try:
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        prepared_batch = task.prepare_batch(batch)
        print("✅ 批次处理成功")
        print(f"   锚点形状: {prepared_batch['anchor'].shape}")
        print(f"   正样本形状: {prepared_batch['positive'].shape}")
    except Exception as e:
        print(f"❌ 批次处理失败: {e}")
        return

    # 步骤5: 前向传播测试
    try:
        z_anchor = task.model(prepared_batch['anchor'])
        z_positive = task.model(prepared_batch['positive'])
        loss = task.infonce_loss(z_anchor, z_positive)
        accuracy = task.compute_accuracy(z_anchor, z_positive)
        print("✅ 前向传播成功")
        print(f"   损失值: {loss.item():.4f}")
        print(f"   准确率: {accuracy.item():.2%}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return

    print("\n🎉 所有组件运行正常！")

if __name__ == "__main__":
    debug_step_by_step()
```

### 2. 性能监控脚本

```python
import psutil
import torch
import time

def monitor_training_performance(task, dataloader, num_batches=10):
    """监控训练性能指标"""

    # GPU内存监控
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

    # CPU和系统内存监控
    process = psutil.Process()
    initial_cpu_memory = process.memory_info().rss / 1024 / 1024  # MB

    task.train()
    times = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        start_time = time.time()

        # 前向传播
        prepared_batch = task.prepare_batch(batch)
        z_anchor = task.model(prepared_batch['anchor'])
        z_positive = task.model(prepared_batch['positive'])
        loss = task.infonce_loss(z_anchor, z_positive)

        # 反向传播
        loss.backward()

        batch_time = time.time() - start_time
        times.append(batch_time)

        # 内存使用
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        cpu_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"Batch {i+1}/{num_batches}:")
        print(f"  时间: {batch_time:.3f}s")
        print(f"  损失: {loss.item():.4f}")
        print(f"  CPU内存: {cpu_memory:.1f}MB")
        if torch.cuda.is_available():
            print(f"  GPU内存: {gpu_memory:.1f}MB")
            print(f"  峰值GPU内存: {peak_gpu_memory:.1f}MB")
        print()

    # 统计信息
    avg_time = sum(times) / len(times)
    print(f"平均批次时间: {avg_time:.3f}s")
    print(f"预估每epoch时间: {avg_time * len(dataloader) / 60:.1f}分钟")
```

### 3. 数据质量检查

```python
def check_data_quality(dataset):
    """检查数据集质量"""
    print("📊 数据质量检查")
    print("-" * 40)

    # 采样检查
    sample_ids = []
    for i in range(min(100, len(dataset))):
        sample_id, _, metadata = dataset[i]
        sample_ids.append(sample_id)

    # 检查ID分布
    from collections import Counter
    id_counts = Counter(sample_ids)
    print(f"样本总数: {len(sample_ids)}")
    print(f"唯一ID数: {len(id_counts)}")
    print(f"平均每ID样本数: {len(sample_ids) / len(id_counts):.2f}")
    print(f"ID分布 (前10): {dict(list(id_counts.most_common(10)))}")

    # 检查数据完整性
    missing_data = 0
    for sample_id in list(id_counts.keys())[:10]:
        try:
            data = dataset._get_data_for_id(sample_id)
            if data is None or len(data) == 0:
                missing_data += 1
        except:
            missing_data += 1

    print(f"缺失数据ID数: {missing_data}/10")

    if missing_data > 0:
        print("⚠️  发现数据完整性问题，请检查数据文件")
    else:
        print("✅ 数据完整性检查通过")
```

## 📋 故障排除清单

### 启动前检查

- [ ] Python版本 ≥ 3.8
- [ ] PyTorch版本 = 2.6.0
- [ ] CUDA版本兼容 (如使用GPU)
- [ ] 数据文件存在且可读
- [ ] 配置文件语法正确
- [ ] 必需字段完整

### 运行时检查

- [ ] GPU内存充足 (至少2GB空闲)
- [ ] 系统内存充足 (至少8GB)
- [ ] 磁盘空间充足 (结果保存)
- [ ] 数据加载正常
- [ ] 模型初始化成功

### 训练过程检查

- [ ] 损失值合理 (0.1-10范围)
- [ ] 准确率逐步提升
- [ ] 内存使用稳定
- [ ] 无警告或异常

## 🆘 获取帮助

### 日志分析

1. **查看详细错误信息**:
```bash
python main.py --config configs/id_contrastive/debug.yaml --verbose
```

2. **保存日志到文件**:
```bash
python main.py --config configs/id_contrastive/debug.yaml 2>&1 | tee training.log
```

3. **分析日志模式**:
```bash
grep "ERROR\|WARNING" training.log
grep "loss\|accuracy" training.log
```

### 联系支持

- **GitHub Issues**: [PHM-Vibench Issues](https://github.com/your-repo/issues)
- **文档**: [technical_guide.md](technical_guide.md)
- **API参考**: [api_reference.md](api_reference.md)

---

**更新时间**: 2024年9月
**适用版本**: ContrastiveIDTask v1.0