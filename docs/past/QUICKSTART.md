# PHM-Vibench 快速开始指南

> ⚡ **5分钟上手** PHM-Vibench，立即开始工业信号分析和故障诊断研究

## 🎯 适用对象

- **PHM基础模型开发者**: 想要使用或改进工业信号基础模型
- **故障诊断研究者**: 需要快速验证算法效果
- **工程师**: 希望应用PHM技术到实际问题

## 🚀 超快开始 (3分钟)

### 第一步：运行快速示例

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/PHM-Vibench.git
cd PHM-Vibench

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行归档快速示例
python docs/past/examples/quickstart.py
```

就这么简单！该示例将运行两个实验：
- 🔰 基础实验：ResNet1D + 分类任务
- 🚀 进阶实验：ISFM基础模型

### 第二步：理解输出

实验完成后，你会看到：
```
✅ 基础ResNet1D 实验完成!
✅ 进阶ISFM 实验完成!

📈 结果总结:
   - 基础实验准确率: 95.2%
   - 进阶实验准确率: 97.8%
```

## 📊 三大核心概念 

### 1. 数据体系：三文件架构

PHM-Vibench使用统一的三文件数据格式：

```
data/
├── metadata.xlsx    # 📋 数据索引（核心）
├── data.h5         # 📊 信号数据
└── corpus.xlsx     # 📝 文本描述（可选）
```

**关键理解**：
- `metadata.xlsx` 是一切的核心，包含每个样本的所有元信息
- `Id` 字段链接三个文件
- `data.h5` 按 Id 存储实际信号数据

### 2. 任务类型：从简单到复杂

| 任务类型 | 简称 | 描述 | 适用场景 |
|---------|------|------|----------|
| **分类** | `classification` | 故障类型分类 | ✅ **新手推荐** |
| **领域泛化** | `DG` | 单域到单域 | 🏃‍♂️ 进阶 |
| **跨数据集** | `CDDG` | 跨数据集泛化 | 🚀 高级 |
| **少样本** | `FS` | 少样本学习 | 🧠 研究级 |

**建议学习路径**：`classification` → `DG` → `CDDG` → `FS`

### 3. 模型架构：传统到基础模型

#### 传统模型（快速上手）
```yaml
model:
  name: "ResNet1D"      # 经典CNN
  type: "CNN"           
  depth: 18
  num_classes: 4
```

#### ISFM基础模型（推荐）
```yaml
model:
  name: "M_01_ISFM"         # 工业信号基础模型
  type: "ISFM"
  embedding: "E_01_HSE"     # 层次信号嵌入
  backbone: "B_08_PatchTST" # Transformer骨干
  task_head: "H_01_Linear_cla" # 任务头
```

## ⚙️ 配置文件详解

### 最小配置模板

```yaml
# configs/my_experiment.yaml
environment:
  WANDB_MODE: "disabled"    # 简化输出
  seed: 42                  # 可重现
  iterations: 1

data:
  data_dir: "./data"
  metadata_file: "metadata.xlsx"
  batch_size: 32
  window_size: 1024

model:
  name: "ResNet1D"
  type: "CNN"
  num_classes: 4

task:
  name: "classification"
  type: "DG"
  epochs: 50
  lr: 0.001

trainer:
  name: "Default_trainer"
  num_epochs: 50
  gpus: 1
```

### 运行你的配置

```bash
python main.py --config_path configs/my_experiment.yaml
```

## 🛠️ 常见使用场景

### 场景1：验证新算法

```python
# 1. 修改模型配置
model:
  name: "YourNewModel"
  type: "CNN"  # 或 ISFM, RNN, Transformer
  # ... 你的参数

# 2. 运行实验
python main.py --config_path your_config.yaml

# 3. 查看结果
# save/metadata_xxx/YourNewModel/results/
```

### 场景2：跨数据集验证

```yaml
task:
  type: "CDDG"                    # 跨数据集
  source_domain_id: [1, 5, 6]    # 训练数据集
  target_domain_id: [19]         # 测试数据集
```

### 场景3：少样本学习

```yaml
task:
  type: "FS"                     # Few-Shot
  num_support: 5                 # 支撑样本数
  num_query: 15                  # 查询样本数
  num_episodes: 1000             # 训练episodes
```

## 🏗️ 自定义开发

### 添加新模型

1. **创建模型文件**：
```python
# src/model_factory/YourType/YourModel.py
class Model(nn.Module):  # 必须命名为 Model
    def __init__(self, args_m, metadata=None):
        # 你的实现
    
    def forward(self, x):
        # 你的前向传播
```

2. **更新配置**：
```yaml
model:
  name: "YourModel"
  type: "YourType"
  # 你的参数会自动传给 __init__
```

### 添加新数据集

1. **创建Reader**：
```python
# src/data_factory/reader/RM_XXX_YourDataset.py
class RM_XXX_YourDataset:
    def read(self, file_path, args_data):
        # 返回标准格式数据
        return data_array  # shape: (L, C)
```

2. **更新元数据**：在Excel中添加数据集信息

## 📈 理解实验结果

### 结果目录结构
```
save/
└── metadata_xxx/
    └── ModelName/
        └── TaskType_TrainerName_timestamp/
            ├── checkpoints/     # 模型权重
            ├── metrics.json     # 性能指标
            ├── log.txt         # 训练日志  
            └── config.yaml     # 实验配置
```

### 关键指标解读

| 指标 | 含义 | 期望值 |
|------|------|--------|
| **Accuracy** | 分类准确率 | >90% (良好) |
| **F1-Score** | 平衡精确率和召回率 | >0.9 (良好) |
| **Loss** | 训练损失 | 持续下降 |
| **Val_Loss** | 验证损失 | 不应持续上升 |

## 🐛 故障排除

### 常见问题

#### 1. 导入错误
```bash
ImportError: No module named 'src'
```
**解决**：确保在项目根目录运行
```bash
cd PHM-Vibench  # 确保在根目录
python docs/past/examples/quickstart.py
```

#### 2. 数据文件不存在
```bash
FileNotFoundError: metadata_dummy.csv
```
**解决**：创建dummy数据或使用真实数据
```bash
# 方案1: 创建dummy数据
python scripts/create_dummy_data.py

# 方案2: 使用真实数据
# 修改配置文件中的 metadata_file 路径
```

#### 3. CUDA内存不足
```bash
RuntimeError: CUDA out of memory
```
**解决**：减少批次大小
```yaml
data:
  batch_size: 16  # 从32减少到16
```

#### 4. 训练不收敛
**可能原因**：
- 学习率过大：尝试 `lr: 0.0001`
- 数据未标准化：确保 `normalization: true`
- 模型过大：减少层数或隐藏单元

## 🔄 进阶工作流

### 完整研究流程

```bash
# 1. 数据探索
# 旧 examples 已归档到 docs/past/examples/；维护中的入口见 configs/demo/

# 2. 基线实验
python main.py --config configs/baseline.yaml

# 3. 模型调优
python scripts/hyperparameter_search.py

# 4. 跨数据集验证
python main.py --config configs/cross_dataset.yaml

# 5. 结果分析
python scripts/result_analysis.py
```

### 批量实验

```bash
# 运行多个配置
for config in configs/experiments/*.yaml; do
    python main.py --config_path "$config"
done
```

## 🎓 学习资源

### 必读文档
1. **[MODEL_INTERFACE.md](MODEL_INTERFACE.md)** - 模型接口规范
2. **[DATA_GUIDE.md](DATA_GUIDE.md)** - 数据系统详解
3. **[TASK_GUIDE.md](TASK_GUIDE.md)** - 任务类型说明

### 示例代码
- `docs/past/examples/quickstart.py` - 归档快速开始
- 维护中的可运行实验模板见 `configs/demo/`

### 配置模板
- `configs/template/minimal.yaml` - 最小配置
- `configs/template/research.yaml` - 研究配置
- `configs/template/production.yaml` - 生产配置

## 💡 最佳实践

### 开发建议

1. **从简单开始**：先用ResNet1D验证数据和流程
2. **逐步升级**：然后尝试ISFM基础模型
3. **记录实验**：使用有意义的实验名称和备注
4. **版本控制**：保存每次实验的配置文件

### 性能优化

1. **数据加载**：
   ```yaml
   data:
     num_workers: 8        # 增加workers
     pin_memory: true      # 启用pin memory
     persistent_workers: true  # 保持workers
   ```

2. **模型训练**：
   ```yaml
   trainer:
     mixed_precision: true  # 混合精度
     gradient_clip_val: 1.0 # 梯度裁剪
   ```

## 🚀 下一步

现在你已经掌握了基础！建议继续：

1. **深入学习**：阅读 [MODEL_INTERFACE.md](MODEL_INTERFACE.md) 了解模型开发
2. **实际应用**：使用你的数据集进行实验
3. **参与社区**：提交issues和pull requests
4. **扩展框架**：开发新的模型或数据集

---

🎉 **恭喜！你已经入门PHM-Vibench！**

如有问题，欢迎：
- 📖 查看 [FAQ.md](FAQ.md)
- 🐛 提交 [GitHub Issues](https://github.com/your-repo/issues)
- 💬 参与社区讨论
