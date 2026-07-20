# 中文Prompt：论文项目README生成器

## 完整版Prompt模板

### 版本1：详细版（推荐用于复杂项目）

```
# 任务：创建论文项目README文档

你是一个专业的科研文档撰写助手，请为一篇关于"[研究主题]"的论文创建一个完整的项目README文档。

## 项目信息
- 论文标题：[论文完整标题]
- 项目名称：[项目名称/代码库名]
- 研究领域：[如：故障诊断、机器学习、深度学习、计算机视觉等]
- 研究目标：[用1-2句话描述核心研究目标]
- 主要贡献：[列出2-3个主要创新点]
- 项目状态：[开发中/已发布/维护中]

## 要求

请按照以下结构创建README文档，使用中文撰写：

### 1. 项目标题与概述
```
# [项目名称] - [项目简短描述]

![状态](https://img.shields.io/badge/状态-活跃-brightgreen)
![版本](https://img.shields.io/badge/版本-v1.0-blue)
![许可](https://img.shields.io/badge/许可-MIT-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)

## 📋 项目概述

[简要描述项目背景（2-3段），包括：]
- 研究领域现状和挑战
- 现有方法的局限性
- 本项目的解决方案和动机

**主要特性**：
- ✅ [特性1：具体说明，如"创新的注意力机制，提升特征提取能力"]
- ✅ [特性2：具体说明，如"跨域泛化能力强，适应多种工业场景"]
- ✅ [特性3：具体说明，如"少样本学习，降低标注成本"]
- ✅ [特性4：具体说明，如"完整的实验框架，支持快速验证"]
- ✅ [特性5：具体说明，如"详细的文档和示例代码"]

**项目结构**：
```
[项目名称]/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖列表
├── setup.py                 # 安装脚本
├── configs/                 # 配置文件
│   ├── baseline.yaml        # 基线配置
│   ├── experiment_1.yaml    # 实验1配置
│   └── ...
├── src/                     # 源代码
│   ├── models/              # 模型实现
│   ├── data/                # 数据处理
│   ├── utils/               # 工具函数
│   └── train.py             # 训练脚本
├── scripts/                 # 执行脚本
│   ├── run_all.sh           # 完整实验脚本
│   └── run_baseline.sh      # 基线实验脚本
├── experiments/             # 实验设计
│   ├── experiment_design.md
│   └── results/             # 实验结果
├── docs/                    # 详细文档
│   ├── api.md               # API文档
│   └── tutorials/           # 教程
└── tests/                   # 测试代码
```

### 2. 科学研究框架
```
## 🎯 科学研究框架

### 核心研究问题
本研究通过系统的实验设计回答以下核心问题：

#### 问题一：[问题1标题，如"基线方法性能评估"]
**核心问题**: [详细描述研究问题]

**具体假设**：
- **H0**: [零假设，如"传统方法在跨域场景下性能有限（准确率<70%）"]
- **H1**: [备择假设1，如"提出的特征学习方法能提升跨域性能（70-80%）"]
- **H2**: [备择假设2，如"引入注意力机制进一步提升性能（80-85%）"]
- **H3**: [备择假设3，如"完整方法达到最优性能（>90%）"]

#### 问题二：[问题2标题，如"方法泛化性验证"]
**核心问题**: [详细描述研究问题]

**具体假设**：
- [列出其他假设]

### 实验设计方案
[描述实验设计的核心思路，包括：]
- 渐进式验证策略
- 对照实验设置
- 评估指标选择
- 统计显著性检验方法

### 预期贡献
本研究的主要贡献包括：
1. [贡献1：理论创新]
2. [贡献2：方法创新]
3. [贡献3：实践价值]
4. [贡献4：开源贡献]
```

### 3. 实验体系
```
## 📊 实验体系设计

### 精确实验矩阵
| 实验 | 研究目标 | 方法对比 | 数据集 | 预期性能 | 运行次数 | 配置文件 |
|------|----------|----------|--------|----------|----------|----------|
| 实验0 | 基线建立 | Backbone+Head | 数据集A | 65-70% | 5次 | configs/exp0.yaml |
| 实验1 | 特征学习 | +特征提取 | 数据集A | 70-75% | 5次 | configs/exp1.yaml |
| 实验2 | 注意力机制 | +注意力模块 | 数据集A | 75-80% | 5次 | configs/exp2.yaml |
| 实验3 | 完整方法 | 所有组件 | 数据集A | 80-90% | 5次 | configs/exp3.yaml |
| 实验4 | 泛化验证 | 完整方法 | 数据集B | >75% | 5次 | configs/exp4.yaml |
| 实验5 | 消融研究 | 组件组合 | 数据集A | 量化分析 | 30次 | configs/exp5.yaml |

### 资源配置与时间估算
基于单张NVIDIA RTX 3090:

| 实验 | GPU时间/次 | 总GPU时间 | 内存需求 | 批次大小 | 训练轮数 |
|------|------------|------------|----------|----------|----------|
| 实验0 | 0.5小时 | 2.5小时 | 8GB | 32 | 100 |
| 实验1 | 0.6小时 | 3.0小时 | 10GB | 32 | 100 |
| 实验2 | 0.8小时 | 4.0小时 | 12GB | 32 | 100 |
| 实验3 | 1.0小时 | 5.0小时 | 12GB | 32 | 100 |

**总资源需求**：
- GPU小时数：约15小时
- 内存需求：最高12GB
- 存储空间：约20GB

### 论文表格对应关系
| 表格编号 | 表格标题 | 对应实验 | 验证要点 | 评估指标 |
|----------|----------|----------|----------|----------|
| 表1 | 基线方法对比 | 实验0 | 性能下限 | 准确率、F1 |
| 表2 | 消融研究结果 | 实验5 | 组件贡献 | 性能提升 |
| 表3 | 跨数据集泛化 | 实验4 | 泛化能力 | 领域适应性 |
```

### 4. 快速开始
```
## 🚀 快速开始

### 1. 环境配置
```bash
# 克隆项目
git clone https://github.com/[用户名]/[项目名].git
cd [项目名]

# 创建conda环境
conda create -n [环境名] python=3.9
conda activate [环境名]

# 安装PyTorch（根据CUDA版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. 数据准备
```bash
# 下载数据集
python scripts/download_data.py --dataset [数据集名称]

# 数据预处理
python scripts/preprocess_data.py --input_dir [原始数据] --output_dir [处理数据]
```

### 3. 快速运行
```bash
# 运行基线实验（单数据集，快速验证）
python src/train.py --config configs/baseline.yaml \
                   --dataset [数据集名称] \
                   --epochs 10 \
                   --debug

# 运行完整实验（所有配置，多数据集）
bash scripts/run_all_experiments.sh
```

### 4. 结果可视化
```bash
# 生成性能报告
python scripts/generate_report.py --results_dir experiments/results \
                                 --output_dir reports \
                                 --format pdf
```
```

### 5. 配置系统
```
## ⚙️ 配置系统详解

### 配置文件结构
```yaml
# configs/experiment_template.yaml
# =============================================================================
# 实验[编号]: [实验名称]
# 目标: [实验目标]
# =============================================================================

# 环境配置
environment:
  project: "[项目名称]"
  seed: 42
  output_dir: "results/experiment_[编号]"
  wandb_project: "wandb_project_name"

# 数据配置
data:
  data_dir: "/path/to/dataset"
  dataset_name: "[数据集名称]"
  batch_size: 32
  num_workers: 8
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  normalization: "standardization"
  augmentation: true

# 模型配置
model:
  name: "[模型名称]"
  backbone: "[骨干网络]"
  embedding_dim: 256
  num_layers: 4
  dropout: 0.1
  # 实验特定参数
  use_attention: true
  attention_heads: 8
  prompt_dim: 128

# 训练配置
training:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.0001
  max_epochs: 100
  early_stopping: true
  patience: 15
  scheduler: "cosine"

# 实验配置
experiment:
  name: "实验名称"
  description: "实验描述"
  target_metrics: ["accuracy", "f1_score"]
  baseline_comparison: true
```

### 参数覆盖系统
```bash
# 使用--override参数动态调整配置
python src/train.py --config configs/base.yaml \
                   --override data.batch_size=64 \
                   --override training.learning_rate=0.0005 \
                   --override model.use_attention=true \
                   --override environment.seed=123
```

### 配置验证
```python
# 验证配置文件完整性
from utils.config import validate_config

config = load_config("configs/experiment.yaml")
is_valid, issues = validate_config(config)
if not is_valid:
    print("配置验证失败:", issues)
```
```

### 6. 执行指南
```
## 🎯 零歧义执行指南

### 完整执行流程
```bash
# 进入项目目录
cd /path/to/project

# 阶段1：环境验证（5分钟）
python scripts/check_environment.py

# 阶段2：单数据集验证（30分钟）
python src/train.py --config configs/quick_test.yaml \
                   --dataset test_dataset \
                   --epochs 5

# 阶段3：基线实验（2小时）
for dataset in dataset1 dataset2 dataset3; do
    python src/train.py --config configs/baseline.yaml \
                       --dataset $dataset \
                       --seed 42
done

# 阶段4：完整实验（4小时）
bash scripts/run_full_experiments.sh

# 阶段5：结果收集（10分钟）
python scripts/collect_results.py \
    --input_dir results \
    --output_dir final_results \
    --format both
```

### 批量实验脚本示例
```bash
#!/bin/bash
# scripts/run_experiments.sh

# 定义参数
datasets=("dataset1" "dataset2" "dataset3")
seeds=(42 123 456)
configs=("baseline.yaml" "method1.yaml" "method2.yaml")

# 循环执行所有实验组合
for config in "${configs[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "运行: config=$config, dataset=$dataset, seed=$seed"
            python src/train.py \
                --config configs/$config \
                --dataset $dataset \
                --seed $seed \
                --output_dir results/${config%.*}/$dataset/seed_$seed
        done
    done
done
```
```

### 7. 结果组织规范
```
## 📊 结果组织规范

### 文件命名规范
```
results/
├── experiment_[编号]_[名称]/
│   ├── dataset_[数据集名]/
│   │   ├── seed_[随机种子]/
│   │   │   ├── config.yaml          # 使用的配置文件
│   │   │   ├── model.pth            # 模型权重
│   │   │   ├── training_log.csv     # 训练日志
│   │   │   ├── metrics.json         # 评估指标
│   │   │   ├── predictions.npy      # 预测结果
│   │   │   └── visualizations/      # 可视化结果
│   │   │       ├── confusion_matrix.png
│   │   │       └── learning_curve.png
│   │   └── aggregated_results.json  # 多种子聚合结果
│   └── summary.json                # 实验汇总
└── all_experiments_summary.csv      # 全实验汇总
```

### Metrics文件格式
```json
{
  "experiment_name": "experiment_3_full_method",
  "dataset": "dataset1",
  "seed": 42,
  "model_config": {
    "backbone": "ResNet50",
    "embedding_dim": 256,
    "use_attention": true
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "max_epochs": 100
  },
  "results": {
    "accuracy": 0.9234,
    "f1_macro": 0.9198,
    "precision_macro": 0.9212,
    "recall_macro": 0.9185,
    "auc_macro": 0.9678,
    "training_time": 5423.7,
    "inference_time": 15.6,
    "peak_memory": 12288
  },
  "class_wise_results": {
    "class_0": {"precision": 0.95, "recall": 0.92, "f1": 0.93},
    "class_1": {"precision": 0.89, "recall": 0.94, "f1": 0.91}
  },
  "timestamp": "2025-01-29T10:30:45Z",
  "git_commit": "abc123def456",
  "hardware": "NVIDIA RTX 3090"
}
```
```

### 8. 故障排除
```
## 🔧 故障排除

### 常见问题及解决方案

#### 1. 环境问题
**问题**: 依赖版本冲突
```bash
# 解决方案：使用虚拟环境
conda create -n [项目名] python=3.9
conda activate [项目名]
pip install -r requirements.txt
```

**问题**: CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi
nvcc --version

# 安装对应版本的PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取正确命令
```

#### 2. 内存问题
**问题**: GPU内存不足 (OOM)
```yaml
# 解决方案：调整配置
data:
  batch_size: 16  # 从32减少到16
model:
  gradient_checkpointing: true  # 启用梯度检查点
training:
  accumulate_grad_batches: 2  # 梯度累积
```

#### 3. 数据问题
**问题**: 数据路径错误
```bash
# 检查数据路径
python -c "import os; print(os.path.exists('/path/to/data'))"

# 使用绝对路径或设置环境变量
export DATA_DIR="/absolute/path/to/data"
```

#### 4. 性能问题
**问题**: 训练速度慢
```python
# 解决方案：性能优化
# 1. 使用混合精度训练
training:
  precision: 16  # 使用FP16

# 2. 增加num_workers
data:
  num_workers: 8  # 根据CPU核心数调整

# 3. 使用pin_memory
data:
  pin_memory: true
```

### 调试工具
```bash
# 1. 查看实时日志
tail -f logs/experiment.log

# 2. 监控GPU使用
watch -n 1 nvidia-smi

# 3. 性能分析
python -m torch.utils.bottleneck src/train.py --config configs/debug.yaml

# 4. 内存分析
python -c "import torch; print(torch.cuda.memory_summary())"

# 5. 配置验证
python scripts/validate_config.py --config configs/experiment.yaml
```

### 性能优化建议
1. **数据加载优化**：
   - 使用HDF5或LMDB格式存储数据
   - 预计算数据增强
   - 使用多进程数据加载

2. **训练优化**：
   - 使用混合精度训练
   - 实现梯度累积
   - 启用梯度检查点

3. **模型优化**：
   - 使用更高效的backbone
   - 实现模型剪枝
   - 量化模型权重
```

### 9. 文档与贡献
```
## 📚 文档结构

### 详细文档
- **[安装指南](docs/installation.md)** - 详细的环境配置说明
- **[API文档](docs/api.md)** - 完整的API参考
- **[教程](docs/tutorials/)** - 从入门到高级的使用教程
- **[FAQ](docs/faq.md)** - 常见问题解答

### 示例代码
- **[基础示例](examples/basic_usage.py)** - 简单的使用示例
- **[高级示例](examples/advanced_usage.py)** - 复杂场景的使用
- **[自定义模型](examples/custom_model.py)** - 如何扩展模型

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献
1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装pre-commit钩子
pre-commit install

# 运行测试
pytest tests/ -v

# 代码格式化
black .
isort .

# 代码检查
flake8 .
mypy .
```

### 提交规范
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关
- chore: 构建过程或辅助工具的变动
```

### 10. 附录
```
## 📄 附录

### 许可证
本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

### 引用
如果你在研究中使用了本项目，请引用：

```bibtex
@article{[作者]2025,
  title={论文标题},
  author={[作者列表]},
  journal={期刊名},
  year={2025}
}
```

### 联系方式
- **项目维护者**: [您的姓名]
- **邮箱**: [您的邮箱]
- **GitHub Issues**: [项目Issues链接]
- **讨论区**: [Discussions链接]

### 致谢
感谢以下开源项目和贡献者：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Weights & Biases](https://wandb.ai/) - 实验跟踪平台
- 所有为本项目做出贡献的研究人员

### 更新日志
- **v1.0.0** (2025-01-29) - 初始版本发布
- **v1.1.0** (计划中) - 新功能特性
- **v1.2.0** (计划中) - 性能优化

---

**最后更新**: 2025-01-29
**版本**: v1.0
**项目状态**: 活跃开发中
**CI/CD**: [![CI](https://github.com/[用户]/[项目]/workflows/CI/badge.svg)](https://github.com/[用户]/[项目]/actions)
```

## 特殊要求

1. **语言要求**：
   - 使用中文撰写
   - 技术术语保持准确性
   - 专业表达清晰易懂

2. **格式要求**：
   - 使用emoji增强可读性（📋、🎯、🚀、⚙️、🔧等）
   - 重要信息使用**粗体**标记
   - 代码块指定语言类型（bash、yaml、python、json等）
   - 表格对齐整齐，使用Markdown表格语法

3. **内容要求**：
   - 提供具体可执行的命令示例
   - 包含实际的项目结构和配置
   - 实验设计科学合理
   - 故障排除实用有效

4. **风格要求**：
   - 专业且易于理解
   - 逻辑清晰，层次分明
   - 避免冗余，突出重点
   - 保持积极友好的语调

请根据提供的项目信息，生成完整的README文档。
```

### 版本2：简化版（用于快速生成）

```
请为论文项目"[项目名称]"创建README文档。

项目信息：
- 研究主题：[研究主题]
- 主要目标：[1-2句话描述目标]
- 创新点：[列出2-3个创新点]
- 项目状态：[开发状态]

要求包含以下核心部分：
1. 项目概述（含特性列表和项目结构）
2. 研究框架（研究问题和假设）
3. 实验设计（实验矩阵表格）
4. 快速开始（环境配置和运行命令）
5. 配置说明（YAML配置示例）
6. 执行指南（分阶段执行流程）
7. 结果组织（文件命名规范）
8. 故障排除（常见问题解决）

使用中文撰写，包含具体代码示例，添加emoji增强可读性。
```

### 版本3：极简版（用于快速原型）

```
# 生成README - [项目名称]

项目：[项目名称]
主题：[研究主题]
目标：[主要目标]
创新：[创新点1、创新点2、创新点3]

请创建包含以下内容的README：
- 项目简介（100字内）
- 3-5个主要特性（✅列表）
- 安装命令
- 使用示例
- 实验表格
- 联系方式

要求：中文、简洁、专业。
```

## 使用指南

1. **选择合适的版本**：
   - 复杂项目使用完整版
   - 中等项目使用简化版
   - 原型项目使用极简版

2. **填写占位符**：
   - 将`[ ]`中的内容替换为实际信息
   - 保持格式一致性

3. **自定义调整**：
   - 根据项目特点增减章节
   - 调整技术细节深度

4. **生成后检查**：
   - 验证命令是否可执行
   - 检查链接是否有效
   - 确保格式正确

---

**创建时间**: 2025-01-29
**版本**: v1.0