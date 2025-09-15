# HSE Pipeline使用指南

## 📋 概述

本指南详细说明如何使用HSE Industrial Contrastive Learning pipeline进行工业振动信号分析。涵盖从环境设置到生产实验的完整流程。

## 🚀 快速开始

### 1. 环境准备

```bash
# 1. 激活conda环境
conda activate P

# 2. 验证关键依赖
python -c "import wandb, swanlab; print('✅ wandb和swanlab已安装')"

# 3. 验证数据目录
ls data/  # 确保包含metadata文件和H5数据文件
```

### 2. 基础验证

```bash
# 运行合成数据演示（推荐首次使用）
python scripts/hse_synthetic_demo.py

# 预期输出：
# ✅ 系统提示编码: 成功
# ✅ 样本提示编码: 成功
# ✅ 提示融合: 成功
# ✅ 对比学习: 成功 (准确度提升: 14.3%)
# ✅ 验证测试: 成功 (内存: <0.1GB, 速度: >1400 samples/sec)
```

### 3. Pipeline集成测试

```bash
# 运行Pipeline_03集成测试
python scripts/test_pipeline03_integration.py

# 预期输出：
# ✅ 配置加载测试: 通过
# ✅ 组件集成测试: 通过
# ✅ 检查点处理测试: 通过
# 测试成功率: 55.6% (5/9 tests passing)
```

## 🎛️ Pipeline选择

### Pipeline_03: 多任务预训练微调

**适用场景**:
- 跨域工业信号分析
- 少样本学习任务
- 需要强泛化能力的应用

**核心特性**:
- 两阶段训练: 预训练 → 微调
- 提示引导的特征学习
- 多任务联合训练

```bash
# 使用Pipeline_03运行HSE实验
python scripts/run_hse_prompt_pipeline03.py

# 或使用配置文件方式
python main.py --pipeline Pipeline_03 \
  --config configs/pipeline_03/hse_prompt_multitask_config.yaml
```

### Pipeline_01: 标准训练

**适用场景**:
- 单域任务
- 基线对比实验
- 快速验证

```bash
# 基线实验示例
python main.py --config configs/baseline/cwru_baseline.yaml
```

## 📊 实验配置

### 主配置文件

#### 1. HSE多任务配置
**文件**: `configs/pipeline_03/hse_prompt_multitask_config.yaml`

```yaml
# 核心配置项
pipeline: "Pipeline_03_multitask_pretrain_finetune"

data:
  dataset_names: ["CWRU", "XJTU", "THU", "Ottawa", "JNU"]  # 5个数据集联合训练
  unified_loading: true

model:
  backbone:
    name: "B_11_MomentumEncoder"
    base_encoder: "E_01_HSE_v2"
  task_head:
    name: "H_10_ProjectionHead"

task:
  task_type: "hse_contrastive"
  loss_type: "infonce"

trainer:
  max_epochs: 50
  batch_size: 32
  learning_rate: 1e-4
```

#### 2. 消融研究配置

**目录**: `configs/pipeline_03/ablation/`

```bash
# 无提示基线
configs/pipeline_03/ablation/hse_no_prompt_baseline.yaml

# 仅系统提示
configs/pipeline_03/ablation/hse_system_prompt_only.yaml

# 仅样本提示
configs/pipeline_03/ablation/hse_sample_prompt_only.yaml
```

#### 3. HSE对比学习演示

**目录**: `configs/demo/HSE_Contrastive/`

```bash
# 高对比度实验
configs/demo/HSE_Contrastive/high_contrast.yaml

# 跨数据集域泛化
configs/demo/HSE_Contrastive/hse_cddg.yaml

# 提示融合消融
configs/demo/HSE_Contrastive/hse_prompt_ablation_fusion.yaml
```

## 🎯 两阶段训练流程

### 阶段1: 对比学习预训练

```bash
# 1. 配置预训练参数
vim configs/demo/HSE_Contrastive/hse_prompt_pretrain.yaml

# 关键配置
stage1:
  epochs: 100
  learning_rate: 1e-3
  task_type: "contrastive"
  datasets: ["CWRU", "XJTU", "THU", "Ottawa", "JNU"]

# 2. 启动预训练
python scripts/run_hse_prompt_pipeline03.py \
  --stage pretrain \
  --config configs/demo/HSE_Contrastive/hse_prompt_pretrain.yaml
```

### 阶段2: 下游任务微调

```bash
# 1. 配置微调参数
vim configs/demo/HSE_Contrastive/hse_prompt_finetune.yaml

# 关键配置
stage2:
  epochs: 20
  learning_rate: 1e-4
  task_type: "classification"
  freeze_backbone: false
  pretrained_path: "save/stage1_checkpoint.pth"

# 2. 启动微调
python scripts/run_hse_prompt_pipeline03.py \
  --stage finetune \
  --config configs/demo/HSE_Contrastive/hse_prompt_finetune.yaml
```

## 🔍 提示系统配置

### 1. 系统级提示

```yaml
system_prompt:
  # 数据集提示
  dataset_embedding:
    vocab: ["CWRU", "XJTU", "THU", "Ottawa", "JNU"]
    embedding_dim: 32

  # 域提示
  domain_embedding:
    vocab: ["bearing", "gearbox", "motor", "pump"]
    embedding_dim: 32

  # 工况提示
  condition_embedding:
    vocab: ["normal", "fault", "degraded"]
    embedding_dim: 16
```

### 2. 样本级提示

```yaml
sample_prompt:
  # 采样率提示
  sample_rate_embedding:
    min_rate: 1000
    max_rate: 50000
    embedding_dim: 16

  # 序列长度提示
  sequence_length_embedding:
    min_length: 512
    max_length: 4096
    embedding_dim: 16

  # 噪声水平提示
  noise_level_embedding:
    levels: [0.0, 0.1, 0.2, 0.5]
    embedding_dim: 8
```

### 3. 提示融合策略

```yaml
prompt_fusion:
  strategy: "attention"  # attention/concat/gate

  attention_config:
    num_heads: 8
    hidden_dim: 128
    dropout: 0.1

  output_dim: 64
```

## 📈 监控和日志

### 1. 实验跟踪

```python
# WandB集成
wandb.init(
    project="hse-industrial-contrastive",
    name=f"hse-{dataset}-{timestamp}",
    config=config
)

# SwanLab集成
swanlab.init(
    project="HSE-Prompt-Learning",
    experiment_name=f"multi-task-{timestamp}"
)
```

### 2. 指标监控

关键监控指标:
- **训练损失**: 对比学习损失趋势
- **验证准确度**: 各数据集上的分类准确度
- **内存使用**: 峰值GPU内存占用
- **训练速度**: samples/second
- **梯度范数**: 训练稳定性指标

### 3. 自动报告生成

```bash
# 生成实验报告
python script/unified_metric/collect_results.py \
  --experiment_dir save/hse_experiment_20250915 \
  --output_format markdown

# 输出位置
# reports/hse_experiment_report_20250915.md
```

## 🛠️ 高级配置

### 1. 内存优化

```yaml
# 混合精度训练
trainer:
  precision: 16

# 梯度检查点
model:
  gradient_checkpointing: true

# 数据加载优化
data:
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
```

### 2. 分布式训练

```bash
# 多GPU训练
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  scripts/run_hse_prompt_pipeline03.py \
  --config configs/pipeline_03/hse_prompt_multitask_config.yaml
```

### 3. 超参数调优

```yaml
# Grid Search配置
hyperparameter_search:
  learning_rate: [1e-5, 1e-4, 1e-3]
  batch_size: [16, 32, 64]
  temperature: [0.05, 0.1, 0.2]
  momentum: [0.9, 0.99, 0.999]
```

## 🔧 故障排除

### 常见问题及解决方案

#### 1. ConfigWrapper兼容性问题
```bash
# 症状: TypeError: 'ConfigWrapper' object is not iterable
# 解决: 确保配置更新使用ConfigWrapper.update()方法
config = load_config('base_config')
config.update({'model.backbone.name': 'new_value'})
```

#### 2. H5数据加载失败
```bash
# 症状: No such file or directory: '*.h5'
# 解决: 检查数据目录配置
export DATA_DIR=/path/to/your/data
python scripts/check_data_paths.py
```

#### 3. 内存不足
```bash
# 症状: RuntimeError: CUDA out of memory
# 解决: 调整batch_size或使用梯度累积
trainer:
  batch_size: 16  # 减小batch size
  accumulate_grad_batches: 2  # 梯度累积
```

#### 4. seaborn导入错误
```bash
# 症状: ModuleNotFoundError: No module named 'seaborn'
# 解决: 安装可选依赖或禁用可视化
pip install seaborn
# 或在代码中已做容错处理，可忽略此警告
```

## 📊 性能基准

### 期望性能指标

| 指标 | 目标值 | 当前状态 |
|------|--------|----------|
| 内存使用 | < 1GB | ✅ < 0.1GB |
| 训练速度 | > 1000 samples/sec | ✅ > 1400 samples/sec |
| 验证成功率 | > 80% | ⚠️ 55.6% |
| 准确度提升 | > 10% | ✅ 14.3% |

### 基准测试命令

```bash
# 运行完整基准测试
python tests/performance/prompt_benchmarks.py

# 生成性能报告
python scripts/generate_benchmark_report.py \
  --results benchmark_results/ \
  --output benchmark_report.md
```

## 📝 实验记录

### 实验命名规范

```
实验名称格式: hse_{dataset}_{task}_{timestamp}
示例: hse_cwru_classification_20250915_1430
```

### 结果保存结构

```
save/
├── hse_cwru_classification_20250915_1430/
│   ├── checkpoints/           # 模型权重
│   ├── logs/                  # 训练日志
│   ├── metrics.json           # 性能指标
│   ├── config.yaml           # 实验配置
│   └── figures/              # 可视化图表
```

## 🎯 最佳实践

### 1. 实验设计
- 始终先运行合成数据验证
- 使用OneEpochValidator快速检查配置
- 对比基线和HSE方法的性能

### 2. 配置管理
- 使用版本控制管理配置文件
- 为每个实验创建独立的配置文件
- 记录关键超参数的选择理由

### 3. 结果分析
- 关注跨域泛化性能
- 分析不同提示策略的影响
- 定期生成性能对比报告

---

*本指南涵盖了HSE pipeline的完整使用流程。如遇问题，请参考 [故障排除](#故障排除) 部分或联系开发团队。*