# 🚀 Flow预训练模块快速开始指南

## 概述

Flow预训练模块为PHM-Vibench提供了基于Flow的生成式预训练功能，支持联合对比学习训练和Pipeline_02兼容性。

## 🏁 快速验证

首先运行验证脚本确保系统准备就绪：

```bash
# 验证Flow模块设置
python validate_flow_setup.py
```

如果看到 `🎉 Flow设置验证完成! 系统已准备就绪`，则可以开始实验。

## 🎯 实验类型

### 1. 快速验证 (5分钟)
适合快速验证功能是否正常：
```bash
./run_flow_experiments.sh quick
```

### 2. 基线实验 (1小时)
标准Flow预训练实验：
```bash
./run_flow_experiments.sh baseline
```

### 3. 对比学习实验 (1.5小时)
Flow + 对比学习联合训练：
```bash
./run_flow_experiments.sh contrastive
```

### 4. Pipeline_02预训练 (2.5小时)
为Few-shot学习准备的预训练：
```bash
./run_flow_experiments.sh pipeline02
```

### 5. 研究级实验 (5小时)
论文发表级别的完整实验：
```bash
./run_flow_experiments.sh research --wandb
```

## 📊 批量实验管理

### 验证套件
运行3个核心实验 (quick, baseline, contrastive)：
```bash
python run_flow_experiment_batch.py validation
```

### 研究管道
运行4个研究实验 (baseline, contrastive, pipeline02, research)：
```bash
python run_flow_experiment_batch.py research --wandb
```

### 自定义批次
指定特定实验组合：
```bash
python run_flow_experiment_batch.py custom --experiments quick baseline contrastive
```

## 🔧 高级选项

### GPU选择
```bash
# 使用GPU 1
./run_flow_experiments.sh baseline --gpu 1

# 批量实验指定GPU
python run_flow_experiment_batch.py validation --gpu 1
```

### WandB跟踪
```bash
# 启用WandB (会自动修改配置)
./run_flow_experiments.sh baseline --wandb

# 批量实验启用WandB
python run_flow_experiment_batch.py research --wandb
```

### 试运行模式
```bash
# 查看将要执行的命令，不实际运行
./run_flow_experiments.sh baseline --dry-run
```

### 添加实验备注
```bash
./run_flow_experiments.sh baseline --notes "测试新的超参数配置"
```

## 📁 实验结果

实验结果自动保存在 `results/` 目录下：

```
results/
├── flow_quick_validation/     # 快速验证结果
├── flow_baseline/            # 基线实验结果
├── flow_contrastive/         # 对比学习实验结果
├── flow_pipeline02_pretrain/ # Pipeline预训练结果
└── flow_research/           # 研究级实验结果
```

每个实验目录包含：
- `checkpoints/`: PyTorch Lightning检查点
- `log.txt`: 详细训练日志
- `metrics.json`: 性能指标摘要
- `figures/`: 可视化图表 (如果启用)

## 🔍 监控和调试

### 实时监控
使用WandB在线监控训练过程：
```bash
./run_flow_experiments.sh baseline --wandb
# 查看: https://wandb.ai/your-project/flow_baseline_experiment
```

### 查看日志
```bash
# 实时查看训练日志
tail -f results/flow_baseline/log.txt

# 查看最新检查点
ls -la results/flow_baseline/checkpoints/
```

### 故障排除
```bash
# 重新验证设置
python validate_flow_setup.py

# 检查GPU状态
nvidia-smi

# 验证配置文件
python -c "import yaml; print(yaml.safe_load(open('configs/demo/Pretraining/Flow/flow_baseline_experiment.yaml')))"
```

## 📈 配置自定义

配置文件位于 `configs/demo/Pretraining/Flow/`：

- `flow_quick_validation.yaml`: 快速测试配置
- `flow_baseline_experiment.yaml`: 标准基线配置  
- `flow_contrastive_experiment.yaml`: 对比学习配置
- `flow_pipeline02_pretrain.yaml`: Pipeline预训练配置
- `flow_research_experiment.yaml`: 研究级配置

### 关键参数说明

```yaml
# Flow核心参数
task:
  num_steps: 100              # Flow采样步数
  flow_lr: 5e-4              # Flow学习率
  
  # 对比学习 
  use_contrastive: true      # 启用对比学习
  contrastive_weight: 0.3    # 对比损失权重
  temperature: 0.1           # 对比学习温度
  
  # 训练设置
  epochs: 50                 # 训练轮次
  lr: 5e-4                  # 主学习率
  batch_size: 32            # 批次大小
```

## 🚨 常见问题

### Q: 提示"Flow任务未注册"怎么办？
A: 确保在项目根目录运行命令，检查 `src/task_factory/task/pretrain/__init__.py` 是否正确导入了Flow任务。

### Q: GPU内存不足怎么办？
A: 减少batch_size或使用梯度累积：
```yaml
trainer:
  accumulate_grad_batches: 2  # 梯度累积
task:
  batch_size: 16             # 减少批次大小
```

### Q: 训练过程中断如何恢复？
A: PyTorch Lightning会自动保存检查点，重新运行相同命令即可恢复。

### Q: 如何查看训练进度？
A: 启用WandB或查看终端输出，也可以查看日志文件。

## 📊 预期性能基准

基于CWRU等标准数据集的参考性能：

| 实验类型 | 训练轮次 | 预期时间 | 验证准确率 |
|---------|---------|----------|-----------|
| quick   | 5       | ~5分钟   | 60-70%    |
| baseline| 50      | ~1小时   | 75-85%    |
| contrastive| 60   | ~1.5小时 | 80-90%    |
| pipeline02| 100   | ~2.5小时 | 85-92%    |
| research| 200     | ~5小时   | 90-95%    |

## 📖 进一步学习

- **源代码**: `src/task_factory/task/pretrain/`
- **配置系统**: `src/configs/CLAUDE.md`
- **模型架构**: `src/model_factory/ISFM/M_04_ISFM_Flow.py`
- **测试样例**: `test_flow_*`

---

🎯 **开始第一个实验**: `./run_flow_experiments.sh quick`