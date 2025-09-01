# HSE异构对比学习 Cross-System Generalization

## 概述

本目录包含基于HSE异构信号嵌入与对比学习的跨系统域泛化实验配置。该方案通过系统级对比学习增强模型在不同工业设备间的泛化能力。

## 配置文件说明

### 1. hse_cddg.yaml（基础配置）
- **用途**：标准HSE对比学习实验
- **对比权重**：0.1
- **温度参数**：0.07
- **特征**：启用困难负样本挖掘

### 2. ablation_no_contrast.yaml（消融实验）
- **用途**：禁用对比学习的基线实验
- **对比权重**：0.0（禁用对比损失）
- **特征**：仅使用分类损失，用于对比验证

### 3. high_contrast.yaml（高权重实验）
- **用途**：强对比学习实验
- **对比权重**：0.3
- **温度参数**：0.05（更强的对比效果）

## 运行方式

### 基础实验
```bash
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml
```

### 消融实验（无对比学习）
```bash
python main.py --config configs/demo/HSE_Contrastive/ablation_no_contrast.yaml
```

### 高对比权重实验
```bash
python main.py --config configs/demo/HSE_Contrastive/high_contrast.yaml
```

### 参数调整
```bash
# 调整对比权重
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml \
               --override "{'task.contrast_weight': 0.2}"

# 调整温度参数
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml \
               --override "{'task.temperature': 0.05}"

# 快速测试（少量epoch）
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml \
               --override "{'task.epochs': 2, 'data.batch_size': 8}"
```

## 核心技术特点

### HSE异构嵌入 (E_01_HSE)
- **多尺度patch采样**：patch_size_L=256, num_patches=128
- **时间嵌入融合**：结合采样频率信息
- **异构信号支持**：适应不同类型的工业信号

### 系统级对比学习
- **正样本**：同系统内的不同样本
- **负样本**：不同系统间的样本
- **InfoNCE损失**：支持困难负样本挖掘

### 跨系统域泛化 (CDDG)
- **多源域训练**：利用多个源系统的数据
- **目标域测试**：在新系统上评估泛化能力
- **无标签适应**：不需要目标域的标签信息

## 实验结果存储

结果将自动保存在以下位置：
```
save/metadata_6_1/M_01_ISFM/CDDG_hse_contrastive_Default_trainer_[timestamp]/
├── checkpoints/     # 模型权重
├── metrics.json     # 性能指标
├── log.txt         # 训练日志
├── figures/         # 可视化图表
└── config.yaml     # 实验配置备份
```

## 监控指标

### 训练期间
- `train/cls_loss`：分类损失
- `train/contrast_loss`：对比学习损失
- `train/total_loss`：总损失
- `train/feature_norm`：特征范数
- `train/contrast_valid_ratio`：有效对比批次比例

### 验证/测试
- `val/acc`：验证准确率
- `test/acc`：测试准确率
- `test/f1`：F1分数

## 参数调优建议

### 对比权重 (contrast_weight)
- **0.0**：禁用对比学习（基线）
- **0.05-0.1**：轻度对比学习（推荐起始值）
- **0.1-0.2**：中等强度对比学习
- **0.2-0.5**：强对比学习（可能需要调整学习率）

### 温度参数 (temperature)
- **0.05**：强对比（更尖锐的分布）
- **0.07**：中等对比（推荐值）
- **0.1-0.2**：软对比（更平滑的分布）

### HSE参数
- **patch_size_L**：128/256/512（信号patch长度）
- **num_patches**：64/128/256（patch数量）
- **output_dim**：512/1024/2048（嵌入维度）

## 故障排除

### 常见问题
1. **内存不足**：减少batch_size或num_patches
2. **收敛困难**：降低contrast_weight或调整学习率
3. **对比损失为0**：检查数据是否包含多个系统
4. **特征提取失败**：检查ISFM模型配置

### 调试方法
```bash
# 启用详细日志
export PYTHONPATH=.
python -u main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml

# 快速验证
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml \
               --override "{'task.epochs': 1, 'data.batch_size': 4}"
```

## 技术支持

- **框架版本**：PHM-Vibench v5.0
- **依赖要求**：PyTorch 2.6.0+, PyTorch Lightning
- **任务实现**：`src/task_factory/task/CDDG/hse_contrastive.py`
- **配置系统**：基于ConfigWrapper的统一配置管理

---
*本实验基于PHM-Vibench框架，实现了HSE异构信号嵌入与系统级对比学习的完美集成。*