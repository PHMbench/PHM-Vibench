# Bug Report

## Bug Summary
Multi-task PHM模型训练完成但多个任务性能指标异常，包括异常检测AUROC接近随机、RUL预测R²严重负值、信号预测性能不佳，以及TorchMetrics样本不平衡警告。

## Bug Details

### Expected Behavior
- 异常检测任务AUROC应在0.5以上，理想情况0.8+
- RUL预测R²应为正值，表明模型优于简单平均值
- 信号预测R²应为正值或接近0
- 训练过程不应出现样本不平衡警告

### Actual Behavior
```
anomaly_detection_test_auroc        0.02027323469519615  # 接近随机水平
rul_prediction_test_r2              -2641.96044921875    # 严重负值
signal_prediction_test_r2           -0.4496903419494629  # 负值，差于基线

TorchMetrics警告：
- "No positive samples in targets, true positive value should be meaningless"
- "No negative samples in targets, false positive value should be meaningless"
```

### Steps to Reproduce
1. 运行多任务PHM训练：`python main.py --config configs/demo/Multiple_DG/all.yaml`
2. 观察训练完成后的测试指标
3. 注意TorchMetrics在训练过程中的警告信息

### Environment
- **Framework**: PyTorch Lightning with PHM-Vibench
- **Model**: M_01_ISFM multi-task
- **Tasks**: ['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction']
- **Dataset**: metadata_6_11.xlsx (多数据集混合)
- **Hardware**: NVIDIA GeForce RTX 3090

## Impact Assessment

### Severity
- [x] High - Major functionality broken
- [ ] Critical - System unusable
- [ ] Medium - Feature impaired but workaround exists  
- [ ] Low - Minor issue or cosmetic

### Affected Users
所有使用多任务PHM训练功能的研究人员和开发者

### Affected Features
- 异常检测任务完全失效
- RUL预测任务性能极差
- 信号预测任务性能不佳
- 多任务训练整体可靠性受损

## Additional Context

### Error Messages
```
/home/lq/.conda/envs/P/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score
  warnings.warn(*args, **kwargs)

/home/lq/.conda/envs/P/lib/python3.10/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: No negative samples in targets, false positive value should be meaningless. Returning zero tensor in false positive score
  warnings.warn(*args, **kwargs)
```

### Performance Metrics Analysis
```
# 分类任务 - 正常
classification_test_acc    0.923159658908844

# 异常检测任务 - 异常
anomaly_detection_test_acc     0.929   # 高准确率但...
anomaly_detection_test_auroc   0.020   # AUROC接近随机
anomaly_detection_test_f1      0.067   # F1极低

# RUL预测任务 - 严重异常
rul_prediction_test_r2     -2641.96   # R²严重负值
rul_prediction_test_mape   0.998      # MAPE接近1

# 信号预测任务 - 性能不佳  
signal_prediction_test_r2  -0.450     # 负R²
```

### Related Issues
- 可能与之前修复的tensor view/reshape问题相关
- 可能涉及system_id在批处理中的处理问题
- 多任务损失权重或归一化问题

## Initial Analysis

### Suspected Root Cause
1. **样本不平衡问题** - TorchMetrics警告显示某些批次只有正样本或负样本
2. **任务头初始化问题** - 异常检测和回归任务的权重初始化可能不当
3. **损失函数配置** - 多任务损失权重可能导致某些任务被忽略
4. **数据预处理** - 不同数据集的标签格式或分布可能不一致

### Affected Components
- `src/task_factory/task/In_distribution/multi_task_phm.py`
- `src/model_factory/ISFM/task_head/H_09_multiple_task.py`
- `src/model_factory/ISFM/task_head/H_01_Linear_cla.py`
- `src/model_factory/ISFM/M_01_ISFM.py`
- 数据加载和预处理模块