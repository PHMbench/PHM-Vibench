# Bug Report

## Bug Summary
ModelCheckpoint监控指标配置错误：配置文件中监控`val_total_loss`，但多任务Lightning模块实际记录的是`val_loss`

## Bug Details

### Expected Behavior
ModelCheckpoint应该能够正常监控验证损失并保存最佳模型检查点，训练过程应该正常进行

### Actual Behavior  
训练在初始化ModelCheckpoint时失败，抛出MisconfigurationException，无法找到配置的监控指标`val_total_loss`

### Steps to Reproduce
1. 使用多任务配置文件运行训练：`python main_LQ.py --config script/Vibench_paper/foundation_model/multitask_B_04_Dlinear_debug.yaml`
2. 系统尝试初始化ModelCheckpoint组件
3. ModelCheckpoint查找配置的`monitor='val_total_loss'`指标
4. 抛出MisconfigurationException错误，训练终止

### Environment
- **Version**: PHM-Vibench current version
- **Platform**: Linux, CUDA environment
- **Configuration**: 多任务训练配置 (multitask_B_04_Dlinear_debug.yaml)
- **Framework**: PyTorch Lightning

## Impact Assessment

### Severity
- [x] High - Major functionality broken

### Affected Users
所有使用多任务配置进行模型训练的用户

### Affected Features
- 多任务模型训练pipeline
- ModelCheckpoint功能 
- 自动最佳模型保存

## Additional Context

### Error Messages
```
lightning_fabric.utilities.exceptions.MisconfigurationException: `ModelCheckpoint(monitor='val_total_loss')` could not find the monitored key in the returned metrics: ['train_classification_loss', 'train_classification_loss_step', 'train_anomaly_detection_loss', 'train_anomaly_detection_loss_step', 'train_loss', 'train_loss_step', 'val_classification_loss', 'val_anomaly_detection_loss', 'val_loss', 'train_classification_loss_epoch', 'train_anomaly_detection_loss_epoch', 'train_loss_epoch', 'epoch', 'step']. HINT: Did you call `log('val_total_loss', value)` in the `LightningModule`?
```

### Stack Trace
```
File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/src/Pipeline_01_default.py", line 127, in pipeline
    trainer.fit(
File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/main_LQ.py", line 45, in main
    results = pipeline.pipeline(args)
File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/main_LQ.py", line 51, in <module>
    main()
```

### Available Metrics Analysis
从错误信息可以看出，实际记录的验证指标包括：
- `val_classification_loss`
- `val_anomaly_detection_loss` 
- `val_loss`

但配置文件试图监控不存在的`val_total_loss`指标。

### Related Issues
- 之前的多任务训练模块重构可能影响了指标命名约定
- 配置文件与实际Lightning模块日志记录不一致

## Initial Analysis

### Suspected Root Cause
配置文件中的监控指标名称(`val_total_loss`)与多任务Lightning模块中实际记录的指标名称(`val_loss`)不匹配

### Affected Components
- 配置文件：`script/Vibench_paper/foundation_model/multitask_B_04_Dlinear_debug.yaml`
- 多任务Lightning模块：`src/task_factory/task/In_distribution/multi_task_phm.py`  
- Training pipeline：`src/Pipeline_01_default.py`
- ModelCheckpoint配置逻辑

### Immediate Fix Options
1. 修改配置文件中的monitor参数：`val_total_loss` → `val_loss`
2. 或修改Lightning模块记录指标名称：`val_loss` → `val_total_loss`
3. 检查并统一所有相关配置文件的monitor设置