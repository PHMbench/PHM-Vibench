# Bug Report

## Bug Summary
Multi-task PHM training fails to start due to torchmetrics type error - `num_classes` parameter receives float value instead of required integer type.

## Bug Details

### Expected Behavior
Multi-task training should initialize successfully and begin training across all enabled tasks (classification, anomaly_detection, signal_prediction, rul_prediction).

### Actual Behavior  
Training fails during task initialization with error:
```
Failed to create task In_distribution.multi_task_phm: Optional arg `num_classes` must be type `int` when task is multiclass. Got <class 'float'>
```
This results in `NoneType` being passed to PyTorch Lightning trainer, causing a subsequent TypeError.

### Steps to Reproduce
1. Run any multi-task experiment with command: `python main_LQ.py --config_path script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml`
2. System processes successfully through data loading and model creation
3. Task creation fails with torchmetrics type error
4. Training cannot start due to NoneType model

### Environment
- **Version**: PHM-Vibench v5.0
- **Platform**: Linux with NVIDIA GeForce RTX 3090
- **Configuration**: Multi-task foundation model experiments
- **Python**: 3.10.18
- **PyTorch**: 2.7.1+cu126
- **torchmetrics**: Latest version

## Impact Assessment

### Severity
- [x] Critical - System unusable
- [ ] High - Major functionality broken
- [ ] Medium - Feature impaired but workaround exists
- [ ] Low - Minor issue or cosmetic

### Affected Users
All users attempting to run multi-task PHM experiments with foundation models.

### Affected Features
- Multi-task training pipeline
- All backbone models (B_04_Dlinear, B_06_TimesNet, B_08_PatchTST, B_09_FNO)
- Metric computation initialization

## Additional Context

### Error Messages
```
Failed to create task In_distribution.multi_task_phm: Optional arg `num_classes` must be type `int` when task is multiclass. Got <class 'float'>
[INFO] 构建训练器...
[INFO] 开始训练...
Traceback (most recent call last):
  File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/main_LQ.py", line 51, in <module>
    main()
  File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/main_LQ.py", line 45, in main
    results = pipeline.pipeline(args)
  File "/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/src/Pipeline_01_default.py", line 145, in pipeline
    trainer.fit(
  File "/home/lq/.conda/envs/P/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 554, in fit
    model = _maybe_unwrap_optimized(model)
  File "/home/lq/.conda/envs/P/lib/python3.10/site-packages/pytorch_lightning/utilities/compile.py", line 111, in _maybe_unwrap_optimized
    raise TypeError(
TypeError: `model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `NoneType`
```

### Screenshots/Media
Terminal output shows the failure occurring consistently across different backbone models (B_04_Dlinear, B_06_TimesNet, B_08_PatchTST).

### Related Issues
This appears to be related to the recent refactoring of task-specific metrics implementation and the interaction with torchmetrics library requirements.

## Initial Analysis

### Suspected Root Cause
In `src/task_factory/task/In_distribution/multi_task_phm.py` lines 158-161, the `max_classes` calculation returns a float value when metadata labels contain float values, but torchmetrics strictly requires integer type for `num_classes` parameter in multiclass classification metrics.

### Affected Components
- File: `src/task_factory/task/In_distribution/multi_task_phm.py`
- Function: `_initialize_task_metrics()`
- Lines: 158-161
- torchmetrics library integration