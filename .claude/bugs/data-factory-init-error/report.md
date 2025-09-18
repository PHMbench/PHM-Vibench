# Bug Report

## Bug Summary
Data factory initialization fails during training/validation dataset creation in data_factory.py line 51, causing complete pipeline failure.

## Bug Details

### Expected Behavior
The data factory should successfully initialize and create training/validation datasets from the cache.h5 file for multi-task PHM experiments.

### Actual Behavior  
The data factory crashes during initialization with a traceback pointing to data_factory.py line 51, immediately after completing the cache integration phase.

### Steps to Reproduce
1. Run PHM-Vibench experiment with multi-task configuration
2. Use config: `script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml`
3. Execute: `python main_LQ.py --config_path <config_file>`
4. Wait for cache integration to complete (successful)
5. Observe crash during "Creating train/val datasets" phase

### Environment
- **Version**: PHM-Vibench latest (main_LQ.py)
- **Platform**: HPC cluster (SLURM environment)
- **Configuration**: 
  - Data dir: `/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/data/`
  - Cache file: `cache.h5` (8742 samples)
  - Config: `multitask_B_04_Dlinear.yaml`
  - Pipeline: `Pipeline_01_default`

## Impact Assessment

### Severity
- [x] Critical - System unusable
- [ ] High - Major functionality broken
- [ ] Medium - Feature impaired but workaround exists
- [ ] Low - Minor issue or cosmetic

### Affected Users
All users attempting to run multi-task PHM experiments, particularly on HPC clusters with the updated data path configuration.

### Affected Features
- Complete pipeline execution blocked
- Multi-task model training cannot proceed
- Data factory initialization process
- Training/validation dataset creation

## Additional Context

### Error Messages
```
2025-09-07 01:03:54
整合 cache.h5: 100%|██████████| 8742/8742 [00:36<00:00, 238.01it/s]
2025-09-07 01:04:30
数据整合完成。最终缓存文件: /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/data/cache.h5
2025-09-07 01:04:30
Using ID_dataset for on-demand processing.
2025-09-07 01:04:30
Warning: Task type In_distribution not specifically handled for ID searching. Defaulting to all keys.
2025-09-07 01:04:30
Initializing training and validation datasets...
2025-09-07 01:04:30
Creating train/val datasets:   0%|          | 0/8742 [00:00<?, ?it/s]
2025-09-07 01:04:30
Traceback (most recent call last):
  File "/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/main_LQ.py", line 51, in <module>
    main()
  File "/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/main_LQ.py", line 45, in main
    results = pipeline.pipeline(args)
  File "/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/src/Pipeline_01_default.py", line 93, in pipeline
    data_factory = build_data(args_data, args_task)
  File "/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/src/data_factory/__init__.py", line 55, in build_data
    return factory_cls(args_data, args_task)
  File "/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/src/data_factory/data_factory.py", line 51, in __init__
```

### Screenshots/Media
N/A - Terminal error output provided above.

### Related Issues
- This may be related to the recent path configuration updates in the YAML config
- Data path changed to HPC location: `/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/data/`
- Could be related to the multi-task configuration or IdIncludedDataset processing

## Initial Analysis

### Suspected Root Cause
The error occurs at data_factory.py line 51 during __init__ method execution. This is likely one of:
1. **Path/File Access Issue**: New HPC data path may have permission or access issues
2. **Configuration Mismatch**: multi-task config parameters not compatible with data factory
3. **Dataset Creation Error**: IdIncludedDataset initialization failure during train/val split
4. **Memory/Resource Issue**: Insufficient resources for 8742 samples processing

### Affected Components
- `src/data_factory/data_factory.py` (line 51)
- `src/data_factory/__init__.py` (build_data function)  
- `src/Pipeline_01_default.py` (line 93)
- Configuration file: `multitask_B_04_Dlinear.yaml`
- Cache file: `/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/data/cache.h5`

---

**Status**: Bug documented, ready for analysis phase.
**Next Step**: Execute `/bug-analyze` to investigate root cause.