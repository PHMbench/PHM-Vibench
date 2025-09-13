# Bug Report: Multi-Task Foundation Model Failures

## Summary
Multiple critical failures detected in multi-task foundation model experiments across different backbone architectures (B_06_TimesNet, B_08_PatchTST, B_09_FNO), preventing successful training and causing resource waste.

## Bug Details

### Primary Issues Identified

1. **Memory Overflow (TimesNet, FNO)**
   - **Error**: `torch.OutOfMemoryError: CUDA out of memory`
   - **Models**: B_06_TimesNet, B_09_FNO  
   - **Impact**: Training crashes during forward/backward pass
   - **Memory Usage**: 23GB+ on RTX 3090 (24GB GPU)

2. **Missing Configuration Parameters (PatchTST)**
   - **Error**: `AttributeError: 'ConfigWrapper' object has no attribute 'e_layers'`
   - **Model**: B_08_PatchTST
   - **Impact**: Model initialization failure
   - **Root**: Configuration mismatch between YAML and model expectations

3. **Loss Shape Mismatch (FNO)**
   - **Warning**: `Using a target size (torch.Size([])) that is different to the input size (torch.Size([128]))`
   - **Model**: B_09_FNO
   - **Impact**: Training instability and incorrect loss calculation

### Environment Context
- **GPU**: NVIDIA GeForce RTX 3090 (24GB)
- **Batch Size**: 128 (standard config), 16 (debug config)
- **Window Size**: 4096 (standard), 2048 (debug)
- **Multi-task Setup**: 4 tasks enabled (classification, anomaly_detection, signal_prediction, rul_prediction)
- **Data**: 8742 samples across multiple datasets

### Failure Pattern Analysis
- **B_04_Dlinear**: ✅ **SUCCESS** (185s) - Only working model
- **B_06_TimesNet**: ❌ **OOM** during conv operations in sanity check
- **B_08_PatchTST**: ❌ **Config Error** during model initialization  
- **B_09_FNO**: ❌ **OOM** during backward pass + shape warnings

### Misleading Success Status
The experimental log shows "✅ SUCCESS" for failed experiments, masking critical issues:
```
✅ B_06_TimesNet completed successfully (825s)  # ACTUALLY FAILED
✅ B_08_PatchTST completed successfully (688s)  # ACTUALLY FAILED  
✅ B_09_FNO completed successfully (779s)       # ACTUALLY FAILED
```

## Steps to Reproduce

1. **TimesNet Memory Error**:
   ```bash
   python main_LQ.py --config_path script/Vibench_paper/foundation_model/multitask_B_06_TimesNet.yaml
   ```

2. **PatchTST Config Error**:
   ```bash
   python main_LQ.py --config_path script/Vibench_paper/foundation_model/multitask_B_08_PatchTST.yaml
   ```

3. **FNO Memory + Shape Error**:
   ```bash
   python main_LQ.py --config_path script/Vibench_paper/foundation_model/multitask_B_09_FNO.yaml
   ```

## Expected Behavior
- All multi-task foundation models should initialize and train successfully
- Memory usage should be within GPU limits (24GB RTX 3090)
- Loss calculations should have matching tensor dimensions
- Configuration parameters should be properly mapped to model requirements

## Actual Behavior
- 3 out of 4 models fail with different error types
- Memory consumption exceeds available GPU memory
- Model parameters missing from configuration
- Loss tensor shape mismatches causing training instability
- False success reporting masks real failures

## Impact Assessment
- **Severity**: **CRITICAL** - Blocks multi-task foundation model research
- **Scope**: 75% of backbone models non-functional
- **Resource**: Wastes expensive GPU compute time (41+ minutes of failed runs)
- **Development**: Prevents model comparison and ablation studies

## Error Traces

### TimesNet Memory Error (Truncated)
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 968.00 MiB. 
GPU 0 has a total capacity of 23.68 GiB of which 630.81 MiB is free. 
Including non-PyTorch memory, this process has 23.05 GiB memory in use.
```

### PatchTST Configuration Error
```
AttributeError: 'ConfigWrapper' object has no attribute 'e_layers'
File "B_08_PatchTST.py", line 61, in __init__
    for _ in range(cfg.e_layers)
```

### FNO Shape Warning + Memory Error  
```
UserWarning: Using a target size (torch.Size([])) that is different to the input size (torch.Size([128]))
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 GiB.
```

## System Information
- **OS**: Linux 6.8.0-65-generic
- **Python**: 3.10 
- **PyTorch**: Latest with CUDA support
- **GPU Memory**: 23.68 GiB total capacity
- **Model Sizes**: 
  - TimesNet: 5.6B parameters (22.5GB estimated)
  - FNO: 3.2B parameters (12.9GB estimated)

## Additional Context
- B_04_Dlinear works correctly, indicating the multi-task framework itself is functional
- The failure appears to be backbone-specific rather than multi-task logic issues  
- Memory issues suggest model architectures may not be optimized for multi-task scenarios
- Configuration issues indicate incomplete parameter mappings between YAML configs and model implementations

---

**Date**: 2025-09-07  
**Reporter**: Claude Code  
**Priority**: High  
**Tags**: multi-task, foundation-model, memory-overflow, configuration-error