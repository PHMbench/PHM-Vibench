# Bug Report

## Bug Summary
Multi-task ISFM model is generating an excessively large model (3.2B parameters) causing memory issues during training, and experiencing tensor shape mismatches in loss computation for certain tasks.

## Bug Details

### Expected Behavior
- Multi-task ISFM model should have reasonable parameter count (expected ~100-500M parameters)
- All task heads should produce outputs with shapes compatible with their respective loss functions
- Training should proceed smoothly without shape mismatch warnings or memory issues

### Actual Behavior  
- Model reports 3.2 billion parameters (3.2 B), which is unexpectedly large
- MSE loss warning: "Using a target size (torch.Size([])) that is different to the input size (torch.Size([128])). This will likely lead to incorrect results due to broadcasting"
- Potential memory pressure during training phase

### Steps to Reproduce
1. Run multi-task training with command: `python main_LQ.py`
2. Use configuration: `script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml`
3. Enable multiple tasks: `['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction']`
4. Observe model parameter count and loss computation warnings during sanity checking

### Environment
- **Version**: PHM-Vibench current development branch (loop_id)
- **Platform**: Linux 6.8.0-65-generic, Python 3.10, CUDA device 'NVIDIA GeForce RTX 3090'
- **Configuration**: Multi-task B_04_Dlinear backbone with 4 enabled tasks

## Impact Assessment

### Severity
- [x] High - Major functionality broken
- [ ] Critical - System unusable
- [ ] Medium - Feature impaired but workaround exists
- [ ] Low - Minor issue or cosmetic

### Affected Users
- Researchers attempting multi-task learning experiments
- Users with limited GPU memory (models requiring >12GB VRAM)
- Anyone running the multi-task ISFM configuration

### Affected Features
- Multi-task model training efficiency
- Memory-constrained training environments
- Loss computation accuracy for regression tasks
- Model deployment and inference performance

## Additional Context

### Error Messages
```
  | Name    | Type  | Params | Mode 
------------------------------------------
0 | network | Model | 3.2 B  | train
------------------------------------------
3.2 B     Trainable params
0         Non-trainable params
3.2 B     Total params
12,899.069Total estimated model params size (MB)

/home/lq/.conda/envs/P/lib/python3.10/site-packages/torch/nn/modules/loss.py:610: UserWarning: Using a target size (torch.Size([])) that is different to the input size (torch.Size([128])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
```

### Screenshots/Media
PyTorch Lightning training output shows unexpectedly large parameter count and tensor shape warnings.

### Related Issues
- Recent ISFM task head refactoring may have introduced parameter multiplication
- Multi-task head initialization could be creating redundant parameters
- Loss function shape handling needs verification across all task types

## Initial Analysis

### Suspected Root Cause
1. **Parameter Explosion**: Multi-task setup may be instantiating multiple copies of large components (backbone, embedding) instead of sharing them
2. **Shape Mismatch**: Task heads producing scalar outputs while loss functions expect vector targets, or vice versa
3. **Configuration Issue**: Model hyperparameters (d_model, hidden dimensions) set too high in configuration

### Affected Components
- `src/model_factory/ISFM/M_01_ISFM.py` - Main ISFM model class with multi-task head management
- `src/model_factory/ISFM/task_head/` - Individual task head implementations
- `src/task_factory/task/In_distribution/multi_task_phm.py` - Multi-task training logic
- `script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml` - Configuration file
- Loss function implementations in task or trainer components

## Priority Actions Required
1. **Immediate**: Analyze parameter count breakdown to identify source of 3.2B parameters
2. **High**: Fix tensor shape mismatches in loss computation
3. **Medium**: Optimize model architecture for memory efficiency
4. **Low**: Add parameter count validation and warnings for unexpectedly large models