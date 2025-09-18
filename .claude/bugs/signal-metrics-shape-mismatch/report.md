# Bug Report

## Bug Summary
Multi-task training fails during metrics computation for signal_prediction task due to shape mismatch between model predictions (2 channels) and target signals (3 channels). Additional issues with system ID handling and RUL label validation during batch processing.

## Bug Details

### Expected Behavior
- Signal prediction metrics (MSE, MAE, R²) should compute successfully during training
- Multi-channel signals should be handled consistently across loss computation and metrics
- System ID differences should not prevent proper metrics aggregation
- RUL task should gracefully handle batches without valid labels

### Actual Behavior  
During multi-task training (Epoch 0), the following errors occur repeatedly:

```
Warning: Failed to compute train_mse for signal_prediction: Predictions and targets are expected to have the same shape, but got torch.Size([65536, 2]) and torch.Size([65536, 3]).
Warning: Failed to compute train_mae for signal_prediction: Predictions and targets are expected to have the same shape, but got torch.Size([65536, 2]) and torch.Size([65536, 3]).
Warning: Failed to compute train_r2 for signal_prediction: Predictions and targets are expected to have the same shape, but got torch.Size([65536, 2]) and torch.Size([65536, 3]).
```

Additional warnings:
- "Warning: No valid RUL labels in current batch, skipping RUL task"
- "不同system的 id不同" (Different system IDs are different)

### Steps to Reproduce
1. Set up multi-task PHM training with signal_prediction enabled
2. Use a dataset with 3-channel vibration signals (XYZ components)
3. Configure model with memory constraints (max_out=2) 
4. Start training with multi-task configuration
5. Observe metrics computation failures during training

### Environment
- **Version**: PHM-Vibench v0.2.0-alpha
- **Platform**: Linux with CUDA-capable GPU
- **Configuration**: Multi-task training with ISFM M_01 model using H_03_Linear_pred head

## Impact Assessment

### Severity
- [x] High - Major functionality broken

### Affected Users
- Researchers using multi-task training with signal prediction
- Users with multi-channel vibration datasets (3+ channels)
- Anyone using memory-constrained configurations with signal reconstruction

### Affected Features
- Signal prediction metrics computation
- Multi-task training progress monitoring
- System-level performance tracking
- RUL prediction in mixed-label scenarios

## Additional Context

### Error Messages
```
Warning: Failed to compute train_mse for signal_prediction: Predictions and targets are expected to have the same shape, but got torch.Size([65536, 2]) and torch.Size([65536, 3]).
Warning: Failed to compute train_mae for signal_prediction: Predictions and targets are expected to have the same shape, but got torch.Size([65536, 2]) and torch.Size([65536, 3]).
Warning: Failed to compute train_r2 for signal_prediction: Predictions and targets are expected to have the same shape, but got torch.Size([65536, 2]) and torch.Size([65536, 3]).
Epoch 0: 100%|█████████▉| 1631/1636 [01:04<00:00, 25.29it/s, v_num=tr81]Warning: No valid RUL labels in current batch, skipping RUL task
Epoch 0: 100%|██████████| 1636/1636 [01:04<00:00, 25.27it/s, v_num=tr81]  不同system的 id不同
```

### Screenshots/Media
Training progress shows successful completion but with repeated metric computation warnings.

### Related Issues
- Previous signal prediction dimension mismatch fix in loss computation
- System-specific metrics tracking implementation
- Memory optimization constraints (max_out=2 in H_03_Linear_pred)

## Initial Analysis

### Suspected Root Cause
1. **Metrics Computation Gap**: The dimension mismatch fix was applied to loss computation but not to metrics computation in `_compute_task_metrics()` method
2. **Inconsistent Channel Handling**: Loss computation truncates targets to 2 channels, but metrics still receive 3-channel targets
3. **System ID Validation**: Different system IDs within batches may cause issues with aggregated metrics

### Affected Components
- `src/task_factory/task/In_distribution/multi_task_phm.py` - `_compute_task_metrics()` method
- `src/model_factory/ISFM/task_head/H_03_Linear_pred.py` - Channel output limitation
- System metrics tracking code (recently implemented)

## Technical Context

### Previous Fixes Applied
- Loss computation now handles channel dimension mismatch by truncating targets
- Model output dispatch fixed to return dict for list inputs
- System-specific metrics tracking implemented

### Current Issue
The metrics computation (`_compute_task_metrics()`) still receives the original 3-channel targets while predictions are limited to 2 channels by the model's max_out constraint. This creates a mismatch specifically during metrics calculation, even though loss computation works correctly.

### Memory Constraint Background
The max_out=2 limitation exists in foundation model configs as a memory optimization to prevent GPU OOM errors. This constraint affects H_03_Linear_pred output channels but needs to be consistently handled across both loss and metrics computation.

---

**Next Steps**: Proceed to analysis phase to investigate the metrics computation code and develop a fix that maintains consistency with the existing loss computation solution.