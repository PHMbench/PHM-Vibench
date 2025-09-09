# Bug Report

## Bug Summary
Signal prediction task fails with tensor dimension mismatch error during multi-task PHM training. The error "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2" occurs when computing MSE loss between predicted and target signals with mismatched channel dimensions.

## Bug Details

### Expected Behavior
Signal prediction task should reconstruct the input signal with matching dimensions. If input has shape `[batch_size, sequence_length, channels]`, the output should have the same shape to enable proper MSE loss computation.

### Actual Behavior  
The model outputs signal prediction tensor with shape `[16, 4096, 2]` while the target signal (input) has shape `[16, 4096, 3]`, causing tensor dimension mismatch during loss computation.

### Steps to Reproduce
1. Run multi-task PHM training with signal_prediction task enabled
2. Use dataset with 3-channel input signals (e.g., vibration data with XYZ components)
3. Observe tensor dimension mismatch error during training step
4. Training fails with RuntimeError about tensor size mismatch

### Environment
- **Version**: PHM-Vibench current version
- **Platform**: Linux with PyTorch 2.6.0
- **Configuration**: Multi-task training with signal_prediction enabled

## Impact Assessment

### Severity
- [x] High - Major functionality broken

### Affected Users
Users running multi-task PHM training that includes signal_prediction task with multi-channel input data.

### Affected Features
- Multi-task training with signal_prediction
- Signal reconstruction tasks
- MSE loss computation for signal prediction

## Additional Context

### Error Messages
```
ERROR: signal_prediction train step failed - RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2
  Task output shape: torch.Size([16, 4096, 2])
ERROR: signal_prediction val step failed - RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 2
  Task output shape: torch.Size([16, 4096, 2])
```

### Screenshots/Media
Training logs show repeated failures with consistent tensor shape mismatch pattern.

### Related Issues
This may be related to configuration of max_out parameter in H_03_Linear_pred and how shape parameters are passed to task heads.

## Initial Analysis

### Suspected Root Cause
The H_03_Linear_pred task head is configured with a hardcoded or misconfigured max_out parameter that doesn't match the input signal's channel count. The model outputs 2 channels while input has 3 channels.

### Affected Components
- `src/model_factory/ISFM/M_01_ISFM.py` - Shape parameter preparation for signal_prediction
- `src/model_factory/ISFM/task_head/H_03_Linear_pred.py` - Prediction head output channels
- `src/model_factory/ISFM/task_head/multi_task_head.py` - Default shape parameter (96, 2)
- `src/task_factory/task/In_distribution/multi_task_phm.py` - Signal prediction loss computation

### Data Flow Analysis
1. Input signal: `[batch_size, 4096, 3]` (3 channels)
2. M_01_ISFM sets shape parameter: `(4096, 3)` based on input
3. H_03_Linear_pred task head outputs: `[batch_size, 4096, 2]` (only 2 channels)
4. Loss computation fails: MSE expects matching dimensions

## Key Technical Details

### Shape Parameter Flow
In `M_01_ISFM.py:151`:
```python
params['shape'] = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
```
This correctly extracts `(sequence_length, channels)` from input shape.

### Default Configuration Issue
In `multi_task_head.py:202`:
```python
shape = kwargs.get('shape', (96, 2))  # Default: predict 96 timesteps, 2 channels
```
The default assumes 2 channels, which may override the correct shape parameter.

### H_03_Linear_pred Configuration
In `H_03_Linear_pred.py:26`:
```python
max_out = getattr(args, "max_out", 3)  # Default allows up to 3 channels
```
The head should dynamically adjust to match input channels.